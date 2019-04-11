"""
parameters.py
=============
Copyright 2013-2019 Sebastian Kamann

This file is part of PampelMuse.

PampelMuse is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

PampelMuse is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PampelMuse. If not, see <http://www.gnu.org/licenses/>.


+++
Provides
--------
The class pampelmuse.core.parameters.Parameters

Purpose
-------
The module provides a general framework for dealing with fit parameters, such
as the parameters that define the PSF profile or the parameters of the
coordinate transformation from reference to IFS coordinates.

Latest Git revision
-------------------
2019/04/11
"""
import logging
import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Legendre, Hermite, Polynomial
from astropy.io import fits
from ..utils.statistics import robust_fit


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20190411


logger = logging.getLogger(__name__)


class Parameters(object):
    """
    Class to handle the values for a set of parameters.
    
    Each instance of the class stores the parameters as a pandas data frame.
    The dispersion information is stored in the individual rows of the data
    frame, while the individual parameters are stored in its columns. As each
    parameter can have a value and a polynomial fit, the columns use a pandas
    MultiIndex.

    The Parameters-class defines methods to read in the parameters, modify
    them, access them and store them as a FITS-table. The set of parameters
    must be provided when a new instance is initialized.  It is currently not
    possible to add parameters afterwards.
    """

    def __init__(self, data, free, **kwargs):
        """
        Initialize a new instance of the Parameters class.
        
        data : pandas.DataFrame
            The data used to initialize a new instance should be provided as a
            pandas DataFrame containing one parameter per column.
        free : array_like,
            A boolean array containing a flag for each parameter in 'data'
            indicating if the parameter is allowed to vary in a fit
        kwargs
            This method does not support any additional keyword arguments.
        """
        # make sure no unsupported arguments were provided
        if kwargs:
            logger.error('Unsupported keyword argument(s): {0}'.format(', '.join(kwargs.keys())))
            return

        # Sanity check on provided data
        if not isinstance(data, pd.DataFrame):
            logger.error('Parameter "data" must be pandas DataFrame, not {0}.'.format(type(data)))
            return

        # check if data were provided with polynomial representations. If not, reindex data
        if isinstance(data.columns, pd.MultiIndex):
            # if so, make sure it has right format
            if len(data.columns.levshape) != 2:
                logger.error('Invalid column format in provided data.')
                logger.error('MultiIndex must have depth=2.')
                return
            elif data.columns.levshape[1] != 2:
                logger.error('Invalid column format in provided data.')
                logger.error('Second level of MultiIndex must have length=2')
                return
            elif 'value' not in data.columns.levels[1].values or 'fit' not in data.columns.levels[1].values:
                logger.error('Invalid column format in provided data.')
                logger.error('Second level of MultiIndex must contain "value" and "fit".')
                return
            self.data = data
        else:
            # first add new level for type: value or fit
            _columns = pd.MultiIndex.from_product([data.columns, ['value']])
            _data = data.reindex(columns=_columns, level=0)

            # then add new columns for polynomial fits
            columns = pd.MultiIndex.from_product([data.columns, ['fit', 'value']])
            self.data = _data.reindex(columns=columns)

        # if no information on free parameters was provided, assume none is free
        if free is not None:
            self.free = pd.Series(free, index=self.names)
        else:
            self.free = pd.Series(np.zeros(self.n_parameters, dtype=np.bool), index=self.names)

        # check if at least one parameter was provided.
        if len(self.names) == 0:
            logger.error("Cannot initialize instance of Parameters - no data provided.")
            logger.error("Either use 'hdu' or the keyword arguments to provide parameters.")
            return

        # check if data frame is sufficiently ordered for slicing
        # check: http://pandas.pydata.org/pandas-docs/stable/advanced.html#using-slicers
        if not self.data.columns.is_lexsorted():
            logger.warning('Data frame containing parameters not sorted. Reordering ...')
            self.data = self.data.sort_index(level=0, axis=1)  # axis=1 = columns

    @classmethod
    def from_hdu(cls, hdu, **kwargs):
        """
        Initialises a new instance of the Parameters-class from a FITS table
        HDU.
        
        Parameters
        ----------
        hdu : instance of fits.table.BinTableHDU
            The table must at least contain columns named 'name', 'value', and
            'freepar', providing the names, values, and variability flags of
            the various parameters, respectively. Additional fields that are
            used when available are 'isfitted' and 'polyfit'. The former
            indicates if the parameter behaviour with wavelength has been
            modelled with a polynomial. If so, the code assumes the polynomial
            fit to be contained in 'polyfit'.
        kwargs
            Any additional keyword arguments are passed on to the
            initialisation sequence of the class.

        Returns
        -------
        instance of Parameters
            The new instance of the class.
        """
        if "EXTNAME" in hdu.header:
            logger.info("Initialising parameters from HDU '{0}' ...".format(hdu.header["EXTNAME"]))
        else:
            logger.info("Initializing parameters from unnamed HDU ...")

        # get names of all parameters
        column_names = hdu.columns.names
        logger.info('Found the following columns: {0}'.format(', '.join(column_names)))

        # check if required fields 'name' and 'value exist':
        for field in ['name', 'value', 'freepar']:
            if field not in column_names:
                logger.error('Cannot find required field "{0}" in provided HDU.'.format(field))
                return None

        # get values of all parameters, make sure it has correct shape (even when no dispersion information available)
        hdu_data = hdu.data
        data = np.atleast_2d(hdu_data['value'].T)  # has shape (n_dispersion, n_parameters)

        # define indices for data frame
        index = pd.Index(np.arange(data.shape[0]))

        # check if polynomial fits are available
        contains_polynoms = 'isfitted' in column_names and 'polyfit' in column_names

        if contains_polynoms:
            # if polynomial fits are included, add them to the data array (append a new axis)
            data = np.dstack((np.atleast_2d(hdu_data['polyfit'].T, data)))  # has shape (n_dispersion, n_parameters, 2)

            # must define a pandas MultiIndex: 0th level - parameter 1st level: value or fit
            columns = pd.MultiIndex.from_product([hdu_data['name'], ['fit', 'value']], names=['parameter', 'type'])
        else:
            # otherwise a plain pandas Index is sufficient for the columns
            columns = column_names

        # initialise pandas DataFrame. For this, the shape of data must be (n_dispersion, n_columns)
        data_frame = pd.DataFrame(data.reshape(data.shape[0], -1), index=index, columns=columns)

        # get information on free parameters
        free = hdu_data['freepar']

        # return instance
        return cls(data=data_frame, free=free, **kwargs)

    @classmethod
    def from_dictionary(cls, parameters, free=None):
        """
        parameters : dict
            The dictionary containing the parameter names as keys and their
            values as values.
        free : array_like,
            A boolean array containing a flag for each parameter in 'data'
            indicating if the parameter is allowed to vary in a fit.

        Returns
        -------
        instance of Parameters
            The new instance of the class.
        """
        return cls(pd.DataFrame.from_dict(parameters), free=free)

    @property
    def names(self):
        """
        Returns
        -------
        pandas.Index
            The names of the parameters defined in the instance of Parameters.
        """
        return self.data.columns.levels[0]

    @property
    def n_dispersion(self):
        """
        Returns
        -------
        int
            The number of dispersion elements defined in the instance of
            Parameters.
        """
        return self.data.shape[0]

    @property
    def n_parameters(self):
        """
        Returns
        -------
        int
            The number of parameters defined in the instance of Parameters.
        """
        return len(self.names)

    @property
    def mask(self):
        """
        Returns
        -------
        pandas.DataFrame
            Boolean mask of same size as data array indicating which data
            values are invalid, i.e. NaN.
        """
        return np.isnan(self.data)

    @property
    def defined(self):
        """
        Returns
        -------
        pandas.Series
            A series of length Parameters.n_parameters that specifies if a
            value for each parameter has been defined.
        """
        _defined = (~self.mask.loc[:, (slice(None), 'value')]).any(axis=0)
        # note that the Series will still have a MultiIndex with the first level containing only 'value'.  Remove it:
        _defined.index = _defined.index.droplevel(1)
        return _defined

    @property
    def fitted(self):
        """
        Returns
        -------
        pandas.Series
            A series of length Parameters.n_parameters that specifies if the
            values of a parameter have been fitted.
        """
        _fitted = (~self.mask.loc[:, (slice(None), 'fit')]).all(axis=0)
        # note that the Series will still have a MultiIndex with the first level containing only 'fit'.  Remove it:
        _fitted.index = _fitted.index.droplevel(1)
        return _fitted

    @property
    def contains_free_parameters(self):
        """
        Returns
        -------
        bool
            A flag indicating if any parameter should be varied in a fit,
            i.e. is defined as a free parameter and has NOT been modeled
            with a polynomial.
        """
        return (self.free & ~self.fitted).any()

    @property
    def valid(self):
        """
        Returns
        -------
        boolean
            A boolean flag indicating if all parameters have valid data
            available, i.e. either have useful values or have been modeled
             with a polynomial.
        """
        return (self.fitted | self.defined).all()

    def interpolate(self):
        """
        This method replaces NaN values (if any) in the data array via
        interpolation of the valid data along the dispersion axis.
        
        The interpolation is done independently for each parameter.
        """
        for parameter in self.names:
            if self.mask[(parameter, 'value')].all():
                logger.error('Cannot interpolate. Parameter "{0}" has no valid data.'.format(parameter))
            elif self.mask[(parameter, 'value')].any():
                valid_data = ~self.mask[(parameter, 'value')]
                self.data[(parameter, 'value')] = np.interp(
                    self.data.index, self.data.index[valid_data], self.data.loc[valid_data, (parameter, 'value')])

    def resize(self, index):
        """
        Change the length of the dispersion axis of the data.
         
        Newly added rows will be filled using data from the existing rows (of
        the same parameter and type) if possible. Otherwise, they will be set
        to NaN.

        Parameters
        ----------
        index : pandas.Index
            The new index used to label the data rows.
        """
        # issue warning if reindexing would remove existing data
        if not self.data.index.isin(index).all():
            logger.warning('Changing dispersion axis as requested will alter existing values.')

        self.data = self.data.reindex(index=index, method='nearest')
        # with method='nearest', the code will fill new rows of the data frame
        # using existing ones. If a column contains only NaN, new values will
        # also be filled with NaNs.

    def info(self, names=None):
        """
        Print out some information about the current status of the parameters,
        i.e. whether they have been defined, what is their (mean) value, are
        they free parameters, and have the been fitted with a smooth function.
        
        names : list, optional
            The parameter can be used to restrict the output to a subset of
            parameters or to print them in a given order. By default, all
            parameters are printed in random order.
        """
        logger.info(" Parameter  defined?  (mean)value  free?  fitted?")

        # check if names were provided; if not, use all available
        if names is None:
            names = self.names

        # loop over names
        for name in names:

            # skip if not defined in this instance
            if name not in self.names:
                continue

            # add name to current logging row
            log_string = " *{0:8s}".format(name)

            # check if parameter defined
            if self.defined[name]:
                log_string += "       [x]"
            else:
                log_string += "       [ ]          NaN    [ ]      [ ]"
                logger.info(log_string)
                continue

            # get (mean) value
            log_string += "{0:13.2f}".format(np.nanmean(self.data[(name, 'value')]))

            # check if parameter fixed/free
            if self.free[name]:
                log_string += "    [x]"
            else:
                log_string += "    [ ]"

            # check if parameter has been fitted with polynomial
            if self.fitted[name]:
                log_string += "      [x]"
            else:
                log_string += "      [ ]"

            logger.info(log_string)

    def polynomial_fit(self, name, order, mask=None, polynom="plain"):
        """
        Fit a parameter using a polynomial.
        
        Currently, the smooth function can be either a plain polynomial, a
        Chebyshev, a Legendre, or a Hermite polynomial. A mask may be
        provided to exclude certain data from the fit.
        
        Parameters
        ----------
        name : string
            The name of the parameter that should be fitted.
        order : int
            The order of the polynomial that should be fitted to the data.
        mask : pandas.Series, optional
            An optional bad-pixel mask. If specified, it must be a boolean
            pandas Series that uses the same indices as the data. Bad pixels
            should be set to True, good ones to False.
        polynom : str, optional
            The function used in the fitting. May be either 'plain',
            'legendre', 'hermite', or 'chebyshev'.
        """
        # sanity check on provided parameter
        if name not in self.names:
            logger.error("Unknown parameter name '{0}' provided, cannot fit data.".format(name))
            return

        # The fitting function that are currently available:
        available = {"chebyshev": Chebyshev, "legendre": Legendre, "hermite": Hermite, "plain": Polynomial}

        # check if fit function is valid:
        if not polynom.lower() in available.keys():
            logger.error("Unknown fitting function '{0}' provided., cannot fit data.".format(polynom))
            return
        else:
            fit_function = available[polynom.lower()]

        abscissa = self.data.index

        # create pseudo-mask if None provided
        if mask is not None:
            mask |= self.mask[(name, 'value')]
        else:
            mask = self.mask[(name, 'value')]

        # carry out fit
        fit = fit_function.fit(abscissa[~mask], self.data.loc[~mask, (name, 'value')], order)
        self.data[(name, 'fit')] = fit(abscissa)

    def make_hdu(self, include_fit=True, header_keys=None):
        """
        Write the current state of the parameters (names, values, status, ...)
        to a FITS table HDU.
        
        The HDU that is created contains a binary table with at least 3
        columns, 'name', 'value', and 'freepar'. The parameter names are
        written as plain strings to 'name', their values are written as
        either floats or arrays of floats (depending on whether dispersion
        information is available) to 'values', and 'freepar' contains a
        boolean flag indicating whether the parameter should be optimized
        in a fit.
        
        Optionally, two additional columns can be included in the table:
        'isfitted', containing a boolean flag for each parameter that
        indicates if the parameter values have been fitted with a smooth
        function, and 'polyfit' containing that function (if available).
        
        Parameters
        ----------
        include_fit : boolean, optional
            Whether or not the fits to the parameters should  be included
            in the table.
        header_keys : dict, optional
            Any additional keywords that should be included in the FITS
            header of the  table.
            
        Returns
        -------
        astropy.io.fits.BinTableHDU instance
            The HDU containing the parameters.
        """
        # check if dispersion information exits or not, and adapt column format
        if self.n_dispersion == 1:
            data_format = "D"
        else:
            data_format = "{0}D".format(self.n_dispersion)

        # for output, get data as plain numpy array with shape (n_dispersion, n_parameters, 2)
        # the first index of the last axis contains the fits, the second one the values
        data = self.data.values.reshape((self.n_dispersion, self.n_parameters, 2))
        # -> (has shape n_dispersion, n_parameters, 2)

        # collect columns 'name', 'value', and 'freepar' that are always included in  the HDU
        out_columns = [fits.Column(name="name", format="10A", array=self.names),
                       fits.Column(name="value", format=data_format, array=data[..., 1].swapaxes(0, 1).squeeze()),
                       fits.Column(name="freepar", format="L", array=self.free)
                       ]

        # The columns 'isfitted' and 'polyfit' are only included upon request
        if include_fit:
            out_columns.extend([
                fits.Column(name="isfitted", format="L", array=self.fitted),
                fits.Column(name="polyfit", format=data_format, array=data[..., 0].swapaxes(0, 1).squeeze()),
            ])

        # Store additional keyword arguments (if any) in the FITS header
        cards = []
        if header_keys is not None:
            for key, value in header_keys.items():
                cards.append(fits.Card(key=key.upper(), value=value))

        # return HDU
        return fits.BinTableHDU.from_columns(out_columns, header=fits.Header(cards=cards))
