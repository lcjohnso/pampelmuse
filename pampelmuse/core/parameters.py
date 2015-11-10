"""
parameters.py
=============

Provides
--------
The class pampelmuse.core.parameters.Parameters

Purpose
-------
The module provides a general framework for dealing with fit parameters,
such as the parameters that define the PSF profile or the parameters of
the coordinate transformation from reference to spaxel coordinates. The
general idea is to store these parameters as a three-dimensional numpy
array. The first axis of this arrays is the dispersion axis, the second
axis represents the parameters, and the third axis always has length 2
and contains the parameters value under index 0 and a potential fit of
it under index 1.
In the class Parameters, methods are defined to read in the parameters,
modify them, access them and store them as a FITS-table. The set of
parameters must be provided when a new instance is initialized.  It is
currently not possible to add parameters afterwards.


Latest SVN revision
-------------------
281, 2015/11/10
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 281

# import required python modules
import logging
import re

import numpy as np
from numpy.polynomial import Chebyshev, Legendre, Hermite, Polynomial
from astropy.io import fits


class Parameters(object):
    """
    Class to handle the values for a set of parameters. The values are contained
    in a 3dim. numpy array. The first axis is the dispersion axis, the second the
    parameter axis, and the third one contains the values under the first index
    and a fit to them (if available) under the second index.
    """

    def __init__(self, hdu=None, order=None, **kwargs):
        """
        Initialize a new instance of the Parameters class.
        
        hdu : an astropy.io.fits.BinTableHDU instance, optional
            The FITS-Table HDU that contains the Parameters. The table must contain
            fields named 'name' and 'value', providing the names and values, resp.,
            of the individual parameters that define the profile. Additional fields
            that are used when available are 'freepar', indicating whether the para-
            meter should be varied when the PSF is fitted, 'isfitted', indicating
            whether the parameter has been fitted as a function of wavelength, and
            'polyfit', containing the fit (if any).
        order : list, optional
            In case it is required that the parameters enter the structure in a
            certain order, a list indicating this order may be provided. By default,
            the parameters are unsorted.
        kwargs
            The keyword arguments may also be used to provide names and values for
            the individual parameters. Note that values provided here supersede
            those in the FITS table. If no FITS table is provided, at least one
            parameter and its value must be provided.
        """
        # Initial definitions:
        self.hdu = hdu
        self.ndisp = 1  # The number of data points in dispersion direction
        self._data = None  # The parameter values (and potentially smooth representations in dispersion direction)

        if order is not None:  # The names of the parameters, check if any order is requested
            names = order
            for name in kwargs.keys():
                if name not in names:
                    names.append(name)
        else:
            names = kwargs.keys()

        # make sure 'hdu' is a fits.BinTableHDU if provided
        if hdu is not None and not isinstance(hdu, fits.BinTableHDU):
            logging.critical("If provided, 'hdu' must be a FITS Table-HDU.")
            assert isinstance(hdu, fits.BinTableHDU), "Invalid type for argument 'hdu': {0}".format(
                re.findall(r"'(.*)'", str(type(hdu)))[0])

        # read FITS Table if it has been provided:
        if hdu is not None:
            if "EXTNAME" in hdu.header:
                logging.info("Reading parameters from HDU '{0}'...".format(hdu.header["EXTNAME"]))
            else:
                logging.info("Reading parameters from unnamed HDU...")

            # check if required fields 'name' and 'value exist':
            for field in ["name", "value"]:
                if field not in hdu.data.names:
                    logging.critical("Cannot find field '{0}' in provided HDU.".format(field))
                    assert field in hdu.data.names, "Required field '{0}' missing in HDU.".format(field)

            # get dispersion information using the TFORM{i} header key (where i is the index of the field 'values')
            for i in xrange(hdu.header["TFIELDS"]):
                if hdu.header["TTYPE{0}".format(i + 1)] == "value":
                    size = re.findall(r"([0-9]*)[A-Z]", hdu.header["TFORM{0}".format(i + 1)])[0]
                    if size != "":
                        self.ndisp = int(size)
                    break

            # parameters can either be defined
            self.names = list(np.union1d(names, hdu.data["name"]))

            # initialize the data array
            self._data = np.zeros((self.ndisp, len(self.names), 2), dtype=np.float32)
            self._data[:] = np.nan  # no values have been provided so far

            # initialize a boolean array indicating if a parameter is free, i.e. should be optimized in a fit.
            self.free = np.zeros(self._data.shape[1], dtype=np.bool)

            # read the parameter values into the structure
            for i in xrange(hdu.data.size):
                name = hdu.data.field("name")[i]

                if name not in self.names:
                    self.names.append(name)
                j = self.names.index(name)

                self._data[:, j, 0] = hdu.data.field("value")[i]
                if "freepar" in hdu.data.names:  # check for optional column 'freepar'
                    self.free[j] = hdu.data.field("freepar")[i]
                if "isfitted" in hdu.data.names:  # check for optional column 'isfitted'
                    if hdu.data.field("isfitted")[i]:
                        if "polyfit" in hdu.data.names:  # check for optional column 'polyfit'
                            self._data[:, j, 1] = hdu.data.field("polyfit")[i]
                        else:
                            logging.error("Cannot find fit to parameter '{0}'.".format(name))

                logging.info("Added parameter '{0}' from HDU to the structure.".format(name))
        else:
            self.names = names
            self._data = np.zeros((1, self.npars, 2), dtype=np.float32)
            self._data[:] = np.nan
            self.free = np.zeros((self.npars,), dtype=np.bool)

        # iterate over provided parameters
        for name, value in kwargs.iteritems():

            # get number of spectral pixels
            ndisp = len(np.atleast_1d(value))  # use np.atleast_1d to get len=1 for single values

            # get the index of the parameter in the array
            j = self.names.index(name)

            # add the parameter values(s) to the data. If the data does not contain dispersion information
            # but the parameter does, the data is expanded in dispersion direction
            if self.ndisp == 1 and ndisp > 1:
                self.resize(ndisp)
            elif self.ndisp > 1 and ndisp > 1 and self.ndisp != ndisp:
                logging.error("Data array of keyword parameter '{0}' has invalid shape.".format(name))
            self._data[:, j, 0] = value

        # check if at least one parameter was provided.
        if self.npars == 0:
            logging.critical("Cannot initialize instance of Parameters - no data provided.")
            logging.critical("Either use 'hdu' or the keyword arguments to provide parameters.")
        assert self.npars > 0, "Cannot create an empty instance of Parameters."

    @property
    def npars(self):
        """
        Returns
        -------
        npars : int
            The number of parameters in the definition of the PSF profile.
        """
        return len(self.names)

    def set_data(self, value):
        """
        Set the data values for the parameters. This method sets the internal
        parameter '_data'. To ensure that it stays valid, a number of checks
        are performed on the provided value.
        
        Parameters
        ----------
        value : array_like
            The data values of the parameters, provided as a numpy array. The first
            axis should be the dispersion axis (can be omitted for monochromatic
            data), the second axis must have a length equal to the number of para-
            meters of the profile. The third axis (if any), must be of length 1 or 2,
            depending on whether only data are provided or data and fits.
            Note that if for 2dim. the length of both axes is equal to the number of
            parameters, it is assumed that the second axis is the parameters-axis.
        """
        # sanity check: either a list or a numpy array must be provided
        assert isinstance(value, (list, np.ndarray)), "'data' must be a list or a numpy array, not {1}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.atleast_1d(value)

        # sanity check: data must be 1D, 2D, or 3D.
        assert value.ndim in [1, 2, 3], "'data' must have dimensionality 1, 2, or 3, but has {0}".format(value.ndim)

        # 1dim. data provided, assume one value provided per parameter, no fit, no dispersion information
        if value.ndim == 1:
            assert value.size == self.npars, "Shape of 'data' does not match no. of parameters in profile."
            self._data = np.ones((1, self.npars, 2), dtype=np.float32)
            self._data[..., 0] = value
            self._data[..., 1] = np.nan

        # 2dim. data provided, either no fit (if length of second axis equals number of parameters in profile) or no
        # dispersion information provided (else)
        if value.ndim == 2:
            assert value.shape[0] == self.npars or value.shape[1] == self.npars,\
                "Shape of 'data' does not match no. of parameters in profile."
            if value.shape[1] == self.npars:  # no fit provided, second axis has same length as no. of parameters
                self._data = np.ones((value.shape[0], self.npars, 2), dtype=np.float32)
                self._data[..., 0] = value
                self._data[..., 1] = np.nan
            else:  # no dispersion information provided, first axis has same length as no. of parameters
                assert value.shape[1] in [1, 2],\
                    "Second axis of 'data' must have length 1 (only data) or 2 (data and fit)."
                assert value.shape[0] == self.npars, "Shape of 'data' does not match no. of parameters in profile."
                self._data = np.ones((1, self.npars, 2), dtype=np.float32)
                if value.shape[1] == 1:
                    self._data[..., 0] = value
                    self._data[..., 1] = np.nan
                else:
                    self._data[:] = value

        # 3dim. data provided, includes fit and dispersion information
        if value.ndim == 3:
            assert value.shape[2] in [1, 2], "Last axis of 'data' must have length 1 (only data) or 2 (data and fit)."
            assert value.shape[1] == self.npars, "Shape of 'data' does not match no. of parameters in profile."
            self._data = np.zeros((value.shape[0], self.npars, 2), dtype=np.float32)
            self._data[:] = value

        # update dispersion information
        self.ndisp = self._data.shape[0]

    def get_data(self):
        """
        Returns
        -------
        data : ndarray
            The data values of the parameters that define the PSF profile. 
        """
        return self._data

    data = property(get_data, set_data, None, "The data values of the parameters in the definition.")

    @property
    def mask(self):
        """
        :return:
        Boolean array of same size as data array indicating which data values are
        invalid.
        """
        return np.isnan(self._data)

    @property
    def defined(self):
        """
        Returns
        -------
        defined : ndarray
            A 1dim. boolean array of length 'npars' that specifies if the
            value of each PSF parameter has been defined.
        """
        return (~self.mask[..., 0]).any(axis=0)

    @property
    def fitted(self):
        """
        Returns
        -------
        fitted : ndarray
            A 1dim. boolean array of length 'npars' that specifies if the
            value of each PSF parameter has been fitted.
        """
        return (~self.mask[..., 1]).all(axis=0)

    @property
    def contains_free_parameters(self):
        """
        Returns
        -------
        contains_free_parameters : boolean
            True or False depending on whether any of the parameters that
            define the profile should  be fitted, i.e. is a free parameter
            and has NOT been fitted with a smooth function.
        """
        return (self.free & ~self.fitted).any()

    @property
    def valid(self):
        """
        Returns
        -------
        valid : boolean
            True or False depending on whether for each parameter either values
            or a fit exists.
        """
        return (self.fitted | self.defined).all()

    def interpolate(self):
        """
        This method replaces NaN values (if any) in the data array via interpolation of
        the valid array entries along the first axis. This is done independently for each
        parameter, i.e. each chunk data[:, i, 0], with i >= 0 > npars.
        """
        for i in xrange(self.npars):
            if 0 < (~self.mask[:, i, 0]).sum() < self.ndisp:
                self.data[:, i, 0] = np.interp(np.arange(self.ndisp), (~self.mask[:, i, 0]).nonzero()[0],
                                               self.data[~self.mask[:, i, 0], i, 0])
            elif self.mask[:, i, 0].all():
                logging.error("Data for parameter '{0}' only contains NaNs, cannot interpolate.".format(self.names[i]))

    def resize(self, ndisp):
        """
        Change  the size of the array containing the parameter values in dispersion
        direction, i.e. along the first axis. Note that if the new length of the
        dispersion axis is smaller than the old length, the data in the last <old minus
        new> layers is lost. If the new length exceeds the old one, the new entries will
        be initialized with the values from the last layer
        
        Parameters
        ----------
        ndisp : int
            The new length of the dispersion axis.
        """
        # sanity checks on provided value:
        assert isinstance(ndisp, (int, np.int)), "'ndisp' must be an integer, not {0}.".format(
            re.findall(r"'(.*)'", str(type(ndisp)))[0])
        assert ndisp > 0, "'ndisp' must be >0 but is {0}".format(ndisp)

        # an array can only be resized in place if it owns its data.
        assert self._data.base is None, "Cannot resize array because it does not own its data."

        if self.ndisp == ndisp:
            return  # do nothing if old and new shape are the same
        elif self.ndisp > ndisp:
            logging.warning("Reducing size of parameter data. Some data will be lost.")

        self._data.resize((ndisp, self.npars, 2))

        # if the data array grew in size, fill additional entries with nearest values
        if self.ndisp < ndisp:
            self._data[self.ndisp:] = self._data[self.ndisp - 1]

        # update dispersion information
        self.ndisp = ndisp

    def info(self, names=None):
        """
        Print out some information about the current status of the parameters,
        i.e. whether they have been defined, what is their (mean) value, are
        they free parameters, and have the been fitted with a smooth function.
        
        names : list, optional
            The parameter can be used to restrict the output to a subset of
            parameters or to print them in a given order. By default, all para-
            meters are printed in random order.
        """

        logging.info(" Parameter  defined?  (mean)value  free?  fitted?")

        if names is None:
            names = self.names

        for name in names:
            if name not in self.names:
                continue

            i = self.names.index(name)

            logstring = " *{0:8s}".format(name)
            if self.defined[i]:
                logstring += "       [x]"
            else:
                logstring += "       [ ]          NaN    [ ]      [ ]"
                logging.info(logstring)
                continue

            logstring += "{0:13.2f}".format(float(self._data[~self.mask[:, i, 0], i, 0].mean(axis=-1)))

            if self.free[i]:
                logstring += "    [x]"
            else:
                logstring += "    [ ]"

            if self.fitted[i]:
                logstring += "      [x]"
            else:
                logstring += "      [ ]"

            logging.info(logstring)

    def provideParNames(self):
        """
        Provide the names of the parameters in the definition in the form of
        QStrings. This is needed to load the names in the combo box of the GUI
        of PampelMuse. Note that this method requires PyQt4 to be installed
        on the system.
        
        Returns
        -------
        names : PyQt4.QtCore.QStringList
            The names of the parameters that define the profile.
        """
        try:
            # check if PyQt4 is available on the system
            from PyQt4 import QtCore

            return QtCore.QStringList(self.names)
        except ImportError:
            logging.error("PyQt4 is not available on the system.")
            logging.error("Cannot provide names of PSF parameters as QStrings.")
            return None

    def polynomial_fit(self, name, order, mask=None, function="step", **kwargs):
        """
        Fit a parameter using a smooth function. Currently, the smooth function
        can be either a plain polynomial, a broken polynomial, or a Chebyshev
        polynomial. A mask may be provided to exclude certain data from the fit.
        
        Parameters
        ----------
        name : string
            The name of the parameter that should be fitted.
        order : int
            The order of the polynomial that should be fitted to the data. Note
            that for a broken polynomial, 'order' applies to each individual
            polynomial.
        mask : ndarray, optional
            If provided, it must be a 1dim. boolean array of the same length as
            the parameter value array. Each data point where 'mask' is True will
            be excluded from the fit.
        function : str, optional
            The function used in the fitting. May be either 'plain', 'legendre',
            'hermite', or 'chebyshev'.
        """
        # sanity check on provided parameter
        if name not in self.names:
            logging.error("Unknown parameter name '{0}' provided, cannot fit data.".format(name))
            return None

        # The index of the requested parameter
        i = self.names.index(name)

        # The fitting function that are currently available:
        available = {"chebyshev": Chebyshev, "legendre": Legendre, "hermite": Hermite, "plain": Polynomial}

        # check if fit function is valid:
        if not function.lower() in available.keys():
            logging.error("Unknown fitting function '{0}' provided., cannot fit data.".format(function))
            return None
        else:
            fitfunction = available[function.lower()]

        abscissa = np.arange(self.ndisp, dtype=np.float32)

        # create pseudo-mask if None provided
        if mask is not None:
            mask |= self.mask[:, i, 0]
        else:
            mask = self.mask[:, i, 0]
        print mask.sum()
        fit = fitfunction.fit(abscissa[~mask], self._data[~mask, i, 0], order)

        self._data[:, i, 1] = fit(abscissa)

    def make_hdu(self, include_fit=True, **kwargs):
        """
        Write the current definition of the parameters (names, values, status, ...) to
        a fits.BinTableHDU. The HDU that will be created contains a binary table with
        at least 3 columns, 'name', 'value', and 'freepar'. The parameter names are written
        as plain strings to 'name', their values are written as either floats or arrays of
        floats (depending on whether dispersion information is available) to 'values', and
        'freepar' contains a boolean flag indicating whether the parameter should  be
        optimized in a fit. Optionally, two additional columns may be included in the table:
        'isfitted', containing a boolean flag for each parameter that indicates if the
        parameter values have been fitted with a smooth function, and 'polyfit' containing
        that function (if available).
        
        Parameters
        ----------
        include_fit : boolean, optional
            Whether or not the fits to the parameters should  be included in the table.
        kwargs
            Any additional keyword arguments provided to this function will be included
            in the FITS header of the  table.
            
        Returns
        -------
        out_hdu : an  astropy.io.fits.BinTableHDU instance
            The HDU containing the parameters.
        """
        # check if dispersion information exits or not, and adapt column format
        if self.ndisp == 1:
            dataformat = "D"
        else:
            dataformat = "{0}D".format(self.ndisp)

        # collect columns 'name', 'value', and 'freepar' that are always included in  the HDU

        out_columns = [fits.Column(name="name", format="10A", array=self.names),
                       fits.Column(name="value", format=dataformat,
                                   array=np.squeeze(self._data[..., 0].swapaxes(0, 1))),
                       fits.Column(name="freepar", format="L", array=self.free)
                       ]

        # The columns 'isfitted' and 'polyfit' are only included upon request
        if include_fit:
            out_columns.extend([fits.Column(name="isfitted", format="L", array=self.fitted),
                                fits.Column(name="polyfit", format=dataformat,
                                            array=np.squeeze(self._data[..., 1].swapaxes(0, 1)))])

        # Store additional keyword arguments (if any) in the FITS header
        cards = []
        for key, value in kwargs.iteritems():
            cards.append(fits.Card(key=key.upper(), value=value))

        # return HDU
        out_hdu = fits.new_table(out_columns, header=fits.Header(cards=cards))
#         out_hdu.add_checksum()

        return out_hdu            
