"""
sources.py
==========
Copyright 2013-2018 Sebastian Kamann

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


Provides
--------
class pampelmuse.core.sources.Sources

Purpose
-------
The module contains the class 'Sources' that is used by PampelMuse to handle
a catalogue of sources. It keeps track of the loaded sources, their grouping
into the individual components, their spectra and positions. It also provides
some methods to write the individual properties into FITS-HDUs.

The class 'Sources' is designed so that it handles the available memory
efficiently. For this reason, when the spectra or the IFS coordinates of the
sources are provided via FITS HDUs, the data are not automatically loaded to
memory but only on request.

Latest Git revision
-------------------
2018/01/18
"""
import collections
import logging
import time
import warnings
import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Legendre, Hermite, Polynomial
from astropy.io import fits
from astropy.table import Table
from ..core.coordinates import Transformation
from ..filters import get_filter

# check if PyQt is available 
try:
    from PyQt5 import QtCore
except ImportError:
    QtCore = None


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20180118


logger = logging.getLogger(__name__)


class Sources(object):
    """
    The class is designed for the following purposes:
    * Initialize and maintain a coherent catalogue of sources. To this aim,
      the class uses several properties that are either inherent to the sources
      (ID, x, y, mag, ra, dec) or assigned to each source depending on its
      usage in the analysis (status, specrow, primary)
    * Load, maintain and save the set of spectra that is associated with the
      source catalog
    * Load, maintain and save the set of IFU coordinates that is associated
      with the source catalog.
    """
    # available functions for polynomial fits:
    available_polynoms = collections.OrderedDict()
    available_polynoms["chebyshev"] = Chebyshev
    available_polynoms["legendre"] = Legendre
    available_polynoms["hermite"] = Hermite
    available_polynoms["plain"] = Polynomial

    # for writing to/reading from FITS files, each polynomial is assigned an index
    polyident = collections.OrderedDict(zip(available_polynoms.keys(), np.arange(len(available_polynoms))))

    def __init__(self, catalog):
        """
        Initialize a new instance of the `Sources`-class.

        Parameters
        ----------
        catalog : a pandas.DataFrame
            The source catalog that should be used to initialize the instance.
            It is assumed that the index is equal to the source IDs. Other
            columns are optional and the corresponding properties might be
            defined later.
        """
        warnings.simplefilter("always", DeprecationWarning)  # By default this type of warning is ignored

        # open the data, check if provided as DataFrame
        if not isinstance(catalog, pd.DataFrame):
            catalog = pd.DataFrame(catalog)

        # use source IDs as index if they exist
        if 'id' in catalog.columns:
            self.catalog = catalog.set_index('id', drop=False)  # Note that astropy does not store the index column!
        else:
            self.catalog = catalog

        # assert that data frame is NOT a view of another one.
        if self.catalog.values.base.size != self.catalog.values.size:
            logger.warning('Need to copy catalog as it is a view to another pandas.DataFrame.')
            self.catalog = self.catalog.copy()

        # check that source IDs are unique
        if self.ids.duplicated().any():
            logger.critical('Provided source catalog contains {0} duplicated IDs.'.format(self.ids.duplicated().sum()))
            return

        # check if other columns are available - if not, add them and fill with NaN
        for column in ['x', 'y', 'ra', 'dec', 'mag']:
            if column not in self.catalog.columns:
                dtype = np.float64 if column in ['ra', 'dec'] else np.float32
                self.catalog[column] = pd.Series(np.zeros(self.catalog.shape[0], dtype=dtype),
                                                 index=self.catalog.index)
                self.catalog[column] = np.nan

        # if available, get status of each source in analysis
        if 'status' not in self.catalog.columns:
            self.catalog.loc[:, 'status'] = pd.Series(np.zeros((self.n_sources,), dtype=np.int16),
                                                      index=self.catalog.index)
            self.catalog.at[self.ids < 0, 'status'] = 4  # assume IDs<0 are used for background components

        # if available, get primary source of each component, otherwise assume each source has its own component
        if 'primary' not in self.catalog.columns:
            self.catalog.loc[:, 'primary'] = self.ids

        self._weights = None  # a 1dim. array of length Sources.nsrc containing the relative contribution of each
        # source to its component. This attribute is determined internally when the first call to its getter method is
        # made.

        # initialize coordinate transformation
        self.transformation = Transformation()

        # initialize arrays that will contain important information about source spectra and coordinates:
        # (a) whether a spectrum has been loaded for a component;
        #     a boolean series will be created by Sources.load_spectra()
        self.spectrum_loaded = None

        # (b) a boolean series indicating if IFS coordinates have been loaded for a source
        self.ifs_coordinates_loaded = pd.Series(np.zeros((self.n_sources,), dtype=np.bool), index=self.ids)

        # (c) a boolean series indicating if a polynomial fit has been performed to the IFU coordinates of a source
        self.ifs_coordinates_fitted = pd.Series(np.zeros((self.n_sources,), dtype=np.bool), index=self.ids)

        # (d) a boolean series indicating if the IFU coordinates of a source are a free fit parameter
        self.ifs_coordinates_free = pd.Series(np.zeros((self.n_sources,), dtype=np.bool), index=self.ids)

        # (e) a dictionary of ID:<function> pairs that will contain callable functions that return the polynomial fits
        #  to the positions of a source, obtained using Sources.fitIfsCoordinates()
        self.ifs_coordinates_fit_functions = {}

        # place-holders for data arrays
        self._wave = None  # The internal variable for the (global) wavelength array
        self.data = None  # The spectra of the components and their uncertainties
        self.ifs_coordinates = None  # The x/y coordinates of the sources on the IFS spaxel array

        # place-holders for HDUs containing fluxes or IFS coordinates
        self.data_hdu = None
        self.data_wave_hdu = None
        self.ifs_coordinates_data_hdu = None
        self.ifs_coordinates_fit_hdu = None

        # Initialize the dispersion information
        self.n_dispersion = 1
        self.crval = 1.
        self.crpix = 1
        self.cdelt = 1.
        self.ctype = None
        self.cunit = None

    @property
    def n_sources(self):
        """
        Returns
        -------
        n_sources : int
            The total number of sources in the catalogue (incl. background
            components). 
        """
        return self.catalog.shape[0]

    @property
    def ids(self):
        """
        Returns
        -------
        ids : pandas.Index
            The IDs of the sources in the catalog.
        """
        return self.catalog.index

    def _check_consistency(self, value):
        """
        Checks the consistency of updated source properties before they are
        written to the source catalog.
        
        Parameters
        ----------
        value : array-like or pandas Series
            The updated values of a source property

        Returns
        -------
        value : pandas.Series
            The input value as a pandas Series with correct indices

        Raises
        ------
        IOError
            If value cannot be broadcast to an array.
        ValueError
            If value has an incorrect shape.
        """
        if not isinstance(value, (list, np.ndarray, pd.Series)):
            raise IOError('Expected pandas Series or array-like variable, got {0}'.format(type(value)))
        if not isinstance(value, pd.Series):
            if len(value) != self.n_sources:
                raise ValueError('Array-size does not match number of sources in catalog.')
            else:
                return pd.Series(value, index=self.catalog.index)
        else:
            return value

    # Reference x-coordinates
    @property
    def x(self):
        """
        Returns
        -------
        x : pandas.Series
            A 1dim. of length Sources.n_sources containing the coordinate of
            each source along the x-axis of the reference system.
        """
        return self.catalog['x']

    @x.setter
    def x(self, value):
        """
        The setter method for the Sources.x-property, containing the
        coordinate of each source along the x-axis of the reference system.

        Parameters
        ----------
        value : array_like
            The x-coordinates. Must be a 1dim. list of length
            Sources.n_sources.
        """
        try:
            value = self._check_consistency(value)
        except (IOError, ValueError) as msg:
            logger.error('Cannot update values for source property "x": {0}'.format(msg))
        else:
            self.catalog.loc[:, 'x'] = value

    # Reference y-coordinates
    @property
    def y(self):
        """
        Returns
        -------
        y : pandas.Series
            A 1dim. of length Sources.n_sources containing the coordinate of
            each source along the y-axis of the reference system.
        """
        return self.catalog['y']

    @y.setter
    def y(self, value):
        """
        The setter method for the Sources.y-property, containing the
        coordinate of each source along the x-axis of the reference system.

        Parameters
        ----------
        value : array_like
            The y-coordinates. Must be a 1dim. list of length
            Sources.n_sources.
        """
        try:
            value = self._check_consistency(value)
        except (IOError, ValueError) as msg:
            logger.error('Cannot update values for source property "y": {0}'.format(msg))
        else:
            self.catalog.loc[:, 'y'] = value

    # Broadband magnitudes
    @property
    def mag(self):
        """
        Returns
        -------
        mag : pandas.Series
            A 1dim. of length Sources.n_sources containing the magnitude of
            each source in the catalog.
        """
        return self.catalog['mag']

    @mag.setter
    def mag(self, value):
        """
        The setter method for the Sources.mag-property, containing the
        magnitude of each source in the catalog.

        Parameters
        ----------
        value : array_like
            The magnitudes. Must be a 1dim. list of length Sources.n_sources.
        """
        try:
            value = self._check_consistency(value)
        except (IOError, ValueError) as msg:
            logger.error('Cannot update values for source property "mag": {0}'.format(msg))
        else:
            self.catalog['mag'] = value

    # WCS coordinates: RA
    @property
    def ra(self):
        """
        Returns
        -------
        ra : pandas.Series
            A 1dim. of length Sources.n_sources containing the coordinate of
            each source along the RA-axis.
        """
        return self.catalog['ra']

    @ra.setter
    def ra(self, value):
        """
        The setter method for the Sources.ra-property, containing the
        coordinate of each source along the RA-axis.

        Parameters
        ----------
        value : array_like
            The RA-coordinates. Must be a 1dim. list of length
            Sources.n_sources.
        """
        try:
            value = self._check_consistency(value)
        except (IOError, ValueError) as msg:
            logger.error('Cannot update values for source property "ra": {0}'.format(msg))
        else:
            self.catalog.loc[:, 'ra'] = value

    # WCS coordinates: Dec
    @property
    def dec(self):
        """
        Returns
        -------
        dec : pandas.Series
            A 1dim. of length Sources.n_sources containing the coordinate of
            each source along the Dec-axis.
        """
        return self.catalog['dec']

    @dec.setter
    def dec(self, value):
        """
        The setter method for the Sources.Dec-property, containing the
        coordinate of each source along the Dec-axis.

        Parameters
        ----------
        value : array_like
            The Dec-coordinates. Must be a 1dim. list of length
            Sources.n_sources.
        """
        try:
            value = self._check_consistency(value)
        except (IOError, ValueError) as msg:
            logger.error('Cannot update values for source property "ra": {0}'.format(msg))
        else:
            self.catalog.loc[:, 'ra'] = value

    # The status of the sources during the fit
    @property
    def status(self):
        """
        Returns
        -------
        status : pandas.Series
            A 1dim. of length Sources.n_sources containing the status of each
            source in the catalog.
        """
        return self.catalog['status']

    @status.setter
    def status(self, value):
        """
        The setter method for the Sources.status-property, containing the
        status of each source during the fit. The classification scheme for
        the status of a source is as follows.
            [-1] - source is not used
            [ 0] - source contributes to unresolved component
            [ 1] - source is fitted together with a brighter neighbor.
            [ 2] - source is fitted independently
            [ 3] - source is used as a PSF source.
            [ 4] - source is a background (e.g. sky) component

        Parameters
        ----------
        value : array_like
            A 1dim. list of length Sources.n_sources.
        """
        try:
            value = self._check_consistency(value)
        except (IOError, ValueError) as msg:
            logger.error('Cannot update values for source property "status": {0}'.format(msg))
        else:
            self.catalog.loc[:, 'status'] = value
            if 'specrow' in self.catalog.columns:
                # if Sources.status is (re)set after attribute Sources.specrow has been set internally,
                # the latter is not valid anymore.
                logger.warning("Setting attribute Sources.status will render attribute Sources.specrow invalid.")
                del self.catalog['specrow']

    # The primary source in each component
    @property
    def primary(self):
        """
        Returns
        -------
        status : pandas.Series
            A 1dim. of length Sources.n_sources containing the ID/index of the
            primary source for each source in the catalog.
        """
        return self.catalog['primary']

    @primary.setter
    def primary(self, value):
        """
        The setter method for the Sources.primary-property, containing the
        index/ID of the primary source for each entry in the source catalog.
        This index is identical to the index of the source itself for sources
        with status > 1.

        Parameters
        ----------
        value : array_like
            The indices/IDs of the primary sources. Must be a 1dim. list of
            integers of length Sources.n_sources.
        """
        try:
            value = self._check_consistency(value)
        except (IOError, ValueError) as msg:
            logger.error('Cannot update values for source property "status": {0}'.format(msg))
        else:
            self.catalog.loc[:, 'primary'] = value
            # if Sources.primary is (re)set after attribute Sources.specrow has been set internally,
            # the latter is not valid anymore.
            if 'specrow' in self.catalog.columns:
                logger.warning("Setting attribute Sources.primary will render attribute Sources.specrow invalid.")
                del self.catalog['specrow']

    @property
    def specrow(self):
        """
        The getter method of the Sources.specrow-property.

        Returns
        -------
        specrow : pandas.Series
            A series of integers and of length Sources.n_sources that
            contains, for each source in the catalog with status>=0, the
            index/ID of the row that contains the spectrum in the data frame
            Sources.spectra. Unless they were defined previously, the indices
            are automatically assigned using the information from the
            attributes Source.primary and Sources.status.
        """
        if 'specrow' not in self.catalog.columns:

            # get status for each source
            status = self.catalog['status']

            # prepare column
            specrow = pd.Series(np.zeros(self.n_sources, dtype=np.int32), index=self.catalog.index)

            # set row for independent ('primary') sources
            specrow[status > 1] = np.arange((status > 1).sum())

            # if unresolved component exists, make sure its spectrum comes before any sky spectrum
            if self.n_background > 0:
                if (status == 0).any():
                    specrow[status >= 4] += 1
                unresolved_row = specrow[status >= 4].min() - 1
            else:
                unresolved_row = specrow.max() + 1

            # now deal with sources that are either next to a brighter neighbor(status=1) or included in the unresolved
            # component (status=0)
            for i in self.ids:
                if status[i] == 1:
                    # find spectrum of primary component; the following loop is necessary because the initially
                    # assigned primary component might have become the secondary component of another source.
                    secondary = []
                    j = i
                    while self.catalog.at[j, 'primary'] != j:  # continue until current source is primary

                        # exclude source if primary source has status = -1.
                        if self.catalog.at[j, 'primary'] not in self.ids:
                            logging.error("No 'primary' counterpart for source #{0}".format(i))
                            logging.error("Source #{0} excluded".format(i))
                            self.status[i] = -1
                            break
                        secondary.append(j)
                        j = self.catalog.at[j, 'primary']

                    specrow[secondary] = specrow[j]
                elif status[i] == 0:
                    specrow[i] = unresolved_row

            # set column in sources catalog
            self.catalog.loc[:, 'specrow'] = specrow

        return self.catalog['specrow']

    # The mapping from a spectrum to the indices of contributing sources in the catalogue.
    @property
    def indexmap(self):
        """
        Returns
        -------
        indexmap : dictionary
            The mapping from the Sources.specrow (keys) to the index/indices (values)
            of the corresponding source IDs in the catalogue.
        """
        grouped = self.specrow.groupby(self.specrow)
        return grouped.groups

    # The magnitude weights of the individual contributions to a spectrum
    @property
    def weights(self):
        """
        Returns
        -------
        weights: pandas.Series
            A series of the same length as the whole source list containing
            the magnitude weight of each source in its component. The magnitude
            weight is defined as 10**(-0.4*df), where df is the fraction of
            flux the source contributes to the component.
        """
        if 'weights' not in self.catalog.columns:
            # group sources by index/ID of primary source
            grouped = self.catalog[['mag', 'primary']].groupby('primary')

            # calculate total fluxes in each group and broadcast to full catalog
            component_fluxes = grouped['mag'].transform(lambda x: np.sum(10**(-0.4*x)))

            # get flux ratio between each source and its component
            flux_weights = 10**(-0.4*self.catalog['mag'])/component_fluxes

            self.catalog['weights'] = -2.5*np.log10(flux_weights)
        return self.catalog['weights']

    # The number of independent components in the fit.
    @property
    def n_components(self):
        """
        Returns
        -------
        n_components : int
            The number of components (i.e. independent spectra) in the
            catalogue.
        """
        return self.specrow.max() + 1

    # Return a boolean array of len(n_sources) that is true if the source is resolved.
    @property
    def is_resolved(self):
        """
        Returns
        -------
        resolved : pandas.Series
            A boolean series of len(Spectrum.n_sources) indicating whether a
            source is resolved. If Spectrum.status is undefined, all sources
            are considered unresolved.
        """
        return (self.status > 0) & (self.status < 4)

    #  A flag indicating for each source if it is part of the unresolved stellar component
    @property
    def is_unresolved(self):
        """
        Returns
        -------
        is_unresolved : pandas.Series
            A boolean array of len(Spectrum.n_sources) indicating whether a source
            is part of the unresolved stellar component.
            """
        return self.status == 0

    # A flag indicating for each source if it is part of the background
    @property
    def is_background(self):
        """
        Returns
        -------
        is_background : pandas.Series
            A boolean series of len(Spectrum.n_sources) indicating whether a
            source is part of the background.
            """
        return self.status > 3

    # A flag on whether a unresolved stellar component is included in the catalogue.
    @property
    def includes_unresolved(self):
        """
        Returns
        -------
        includes_unresolved : bool
            A flag indicating whether an unresolved stellar component is
            included in the catalogue.
        """
        return self.is_unresolved.any()

    # The row of the unresolved stellar component (if any)
    @property
    def specrow_unresolved(self):
        """
        Returns
        -------
        specrow_unresolved : int or None
            The row(s) of the spectrum/spectra of the unresolved component(s)
            (if any, otherwise returns an empty array)
        """
        rows = self.specrow[self.is_unresolved]
        assert (np.unique(rows).size == 1)
        return rows[rows.index[0]]

    # A flag on whether one or more background components are included in the catalogue. 
    @property
    def includes_background(self):
        """
        Returns
        -------
        includes_background : bool
            A flag indicating whether one or more background components are
            included in the catalog.
        """
        return self.is_background.any()

    # The number of background components included in the catalogue
    @property
    def n_background(self):
        """
        Returns
        -------
        n_background : int
            The number of background components that is included in the
            catalogue."""
        return self.is_background.sum()

    # The rows of the spectra of the background components (if any)
    @property
    def specrow_background(self):
        """
        Returns
        -------
        specrow_background : pandas.Series
            The rows containing the background spectra (if any, otherwise
            returns an empty array)
        """
        return self.specrow[self.is_background]

    # The indices of the sources used to determine the PSF (if any)
    @property
    def psf_indices(self):
        """
        Returns
        -------
        psf_indices : pandas.Index
            A 1dim. index array containing the IDs/indices of the sources that
            are used to determine the PSF (i.e. sources whose `status` is 3).
        """
        return self.ids[self.status == 3]

    ############################################################################
    #  Methods providing the user with information about the source catalogue  #
    ############################################################################
    def info(self):
        """
        Print out some information about the loaded source catalogue. Also
        check if all necessary properties have been defined.
        """
        logger.info("Current status of catalogue of sources:")
        logger.info("   no. of PSF sources: {0}".format(self.psf_indices.size))
        logger.info("   no. of resolved sources: {0}".format(self.is_resolved.sum()))
        logger.info("   no. of nearby sources: {0}".format((self.status == 1).sum()))
        logger.info("   no. of unresolved sources: {0}".format(self.is_unresolved.sum()))
        logger.info("   no. of background components: {0}".format(self.n_background))

        logger.debug("Overview of all components used in fit:")
        logger.debug("   Component:   IDs (& mags):")

        if logger.getEffectiveLevel() < 20:
            for specrow in range(self.n_components):
                log_string = "   {0:8d}".format(specrow)

                for i, source_id in enumerate(self.ids[self.specrow == specrow]):
                    log_string += "   {0:6d} ({1:.2f})".format(source_id, self.weights[source_id])

                    # new line after every second source
                    if np.mod(i + 1, 2) == 0:
                        logger.debug(log_string)
                        log_string = 11 * " "

                # log remaining source if number of sources was odd
                if len(log_string.split()) > 1:
                    logger.debug(log_string)

        if self.transformation.valid:
            self.transformation.info()

    #############################################################################
    #              Methods dealing with the spectra of the sources              #
    #############################################################################
    def open_spectra_hdu(self, hdu, data_wave_hdu=None):
        """
        Open a FITS HDU containing the spectra of the sources. Note that the
        spectra must be provided in the form of a single HDU containing both
        spectra and uncertainties.
          
        Parameters
        ----------
        hdu : instance of astropy.io.fits.ImageHDU
            The HDU containing the spectra and their uncertainties. The shape
            of the data array must be [python counting]:
            (Sources.n_dispersion, Sources.n_components, 2)
            The HDU is accessible via attribute Sources.data_hdu afterwards.
        data_wave_hdu : instance of astropy.io.fits.ImageHDU, optional
            This parameter can be used to provide a separate wavelength array
            for each spectrum in 'hdu'. If provided, must have a shape of
            (Sources.n_dispersion, Sources.n_components).
        """
        # sanity checks on provided HDUs
        for name, value in {'DATA': hdu, 'WAVE': data_wave_hdu}.items():
            if value is None:
                continue
            elif not isinstance(value, fits.hdu.image._ImageBaseHDU):
                logger.error('{0}-HDU has invalid type "{1}".'.format(name, type(value)))
                return
            elif value.header['NAXIS'] not in [2, 3]:
                logger.error('{0}-HDU must be (2/3)D array, not {1}D.'.format(name, value.header['NAXIS']))
            n_dim = value.header['NAXIS']

            if 'EXTNAME' in value.header:
                logger.info('Reading data from HDU {0} ...'.format(value.header['EXTNAME']))
                name = value.header['EXTNAME']

            # check if dispersion information is valid
            if self.n_dispersion == 1:
                self.n_dispersion = value.header['NAXIS{0}'.format(n_dim)]
            elif value.header['NAXIS{0}'.format(n_dim)] != self.n_dispersion:
                logger.error('HDU {0} has invalid dispersion information.'.format(name))
                return

            # check if number of spectra is valid
            if value.header['NAXIS{0}'.format(n_dim - 1)] != self.n_components:
                logger.error('HDU {0} has invalid number of components.'.format(name))
                return

        self.data_hdu = hdu
        self.data_wave_hdu = data_wave_hdu

        # if no dedicated wavelength HDU was provided, try to update dispersion information from header of data HDU
        if self.data_wave_hdu is None:
            for key in ["CRVAL", "CDELT", "CRPIX"]:
                if "{0}3".format(key) in self.data_hdu.header:
                    setattr(self, key.lower(), self.data_hdu.header["{0}3".format(key)])

        # to to infer type and unit of wavelength scale in any case
        for key in ["CTYPE", "CUNIT"]:
            if "{0}3".format(key) in self.data_hdu.header:
                setattr(self, key.lower(), self.data_hdu.header["{0}3".format(key)])

        # create boolean series that keeps track of which sources have spectra loaded
        self.spectrum_loaded = pd.Series(np.zeros((self.n_components,), dtype=np.bool),
                                         index=pd.Index(np.arange(self.n_components)))

    def load_spectra(self, components=None, layers=None, init_wave=False):
        """
        Load the spectra and uncertainties for the requested source(s) and
        layer(s). If an HDU with dedicated wavelength data for each component
        is available, the wavelength data will also be loaded.

        The method will create an empty pandas DataFrame and try to fill it
        using the available HDUs, provided previously via the method
        Sources.open_spectra_hdu(). If no such HDU are available, the array
        will be empty.
        
        The rows of the data frame will contain the wavelength information.
        The columns of the data frame contain the individual components. For
        each component, the spectrum, its uncertainties, and if available the
        wavelength are stored. This is achieved via a pandas MultiIndex.

        Parameters
        ----------
        components : array_like or integer, optional
            The component(s) for which the spectrum should be loaded. If none
            are provided, the method will load the spectrum of every component
            in the instance. Remember that spectra are not loaded per ID but
            per component, for this reason no IDs should be provided.
        layers : integer  optional
            The indices of the layers over which the spectra should be loaded.
            If none are provided, they are loaded for the whole wavelength
            range.
        init_wave : bool, optional
            The flag indicates if columns for the wavelength information per
            source should be included in the spectra array even if no
            dedicated wavelength HDU is available.
        """
        # sanity checks on provided IDs and layers
        if components is not None and not isinstance(components, (int, np.integer, list, np.ndarray, pd.Index)):
            logger.error("Parameter 'components' must be integer or array-like, not {0}.".format(type(components)))
            return
        # load all components if None was provided
        elif components is None:
            components = np.arange(self.n_components, dtype=np.int32)
        # make sure components are available as an array
        else:
            components = np.atleast_1d(components)

        # check if all provided components are actually in the source catalogue
        if components.min() < 0 or components.max() >= self.n_components:
            logger.error('Some requested components fall outside available range.')
            return

        if layers is not None and not isinstance(layers, (int, np.integer, list, np.ndarray)):
            logger.error('Parameter "layers" must be integer or array-like, not {0}.'.format(type(layers)))
            return
        elif layers is None:
            layers = np.arange(self.n_dispersion)
        else:
            layers = np.atleast_1d(layers)
            # layer selection must be consecutive
            if (np.diff(layers) != 1).any():
                logger.error('Layer selection must be consecutive.')

        # initialize axes of data frame for spectra data.
        # rows contain dispersion data using the actual wavelengths as a pandas Float64Index
        index = self.wave[layers]

        # check if dedicated wavelength information per source should be used
        init_wave |= self.data_wave_hdu is not None

        # columns contain components and ('flux', 'sigma'[, 'wave']) in pandas.MultiIndex
        properties = ['flux', 'sigma'] if not init_wave else ['flux', 'sigma', 'wave']
        columns = pd.MultiIndex.from_product([components, properties], names=['component', 'property'])
        # fill array using data from Spectrum HDU if one is available
        if self.data_hdu is not None:
            # get requested chunk of data from available HDU
            # apparently there is an issue with the byte-order between astropy and pandas:
            # https://github.com/astropy/astropy/issues/1156
            logger.debug('Reading spectrum data HDU ...')
            data = self.data_hdu.section[layers[0]:layers[-1] + 1][:, components].byteswap().newbyteorder()
        else:
            # create empty array that can be used to initialize data frame
            data = np.zeros((index.size, components.size, 2), dtype=np.float32)
        # -> array shape (n_layers, n_components, 2)

        if init_wave:
            # add wavelength information if available
            if self.data_wave_hdu is not None:
                logger.debug('Reading wavelength HDU ...')
                wave = self.data_wave_hdu.section[layers[0]:layers[-1] + 1][:, components].byteswap().newbyteorder()
            # otherwise use global wavelength array
            else:
                wave = np.repeat(self.wave[layers].values[:, np.newaxis], components.size, axis=-1)

            data = np.dstack((data, wave))
            # -> array shape (n_layers, n_components, 3)

        # initialize data frame
        self.data = pd.DataFrame(data.reshape(data.shape[0], -1), index=index, columns=columns)

        # keep track of loaded spectra
        self.spectrum_loaded = pd.Series(np.in1d(np.arange(self.n_components, dtype=np.int32), components),
                                         index=np.arange(self.n_components))

    @property
    def wave(self):
        """
        Returns the global wavelength array.

        Note that this is always the wavelength array that is (approximately)
        valid for every spectrum. In case each spectrum has its own wavelength
        solution, this can be accessed from the Spectrum.data data frame,
        using the 'wave' column in the MultiIndex.

        Returns
        -------
        wave : nd_array, shape (k,)
            The wavelength array.
        """
        # in case dispersion information has changed, existing wavelength data must be reset.
        if self._wave is not None and self._wave.size != self.n_dispersion:
            self._wave = None

        if self._wave is None:
            if self.data_wave_hdu is not None:
                self._wave = np.nanmean(self.data_wave_hdu.data, axis=1)
                if np.isnan(self._wave).any():
                    # need to replace NaNs because wavelength is used as an index
                    x = np.arange(self._wave.size)
                    self._wave = np.interp(x, x[np.isfinite(self._wave)], self._wave[np.isfinite(self._wave)])
            else:
                self._wave = np.arange(self.n_dispersion, dtype=np.float64)  # shape (k,)
                self._wave -= (self.crpix - 1)
                self._wave *= self.cdelt
                self._wave += self.crval
        return pd.Float64Index(self._wave)

    @wave.setter
    def wave(self, value):
        """
        Sets the wavelength array for the current instance of Sources.
        
        Parameters
        ----------
        value: list, ndarray or pandas.Index
            The new value for the wavelength array. Its length must match the
            number of dispersion elements in the instance.
        """
        if not isinstance(value, (list, np.ndarray, pd.Index)):
            logger.error('Parameter "wave" must be one of list, array, or pandas Index.')
        elif not len(value) == self.n_dispersion:
            logger.error('Provided parameter "wave" has incompatible shape.')
        else:
            self._wave = pd.Index(value)

    def get_magnitude(self, components, passband=None):
        """
        This method calculates the magnitudes for one or more spectra in a
        given filter.

        The method assumes that the fluxes are provided in erg/s/cm2/Angstrom.
        If that is not the case, the user has to perform the correction for
        the resulting magnitude offset manually. The returned magnitude will
        be in the AB system.

        Parameters
        ----------
        components : int or array_like, shape (N,)
            The index/indices of the component(s) for which the magnitude(s)
            should be determined.
        passband : str
            The name of the passband in which the magnitude should be
            calculated.

        Returns
        -------
        mag : pandas.Series
            Series of floats containing the magnitude(s) in the AB system of
            the component(s) under the assumption that the spectrum have units
            of erg/s/cm2/Angstrom.
        """
        # load _full_ spectra of requested components if necessary
        if not self.spectrum_loaded[components].all() or self.data.shape[0] != self.n_dispersion:
            self.load_spectra(components=components)

        # wavelength must be calculated in Angstrom
        if self.cunit == 'm':
            factor = 1e10
        elif self.cunit == 'nm':
            factor = 10
        else:
            factor = 1

        # if passband is provided, prepare resampling to wavelengths of spectra
        throughput_curve = get_filter(passband) if passband else None

        # it is assumed that the spectra are in units of erg/s/cm^2/Angstrom
        # if that is not the case, the user has to do the conversion itself
        # returned are AB magnitudes, for the calculation of AB magnitudes from fluxes per Angstrom, see
        # https://en.wikipedia.org/wiki/AB_magnitude#Expression_in_terms_of_f.CE.BB
        magnitudes = []

        # loop over components
        for component in np.atleast_1d(components):
            # get flux
            flux = self.data[(component, 'flux')]

            # get wavelength, check if each spectrum requires own wavelength array or global one can be used.
            if self.data_wave_hdu is None:
                wave = factor*self.wave
            else:
                wave = factor*self.data[(component, 'wave')]

            # resample throughput curve (if available, otherwise create whitelight image)
            if throughput_curve is not None:
                throughput = throughput_curve(wave)
            else:
                throughput = np.ones(self.n_dispersion, dtype='<f4')

            # calculate magnitude and add to list
            integrated_flux = np.nansum(throughput*flux*(wave**2))/np.nansum(throughput)
            if integrated_flux <= 0:
                logging.warning('Cannot determined magnitude for component #{0} with negative flux.'.format(component))
                magnitudes.append(np.nan)
            else:
                magnitudes.append(-2.5*np.log10(3.34e4*integrated_flux) + 8.9)

        return pd.Series(magnitudes, index=np.atleast_1d(components))

    def provideComponentNames(self):
        """
        Method to provide the IDs of the primary sources for each component as
        a QStringList.

        Returns
        -------
        names : list of strings
           The names of the components
        """
        # add sources with own spectrum, i.e. have status between 2 and 4
        _names = self.ids[(self.status >= 2) & (self.status <= 4)].values.astype("|S10")

        # need to decode numpy_bytes to str
        names = [_names[i].decode('UTF-8') for i in range(len(_names))]

        # if available, add unresolved component
        if self.includes_unresolved:
            names.append("Unresolved")

        # add background components
        for i in range(self.n_background):
            names.append("Back{0:04d}".format(i))

        return names

    def viewSpectrum(self, name=None):
        """
        Return the spectrum corresponding to a provided ID or name of the
        component.
        
        Parameters
        ----------
        name : string
           The name of the requested component. Can either be th ID of a
           source or 'Unresolved' (to fetch the spectrum of the unresolved
           component) or 'BackXXXX' (to fetch the spectrum of background
           component XXXX).

        Returns
        -------
        data : pandas.Series
            The Series containing the the requested spectrum, and using the
            dispersion information as index.
        """
        if self.spectrum_loaded is None and self.data_hdu is None:
            logging.error("No spectra have been loaded using Sources.load_spectra() or Sources.open_spectra_hdu().")
            return None
        # get index of component (i.e. specrow)
        else:
            # check if ID was provided
            try:
                name = int(name)
            # otherwise, check if unresolved or background component requested
            except ValueError:
                if name == 'Unresolved':
                    i = self.specrow_unresolved
                elif name[:4] == 'Back':
                    i = self.specrow_background.iloc[int(name[4:])]
            else:
                i = self.specrow[name]

        # load the spectrum if it has not been loaded previously
        if self.spectrum_loaded is None or not self.spectrum_loaded[i]:
            # check if a spectrum is available for the requested source
            self.load_spectra(components=i)

        # return data, check before if dedicated wavelength array exists for spectrum
        if self.data_wave_hdu is None:
            return self.data[(i, 'flux')]
        else:
            return pd.Series(self.data[(i, 'flux')].values, index=self.data[(i, 'wave')].values)

    #############################################################################
    #            Methods dealing with IFU coordinates of the sources            #
    #############################################################################

    # Check if source coordinates or the coordinate transformation should be fitted
    @property
    def contains_free_coordinates(self):
        """
        Returns
        -------
        contains_free_coordinates : bool
            Flag indicating whether any of the parameters that deal with the
            source coordinates (including the coordinate transformation) are
            free fit parameters. 
        """
        if self.ifs_coordinates_fitted[self.status < 4].all():
            return False
        elif self.ifs_coordinates_free.any():
            return True
        else:
            return self.transformation.contains_free_parameters

    def open_coordinates_hdu(self, hdu):
        """
        Open a FITS HDU containing IFS coordinates for the sources.
        
        There are two possibilities for loading IFS coordinates, either via an
        image HDU or via a table HDU.
        
        In the former case, the HDU data should be 3D, with the 0th axis
        (python counting) containing the wavelength information, the 1st axis
        containing the sources and the 2nd axis containing x and y.
        
        Note that the data are not actually loaded into memory but that the
        method only provided the necessary information to do so using the
        method Sources.load_ifs_coordinates().       
        
        In the latter case, a binary FITS table should contain the parameters
        of the polynomial fits to the individual sources. This way of storing
        the fit parameters instead of the fitted functions is much less memory
        consuming for large source lists. The table must contain five columns,
        named "id", "x_function", "x_coefficients", "y_function", and
        "y_coefficients", and providing ID, the type of polynomial (fct) and
        the fit coefficients (prm) along x and y for each source.
        
        Further, the header of the FITS table must contain HIERARCH-keywords
        that provide the start and end of the domain of the polynomial fits,
        named "PAMPELMUSE POSITIONS DOMAIN START and "PAMPELMUSE POSITIONS
        DOMAIN END".
        
        Parameters
        ----------
        hdu : instance of astropy.io.fits.hdu.table.BintableHDU
              or astropy.io.fits.hdu.image.ImageHDU
            The HDU instance containing the coordinates.
        """
        # check whether FITS table or image provided
        # if table, assume it contains coefficients of polynomial fits
        if isinstance(hdu, fits.hdu.table.BinTableHDU):
            logger.info('Reading polynomial fits to coordinates from HDU {0} ...'.format(hdu.header.get('EXTNAME', '')))

            # for compatibility with older versions, still accept previous column names
            old_names = {'x_function': 'xfct', 'y_function': 'yfct', 'x_coefficients': 'xprm', 'y_coefficients': 'yprm'}

            # check if required columns are present in header
            for name in ["id", "x_function", "x_coefficients", "y_function", "y_coefficients"]:
                if name not in hdu.columns.names and old_names[name] not in hdu.columns.names:
                    logger.error("Missing required column '{0}' in positions table HDU.".format(name))
                    return

            # the fit via Sources.fitIfsCoordinates sets the domain of the polynomial to the min. and max. values of
            # the fitted data interval. These values must be stored in the FITS header and recovered because otherwise
            # the values returned by the polynomials are wrong (the default is domain=[-1,1])
            if "PAMPELMUSE POSITIONS DOMAIN START" in hdu.header and "PAMPELMUSE POSITIONS DOMAIN END" in hdu.header:
                domain = [hdu.header["PAMPELMUSE POSITIONS DOMAIN START"],
                          hdu.header["PAMPELMUSE POSITIONS DOMAIN END"]]
            else:
                logger.error('Domain for polynomial fits not specified.')
                return

            # check header for identifications of different polynomials. Note that in the table, the different types
            # are referenced using integers and the translation to the actual names is done via HIERARCH keywords.
            ident = {}
            i = 0
            while "PAMPELMUSE POSITIONS POLY{0}".format(i) in hdu.header:
                ident[i] = hdu.header["PAMPELMUSE POSITIONS POLY{0}".format(i)]
                i += 1

            # Set up the polynomials for all sources that have data available in the HDU
            for i, row in enumerate(hdu.data):
                # the name of the polynomial as a string:
                xfctname = ident[row['x_function'] if 'x_function' in hdu.columns.names else row['xfct']]
                yfctname = ident[row['y_function'] if 'y_function' in hdu.columns.names else row['yfct']]

                xfct = self.available_polynoms[xfctname]  # the actual polynomial class
                yfct = self.available_polynoms[yfctname]

                # save polynomials for current source
                self.ifs_coordinates_fit_functions[row["id"]] = [
                    xfct(row['x_coefficients'] if 'x_coefficients' in hdu.columns.names else row['xprm'],
                         domain=domain),
                    yfct(row['y_coefficients'] if 'y_coefficients' in hdu.columns.names else row['yprm'],
                         domain=domain)]
                # keep track of which sources have fits available
                self.ifs_coordinates_fitted[row['id']] = True

            self.ifs_coordinates_fit_hdu = hdu

        # if an image HDU is provided, assume it contains results of direct coordinates fit
        elif isinstance(hdu, fits.hdu.image._ImageBaseHDU):

            if hdu.header["NAXIS"] != 3:
                logger.error("HDU containing coordinates must be 3D array, not {0}D.".format(hdu.header["NAXIS"]))
                return

            logger.info('Reading source coordinates from HDU {0} ...'.format(hdu.header.get('EXTNAME', '')))

            # check if dispersion information is valid
            if self.n_dispersion == 1:
                self.n_dispersion = hdu.header["NAXIS3"]
            elif hdu.header["NAXIS3"] != self.n_dispersion:
                logger.error("Invalid number of dispersion elements in provided IFS coordinates.")
                return

            # make sure data contains coordinates for all sources in the catalogue
            if hdu.header["NAXIS2"] != self.n_sources - self.n_background:
                logger.error('Number of sources in coordinates HDU does not match number of sources in catalog.')
                return
            self.ifs_coordinates_free[:] = True

            self.ifs_coordinates_data_hdu = hdu

        else:
            logger.error('Unknown type "{0}" for parameter "hdu".'.format(type(hdu)))

    def load_ifs_coordinates(self, ids=None, layers=None):
        """
        Load the IFS coordinates for an individual source, a subset of sources
        or the entire catalogue. The coordinates might be loaded for the
        entire cube, or an individual layer, or a subset of layers
        .
        The method tries to obtain the IFS coordinates in different ways (and
        in the following order):
         (a) reading the coordinates from an available FITS-ImageHDU. The
             method Sources.open_coordinates_hdu() may be used to make such an
             HDU available.
         (b) using the coordinate transformation if it has been fully defined.
         (c) directly use the reference coordinates (i.e. assuming the coord.
             transformation is unity)
             
        Additionally, the method checks if polynomial fits to the IFS
        coordinates exist. The method Sources.open_coordinates_hdu() may be
        used to make such an HDU available, positions can be fitted using the
        method Sources.fitIfsCoordinates().

        The routine will prepare an empty pandas DataFrame and try to fill it
        using the available data. The rows of the data frame will contain the
        wavelength information. The columns of the data frame contain the
        individual sources. For each source, the position along x and y and
        their polynomial fits (if available) are stored. This is achieved via
        a pandas MultiIndex.

        Parameters
        ----------
        ids : array-like or integer, optional
            The ID(s) of the sources for which the IFS coordinates should be
            loaded. If none are provided, the method will load coordinates for
            all point sources in the catalogue.
        layers : array-like or integer, optional
            The indices of the layers for which the IFS coordinates should be
            loaded. If none are provided, the coordinates are loaded for the
            whole wavelength range.
        """
        # FOR DEBUGGING: check where the call was triggered
        if logger.getEffectiveLevel() < 20:
            import inspect
            t0 = time.time()
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            logging.debug("Call to Sources.load_ifs_coordinates() triggered by: {0} in {1}[@{2}]".format(
                calframe[1][3], calframe[1][1], calframe[1][2]))

        # sanity checks on provided IDs and layers

        # load all (stellar) components if None was provided
        if ids is None:
            ids = self.ids[~self.is_background]
        elif not isinstance(ids, (int, np.integer, list, np.ndarray, pd.Index)):
            logger.error("Parameter 'ids' must be integer or array-like, not {0}.".format(type(ids)))
            return
        # make sure components are available as a pandas index
        else:
            ids = pd.Index(np.atleast_1d(ids))
            # check if all provided components are actually in the source catalogue
            if not np.in1d(ids.values, self.ids.values, assume_unique=True).all():
                logger.error('Some requested components fall outside available range.')
                return

        # check if requested layers are valid
        if layers is not None and not isinstance(layers, (int, np.integer, list, np.ndarray)):
            logger.error('Parameter "layers" must be integer or array-like, not {0}.'.format(type(layers)))
            return
        elif layers is None:
            layers = np.arange(self.n_dispersion)
        else:
            layers = np.atleast_1d(layers)
            # layer selection must be consecutive
            if (np.diff(layers) != 1).any():
                logger.error('Layer selection must be consecutive.')

        # initialize data frame for coordinates data.
        # rows contain dispersion data using the actual wavelengths as a pandas Float64Index
        index = pd.Float64Index(self.wave[layers])

        # columns contain IDs, ('x', 'y'), and ('value', 'fit') in a pandas.MultiIndex
        axes = ['x', 'y']
        types = ['fit', 'value']
        columns = pd.MultiIndex.from_product([ids, axes, types], names=['id', 'axis', 'type'])

        if logger.getEffectiveLevel() < 20:
            t1 = time.time()

        # try to collect data to fill data frame
        # first choice: if available, fill in IFS coordinates using open FITS HDU
        if self.ifs_coordinates_data_hdu is not None:
            # need to get row indices of source in data HDU - note that they are store in order of increasing source ID
            indices_in_hdu = np.searchsorted(sorted(self.ids[~self.is_background]), ids)
            # get required chunk of data from HDU
            data = self.ifs_coordinates_data_hdu.section[layers[0]:layers[-1] + 1][:, indices_in_hdu]
            # data array has shape (n_layers, n_sources, 2)

        # second choice: use coordinate transformation if available
        elif self.transformation.valid:
            # collect input coordinates in array of shape (n_sources, 2)
            xy_in = self.catalog.loc[ids, ['x', 'y']]

            # coordinate transformation may needs to be resized
            if not np.array_equal(self.transformation.data.index, self.wave):
                if len(self.transformation.data.index) == len(self.wave):
                    self.transformation.data.index = self.wave
                else:
                    logger.warning('Adding dispersion information to coordinate transformation.')
                    self.transformation.resize(self.wave)

            # function returns pandas DataFrame, need to get values and reshape
            # TODO: directly use DataFrame
            _data = self.transformation(xy_in, layers=layers)
            data = _data.values.reshape((_data.shape[0], -1, 2))
            # data array has shape (n_layers, n_sources, 2)

        # third choice: just use reference coordinates
        else:
            xy_ref = np.vstack((self.x[ids], self.y[ids])).T
            data = np.repeat(xy_ref[np.newaxis], layers.size, axis=0)
            # data array has shape (n_layers, n_sources, 2)

        if logger.getEffectiveLevel() < 20:
            t2 = time.time()

        # the polynomial fits can be obtained from the dictionary of previously performed fits
        fit = np.zeros(data.shape, dtype=np.float32)
        for i, source_id in enumerate(ids):
            if source_id in self.ifs_coordinates_fit_functions.keys():
                xfit, yfit = self.ifs_coordinates_fit_functions[source_id]
                fit[:, i, 0] = xfit(index)
                fit[:, i, 1] = yfit(index)

        # join values and fits - note that fits come first (to have proper alphanumeric order)
        data = np.stack((fit, data), axis=-1)

        # create data frame
        self.ifs_coordinates = pd.DataFrame(data.reshape(data.shape[0], -1), index=index, columns=columns)

        # check if data frame is sufficiently ordered for slicing
        # check: http://pandas.pydata.org/pandas-docs/stable/advanced.html#using-slicers
        if not self.ifs_coordinates.columns.is_lexsorted():
            logger.warning('Data frame containing IFS coordinates not sorted. Reordering ...')
            self.ifs_coordinates = self.ifs_coordinates.sort_index(level=0, axis=1)  # axis=1 = columns

        # keep track of for which sources IFS coordinates have been loaded
        self.ifs_coordinates_loaded = pd.Series(np.in1d(self.ids, ids), index=self.ids)

        if logger.getEffectiveLevel() < 20:
            t3 = time.time()
            logger.debug("Time required to initialize array: {0}s".format(t1 - t0))
            logger.debug("Time required to load IFS coordinates: {0}s".format(t2 - t1))
            logger.debug("Time required to load fits to IFS coordinates: {0}s".format(t3 - t2))

    def fit_coordinate_transformation(self, layers=None, fixed=None):
        """
        The method initializes a new coordinate transformation using the
        available reference and IFS coordinates.
        
        Prior to calling this method, the IFS coordinates must have been
        loaded using the method Sources.load_ifs_coordinates. Coordinates
        must be available for at least four sources.

        Parameters
        ----------
        layers : array-like or integer, optional
            The indices of the layers for which the IFS coordinates should be
            loaded. If none are provided, the coordinates are loaded for the
            whole wavelength range.
        fixed : dictionary, optional
            In case any parameters should be fixed to given values, provide
            here the names of those parameters and their fixed values.
        """
        if self.x is None or self.y is None:
            logging.error("Cannot determine coordinate transformation. Missing reference coordinates.")
            return

        if self.ifs_coordinates is None:
            logging.error("Cannot determine coordinate transformation. Missing IFS coordinates.")
            return

        # check if IFS coordinates are available for at least 4 sources.
        if self.ifs_coordinates_loaded.sum() < 4:
            logging.error("Cannot determine coordinate transformation. IFS coordinates available for < 4 sources.")
            return

        # check if requested layers are valid
        if layers is not None and not isinstance(layers, (int, np.integer, list, np.ndarray)):
            logger.error('Parameter "layers" must be integer or array-like, not {0}.'.format(type(layers)))
            return
        elif layers is None:
            layers = np.arange(self.n_dispersion)
        else:
            layers = np.atleast_1d(layers)
            # layer selection must be consecutive
            if (np.diff(layers) != 1).any():
                logger.error('Layer selection must be consecutive.')

        # collect input coordinates in (n_sources, 2) array
        xy_in = self.catalog.loc[self.ifs_coordinates_loaded, ['x', 'y']]

        # get output coordinates
        xy_out = self.ifs_coordinates.loc[self.wave[layers], (slice(None), slice(None), 'value')]
        # remove 2nd level of MultiIndex (as it only contains 'value')
        xy_out.columns = xy_out.columns.droplevel(2)

        # define new transformation using coordinates
        self.transformation = Transformation.from_coordinates(xy_in, xy_out, fixed=fixed)

    def providePositionIds(self, min_status=1):
        """
        Provide a list with the IDs sources in the catalog that have their own
        coordinates, i.e. stellar sources.

        This method is only used in combination with the PampelMuse GUI. Its
        purpose is to fill the drop-down menu of available sources when
        displaying IFS coordinates.

        Parameters
        ----------
        min_status : int, optional
            This parameter can be used of to constrain the returned list to
            sources with a certain status or higher. The default will exclude
            unresolved sources (i.e. with status 0) from the list.

        Returns
        -------
        list of strings
            The list is returned as a list of strings usable in PyQt5.
        """
        # get boolean selection of included sources
        selection = np.in1d(self.status, np.arange(min_status, 4))

        return [star_id.decode('UTF-8') for star_id in self.ids[selection].astype("|S10")]

    def viewIfsCoordinates(self, source_id):
        """
        Return the IFS x- and y-coordinate of an individual source.
        
        Note that if a polynomial fit for the requested source exists, it is
        also returned. Otherwise, the second part of the returned array will
        be empty.

        Parameters
        ----------
        source_id : int
            The ID of the source for which the coordinates should be returned.

        Returns
        -------
        pandas.DataFrame
            The IFS coordinates of the requested source and their polynomial
            fits. The wavelength information is provided as the index. The
            columns contain the axis ('x' or 'y') and the type ('value' or
            'fit') as a MultiIndex.
        """
        # sanity check on provided ID
        if not isinstance(source_id, (int, np.integer)):
            logger.error("Source ID must be an integer, not {0}.".format(type(source_id)))
            return

        # check if ID in catalog
        if source_id not in self.ids:
            logger.error("Provided ID #{0} not in source catalogue.".format(source_id))
            return

        # load the coordinates if they have not been loaded previously
        if not self.ifs_coordinates_loaded[source_id] or self.ifs_coordinates.shape[0] < self.n_dispersion:
            self.load_ifs_coordinates(ids=source_id)

        # return it
        return self.ifs_coordinates[source_id]

    def fitIfsCoordinates(self, ids, order, mask=None, polynom="plain", common_slope=False, statusBar=None,
                          instrument=None):
        """
        The method fits the IFS coordinates as a function of wavelength.
        
        There are two ways in which the coordinates can be fitted:
        (1) fit a polynomial individually to each position (the default)
        (2) find a polynomial fit for all resolved sources and shift it along
            the selected axis to obtain a position for each source

        Parameters
        ----------
        ids : array_like or None
            The ID[s] of the sources that should be fitted. Use None to fit
            all stellar sources, i.e. all sources with Sources.status < 4.
        order : int
            The order of the polynomial used to perform the fit
        mask : pandas.Series, optional
            An optional bad-pixel mask. If specified, it must be a boolean
            pandas Series that uses wavelengths as indices. Bad pixels should
            be set to True, good ones to False.
        polynom : string, optional
            The fitting function that should be used. Supported types are
            currently a plain polynomial ('plain', default), a Chebyshev
            polynomial ('chebyshev'), a Legendre polynomial ('legendre'), or
            a Hermite polynomial ('hermite').
        common_slope : boolean, optional
            Flag indicating whether a common polynomial should be used to model
            the wavelength dependency of all sources.
        instrument : instance of pampelmuse.core.instruments.Instrument,
                     optional
            The IFS instrument used to obtain the data. This information is used
            to restrict the selection to sources inside the FoV when determining
            the mean slope of a coordinate with wavelength.
        """
        # find the indices of the source(s) to be fitted in the catalogue
        if ids is None:
            ids = self.ids[~self.is_background]
        else:
            if not np.in1d(ids, self.ids).all():
                logger.error('Cannot fit coordinates. Unknown IDs provided.')
                return

        # sanity check for requested fit function
        if not polynom.lower() in self.available_polynoms.keys():
            logging.error("Unknown fitting function provided: {0}".format(polynom))
            return
        fit_function = self.available_polynoms[polynom.lower()]

        # load IFS coordinates for all sources _WITHOUT_ interpolation
        self.load_ifs_coordinates(ids=ids)

        # for further processing, get values for each source
        xy = self.ifs_coordinates.loc[:, (slice(None), slice(None), 'value')]

        # create pseudo-mask if None provided
        if mask is None:
            mask = pd.Series(False, index=self.wave)

        # determine 'mean' coordinate if option 'common_slope' selected
        if common_slope:

            # get mean coordinates along x and y for each available source
            xy_mean = np.nanmean(xy, axis=0).reshape((-1, 2))

            # sources used for creating mean slope are those with status 2 or 3
            used = ((self.status > 1) & (self.status < 4))[self.ifs_coordinates_loaded]

            # if "instrument" is known, then only use sources within FoV to determine common slope.
            if instrument is not None:
                used &= instrument.infov(xy_mean)

            # get mean coordinate along x and y plus boolean array indicating for which layers valid data exist
            mean_x = np.nanmean(xy.loc[:, (slice(None), 'x')], axis=1)
            mean_y = np.nanmean(xy.loc[:, (slice(None), 'y')], axis=1)
            mean_mask = mask | np.nan(mean_x) | np.isnan(mean_y)

            # fit mean coordinates with requested order
            mean_x_fit = fit_function.fit(self.wave[~mean_mask], mean_x[~mean_mask], order)
            mean_y_fit = fit_function.fit(self.wave[~mean_mask], mean_x[~mean_mask], order)

            order = 0  # all individual coordinates are now "fitted" with zeroth order polynomials

        # loop over selected sources and fir polynomials to x- and y-coordinates
        for n, source_id in enumerate(ids, start=1):

            x = xy[(source_id, 'x', 'value')]
            y = xy[(source_id, 'y', 'value')]

            # for each source, get layers where x- and y-coordinate are defined.
            mask_i = mask | np.isnan(x) | np.isnan(y)
            if common_slope:
                mask_i |= mean_mask

            # make sure number of non-masked data points is larger than degree of polynomial
            if (~mask_i).sum() > order:
                if common_slope:
                    x_fit_i = mean_x_fit + fit_function.fit(
                        self.wave[~mask_i], x[~mask_i] - mean_x_fit(self.wave[~mask_i]), order,
                        domain=mean_x_fit.domain)
                    y_fit_i = mean_y_fit + fit_function.fit(
                        self.wave[~mask_i], y[~mask_i] - mean_y_fit(self.wave[~mask_i]), order,
                        domain=mean_y_fit.domain)
                else:
                    x_fit_i = fit_function.fit(self.wave[~mask_i], x[~mask_i], order)
                    y_fit_i = fit_function.fit(self.wave[~mask_i], y[~mask_i], order)

                # update IFS coordinates data frame
                self.ifs_coordinates[(source_id, 'x', 'fit')] = x_fit_i(self.wave)
                self.ifs_coordinates[(source_id, 'y', 'fit')] = y_fit_i(self.wave)

                # update dictionary containing polynomial fits
                self.ifs_coordinates_fitted[source_id] = True
                self.ifs_coordinates_fit_functions[source_id] = [x_fit_i, y_fit_i]

            # if not, trigger error message and skip source
            else:
                logger.error("Source #{0} has insufficient wavelength information for polynomial fit.".format(
                    source_id))

            # log progress
            logging.debug("Polynomial fit to source #{0} [{1}/{2}]".format(source_id, n, len(ids)))
            if statusBar is not None:
                statusBar.showMessage("Polynomial fit to source #{0} [{1}/{2}]".format(source_id, n, len(ids)), 5000)

    ##############################################################################
    #                             Output methods                                 #
    ##############################################################################
    def make_catalogue_hdu(self):
        """
        Write the source catalogue to an astropy.io.fits.BinTableHDU.

        Note that only sources that are used in a fit, i.e. have
        Sources.status >= 0, are included in the table. The returned table is
        named 'SOURCES'.
        
        Returns
        -------
        sources : astropy.io.fits.BinTableHDU
            The source catalogue as a FITS binary table extension.
        """
        logger.info("Saving catalogue in FITS BinTableHDU 'SOURCES'...")

        try:
            select = self.status >= 0
        except ValueError:
            select = np.arange(self.n_sources, dtype=np.int32)

        # make sure 'id' is among columns (as the index is not stored by the astropy routine)
        if 'id' not in self.catalog.columns:
            self.catalog.loc[:, 'id'] = self.catalog.index.values

        # convert catalog to astropy Table
        out_table = Table.from_pandas(self.catalog[select])

        # then convert to HDU
        out_hdu = fits.BinTableHDU(out_table.as_array(), name='SOURCES')

        # return
        return out_hdu

    def make_spectra_hdu(self, components=None):
        """
        Write the spectra to an astropy.io.fits.ImageHDU.

        The data in the HDU will be 3dimensional, with the zeroth axis (python
        counting) being the dispersion axis, the first axis containing the
        components, and the second containing either the spectrum or its
        uncertainties.

        The HDU that is returned is named 'SPECTRA'. If dedicated wavelength
        data exist for each spectrum, they will be stored in a separate HDU
        named 'WAVE' with a 2dim. data array.
        
        WARNING: If not all spectra that should be included in the HDU are
        available, i.e. have been loaded using Sources.load_spectra(), then
        all requested spectra are reloaded. This means that changes to the
        Sources.data array that were made after the last call to
        Sources.load_spectra() will be lost.

        Parameters
        ----------
        components : int or array_like, optional
            The component(s) for which the spectra should be included in the
            HDU. If None is provided, all components are included. If the
            keyword is set to "current", the current Sources.data array is
            stored.
        
        Returns
        -------
        astropy.io.fits.ImageHDU
            The HDU containing the spectra as a 3dim. array. The standard
            dispersion keywords (CRVAL3, CDELT3, ...) are included in the
            FITS header.
        astropy.io.fits.ImageHDU or None
            If each spectrum has a dedicated wavelength array, they will be
            stored in a separate HDU.
        """
        logger.info("Saving spectra in FITS ImageHDU 'SPECTRA'...")

        # get the components for which the spectra are requested
        if components is None:
            components = self.spectrum_loaded.index
        elif isinstance(components, (int, np.integer, list, np.ndarray)):
            components = pd.Index(np.atleast_1d(components))
        elif components == "current":
            components = self.spectrum_loaded.index[self.spectrum_loaded]
        else:
            logger.error('Unsupported value for keyword "components": {0}'.format(components))

        # check if spectra have been loaded for all requested components
        if self.spectrum_loaded is None:
            self.load_spectra(components=components)
        else:
            if not self.spectrum_loaded[components].all():
                logger.warning("Reloading spectra, modifications to Sources.data will be lost.")
                self.load_spectra(components=components)

        if not self.spectrum_loaded.all():
            # Note that currently the class (via Sources.open_spectra_hdu()) does not support loading HDUs that
            # contain the spectra for a subset of the components. So a warning is triggered if an HDU that does
            # not include all spectra will be created.
            logger.warning("Returned HDU will not contain spectra of all components.")

        # check if dedicated wavelength information exist for each source. If so, it will be stored in a
        # separate HDU
        has_wave = 'wave' in self.data.columns.levels[1]

        # Pandas DataFrames store the data as 2dimensional arrays with shape (index.size, columns.size),
        # even when a MultiIndex is used. For this reason, we have to reshape that data before adding it to
        # the HDU.
        output_data = self.data.values.reshape((self.data.shape[0], -1, 3 if has_wave else 2))

        # prepare HDU(s)
        if has_wave:
            spectra_hdu = fits.ImageHDU(output_data[..., :-1], name='SPECTRA')
            wave_hdu = fits.ImageHDU(output_data[..., -1], name='WAVE')
        else:  # otherwise, include dispersion information in Fits-Header
            spectra_hdu = fits.ImageHDU(output_data, name='SPECTRA')
            wave_hdu = None

        # add dispersion information
        spectra_hdu.header["CRPIX3"] = self.crpix
        spectra_hdu.header["CRVAL3"] = self.crval
        spectra_hdu.header["CDELT3"] = self.cdelt
        if self.ctype is not None:
            spectra_hdu.header["CTYPE3"] = self.ctype
        if self.cunit is not None:
            spectra_hdu.header["CUNIT3"] = self.cunit

        # return
        return spectra_hdu, wave_hdu

    def make_ifs_coordinates_hdu(self, ids=None, fit=False, extname='POSITIONS'):
        """
        Create an astropy.io.fits.ImageHDU containing the IFS coordinates of
        (a subset of) the sources or an astropy.io.fits.BinTableHDU containing
        the parameters of the polynomial fits to them.

        When the IFS coordinates are stored the array will always be 3dim.,
        with the zeroth axis (python counting) containing the dispersion
        information, the first axis containing the individual sources, and the
        second axis providing x or y coordinates.

        When the polynomial fits are stored, the  table  will contain five
        columns, providing the ID, the type of fit function along x and y, and
        the fit parameters along x- and y for every source.

        WARNING: If not all IFS coordinates that should be included in the HDU
        are available, i.e. have been loaded previously via
        Sources.load_ifs_coordinates(), then all requested IFS coordinates are
        reloaded. This means that changes to the Sources.ifs_coordinates array
        that were made after the last call to Sources.load_ifs_coordinates()
        will be lost. This does not concern the polynomial fits to the
        coordinates as the fit parameters are stored in a dictionary and will
        be used to recover the fits when reloading the data.
        
        Parameters
        ----------
        ids : array_like or int, optional
            The ID(s) of the sources for which the IFS coordinates should be
            written to the HDU. If None is provided, all point sources from
            the catalogue are included. If the keyword is set to 'current',
            the IFS coordinates currently available in Sources.ifs_coordinates
            are used.
        fit : boolean, optional
            A flag indicating if the fit to the data should be written to a
            HDU instead of the data.
        extname : string, optional
            The name given to the HDU, i.e. the header keyword EXTNAME.

        Returns
        -------
        astropy.io.fits.ImageHDU or astropy.io.fits.TableHDU
            The HDU containing the coordinates . It will contain a 3dim. array
            unless fit=True. In that case, it will contain a binary FITS table
            with the parameters of the polynomial fits to the individual IFS
            coordinates.
        """
        if not fit:
            logger.info("Saving IFS coordinates to ImageHDU '{0}'...".format(extname))

        # check which coordinates should be saved
        if ids is None:
            ids = self.ids[(self.status > -1) & (self.status < 4)]
        elif isinstance(ids, (int, np.integer, list, np.ndarray)):
            ids = pd.Index(np.atleast_1d(ids))
        elif ids == "current":
            if not fit:
                ids = self.ids[self.ifs_coordinates_loaded]
            else:
                ids = self.ids[self.ifs_coordinates_fitted]

        # if polynomial fits should be saved
        if fit:
            logger.info("Saving fits to IFS coordinates to BinTableHDU '{0}'...".format(extname))

            # fits are saved for the requested sources that have polynomial fits available
            ids_to_save = np.intersect1d(ids.values, list(self.ifs_coordinates_fit_functions.keys()))

            # get the number of sources for which fits should be stored
            n_fit = ids_to_save.size

            if n_fit < ids.size:
                logger.error("Position fits not available for some of the requested sources.")
                if n_fit == 0:
                    logger.error("Cannot create BinTableHDU '{0}'.".format(extname))
                    return None

            # get the maximum polynomial degree used in the fits. Note that this would not be necessary if one would
            # use the feature that FITS tables can store arrays of variable length
            # (http://docs.astropy.org/en/tmp-v0.1-wcython/io/fits/usage/unfamiliar.html)
            # Unfortunately, the implementation in astropy seems buggy and I always get an array full of zeros when
            # opening the file again. Therefore, in case different degrees have been used in the fits, the higher
            # orders of the low-order polynomials are padded with zeros.
            x_max_degree = max([self.ifs_coordinates_fit_functions[i][0].degree() for i in ids_to_save])
            y_max_degree = max([self.ifs_coordinates_fit_functions[i][1].degree() for i in ids_to_save])

            # prepare the arrays to hold the data
            source_ids = ids.values
            x_function = np.zeros((n_fit, ), dtype=np.int16)
            x_coefficients = np.zeros((n_fit, x_max_degree + 1), dtype=np.float32)
            y_function = np.zeros((n_fit, ), dtype=np.int32)
            y_coefficients = np.zeros((n_fit, y_max_degree + 1), dtype=np.float32)

            #  loop over all sources that have polynomial fits available
            domain = None
            for i, source_id in enumerate(ids):

                x_polynom, y_polynom = self.ifs_coordinates_fit_functions[source_id]

                # make sure all fit functions use the same domain, this is needed because the domain set by the fit
                # is the min and max of the fitted range and this if different from the default one used by the
                # __init__ methods. Since currently the start and end of the domain are stored as HIERARCH-keywords,
                # the domain must be the same for all fits.
                if domain is None:
                    domain = x_polynom.domain
                elif (x_polynom.domain != domain).any():
                    x_polynom = x_polynom.convert(domain=domain)
                if (y_polynom.domain != domain).any():
                    y_polynom = y_polynom.convert(domain=domain)

                # to get fit function, first check which class the instance originates from, then translate name to
                # an integer
                for name, polynom_type in self.available_polynoms.items():
                    if isinstance(x_polynom, polynom_type):
                        x_function[i] = self.polyident[name]
                    if isinstance(y_polynom, polynom_type):
                        y_function[i] = self.polyident[name]

                # add coefficients of polynomial, take care of zero-padding
                x_coefficients[i, :(x_polynom.degree() + 1)] = x_polynom.coef
                y_coefficients[i, :(y_polynom.degree() + 1)] = y_polynom.coef

            # prepare the output table
            out_hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name="id", format="J", array=source_ids),
                 fits.Column(name="x_function", format="I", array=x_function),
                 fits.Column(name="x_coefficients", format="{0}E".format(x_coefficients.shape[1]),
                             array=x_coefficients),
                 fits.Column(name="y_function", format="I", array=y_function),
                 fits.Column(name="y_coefficients", format="{0}E".format(y_coefficients.shape[1]),
                             array=y_coefficients)])

            out_hdu.header["EXTNAME"] = extname

            # save domain in header
            out_hdu.header["hierarch PAMPELMUSE POSITIONS DOMAIN START"] = domain[0]
            out_hdu.header["hierarch PAMPELMUSE POSITIONS DOMAIN END"] = domain[1]

            # save the polynomial identifications in the header
            for name, index in self.polyident.items():
                out_hdu.header["hierarch PAMPELMUSE POSITIONS POLY{0}".format(index)] = name

        # directly save IFS coordinates
        else:
            # make sure IFS coordinates are loaded for all sources AND across the entire dispersion range
            if not self.ifs_coordinates_loaded[ids].all() or self.ifs_coordinates.shape[0] != self.n_dispersion:
                logger.warning("Reloading IFS coordinates, modifications to Sources.ifs_coordinates will be lost.")
                self.load_ifs_coordinates(ids=ids)

            # Pandas DataFrames store the data as 2dimensional arrays with shape (index.size, columns.size),
            # even when a MultiIndex is used. For this reason, we have to reshape that data before adding it to
            # the HDU.
            output_data = self.ifs_coordinates.loc[:, (ids, slice(None), 'value')].values

            out_hdu = fits.ImageHDU(output_data.reshape((output_data.shape[0], -1, 2)), name=extname)

            # create entry in header about ID of each source and the row containing its position
            for i, source_id in enumerate(ids):
                out_hdu.header["hierarch PAMPELMUSE POSITIONS ID{0}".format(i)] = source_id

        return out_hdu
