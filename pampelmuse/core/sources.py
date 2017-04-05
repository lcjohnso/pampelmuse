"""
sources.py
==========
Copyright 2013-2017 Sebastian Kamann

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
function pampelmuse.core.sources.match_on_id
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

Latest SVN revision
-------------------
408, 2017/04/03
"""
import collections
import logging
import re
import warnings
import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Legendre, Hermite, Polynomial
from astropy.io import fits
from ..core.coordinates import Transformation
from ..filters import get_filter

# check if PyQt is available 
try:
    from PyQt4 import QtCore
except ImportError:
    QtCore = None


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 408


logger = logging.getLogger(__name__)


def match_on_id(parent, child):
    """
    Compare to 1dim. lists for common entries. For each element in 'child'
    that is also part of the 'parent' list, the index of the element in the
    'parent' list is returned.

    Parameters
    ----------
    parent : array_like, 1dim.
        The list that that is scanned
    child : array_like, 1dim.
        The list containing the entries that 'parent' is scanned for.

    Returns
    -------
    indices : nd_array
        The index of every 'child'-entry in the 'parent' list (if found).
    """
    c = np.asarray(child, dtype=np.int32)
    p = np.asarray(parent, dtype=np.int32)

    # The following statement works but quickly eats up the whole memory if p or c are large
    # return np.nonzero((c[:, np.newaxis]-p) == 0)[1]

    # Following the suggestion from http://stackoverflow.com/questions/17374547/matching-elements-in-numpy-array
    sort_idx = np.argsort(p)
    sorted_insert = np.searchsorted(p[sort_idx], c, side="left")

    # The following is equivalent to unsorted_insert = sort_idx[sorted_insert] but faster
    unsorted_insert = np.take(sort_idx, sorted_insert)
    return unsorted_insert


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

        # open the data
        self.catalog = catalog

        # check that source IDs are unique
        if self.ids.duplicated().any():
            logger.critical('Provided source catalog contains {0} duplicated IDs.'.format(self.ids.duplicated().sum()))
            return

        # check if other columns are available - if not, add them and fill with NaN
        for column in ['x', 'y', 'ra', 'dec', 'mag']:
            if column not in self.catalog.columns:
                dtype = np.float64 if column in ['ra', 'dec'] else np.float32
                self.catalog[column] = pd.Series(np.zeros(self.catalog.shape[1], dtype=dtype),
                                                 index=self.catalog.index)
                self.catalog[column] = np.nan

        # if available, get status of each source in analysis
        if 'status' not in self.catalog.columns:
            self.catalog['status'] = pd.Series(np.zeros((self.n_sources,), dtype=np.int16), index=self.catalog.index)
            self.catalog.at[self.ids < 0, 'status'] = 4  # assume IDs<0 are used for background components

        # if available, get primary source of each component, otherwise assume each source has its own component
        if 'primary' not in self.catalog.columns:
            self.catalog['primary'] = self.ids

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
        self.data_wave = None  # If available, the wavelength information for the spectrum of each component
        self.ifs_coordinates = None  # The x/y coordinates of the sources on the IFS spaxel array

        # place-holders for HDUs containing fluxes or IFS coordinates
        self.data_hdu = None
        self.data_wave_hdu = None
        self.ifs_coordinates_data_hdu = None
        self.ifs_coordinates_fit_hdu = None

        # Initialize the dispersion information
        self.ndisp = 1
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
        value : array-like
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
        if not isinstance(value, (list, np.ndarray)):
            raise IOError('Expected array-like variable, got {0}'.format(type(value)))
        if not len(value) == self.n_sources:
            raise ValueError('Array-size does not match number of sources in catalog.')
        return pd.Series(value, index=self.catalog.index)

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
            self.catalog['x'] = value

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
            self.catalog['y'] = value

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
            self.catalog['ra'] = value

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
            self.catalog['ra'] = value

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
            self.catalog['status'] = value
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
            self.catalog['status'] = value
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
                    specrow[status == 4] += 1
                unresolved_row = specrow[status == 4].min() - 1
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
                    while self.catalog.at[j, 'primary'] != self.ids[j]:  # continue until current source is primary

                        # exclude source if primary source has status = -1.
                        if self.catalog.at[j, 'primary'] not in self.ids:
                            logging.error("No 'primary' counterpart for source #{0}".format(self.ids[i]))
                            logging.error("Source #{0} excluded".format(self.ids[i]))
                            self.status[i] = -1
                            break
                        secondary.append(j)

                    specrow[secondary] = specrow[j]
                elif status[i] == 0:
                    specrow[i] = unresolved_row

            # set column in sources catalog
            self.catalog['specrow'] = specrow

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
        # group sources by index/ID of primary source
        grouped = self.catalog[['mag', 'primary']].groupby('primary')

        # calculate total fluxes in each group and broadcast to full catalog
        component_fluxes = grouped['mag'].transform(lambda x: np.sum(10**(-0.4*x)))

        # get flux ratio between each source and its component
        flux_weights = 10**(-0.4*self.catalog['mag'])/component_fluxes

        return 10**(-0.4*flux_weights)

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
        return self.status == 4

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
            for specrow, ids in self.specrow.items():
                log_string = "   {0:8d}".format(specrow)

                for i, source_id in enumerate(ids):
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
            The HDU containing the spectra and their uncertainties for. The
            shape of the data array must be [python counting]:
            (Sources.n_disp, Sources.n_components, 2)
            The HDU is accessible via attribute Sources.data_hdu afterwards.
        data_wave_hdu : instance of astropy.io.fits.ImageHDU, optional
            This parameter can be used to provide a separate wavelength array
            for each spectrum in 'hdu'. If provided, must have a shape of
            (Sources.n_disp, Sources.n_components).
        """
        # sanity checks on provided HDUs
        for name, value in {'DATA': hdu, 'WAVE': data_wave_hdu}.items():
            if value is not None and not isinstance(value, fits.hdu.image._ImageBaseHDU):
                logger.error("{0}-HDU has invalid type '{1}'.".format(name, type(value)))
                return
            elif value.header["NAXIS"] != 3:
                logger.error("{0}-HDU must be 3D array, not {1}D.".format(name, value.header["NAXIS"]))

            if 'EXTNAME' in value.header:
                logger.info('Reading data from HDU "{0}".'.format(value.header['EXTNAME']))
                name = value.header['EXTNAME']

            # check if dispersion information is valid
            if self.ndisp == 1 and name == 'DATA':
                self.ndisp = hdu.header['NAXIS3']
            elif value.header['NAXIS3'] != self.ndisp:
                logger.error('HDU "{0}" has invalid dispersion information.'.format(name))
                return

            # check if number of spectra is valid
            if value.header['NAXIS2'] != self.n_components:
                logger.error('HDU "{0}" has invalid number of components.'.format(name))
                return

        self.data_hdu = hdu
        self.data_wave_hdu = data_wave_hdu

        # if no dedicated wavelength HDU was provided, try to update dispersion information from header of data HDU
        if self.data_wave_hdu is None:
            for key in ["CRVAL", "CDELT", "CRPIX", "CTYPE", "CUNIT"]:
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
            The component(s) for which the spectrum should be loaded. If none are
            provided, the method will load the spectrum of every component in the
            instance. Remember that spectra are not loaded per ID but per component,
            for this reason no IDs should be provided.
        layers : integer  optional
            The indices of the layers over which the spectra should be loaded. If
            none are provided, they are loaded for the whole wavelength range.
        init_wave : boolean, optional
            Flag indicating if an array containing the wavelength of each component
            should be initialized. Only used if no wavelength HDU has been loaded
            via Sources.open_spectra_hdu so far.
        """
        # sanity checks on provided IDs and layers
        if components is not None and not isinstance(components, (int, np.integer, list, np.ndarray)):
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
            layers = np.arange(self.ndisp)
        else:
            layers = np.atleast_1d(layers)

        # initialize data frame for spectra data.
        # rows contain dispersion data using the actual wavelengths as a pandas Float64Index
        index = pd.Float64Index(self.wave[layers])

        # columns contain components and ('flux', 'sigma'[, 'wave']) in pandas.MultiIndex
        properties = ['flux', 'sigma'] if self.data_wave_hdu is None else ['flux', 'sigma', 'wave']
        columns = pd.MultiIndex.from_product([properties, components], names=['property', 'component'])

        self.data = pd.DataFrame(np.zeros((index.size, columns.size), dtype=np.float32), index=index, columns=columns)

        # fill array using data from Spectrum HDU if one is available
        if self.data_hdu is not None:
            self.data.loc[:, 'flux'] = self.data_hdu.data[layers, components, 0]
            self.data.loc[:, 'sigma'] = self.data_hdu.data[layers, components, 1]
        if self.data_wave_hdu is not None:
            self.data.loc[:, 'wave'] = self.data_wave_hdu.data[layers, components]

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
        if self._wave is not None and self._wave.size != self.ndisp:
            self._wave = None

        if self._wave is None:
            self._wave = np.arange(self.ndisp, dtype=np.float64)  # shape (k,)
            self._wave -= (self.crpix - 1)
            self._wave *= self.cdelt
            self._wave += self.crval

        return self._wave

    def get_magnitude(self, component, passband=None):
        """
        This method calculates the magnitudes for one or more spectra in a
        given filter.

        The method assumes that the fluxes are provided in erg/s/cm2/Angstrom.
        If that is not the case, the user has to perform the correction for
        the resulting magnitude offset manually. The returned magnitude will
        be in the AB system.

        Parameters
        ----------
        component : int or array_like, shape (N,)
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
        # load spectra of requested components if necessary
        if not self.spectrum_loaded[component].all():
            self.load_spectra(components=component)

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
        for component in np.atleast_1d(component):
            # get flux
            flux = self.data[('flux', component)]

            # get wavelength, check if each spectrum requires own wavelength array or global one can be used.
            if self.data_wave_hdu is None:
                wave = factor*self.wave
            else:
                wave = factor*self.data[('wave', component)]

            # resample throughput curve (if available, otherwise create whitelight image)
            if throughput_curve is not None:
                throughput = throughput_curve(wave)
            else:
                throughput = np.ones(self.ndisp, dtype='<f4')

            # calculate magnitude and add to list
            integrated_flux = np.nansum(throughput*flux*(wave**2))/np.nansum(throughput)
            if integrated_flux <= 0:
                logging.warning('Cannot determined magnitude for component #{0} with negative flux.'.format(component))
                magnitudes.append(np.nan)
            else:
                magnitudes.append(-2.5*np.log10(3.34e4*integrated_flux) + 8.9)

        return pd.Series(magnitudes, index=np.atleast_1d(component))

    def provideComponentNames(self):
        """
        Method to provide the IDs of the primary sources for each component as
        a QStringList.

        Returns
        -------
        names : QStringList
           The names of the components
        """
        # add sources with own spectrum, i.e. have status between 2 and 4
        names = QtCore.QStringList(self.ids[(self.status >= 2) & (self.status <= 4)].astype("|S10"))

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
        wave : nd_array
            A 1dim. array containing the wavelength of each pixel of the
            spectrum.
        data : nd_array
            A 1dim. array containing the the requested spectrum.
        """
        if self.spectrum_loaded is None and self.data_hdu is None:
            logging.error("No spectra have been loaded using Sources.load_spectra() or Sources.open_spectra_hdu().")
            return None, None
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
            return self.wave, self.data[('flux', i)]
        else:
            return self.data[('wave', i)], self.data[('flux', i)]

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
        named "id", "xfct", "xprm", "yfct", and "yprm", and providing ID, the
        type of polynomial (fct) and the fit coefficients (prm) along x and y
        for each source.
        
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
            logger.info('Loading polynomial fits to coordinates from HDU {0} ...'.format(hdu.header.get('EXTNAME', '')))

            # check if required columns are present in header
            for name in ["id", "xfct", "xprm", "yfct", "yprm"]:
                if name not in hdu.columns.names:
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
                xfctname = ident[row["xfct"]]  # the name of the polynomial as a string
                yfctname = ident[row["yfct"]]

                xfct = self.available_polynoms[xfctname]  # the actual polynomial class
                yfct = self.available_polynoms[yfctname]

                # save polynomials for current source
                self.ifs_coordinates_fit_functions[row["id"]] = [xfct(row["xprm"], domain=domain),
                                                                 yfct(row["yprm"], domain=domain)]
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
            if self.ndisp == 1:
                self.ndisp = hdu.header["NAXIS3"]
            elif hdu.header["NAXIS3"] != self.ndisp:
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

    def load_ifs_coordinates(self, ids=None, layers=None, interpolate=False):
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
             transformation is A=1, B=C=D=0)
             
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
        interpolate : boolean, optional
            A flag indicating if the positions should be interpolated for
            layers where there is no information available.
        """
        # FOR DEBUGGING: check where the call was triggered
        if logger.getEffectiveLevel() < 20:
            import inspect
            import time
            t0 = time.time()
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            logging.debug("Call to Sources.load_ifs_coordinates() triggered by: {0} in {1}[@{2}]".format(
                calframe[1][3], calframe[1][1], calframe[1][2]))

        # sanity checks on provided IDs and layers

        # load all (stellar) components if None was provided
        if ids is None:
            ids = self.ids[~self.is_background]
        elif not isinstance(ids, (int, np.integer, list, np.ndarray)):
            logger.error("Parameter 'ids' must be integer or array-like, not {0}.".format(type(ids)))
            return
        # make sure components are available as a pandas index
        else:
            ids = pd.Index(ids)
            # check if all provided components are actually in the source catalogue
            if not np.in1d(ids.values, self.ids.values, assume_unique=True).all():
                logger.error('Some requested components fall outside available range.')
                return

        # check if requested layers are valid
        if layers is not None and not isinstance(layers, (int, np.integer, list, np.ndarray)):
            logger.error('Parameter "layers" must be integer or array-like, not {0}.'.format(type(layers)))
            return
        elif layers is None:
            layers = np.arange(self.ndisp)
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
        types = ['value', 'fit']
        columns = pd.MultiIndex.from_product([types, ids, axes], names=['type', 'id', 'axis'])

        self.ifs_coordinates = pd.DataFrame(np.zeros((index.size, columns.size), dtype=np.float32),
                                            index=index, columns=columns)

        if logger.getEffectiveLevel() < 20:
            t1 = time.time()

        # start filling data frame
        # first choice: if available, fill in IFS coordinates using open FITS HDU
        if self.ifs_coordinates_data_hdu is not None:
            # get required chunk of data from HDU
            data = self.ifs_coordinates_data_hdu.section[layers[0]:layers[-1] + 1]
            self.ifs_coordinates['value'] = data

        # second choice: use coordinate transformation if available
        elif self.transformation.valid:
            xy_ref = np.vstack((self.x[ids], self.y[ids]))

            # coordinate transformation may needs to be resized
            if self.transformation.ndisp != self.ndisp:
                logger.warning('Adding dispersion information to coordinate transformation.')
                self.transformation.resize(self.ndisp)
            xy_ifs = self.transformation(xy_ref)
            self.ifs_coordinates.value = xy_ifs

        # third choice: just use reference coordinates
        else:
            xy_ref = np.vstack((self.x[ids], self.y[ids]))
            self.ifs_coordinates[:, 'value'] = xy_ref

        if logger.getEffectiveLevel() < 20:
            t2 = time.time()

        # the polynomial fits can be obtained from the dictionary of previously performed fits
        for source_id in ids:
            if source_id in self.ifs_coordinates_fit_functions.keys():
                xfit, yfit = self.ifs_coordinates_fit_functions[source_id]
                self.ifs_coordinates[('fit', source_id, 'x')] = xfit(self.ifs_coordinates.index)
                self.ifs_coordinates[('fit', source_id, 'x')] = yfit(self.ifs_coordinates.index)

        # keep track of for which sources IFS coordinates have been loaded
        self.ifs_coordinates_loaded = pd.Series(np.in1d(np.arange(self.ids.values, dtype=np.int32), ids),
                                                index=self.ids)

        if logger.getEffectiveLevel() < 20:
            t3 = time.time()
            logger.debug("Time required to initialize array: {0}s".format(t1 - t0))
            logger.debug("Time required to load IFS coordinates: {0}s".format(t2 - t1))
            logger.debug("Time required to load fits to IFS coordinates: {0}s".format(t3 - t2))

        # TODO!
        # # When using an open HDU to fill in the IFS coordinates from an open HDU, the data are loaded for all
        # # requested sources across the full spectral range. This is required to allow for interpolation of NaN
        # # values if requested.
        # hdu_indices = match_on_id(self.ifs_coordinates_data_id, ids[processed])
        # ifs_coords = self.ifs_coordinates_data_hdu.data[:, hdu_indices]
        # if not interpolate or not np.isnan(ifs_coords[layers]).any():
        #     self.ifs_coordinates[:, processed, :, 0:1] = ifs_coords[layers][..., np.newaxis]
        # else:
        #     logging.warning("IFS coordinates include interpolated values.")
        #     for i in np.flatnonzero(processed):
        #         for j in range(2):
        #             valid = ~np.isnan(ifs_coords[:, i, j])
        #             if valid.any():
        #                 self.ifs_coordinates[:, i, j, 0] = np.interp(layers, np.flatnonzero(valid),
        #                                                              ifs_coords[valid, i, j])
        #             else:
        #                 self.ifs_coordinates[:, i, j, 0] = np.asarray([self.x[self.ifs_coordinates_loaded][i],
        #                                                                self.y[self.ifs_coordinates_loaded][i]])[j]
        #
        #         logging.warning("IFS coordinates include interpolated values.")
        #         for i in src_indices:
        #             for j in range(2):
        #                 valid = ~np.isnan(ifs_coords[:, i, j])
        #                 if valid.sum() > 2:
        #                     self.ifs_coordinates[:, i, j, 0] = np.interp(
        #                     layers, np.flatnonzero(valid), ifs_coords[valid, i, j])
        #                 else:
        #                     self.ifs_coordinates[:, i, j, 0] = np.mean(ifs_coords[valid, i, j])
        #         # 4th entry in square brackets must be slice; if an index (i.e. 0) is used, the shape of the resulting
        #         # sub-array is not as expected

    def fit_coordinate_transformation(self, index=slice(None), **kwargs):
        """
        Calculate/Initialize a new coordinate transformation using the input
        coordinates and the IFS coordinates. For this method to work, the
        properties Sources.x and Sources.y must be defined and the method
        Sources.load_ifs_coordinates' must have been called previously. The
        method will fit the transformation from reference to IFS coordinates.

        Parameters
        ----------
        index : int or 'slice'-instance, optional
            The layer(s) for which the coordinate transformation should be
            calculated. The default is to use the entire cube.
        """
        if self.x is None or self.y is None:
            logging.error("Cannot determine coordinate transformation because no reference coordinates are available.")
            return

        if self.ifs_coordinates is None:
            logging.error("Cannot determine coordinate transformation: because no IFS coordinates are available.")
            return

        # check if IFS coordinates are available for at least 2 sources.
        if self.ifs_coordinates_loaded.sum() < 2:
            logging.error(
                "Cannot determine coordinate transformation because <2 sources have IFS coordinates available.")
            return

        # if integer has been provided as keyword 'index', translate it into a slice
        if isinstance(index, (int, np.int)):
            index = slice(index, index + 1, 1)

        # carry out fit
        ref_coordinates = np.vstack((self.x, self.y)).T[self.ifs_coordinates_loaded]
        self.transformation = Transformation.from_coordinates(ref_coordinates, self.ifs_coordinates[index][..., 0],
                                                              ids=self.ID[self.ifs_coordinates_loaded], **kwargs)

    def providePosIDs(self, minStatus=1):
        """
        Provide a list of all sources in the catalogue that have a certain
        'status'.

        Parameters
        ----------
        minStatus : int, optional
            The minimum 'status' of sources still included  the list

        Returns
        -------
        IDs : QtCore.QStringList
            The list is returned as a list of QStrings so that it can be
            used by PyQt-applications.
        """
        posSel = np.in1d(self.status, np.arange(minStatus, 4))

        return QtCore.QStringList(self.ID[posSel].astype("|S10"))

    def viewIfsCoordinates(self, ID):
        """
        Return the IFS x- and y-coordinate of an individual source. If a
        polynomial fit for this source exists, it is also returned. Otherwise,
        the second part of the returned array will be empty.

        Parameters
        ----------
        ID : int
            The ID of the source for which the coordinates should be returned.

        Returns
        -------
        ifs_coordinates : ndarray
            The requested IFU coordinate and its polynomial fit. The first axis
            of the array is the dispersion axis, the second indicates x or y, and
            the third indicates data or the fit.
        """
        # get index belonging to ID
        assert isinstance(ID, (int, np.integer)), "ID must be an integer, not {0}.".format(
            re.findall(r"'(.*)'", str(type(ID)))[0])
        try:
            i = np.flatnonzero(self.ID == ID)[0]
        except IndexError:
            raise IndexError("Provided ID #{0} not in source catalogue.".format(ID))

        # load the coordinates if they have not been loaded previously
        if not self.ifs_coordinates_loaded[i] or self.ifs_coordinates.shape[0] < self.ndisp:
            self.load_ifs_coordinates(ids=ID)

        j = np.cumsum(self.ifs_coordinates_loaded)[i] - 1  # index of the spectrum in the Sources.data array
        return self.ifs_coordinates[:, j]

    def fitIfsCoordinates(self, ids, order, mask=None, function="plain", commonSlope=False, statusBar=None,
                          instrument=None):
        """
        Method to fit the source positions as a function of wavelength. Two
        basic modes are supported currently:
        (a) fit a polynomial individually to each position (still the default)
        (b) find a polynomial fit for all resolved sources and shift it along
            the selected axis to obtain a position for each source
            (recommended)

        Parameters
        ----------
        ids : array_like or None
            The ID[s] of the sources that should be fitted. Use None to fit
            all point sources, i.e. all sources with Sources.status<4.
        order : int
            The order of the polynomial used to perform the fit
        mask : array_like or None, optional
            An optional bad-pixel spectrumMask. If specified, must have as many pixels
            as spectral pixels are available. Bad pixels should be set to
            True, good ones to False.
        function : string, optional
            The fitting function that should be used. Currently a
            step-polynomial (step) or a Chebyshev polynomial (chebyshev) are
            supported.
        commonSlope : boolean, optional
            Flag indicating whether a common polynomial should be used to model
            the wavelength dependency of all sources.
        instrument : instance of pampelmuse.core.instruments.Instrument, optional
            The IFS instrument used to obtain the data. This information is used
            to restrict the selection to sources inside the FoV when determining
            the mean slope of a coordinate with wavelength.
        """
        # find the indices of the source(s) to be fitted in the catalogue
        if ids is None:
            indices_to_fit = np.flatnonzero(self.status < 4)
        else:
            indices_to_fit = match_on_id(self.ID, np.atleast_1d(ids))

        # load IFS coordinates for all sources _WITHOUT_ interpolation
        self.load_ifs_coordinates(ids=self.ID[indices_to_fit])

        if not function.lower() in self.available.keys():
            logging.critical("Unknown fitting function provided: {0}".format(function))
        fit_function = self.available[function.lower()]

        # create pseudo-mask if None provided
        valid = ~mask if mask is not None else np.ones((self.ndisp,), dtype=np.bool)

        # determine 'mean' coordinate if option 'commonSlope' selected
        if commonSlope:
            # need to create masked array from IFS coordinates in order to get useful mean values
            ifs_coordinates_ma = np.ma.MaskedArray(self.ifs_coordinates[..., 0],
                                                   np.isnan(self.ifs_coordinates[..., 0]))

            # sources used for creating mean slope are those with status 2 or 3
            used = ((self.status > 1) & (self.status < 4))[self.ifs_coordinates_loaded]

            # if "instrument" is known, then only use sources within FoV to determine common slope.
            if instrument is not None:
                used &= instrument.infov(ifs_coordinates_ma.mean(axis=0))

            # get mean coordinate along x and y plus boolean array indicating for which layers valid data exist
            meancoord_ma = np.ma.mean(ifs_coordinates_ma, axis=1)
            meancoord = meancoord_ma.filled(np.nan)
            meanvalid = valid[:, np.newaxis] & ~np.isnan(meancoord)

            # fit mean coordinates with requested order
            xfit = fit_function.fit(self.wave[meanvalid[:, 0]], meancoord[meanvalid[:, 0], 0], order)
            yfit = fit_function.fit(self.wave[meanvalid[:, 1]], meancoord[meanvalid[:, 1], 1], order)

            order = 0  # all individual coordinates are now "fitted" with zeroth order polynomials

        # loop over selected sources and get their polynomial coordinate fits
        for n, j in enumerate(indices_to_fit):

            # get index of source in array of IFS coordinates
            i = self.ifs_coordinates_loaded.cumsum()[j] - 1

            # for each source, get layers where x- and y-coordinate are defined. If it is not defined for any layer
            # just use reference coordinate and trigger warning
            valid_i = valid & np.any(~np.isnan(self.ifs_coordinates[:, i, :, 0]), axis=-1)

            if commonSlope:
                valid_i &= meanvalid.all(axis=1)
            if valid_i.sum() > order:
                wave = self.wave[valid_i]
                fx_wave = self.ifs_coordinates[valid_i, i, 0, 0]
                fy_wave = self.ifs_coordinates[valid_i, i, 1, 0]
            else:
                wave = self.wave[meanvalid.all(axis=1)] if commonSlope else self.wave
                fx_wave = self.x[j]*np.ones(meanvalid.all(axis=1).sum() if commonSlope else self.ndisp)
                fy_wave = self.y[j]*np.ones(meanvalid.all(axis=1).sum() if commonSlope else self.ndisp)
                logging.warning("Not enough successful fits for source #{0} - using reference coordinates.".format(
                    self.ID[j]))

            if commonSlope:
                xfit_i = xfit + fit_function.fit(wave, fx_wave - xfit(wave), order, domain=xfit.domain)
                yfit_i = yfit + fit_function.fit(wave, fy_wave - yfit(wave), order, domain=yfit.domain)
            else:
                xfit_i = fit_function.fit(wave, fx_wave, order)
                yfit_i = fit_function.fit(wave, fy_wave, order)

            self.ifs_coordinates_fit_functions[self.ID[j]] = [xfit_i, yfit_i]

            # update coordinate information for this source
            self.ifs_coordinates_fitted[j] = True
            self.ifs_coordinates[:, i, 0, 1] = np.where(valid_i, xfit_i(self.wave), np.nan)
            self.ifs_coordinates[:, i, 1, 1] = np.where(valid_i, yfit_i(self.wave), np.nan)

            logging.debug("Polynomial fit to source #{0} [{1}/{2}]".format(self.ID[j], n, len(indices_to_fit)))
            if statusBar is not None:
                statusBar.showMessage("Polynomial fit to source #{0} [{1}/{2}]".format(
                    self.ID[j], n, len(indices_to_fit)), 5000)

    ##############################################################################
    #              Source selection methods/properties                           #
    ##############################################################################

    def distance_to_brighter(self):
        """
        DEPRECATED - not used anymore since revision 118
        Calculate the distance of a source to the nearest source that has
        is brighter as the source itself. This calculation is performed
        for all sources that have IFS coordinates available.
        
        Returns
        -------
        mindist : ndarray
            The distance to the nearest brighter source in units of spaxels.
        nearsrc : ndarray
            The ID of the nearest brighter source.
        """

        logging.info("Determining distance to nearest brighter source...")

        # average IFS coordinates over dispersion axis
        x = self.ifs_coordinates[..., 0, 0].mean(axis=0)
        y = self.ifs_coordinates[..., 1, 0].mean(axis=0)
        mag = self._mag[self.ifs_coordinates_loaded]

        # prepare output arrays
        mindist = np.zeros(x.shape, dtype=np.float32)
        nearsrc = np.zeros(x.shape, dtype=np.int32)

        # loop over all source that have IFS coordinates available.
        for i in xrange(x.size):
            dist_i = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
            dist_i[(mag - mag[i]) >= 0] = 99.  # exclude all sources that are not brighter

            mindist[i] = dist_i.min()
            nearsrc[i] = self.ID[self.ifs_coordinates_loaded][dist_i.argmin()]

        return mindist, nearsrc

    def shoulder_drop(self, r):
        """
        DEPRECATED - not used anymore since revision 130 (or so)
        This method determines the shoulder drop or topographic prominence
        of the sources that is a measure for how strongly a source dominates
        its surroundings. The measure used here is the flux ratio of the source
        to the brightest source within distance r. This quantity is used to
        select the sources that are used to fit the PSF.
        For the exact definition of the topographic prominence, see
        http://en.wikipedia.org/wiki/Topographic_prominence. The definition
        here is slightly modified.
        
        Parameters
        ----------
        r : float
            The search radius around each source. Only sources with distances
            <r are used to measure the shoulder drop for a source.
            
        Returns
        -------
        shoulder_drop : ndarray
            For each source with IFS coordinates available, the flux ratio of
            the source to its brightest neighbor (within a distance r) is
            returned.
        """
        logging.info("Determining shoulder drop of sources...")

        x = self.ifs_coordinates[..., 0, 0].mean(axis=0)
        y = self.ifs_coordinates[..., 1, 0].mean(axis=0)
        mag = self._mag[self.ifs_coordinates_loaded]

        dist = np.sqrt((x - x[:, np.newaxis]) ** 2 + (y - y[:, np.newaxis]) ** 2)
        np.fill_diagonal(dist, 99.)

        offset = mag[:, np.newaxis] - mag
        offset[dist > r] = 99.

        return 10 ** (-0.4 * offset.min(axis=0))

    ##############################################################################
    #                             Output methods                                 #
    ##############################################################################
    def make_catalogue_hdu(self):
        """
        Write the source catalogue to an astropy.io.fits.BinTableHDU. Note that
        only sources that are used in a fit, i.e. have Sources.status>=0, are
        included in the table. The returned table is named 'SOURCES'.
        
        Returns
        -------
        sources : astropy.io.fits.BinTableHDU
            The source catalogue as a FITS binary table extension.
        """
        logging.info("Saving catalogue in FITS BinTableHDU 'SOURCES'...")

        try:
            select = np.nonzero(self._status >= 0)[0]
        except ValueError:
            select = np.arange(self.n_sources, dtype=np.int32)

        # sort by increasing magnitudes (if possible)
        if not np.isnan(self._mag).all():
            select = select[self._mag[select].argsort()]

        outcols = [fits.Column(name="id", format="J", array=self.ID[select])]

        formats = {"x": "E", "y": "E", "mag": "E", "ra": "D", "dec": "D", "status": "I", "specrow": "I", "primary": "J"}
        for attribute in ["x", "y", "mag", "ra", "dec", "status", "primary", "specrow"]:
            # if getattr(self, "_{0}".format(attribute)) is not None:
            logging.debug("Adding column '{0}' to table...".format(attribute))
            outcols.append(fits.Column(name=attribute, format=formats[attribute],
                                       array=getattr(self, attribute)[select]))

        return fits.BinTableHDU.from_columns(
                outcols, header=fits.Header(cards=[fits.Card(key="EXTNAME", value="SOURCES")]))

    def make_spectra_hdu(self, components=None):
        """
        Write the spectra to an astropy.io.fits.ImageHDU. There are different
        options to specify which spectra are stored in the HDU that can
        be controlled using the keyword 'components'. In any case, the data
        will be 3dim., with the first axis being the dispersion axis, the
        second referencing the sources and the third containing either the
        spectrum or its uncertainties. The HDU that is returned is named
        'SPECTRA'.
        
        WARNING: If not all spectra that should be included in the HDU are
        available, i.e. have been loaded using Sources.load_spectra(), then
        all requested spectra are reloaded. This means that changes to the
        Sources.data array that were made after the last call to
        Sources.load_spectra() will be lost.

        Parameters
        ----------
        components : int or array_like, optional
            The component(s) for which the spectra should be included in the
            HDU. If None is provided, all spectra are included. If the key-
            word is set to "current", the current Sources.data array is
            stored.
        
        Returns
        -------
        spectra : astropy.io.fits.ImageHDU
            The HDU containing the spectra as a 3dim. array. The standard
            dispersion keywords (CRVAL3, CDELT3, ...) are included in the
            FITS header.
        """
        logging.info("Saving spectra in FITS ImageHDU 'SPECTRA'...")

        # get the components for which the spectra are requested
        if components is None:
            components = np.arange(self.n_components, dtype=np.int32)
        elif isinstance(components, (int, np.integer, list, np.ndarray)):
            components = np.atleast_1d(components)
        elif components == "current":
            components = np.arange(self.n_components, dtype=np.int32)[self.spectrum_loaded]

        # check if spectra have been loaded for all requested components
        if self.spectrum_loaded is None:
            self.load_spectra(components=components)
        else:
            if not self.spectrum_loaded[components].all():
                logging.warning("Reloading spectra, modifications to Sources.data will be lost.")
                self.load_spectra(components=components)

        if not self.spectrum_loaded.all():
            logging.warning("HDU does not contain spectra of all components.")
            # Note that currently the class (via Sources.open_spectra_hdu()) does not support loading HDUs that
            # contain the spectra for a subset of the components. So a warning is triggered if an HDU that does
            # not include all spectra will be created.

        spectra_hdu = fits.ImageHDU(self.data, name="SPECTRA")
        if self.data_wave is not None:  # check if dedicated wavelength HDU must be returned
            wave_hdu = fits.ImageHDU(self.data_wave, name='WAVE')
        else:  # otherwise, include dispersion information in Fits-Header
            wave_hdu = None
            spectra_hdu.header["CRPIX3"] = self.crpix
            spectra_hdu.header["CRVAL3"] = self.crval
            spectra_hdu.header["CDELT3"] = self.cdelt
            if self.ctype is not None:
                spectra_hdu.header["CTYPE3"] = self.ctype
            if self.cunit is not None:
                spectra_hdu.header["CUNIT3"] = self.cunit

        return spectra_hdu, wave_hdu

    def make_ifs_coordinates_hdu(self, ids=None, fit=False, name="POSITIONS", **kwargs):
        """
        Create an astropy.io.fits.ImageHDU containing the IFS coordinates of
        (a subset of) the sources or a astropy.io.fits.BinTableHDU containing
        the parameters of the polynomial fits to them. The keyword 'ids' can
        be used to control which sources enter the HDU.
        
        When the IFS coordinates are stored the array will always be 3dim.,
        with the first axis being the dispersion axis, the second axis distin-
        guishing the individual sources, and the third axis giving x or y coor-
        dinates.
        WARNING: If not all IFS coordinates that should be included in the HDU
        are available, i.e. have been loaded via Sources.load_ifs_coordinates(),
        then all requested IFS coordinates are reloaded. This means that changes
        to the Sources.ifs_coordinates array that were made after the last call to
        Sources.load_ifs_coordinates() will be lost. This does not concern the
        polynomial fits to the coordinates as the fit parameters are stored in a
        dictionary and will be used to recover the fits when reloading the data.
        
        When the polynomial fits are stored, the  table  will contain five columns,
        providing the ID, the type of fit function along x and y, and the fit para-
        meters along x- and y for every source.
        
        Parameters
        ----------
        ids : array_like or int, optional
            The ID(s) of the sources for which the IFS coordinates should be written
            to the HDU. If None is provided, all point sources from the catalogue
            are included. If the keyword is set to 'current', the IFS coordinates
            currently available in Sources.ifs_coordinates are used.
        fit : boolean, optional
            A flag indicating if the fit to the data should be written to the HDU
            instead of the data. It will be included in the FITS header as the
            hierarch keyword PAMPELMUSE POSITIONS POLY.
        name : string, optional
            The name given to the HDU, i.e. the header keyword EXTNAME.

        Returns
        -------
        ifs_coordinates_hdu : astropy.io.fits.ImageHDU
            A 3dim. FITS array containing the data or the fits to them. Note that
            the routine will also add the hierarch keywords PAMPELMUSE POSITIONS ID?
            to the FITS header that give the ID of the sources included in each row
            of  the array. An additional hierarch keyword PAMPELMUSE POSITIONS
            FULLFIT is used to indicate if the coordinates are free parameters in a
            fit.
        """
        if not fit:
            logging.info("Saving IFS coordinates to ImageHDU '{0}'...".format(name))

        # check which coordinates should be saved
        if ids is None:
            ids = self.ID[self.status < 4]
        elif isinstance(ids, (int, np.integer, list, np.ndarray)):
            ids = np.atleast_1d(ids)
        elif ids == "current":
            if not fit:
                ids = self.ID[self.ifs_coordinates_loaded]
            else:
                ids = self.ID[self.ifs_coordinates_fitted]

        if fit:
            logging.info("Saving fits to IFS coordinates to BinTableHDU '{0}'...".format(name))

            # fits are saved for the requested sources that have polynomial fits available
            ids_tosave = np.intersect1d(ids, self.ifs_coordinates_fit_functions.keys())

            if ids_tosave.size == 0:
                logging.error("Position fits not available for any of the requested sources.")
                logging.error("Cannot create BinTableHDU '{0}'.".format(name))

            # get the number of sources for which fits should be stored
            nfit = ids_tosave.size

            # get the maximum polynomial degree used in the fits. Note that this would not be necessary if one would
            # use the feature that FITS tables can store arrays of variable length
            # (http://docs.astropy.org/en/tmp-v0.1-wcython/io/fits/usage/unfamiliar.html)
            # Unfortunately, the implementation in astropy seems buggy and I always get an array full of zeros when
            # opening the file again. Therefore, in case different degrees have been used in the fits, the higher
            # orders of the low-order polynomials are padded with zeros.
            xmax = max(self.ifs_coordinates_fit_functions[i][0].degree() for i in ids_tosave)
            ymax = max(self.ifs_coordinates_fit_functions[i][1].degree() for i in ids_tosave)

            # prepare the arrays to hold the data
            src = np.zeros((nfit, ), dtype=np.int32)
            xfct = np.zeros((nfit, ), dtype=np.int16)
            xprm = np.zeros((nfit, xmax + 1), dtype=np.float32)
            yfct = np.zeros((nfit, ), dtype=np.int32)
            yprm = np.zeros((nfit, ymax + 1), dtype=np.float32)

            #  loop over all sources that have polynomial fits available
            domain = None
            for i, ID in enumerate(ids):
                src[i] = ID

                xfit, yfit = self.ifs_coordinates_fit_functions[ID]

                # make sure all fit functions use the same domain, this is needed because the domain set by the fit
                # is the min and max of the fitted range and this if different from the default one used by the
                # __init__ methods. Since currently the start and end of the domain are stored as HIERARCH-keywords,
                # the domain must be the same for all fits.
                if domain is None:
                    domain = xfit.domain
                else:
                    if (xfit.domain != domain).any():
                        xfit = xfit.convert(domain=domain)
                if (yfit.domain != domain).any():
                    yfit = yfit.convert(domain=domain)

                # to get fit function, first check which class the instance originates from, then translate name to
                # an integer
                for fct in self.available.keys():
                    if isinstance(xfit, self.available[fct]):
                        xfct[i] = self.polyident[fct]
                xprm[i, :(xfit.degree() + 1)] = xfit.coef
                for fct in self.available.keys():
                    if isinstance(yfit, self.available[fct]):
                        yfct[i] = self.polyident[fct]
                yprm[i, :(yfit.degree() + 1)] = yfit.coef

            # prepare the output table
            outhdu = fits.BinTableHDU.from_columns(
                    [fits.Column(name="id", format="J", array=src),
                     fits.Column(name="xfct", format="I", array=xfct),
                     fits.Column(name="xprm", format="{0}E".format(xprm.shape[1]), array=xprm),
                     fits.Column(name="yfct", format="I", array=yfct),
                     fits.Column(name="yprm", format="{0}E".format(yprm.shape[1]), array=yprm)])

            outhdu.header["EXTNAME"] = name
            outhdu.header["hierarch PAMPELMUSE POSITIONS POLY"] = True
            outhdu.header["hierarch PAMPELMUSE POSITIONS DOMAIN START"] = domain[0]
            outhdu.header["hierarch PAMPELMUSE POSITIONS DOMAIN END"] = domain[1]

            # write the polynomial identifications
            for name, index in self.polyident.iteritems():
                outhdu.header["hierarch PAMPELMUSE POSITIONS POLY{0}".format(index)] = name
        else:
            # make sure IFS coordinates are loaded for all sources AND across the entire dispersion range
            if not self.ifs_coordinates_loaded[match_on_id(self.ID, ids)].all() \
                    or self.ifs_coordinates.shape[0] != self.ndisp:
                logging.warning("Reloading IFS coordinates, modifications to Sources.ifs_coordinates will be lost.")
                self.load_ifs_coordinates(ids=ids, layers=None)

            to_store = np.flatnonzero(self.ifs_coordinates_free[self.ifs_coordinates_loaded])

            # make sure there is some data to be stored
            if to_store.size == 0:
                logging.critical("Cannot write IFS coordinates to HDU. No data available.")
                return NotImplemented

            outhdu = fits.ImageHDU(self.ifs_coordinates[:, to_store][..., int(fit)], name=name)

            outhdu.header["hierarch PAMPELMUSE POSITIONS POLY"] = False
            outhdu.header["hierarch PAMPELMUSE POSITIONS FULLFIT"] = True

            # create entry in header about ID of each source and the row containing its position
            for i in xrange(len(to_store)):
                outhdu.header["hierarch PAMPELMUSE POSITIONS ID{0}".format(i)] = self.ID[self.ifs_coordinates_loaded][
                    to_store[i]]

        return outhdu
