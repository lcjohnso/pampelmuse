"""
sources.py
==========

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
237, 2015/02/11
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 237

import collections
import inspect
import logging
import os
import re
import time
import warnings

import numpy as np
from numpy.polynomial import Chebyshev, Legendre, Hermite, Polynomial
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.io import fits

# check if PyQt is available 
try:
    from PyQt4 import QtCore
except ImportError:
    pass

# required PampelMuse-packages
from pampelmuse.core.coordinates import Transformation


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
    available = collections.OrderedDict()
    available["chebyshev"] = Chebyshev
    available["legendre"] = Legendre
    available["hermite"] = Hermite
    available["plain"] = Polynomial

    # for writing to/reading from FITS files, each polynomial is assigned an index
    polyident = collections.OrderedDict(zip(available.keys(), np.arange(len(available))))

    def __init__(self, data=None, **kwargs):
        """
        Initialize a new instance of the `Sources`-class.

        Parameters
        ----------
        data : an astropy.io.fits BinTable-HDU instance or a structured array,
               optional
            The source catalog that should be used to initialize the instance.
            It must contain at least a field named 'id', all other fields are
            optional and the corresponding properties might be defined later.
            If 'data' is not provided, than 'ids' must be provided as a keyword
            argument.
        kwargs
            Additional properties of the source catalogue that are not included
            in 'data' can be provided as keyword arguments. Note that if no data
            are provided, at least one keyword argument 'ids' must be provided
            to initialize the IDs of the sources. Also note that if properties
            that are also available in 'data' are provided as keyword arguments,
            these supersede the values in 'data'.
        """
        warnings.simplefilter("always", DeprecationWarning)  # By default this type of warning is ignored

        # open the data
        if isinstance(data, fits.hdu.table.BinTableHDU):
            if "EXTNAME" in data.header:
                logging.info("Initializing catalogue of sources via HDU '{0}'...".format(data.header["EXTNAME"]))
            else:
                logging.info("Initializing catalogue of sources via provided HDU...")
            data = np.array(data.data, dtype=data.data.dtype)
            # if a FITS Table was provided, extract information and store as rec-array
        else:
            logging.info("Initializing catalogue of sources via provided data...")

        # check that Source IDs are provided and initialize property Sources.ID
        if data is not None:
            try:
                self.ID = data[["id"]].view(data["id"].dtype).reshape(-1, )
            except ValueError:
                logging.critical("No field 'id' found in provided data.")
        elif "ids" in kwargs.keys():
            self.ID = np.asarray(kwargs["ids"])
        if self.ID is None:
            logging.critical("Source IDs must be provided upon initialization.")
        logging.info("Catalogue contains {0} sources".format(self.nsrc))

        assert len(np.unique(self.ID)) == len(self.ID), "Source IDs must be unique."

        # other properties are optional may be provided later. If they are not provided, arrays containing NaN entries
        # are initialized. Note that the properties below can be accessed in two ways:
        # a) with an underscore: direct access, no sanity checks performed
        # b) without underscore: check whether data used to update property is valid.

        # Properties that can be set by the user:
        self._x = np.array(self.nsrc * [np.nan], dtype=np.float32)  # input (or reference) x coordinates
        self._y = np.array(self.nsrc * [np.nan], dtype=np.float32)  # input (or reference) y coordinates
        self._ra = np.array(self.nsrc * [np.nan], dtype=np.float64)  # right ascension
        self._dec = np.array(self.nsrc * [np.nan], dtype=np.float64)  # declination
        self._mag = np.array(self.nsrc * [np.nan], dtype=np.float32)  # broadband magnitudes

        self._status = np.zeros((self.nsrc, ), dtype=np.int16)  # the status of each source in the analysis
        self._status[self.ID < 0] = 4  # assume IDs<0 are used for background components

        self._primary = np.copy(self.ID)  # the primary source of each source

        self._specrow = None  # row in the array of spectra that contains the spectrum of a source
        # NOTE that the array is calculated internally upon first request (i.e. call to its getter method unless it is
        # provided upon initialization
        self._indexmap = None  # a dictionary mapping each value in Sources.specrow to the indices of the source IDs
        # that contribute to it. This attribute is determined internally when the first call to its getter method is
        # made.
        self._weights = None  # a 1dim. array of length Sources.nsrc containing the relative contribution of each
        # source to its component. This attribute is determined internally when the first call to its getter method is
        # made.

        # check if some of the above properties are provided with the input (data or keyword arguments)
        for name in ["x", "y", "mag", "ra", "dec", "status", "primary", "specrow"]:
            if name in kwargs.keys():
                getattr(self, "set_{0}".format(name))(kwargs[name])
            elif data is not None and name in data.dtype.names:
                getattr(self, "set_{0}".format(name))(data[[name]].view(data.dtype[name]).reshape(-1, ))

        # initialize coordinate transformation
        self.transformation = Transformation()

        # initialize arrays that will contain important information about source spectra and coordinates:
        # (a) whether a spectrum has been loaded for a component;
        #     a boolean array will be created by Sources.load_spectra()
        self.spectrum_loaded = None

        # (b) a boolean array indicating if IFU coordinates have been loaded for a source
        self.ifs_coordinates_loaded = np.zeros((self.nsrc,), dtype=np.bool)

        # (c) a boolean array indicating if a polynomial fit has been performed to the IFU coordinates of a source
        self.ifs_coordinates_fitted = np.zeros((self.nsrc,), dtype=np.bool)

        # (d) a boolean array indicating if the IFU coordinates of a source are a free fit parameter
        self.ifs_coordinates_free = np.zeros((self.nsrc,), dtype=np.bool)

        # (e) a dictionary of ID:<function> pairs that will contain callable functions that return the polynomial fits
        #  to the positions of a source, obtained using Sources.fitIfsCoordinates()
        self.ifs_coordinates_fit_functions = {}

        # place-holders for data arrays
        self.data = None  # The source fluxes/spectra
        self.sigma = None  # The uncertainties of the source fluxes/spectra
        self.ifs_coordinates = None  # The x/y coordinates of the sources on the IFS spaxel array

        # place-holders for HDUs containing fluxes or IFS coordinates
        self.spectra_data_hdu = None
        self.ifs_coordinates_data_hdu = None
        self.ifs_coordinates_fit_hdu = None
        self._tmp_names = []  # a list of temporary FITS files created by an instance.
        # The files will be deleted when the instance terminates

        # Initialize the dispersion information
        self.ndisp = 1
        self.crval = 1.
        self.crpix = 1
        self.cdelt = 1.
        self.ctype = None
        self.cunit = None

    def __del__(self):
        """
        When the instance is terminated, the temporary files will be deleted.
        """
        for filename in self._tmp_names: os.remove(filename)

    ###############################################################################
    # Definitions of the various properties,                   #
    # and the methods to create & access them                  #
    ###############################################################################
    @property
    def nsrc(self):
        """
        Returns
        -------
        nsrc : int
            The total number of sources in the catalogue (incl. background
            components). 
        """
        return self.ID.size

    # Reference x-coordinates
    def set_x(self, value):
        """
        The setter method for the Sources.x-property, containing the coordinate of
        each source along the x-axis of the reference system.

        Parameters
        ----------
        value : array_like
            The x-coordinates. Must be a 1dim. list of length Sources.nsrc.
        """
        assert isinstance(value, (list, np.ndarray)), "Attribute Sources.x must be array-like, not {0}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.asarray(value)
        if not value.size == self.nsrc and value.ndim == 1:
            logging.error("Cannot set attribute Sources.x, data have invalid shape.")
        else:
            self._x[:] = value

    def get_x(self):
        """
        The getter method of the Sources.x-property.

        Returns
        -------
        x : nd_array
            A 1dim. of length Sources.nsrc containing the coordinate of each source
            along the x-axis of the reference system.
        """
        return self._x

    x = property(get_x, set_x, None, "The source coordinates along the x-axis of the reference system.")

    # Reference y-coordinates
    def set_y(self, value):
        """
        The setter method for the Sources.y-property, containing the coordinate of
        each source along the y-axis of the reference system.

        Parameters
        ----------
        value : array_like
            The y-coordinates. Must be a 1dim. list of length Sources.nsrc.
        """
        assert isinstance(value, (list, np.ndarray)), "Attribute Sources.y must be array-like, not {0}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.asarray(value)
        if not value.size == self.nsrc and value.ndim == 1:
            logging.error("Cannot set attribute Sources.y, data have invalid shape.")
        else:
            self._y[:] = value

    def get_y(self):
        """
        The getter method of the Sources.y-property.

        Returns
        -------
        y : nd_array
            A 1dim. of length Sources.src containing the coordinate of each source
            along the y-axis of the reference system.
        """
        return self._y

    y = property(get_y, set_y, None, "The source coordinates along the y-axis of the reference system.")

    # Broadband magnitudes
    def set_mag(self, value):
        """
        The setter method for the Sources.mag-property, containing the magnitude of
        each source in some reference filter.

        Parameters
        ----------
        value : array_like
            The magnitudes. Must be a 1dim. list of length Sources.nsrc.
        """
        assert isinstance(value, (list, np.ndarray)), "Attribute Sources.mag must be array-like, not {0}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.asarray(value)
        if not value.size == self.nsrc and value.ndim == 1:
            logging.error("Cannot set attribute Sources.mag, data have invalid shape.")
        else:
            self._mag[:] = value
            self._weights = None  # the attribute Sources.weights must be reset because it depends on Sources.specrow

    def get_mag(self):
        """
        The getter method of the Sources.mag-property.

        Returns
        -------
        mag : nd_array
            A 1dim. of length Sources.nsrc containing the magnitude of each source
            in the reference filter.
        """
        return self._mag

    mag = property(get_mag, set_mag, None, "The broadband magnitudes of the sources.")

    # WCS coordinates: RA
    def set_ra(self, value):
        """
        The setter method for the Sources.ra-property, containing the right ascension
        of each source in decimal degrees.

        Parameters
        ----------
        value : array_like
            The RA-coordinates. Must be a 1dim. list of length Sources.nsrc.
        """
        assert isinstance(value, (list, np.ndarray)), "Attribute Sources.ra must be array-like, not {0}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.asarray(value)
        if not value.size == self.nsrc and value.ndim == 1:
            logging.error("Cannot set attribute Sources.ra, data have invalid shape.")
        else:
            self._ra[:] = value

    def get_ra(self):
        """
        The getter method of the Sources.ra-property.

        Returns
        -------
        ra : nd_array
            A 1dim. of length Sources.nsrc containing the right ascension of each
            source in decimal degrees.
        """
        return self._ra

    ra = property(get_ra, set_ra, None, "The right ascension of each source in decimal degrees.")

    # WCS coordinates: Dec
    def set_dec(self, value):
        """
        The setter method for the Sources.dec-property, containing the declination
        of each source in decimal degrees.

        Parameters
        ----------
        value : array_like
            The Dec-coordinates. Must be a 1dim. list of length Sources.nsrc.
        """
        assert isinstance(value, (list, np.ndarray)), "Attribute Sources.dec must be array-like, not {0}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.asarray(value)
        if not value.size == self.nsrc and value.ndim == 1:
            logging.error("Cannot set attribute Sources.dec, data have invalid shape.")
        else:
            self._dec[:] = value

    def get_dec(self):
        """
        The getter method of the Sources.dec-property.

        Returns
        -------
        x : nd_array
            A 1dim. of length Sources.nsrc containing the declination of each source
            in decimal degrees.
        """
        return self._dec

    dec = property(get_dec, set_dec, None, "The declination of each source in decimal degrees.")

    # The status of the sources during the fit
    def set_status(self, value):
        """
        The setter method for the Sources.status-property, containing the status of
        each source during the fit. The classification scheme for the status of a
        sources is as follows.
            [-1] - source is not used
            [ 0] - source contributes to unresolved component
            [ 1] - source is fitted together with a brighter neighbor.
            [ 2] - source is fitted independently
            [ 3] - source is used as a PSF source.
            [ 4] - source is a background (e.g. sky) component

        Parameters
        ----------
        value : array_like
            A 1dim. list of length Sources.nsrc.
        """
        assert isinstance(value, (list, np.ndarray)), "Attribute Sources.status must be array-like, not {0}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.asarray(value)
        if not value.size == self.nsrc and value.ndim == 1:
            logging.error("Cannot set attribute Sources.status, data have invalid shape.")
        else:
            # if Sources.status is (re)set after attribute Sources.specrow has been set internally, the latter is not
            # valid anymore.
            if self._specrow is not None:
                logging.warning("Setting attribute Sources.status will render attribute Sources.specrow invalid.")
            self._status[:] = value

    def get_status(self):
        """
        The getter method of the Sources.status-property.

        Returns
        -------
        status : nd_array
            A 1dim. of length Sources.nsrc containing the status of each source
            in the fit.
        """
        return self._status

    status = property(get_status, set_status, None, "The status of every source during the fitting process.")

    # The primary source in each component
    def set_primary(self, value):
        """
        The setter method for the Sources.primary-property, containing the index
        of the primary source for each entry in the source catalogue. This index
        is identical to the index of the source itself for sources with status>1.

        Parameters
        ----------
        value : array_like
            The indices of the primary sources. Must be a 1dim. list of
            integers of length Sources.nsrc.
        """
        assert isinstance(value, (list, np.ndarray)), "Attribute Sources.primary must be array-like, not {0}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.asarray(value, dtype=np.int32)
        if not value.size == self.nsrc and value.ndim == 1:
            logging.error("Cannot set attribute Sources.primary, data have invalid shape.")
        else:
            # if Sources.primary is (re)set after attribute Sources.specrow has been set internally,
            # the latter is not valid anymore.
            if self._specrow is not None:
                logging.warning("Setting attribute Sources.primary will render attribute Sources.specrow invalid.")
            self._primary[:] = value

    def get_primary(self):
        """
        The getter method of the Sources.primary-property.

        Returns
        -------
        primary : array_like
            A 1dim. array of integers and of length Sources.nsrc containing the
            index of the primary source of each entry in the source catalogue.
        """
        return self._primary

    primary = property(get_primary, set_primary, None, "The primary source of each available source.")

    # The data row containing the spectrum of a source.
    def set_specrow(self, value):
        """
        The setter method for the Sources.specrow-property, containing the
        index of the row that contains the spectrum of a source.

        Parameters
        ----------
        value : array_like
            The indices of the rows containing the spectra. Must be a 1dim.
            array of integers of length Sources.nsrc.
        """
        assert isinstance(value, (list, np.ndarray)), "Attribute Sources.specrow must be array-like, not {0}.".format(
            re.findall(r"'(.*)'", str(type(value)))[0])
        value = np.asarray(value, dtype=np.int32)
        if not value.size == self.nsrc and value.ndim == 1:
            logging.error("Cannot set attribute Sources.specrow, data have invalid shape.")
        else:
            self._specrow = value
            # the attributes Sources.indexmap and sources.weights must be reset because they depend on Sources.specrow
            self._indexmap = None
            self._weights = None

    def get_specrow(self):
        """
        The getter method of the Sources.specrow-property.

        Returns
        -------
        specrow : nd_array
            A 1dim. of integers and of length Sources.nsrc that contains, for each
            source in the scene with status>=0, the index of the row that contains
            the spectrum in the Sources.data array. Unless the property has been
            set before using its setter method, the indices are automatically
            assigned, using the information from the attributes Source.primary and
            Sources.status.
        """
        if self._specrow is None:
            self._specrow = np.zeros(self.nsrc, dtype=np.int32)

            # set row for independent ('primary') sources
            self._specrow[self.status > 1] = np.arange((self.status > 1).sum(), dtype=np.int32)

            unresolved_row = self._specrow.max() + 1

            # now deal with sources that are either next to a brighter neighbor(status=1) or included in the unresolved
            # component (status=0)
            for i in xrange(self.nsrc):
                if self.status[i] == 1:
                    # find spectrum of primary component; the following loop is necessary because the initially assigned
                    # primary component might have become the secondary component of another source.
                    secondary = []
                    j = i
                    while self.primary[j] != self.ID[j]:  # continue until current source is primary

                        # exclude source if primary source has status = -1.
                        if not (self.ID == self.primary[j]).any():
                            logging.error("No 'primary' counterpart for source #{0}".format(self.ID[i]))
                            logging.error("Source #{0} excluded".format(self.ID[i]))
                            self.status[i] = -1
                            break
                        secondary.append(j)

                        # get index of primary source in Sources.ID array
                        j = np.where(self.ID == self.primary[j])[0][0]

                    self._specrow[secondary] = self._specrow[j]
                elif self.status[i] == 0:
                    self._specrow[i] = unresolved_row

        return self._specrow

    specrow = property(get_specrow, set_specrow, None,
                       "The column of the data array containing the spectrum of a source.")

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
        if self._indexmap is None:
            self._indexmap = {}
            for i in xrange(self.ncmp):
                self._indexmap[i] = np.where(self.specrow == i)[0]
        return self._indexmap

    # The magnitude weights of the individual contributions to a spectrum
    @property
    def weights(self):
        """
        Returns
        -------
        weights: nd_array
            A 1dim. array of the same length as the whole source list containing
            the magnitude weight of each source in its component. The magnitude
            weight is defined as 10**(-0.4*df), where df is the fraction of
            flux the source contributes to the component.
        """
        if self._weights is None:
            self._weights = np.zeros((self.nsrc,), dtype=np.float64)
            if self.status.min() <= 1:  # if some sources have status<2, attribute Sources.mag is required.
                if (self.mag != self.mag).any():
                    logging.error("No magnitudes (attribute Sources.mag) are available.")
                    logging.error(
                        "Cannot determine source contributions to individual components (attribute Sources.weights).")
                else:
                    for i in xrange(self.nsrc):  # calculate magnitude weighted contributions per component
                        same_specrow = (self.specrow == self.specrow[i])
                        self._weights[i] = self._mag[i] + 2.5 * np.log10((10 ** (-0.4 * self._mag[same_specrow])).sum())
        return self._weights

    # The number of independent components in the fit.
    @property
    def ncmp(self):
        """
        Returns
        -------
        ncmp : int
            The number of components (i.e. independent spectra) in the
            catalogue.
        """
        return self.specrow.max() + 1

    # Return a boolean array of len(nsrc) that is true if the source is resolved. 
    @property
    def is_resolved(self):
        """
        Returns
        -------
        resolved : ndarray
            A boolean array of len(self.nsrc) indicating whether a source is
            resolved. If self.status is undefined, all sources are considered
            unresolved.
        """
        return (self.status > 0) & (self.status < 4)

    # A flag indicating for each source if it is part of the background
    @property
    def is_background(self):
        """
        Returns
        -------
        is_background : ndarray
            A boolean array of len(self.nsrc) indicating whether a source is
            part of the background.
            """
        return self.status == 4

    # A flag on whether one or more background components are included in the catalogue. 
    @property
    def includes_background(self):
        """
        Returns
        -------
        includes_background : bool
            A flag indicating whether one or more background components are in-
            cluded in the catalogue.
        """
        return self.is_background.any()

    # The number of background components included in the catalogue
    @property
    def nbackground(self):
        """
        Returns
        -------
        nbackground : int
            The number of background components that is included in the
            catalogue."""
        return self.is_background.sum()

    # The rows of the spectra of the background components (if any)
    @property
    def specrow_background(self):
        """
        Returns
        -------
        specrow_background : array
            The rows containing the background spectra (if any, otherwise
            returns an empty array)
        """
        return self.specrow[np.flatnonzero(self.is_background)]

    #  A flag indicating for each source if it is part of the unresolved stellar component
    @property
    def is_unresolved(self):
        """
        Returns
        -------
        is_unresolved : ndarray
            A boolean array of len(self.nsrc) indicating whether a source is
            part of the unresolved stellar component.
            """
        return self.status == 0

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
            The row of the spectrum of the unresolved component (if any,
            otherwise returns an empty array)
        """
        rows = self.specrow[np.flatnonzero(self.is_unresolved)]
        assert (np.unique(rows).size == 1)
        return rows[0]

    # The indices of the sources used to determine the PSF (if any)
    @property
    def psf_indices(self):
        """
        Returns
        -------
        psf_indices : nd_array
            A 1dim. array containing the indices of the sources that are used
            to determine the PSF (i.e. sources whose `status` is 3).
        """
        return np.arange(self.nsrc, dtype=np.int32)[self.status == 3]

    ############################################################################
    #  Methods providing the user with information about the source catalogue  #
    ############################################################################
    def info(self):
        """
        Print out some information about the loaded source catalogue. Also
        check if all necessary properties have been defined.
        """
        logging.info("Current status of catalogue of sources:")
        logging.info("   no. of PSF sources: {0}".format(self.psf_indices.size))
        logging.info("   no. of resolved sources: {0}".format(self.is_resolved.sum()))
        logging.info("   no. of nearby sources: {0}".format((self.status == 1).sum()))
        logging.info("   no. of unresolved sources: {0}".format(self.is_unresolved.sum()))
        logging.info("   no. of background components: {0}".format(self.nbackground))

        logging.debug("Overview of all components used in fit:")
        logging.debug("   Component:   IDs (& mags):")

        if logging.getLogger().getEffectiveLevel() < 20:
            for i in xrange(self.specrow.max() + 1):
                log_string = "   {0:8d}".format(i)

                slc = np.flatnonzero(self.specrow == i)

                for j, k in enumerate(slc):
                    log_string += "   {0:6d} ({1:.2f})".format(int(self.ID[k]), float(abs(self.weights[k])))

                    if np.mod(j + 1, 2) == 0:
                        logging.debug(log_string)
                        log_string = 11 * " "

                if np.mod(j, 2) == 0:
                    logging.debug(log_string)

        if self.transformation.valid:
            self.transformation.info()

    #############################################################################
    #              Methods dealing with the spectra of the sources              #
    #############################################################################
    def open_spectra_hdu(self, hdu, sigma_hdu=None):
        """
        Open a FITS HDU containing the spectra of the sources. Note that the
        spectra may be provided in two different flavours
        (1) old behaviour: spectra and uncertainties are included in different, the
            shape of each data array is (Sources.ncmp, Sources.ndisp) [python
            counting]
        (2) recommended: one HDU for both spectra and uncertainties, the shape of
            the data array is (Sources.ndisp,Sources.ncmp,2) [python counting]
        If data following format (1) are provided, the method will translate it to
        format (2). The HDU is accessible via attribute Sources.spectra_data_hdu
        afterwards.
          
        Parameters
        ----------
        hdu : astropy.io.fits.PrimaryHDU or astropy.io.fits.ImageHDU instance
            The HDU containing the data (and the uncertainties for 3dim. data).
            Note that only 2dim. or 3dim. data are supported (see description
            of method above).
        sigma_hdu : astropy.io.fits.PrimaryHDU or astropy.io.fits.ImageHDU instance, optional
            The HDU containing the uncertainties for 2dim. data. The array in
            the HDU must have the same shape as the data.
        """
        # sanity checks on provided keywords, both must be HDUs that can carry images
        assert isinstance(hdu, fits.hdu.image._ImageBaseHDU), "Invalid type '{0}' for argument 'hdu'.".format(
            re.findall(r"'(.*)'", str(type(hdu)))[0])
        if sigma_hdu is not None:
            assert isinstance(sigma_hdu,
                              fits.hdu.image._ImageBaseHDU), "Invalid type '{0}' for argument 'sigma_hdu'.".format(
                re.findall(r"'(.*)'", str(type(sigma_hdu)))[0])

        if "EXTNAME" in hdu.header:
            logging.info("Reading spectra from HDU `{0}`.".format(hdu.header["EXTNAME"]))
        else:
            logging.info("Reading spectra from unnamed HDU.")

        if hdu.header["NAXIS"] not in [2, 3]:
            logging.error("Spectra must be provided as 2 or 3dim. array.")
        assert hdu.header["NAXIS"] in [2, 3], "Expected 2 or 3dim. data but got {0}dim.".format(hdu.header["NAXIS"])

        # handle "old" format: 2dim. data, first axis (python counting) corresponding to different stars, second axis
        # giving dispersion information create a temporary FITS file containing the data in the new format in the 2nd
        #  HDU and load this HDU. The file will be deleted once the Sources instance is deleted.
        if hdu.header["NAXIS"] == 2:

            tmp_data_array = np.zeros((hdu.header["NAXIS1"], hdu.header["NAXIS2"], 2), dtype=np.float32)
            tmp_data_array[..., 0] = hdu.data.swapaxes(0, 1)

            if sigma_hdu is not None:
                assert sigma_hdu.data.shape == hdu.data.shape, "Array sizes of spectra and uncertainties differ."
                tmp_data_array[..., 1] = sigma_hdu.data.swapaxes(0, 1)

            tmp_spectra_hdu = fits.ImageHDU(tmp_data_array, header=hdu.header)

            # update dispersion information (if any)
            for key in ["CRVAL", "CDELT", "CRPIX", "CTYPE", "CUNIT"]:
                if "{0}1".format(key) in tmp_spectra_hdu.header:
                    tmp_spectra_hdu.header["{0}3".format(key)] = hdu.header["{0}1".format(key)]
                    del tmp_spectra_hdu.header["{0}1".format(key)]

            # write the HDU to a temporary FITS file (fits provides a function to create a tmp-name) and open it again
            self._tmp_names.append(fits.util._tmp_name("./"))
            tmp_spectra_hdu.writeto(self._tmp_names[-1])

            # remove temporary array and HDU (tmp-file will be deleted via __del__ method)
            del tmp_data_array, tmp_spectra_hdu, hdu
            tmpfile = fits.open(self._tmp_names[-1])
            hdu = tmpfile[1]

        # check if dispersion information is valid
        if self.ndisp == 1:
            self.ndisp = hdu.header["NAXIS3"]
        elif hdu.header["NAXIS3"] != self.ndisp:
            logging.error("Invalid number of dispersion elements in provided spectra.")

        assert hdu.header[
                   "NAXIS2"] == self.ncmp, "No. of spectra in HDU does not match no. of components in source catalog."
        assert hdu.header["NAXIS1"] == 2, "Data in provided HDU has invalid shape."

        self.spectra_data_hdu = hdu

        # update dispersion information within class.
        for key in ["CRVAL", "CDELT", "CRPIX", "CTYPE", "CUNIT"]:
            if "{0}3".format(key) in self.spectra_data_hdu.header:
                setattr(self, key.lower(), self.spectra_data_hdu.header["{0}3".format(key)])

    def load_spectra(self, components=None, layers=None):
        """
        Load the spectra and uncertainties for the requested source(s) and layer(s).
        The method will create an empty array and try to fill it using an available
        FITS HDU, provided previously via the method Sources.open_spectra_hdu(). If 
        no such HDU is available, the array will be empty. The shape of the array,
        available via property Sources.data, will be (Sources.ndisp,Sources.ncmp,2).

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
        """
        # sanity checks on provided IDs and layers
        if components is not None:
            assert isinstance(components, (int, np.integer, list, np.ndarray)),\
                "Parameter 'components' must be integer or array-like, not {0}.".format(
                    re.findall(r"'(.*)'", str(type(components)))[0])
            components = np.sort(np.atleast_1d(components))
        else:
            components = np.arange(self.ncmp, dtype=np.int32)

        # check if all provided IDs are actually in the source catalogue
        assert (components[0] >= 0) & (
            components[-1] < self.ncmp), "Not all requested components fall in available range."

        if layers is not None:
            assert isinstance(layers, (int, np.integer, list, np.ndarray)),\
                "Parameter 'layers' must be transformable to an array."
            layers = np.atleast_1d(layers)
            nlayer = 1
        else:
            layers = np.arange(self.ndisp)
            nlayer = self.ndisp

        # create boolean array that keeps track of which sources have spectra loaded
        self.spectrum_loaded = np.zeros((self.ncmp, ), dtype=np.bool)
        self.spectrum_loaded[components] = True

        # initialize array for spectra. The shape of the array is (#<requested layers>, #<requested IDs>, 2)
        # The last axis distinguishes between data or uncertainties.
        self.data = np.zeros((nlayer, components.size, 2), dtype=np.float32)

        # fill array using data from Spectrum HDU if one is available
        if self.spectra_data_hdu is not None:
            if nlayer == self.ndisp and components.size == self.ncmp:
                self.data[:] = self.spectra_data_hdu.data
            else:
                for i in xrange(nlayer):
                    self.data[i] = self.spectra_data_hdu.section[layers[i]][components]

    @property
    def wave(self):
        """
        Returns
        -------
        wave : ndarray
            The wavelength array.
        """
        wave = np.arange(self.ndisp, dtype=np.float64)
        wave -= (self.crpix - 1)
        wave *= self.cdelt
        wave += self.crval
        return wave

    def provideComponentNames(self):
        """
        Short routine to provide the IDs of the primary sources for each component
        as a QStringList. Additionally, the method initializes a dictionary
        Sources.nameMap that maps the name of each components primary source (as a
        QString) to its value in the Sources.specrow attribute.

        Returns
        -------
        names : QStringList
           The names of the coomponents
        """
        ownSpec = np.in1d(self.status, np.arange(2, 4))

        self.names = QtCore.QStringList(self.ID[ownSpec].astype("|S10"))
        self.nameMap = dict((zip(self.ID[ownSpec].astype("|S10"), self.specrow[ownSpec])))
        for i in xrange(self.nbackground):
            self.names.append("Back{0:04d}".format(i))
            self.nameMap["Back{0:04d}".format(i)] = self.specrow_background[i]
        if self.includes_unresolved:
            self.names.append("Unresolved")
            self.nameMap["Unresolved"] = self.specrow_unresolved

        return self.names

    def viewSpectrum(self, name=None, ID=None):
        """
        Return the spectrum corresponding to a provided ID or name of the
        component. This method may only called after Sources.provideComponentNames
        has been called to initialize the attribute Sources.nameMap.
        
        Parameters
        ----------
        name : string, optional
           The name of the requested component. Note that if `name` is not
           provided, `ID` must be given.
        ID : int, optional
           The ID of the requested component. Note that if ID` is not
           provided, `name` must be given.

        Returns
        -------
        data : ndarray
            An array containing the the requested spectrum and its uncertainties.
        """
        if self.spectrum_loaded is None and self.spectra_data_hdu is None:
            logging.critical("No spectra have been loaded using Sources.load_spectra() or Sources.open_spectra_hdu().")

        if name is None and ID is None:
            logging.error("Either name or ID must be provided.")
            return

        # get no. of component (i.e. specrow)
        if name is not None:
            specrow = self.nameMap[name]
        else:
            specrow = self.specrow[np.where(self.ID == ID)[0][0]]

        # load the spectrum if it has not been loaded previously
        if self.spectrum_loaded is None or not self.spectrum_loaded[specrow]:
            # check if a spectrum is available for the requested source
            self.load_spectra(components=specrow)

        i = np.cumsum(self.spectrum_loaded)[specrow] - 1  # index of the spectrum in the Sources.data array
        return self.data[:, i]

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
            source coordinates (incl. coord. transf.) are free fit parameters. 
        """
        if self.ifs_coordinates_fitted[self.status < 4].all():
            return False
        elif self.ifs_coordinates_free.any():
            return True
        else:
            return self.transformation.contains_free_parameters

    def open_coordinates_hdu(self, hdu):
        """
        Prepare a FITS HDU containing IFS coordinates for the sources. Note 
        that if an Image HDU is provided, the data are not actually loaded
        into memory but that the method only provided the necessary information
        to do so using the method Sources.load_ifs_coordinates().
        
        In case an Image HDU is provided, it must contain a 3dim. array. The
        dispersion axis must be either the first or the last axis of the array.
        The second axis should contain the individual sources. Finally, the
        remaining axis must be of length 2 and contain x and y coordinates,
        respectively. Note that it is possible to provide IFS coordinates either
        for all sources or for a subset of them. In the latter case, however, the
        HIERARCH keywords PAMPELMUSE POSITIONS ID<i> should be present, giving
        the ID of the source stored in row i.

        The method will automatically check whether the HDU contains data or
        polynomial fits, using HIERARCH-keyword 'PAMPELMUSE POSITIONS POLY'
        (or keyword 'ISFITTED').
        Also, the method will automatically check whether provided IFU
        coordinates are free parameters in the fit using the HIERARCH-keyword
        'PAMPELMUSE POSITIONS FULLFIT'

        The method also accepts a binary FITS table containing the parameters of
        the polynomial fits to the individual sources. This way of storing the
        fits is much less memory consuming for large source lists. The table must
        contain five columns, named "id", "xfct", "xprm", "yfct", "yprm", provi-
        ding ID, type of polynomial (fct) and fit coefficients (prm) along x and
        y for each source.
        
        Parameters
        ----------
        hdu : an astropy.io.fits.hdu instance
            The HDU instance containing the coordinates.
        """
        # sanity checks on provided data
        assert isinstance(hdu, (fits.hdu.image._ImageBaseHDU, fits.hdu.table.BinTableHDU)), \
            "Invalid type '{0}' for argument 'hdu' in Sources.open_coordinates_hdu().".format(
                re.findall(r"'(.*)'", str(type(hdu)))[0])

        if "EXTNAME" in hdu.header:
            extname = hdu.header["EXTNAME"]
        else:
            extname = "UNNAMED"

        # check if HDU contains polynomial fits or measured IFU coordinates
        try:
            poly = hdu.header["PAMPELMUSE POSITIONS POLY"]
        except KeyError:
            try:
                poly = hdu.header["ISFITTED"]
            except KeyError:
                poly = False

        if isinstance(hdu, fits.hdu.table.BinTableHDU):
            logging.info("Reading position fit coefficients from table HDU '{0}'.".format(extname))

            poly = True
            free = False

            # check if required columns are present in header
            for colname in ["id", "xfct", "xprm", "yfct", "yprm"]:
                assert colname in hdu.data.dtype.names, "Missing required column '{0}' in positions table HDU.".format(
                    colname)

            # the fit via Sources.fitIfsCoordinates sets the domain of the polynomial to the min. and max. values of
            # the fitted data interval. These values must be stored in the FITS header and recovered because otherwise
            # the values returned by the polynomials are wrong (the default is domain=[-1,1])
            assert "PAMPELMUSE POSITIONS DOMAIN START" in hdu.header and "PAMPELMUSE POSITIONS DOMAIN END" in hdu.header
            domain = [hdu.header["PAMPELMUSE POSITIONS DOMAIN START"], hdu.header["PAMPELMUSE POSITIONS DOMAIN END"]]

            # check header for identifications of different polynomials. Note that in the table, the different types
            # are referenced using integers and the translation to the actual names is done via HIERARCH keywords.
            ident = {}
            i = 0
            while "PAMPELMUSE POSITIONS POLY{0}".format(i) in hdu.header:
                ident[i] = hdu.header["PAMPELMUSE POSITIONS POLY{0}".format(i)]
                i += 1

            # Set up the polynomials for all sources that have data available in the HDU
            ifs_coordinates_id = np.zeros((hdu.data.size,), dtype=np.int32)
            for i in xrange(hdu.data.size):
                xfctname = ident[hdu.data["xfct"][i]]  # the name of the polynomial as a string
                yfctname = ident[hdu.data["yfct"][i]]

                xfct = self.available[yfctname]  # the actual polynomial class
                yfct = self.available[yfctname]

                ifs_coordinates_id[i] = hdu.data["id"][i]
                self.ifs_coordinates_fit_functions[hdu.data["id"][i]] = [xfct(hdu.data["xprm"][i], domain=domain),
                                                                         yfct(hdu.data["yprm"][i], domain=domain)]

        else:
            logging.info("Reading IFS positions from image HDU '{0}'".format(extname))

            if poly:
                warnings.warn("Saving full position fits is deprecated.", DeprecationWarning)
                warnings.warn("The fits will be lost if sources are refitted and fits stored in the same file.",
                              DeprecationWarning)

            # check if measured IFU coordinates are free parameters in the fit
            try:
                free = hdu.header["PAMPELMUSE POSITIONS FULLFIT"]
            except KeyError:
                free = False

            if hdu.header["NAXIS"] != 3:
                logging.error("IFS coordinates must be provided as 3dim. array.")
            assert hdu.header["NAXIS"] == 3, "Expected 3dim. data but got {0}dim.".format(hdu.header["NAXIS"])

            # if the dispersion axis is the last one (python counting), then create a temporary FITS file with the
            # first and last axis swapped and reload the HDU from this file.
            if hdu.header["NAXIS3"] == 2:
                tmp_data_array = hdu.data.swapaxes(0, 2)
                tmp_coordinates_hdu = fits.ImageHDU(tmp_data_array, header=hdu.header)

                # update dispersion information (if any)
                for key in ["CRVAL", "CDELT", "CRPIX", "CTYPE", "CUNIT"]:
                    if "{0}1".format(key) in tmp_coordinates_hdu.header:
                        tmp_coordinates_hdu.header["{0}3".format(key)] = hdu.header["{0}1".format(key)]
                        del tmp_coordinates_hdu.header["{0}1".format(key)]

                # write the HDU to a temporary FITS file (fits provides a function to create a tmp-name) and open it
                # again
                self._tmp_names.append(fits.util._tmp_name("./"))
                tmp_coordinates_hdu.writeto(self._tmp_names[-1])
                hdu = fits.open(self._tmp_names[-1], memmap=True)[1]

                # remove temporary array and HDU (tmp-file will be deleted via __del__ method)
                del tmp_data_array, tmp_coordinates_hdu

            # check if dispersion information is valid
            if self.ndisp == 1:
                self.ndisp = hdu.header["NAXIS3"]
            elif hdu.header["NAXIS3"] != self.ndisp:
                logging.error("Invalid number of dispersion elements in provided IFS coordinates.")

            # check for which sources the HDU contains IFS coordinates, using header HIERARCH-keywords
            # 'PAMPELMUSE POSITIONS ID{i}'. If the keywords do not exist, the routine expects that coordinates are
            # provided for every resolved source in the catalogue.
            ifs_coordinates_id = np.zeros((hdu.header["NAXIS2"],), dtype=np.int32)
            for i in xrange(hdu.header["NAXIS2"]):
                try:
                    ID = hdu.header["PAMPELMUSE POSITIONS ID{0}".format(i)]
                    if ID not in self.ID:
                        logging.error("Received IFS coordinates for unknown source #{0}".format(ID))
                    assert ID in self.ID, "Source #{0} is not included in catalogue.".format(ID)
                    ifs_coordinates_id[i] = ID
                except KeyError:
                    if hdu.header["NAXIS2"] != (self.nsrc - self.nbackground):
                        logging.error("Cannot infer IDs for provided coordinates.")
                    assert hdu.header["NAXIS2"] == (
                        self.nsrc - self.nbackground), "Shape of data in HDU not compatible with source catalogue."
                    ifs_coordinates_id = self.ID[self.status < 4]
                    break

        # initialize properties used by the method Sources.load_ifs_coordinates()
        if not poly:
            self.ifs_coordinates_data_hdu = hdu
            self.ifs_coordinates_data_id = np.copy(ifs_coordinates_id)
            self.ifs_coordinates_free[match_on_id(self.ID, ifs_coordinates_id)] = free
        else:
            self.ifs_coordinates_fit_hdu = hdu
            self.ifs_coordinates_fit_id = np.copy(ifs_coordinates_id)
            self.ifs_coordinates_fitted[match_on_id(self.ID, ifs_coordinates_id)] = True

    def load_ifs_coordinates(self, ids=None, layers=None, interpolate=False):
        """
        Load the IFS coordinates for an individual source, a subset of sources
        or the entire catalogue. The coordinates might be loaded for the entire
        cube, or an individual layer.
        The method tries to obtain the IFS coordinates in different ways (and
        in the following order):
         (a) reading the coordinates from an avaiable FITS-ImageHDU. The method
             Sources.open_coordinates_hdu() may be used to make such an HDU
             available.
         (b) using the coordinate transformation if it has been fully defined.
         (c) directly use the reference coordinates (i.e. assuming the coord.
             transformation is A=1, B=C=D=0)
        Additionally, the method checks if polynomial fits to the IFS coordinates
        exists as a FITS-ImageHDU. Again, the method Sources.open_coordinates_hdu()
        may be used to make such an HDU available

        The routine will prepare a 4dim. array of shape (Ndisp, Nsrc, 2,2 ), and
        store it as property Sources.ifs_coordinates. Its first axis respresents
        the dispersion axis, its second axis distinguishes the individual sources,
        its third axis is x/y, and its forth axis contains the data under index 0
        and a potential fit to it under index 1.

        Parameters
        ----------
        ids : array-like or integer, optional
            The ID(s) of the sources for which the IFS coordinates should be
            loaded. If none are provided, the method will load coordinates for
            all point sources in the catalogue.
        layers : array-like  or integer, optional
            The indices of the layers for which the IFS coordinates should be loaded.
            If none are provided, the coordinates are loaded for the whole wavelength
            range.
        """
        # for debugging: check where the call was triggered
        t0 = time.time()
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        logging.debug("Call to Sources.load_ifs_coordinates() triggered by: {0} in {1}[@{2}]".format(
            calframe[1][3], calframe[1][1], calframe[1][2]))

        # sanity checks on provided IDs and layers
        if ids is not None:
            assert isinstance(ids, (int, np.integer, list, np.ndarray)),\
                "If provided, parameter 'ids' must be single integer or array-like."
            ids = np.atleast_1d(ids)
        else:
            ids = self.ID[self.status < 4]

        # check if all provided IDs are actually in the source catalogue
        match = match_on_id(self.ID, ids)
        if match.size < len(ids):
            logging.error("Invalid source IDs have been provided.")
        assert match.size == len(ids), "Not all provided IDs have no counterpart in source catalogue."

        # bring the provided IDs in the correct order wrt the source catalogue (only if needed):
        if (np.diff(match) < 0).any():
            sorter = np.unique(match, return_index=True)
            # 'sorter' contains 2 arrays, one with the indices of the matched IDs in the proper order and one with the
            # indices of those IDs in the provided list of 'ids'.
            ids = self.ID[sorter[0]]
        t1 = time.time()
        if layers is not None:
            assert isinstance(layers, (int, np.integer, list, np.ndarray)), \
                "If provided, parameter 'layers' must be transformable to an array."
            layers = np.atleast_1d(layers)
        else:
            layers = np.arange(self.ndisp)
        nlayer = layers.shape[0]

        # update boolean arrays that keep track of which sources have IFU coordinates and for which sources these
        # coordinates are free parameters in the fit
        self.ifs_coordinates_loaded[:] = False
        self.ifs_coordinates_loaded[match] = True

        # initialize array for IFS coordinates. The shape of the array is
        #    (<no. of requested layers>, <no of requested IDs>, 2, 2)
        # The last two axes distinguish between x or y coordinate and data or fit, respectively.
        self.ifs_coordinates = np.zeros((nlayer, len(ids), 2, 2), dtype=np.float32)
        t2 = time.time()

        # first choice: if available, fill in IFS coordinates using open FITS HDU
        if self.ifs_coordinates_data_hdu is not None:

            # keep track of sources for which coordinates have been calculated
            processed = np.in1d(ids, self.ifs_coordinates_data_id)
            # returns boolean array of length len(ids); processed[i]=True if ids[i] is in ifs_coordinates_data_id

            # to fill in the IFS coordinates from an open HDU, a loop over the requested layers is performed unless the
            # whole data are requested. While python loops are slow, this is the fastest way of getting the requested
            # coordinates without loading the whole coordinates data into memory because a call like hdu.section[i] is
            # much faster than a call like hdu.section[:,j,k]
            hdu_indices = match_on_id(self.ifs_coordinates_data_id, ids[processed])
            if nlayer == self.ndisp and processed.all():
                self.ifs_coordinates[..., 0] = self.ifs_coordinates_data_hdu.data[:, hdu_indices]
            elif hdu_indices.size > 0:
                for i in xrange(nlayer):
                    self.ifs_coordinates[i:i + 1, processed, :, 0:1] = \
                        self.ifs_coordinates_data_hdu.section[layers[i]][hdu_indices][..., np.newaxis]
        else:
            processed = np.zeros((len(ids),), dtype=np.bool)

        if not processed.all():
            ref_coords = np.vstack((self.x, self.y)).T[self.ifs_coordinates_loaded][~processed]
            src_indices = np.flatnonzero(~processed)

            if self.transformation.valid:  # second option: fill in IFS coordinates via coordinate transformation
                ifs_coords = self.transformation(ref_coords)
                if not interpolate or not np.isnan(ifs_coords[layers]).any():
                    self.ifs_coordinates[:, src_indices, :, 0:1] = ifs_coords[layers][..., np.newaxis]
                else:
                    logging.warning("IFS coordinates include interpolated values.")
                    for i in src_indices:
                        for j in xrange(2):
                            valid = ~np.isnan(ifs_coords[:, i, j])
                            fct = InterpolatedUnivariateSpline(np.flatnonzero(valid), ifs_coords[valid, i, j], k=2)
                            self.ifs_coordinates[:, i, j, 0] = fct(layers)
                # 4th entry in square brackets must be slice; if an index (i.e. 0) is used, the shape of the resulting
                # subarray is not as expected
            else:  # third option: just use reference coordinates
                self.ifs_coordinates[:, src_indices, :, 0:1] = ref_coords[:, :, np.newaxis]
        t3 = time.time()

        # the fits to the IFS coordinates can either be loaded from an available image HDU (deprecated)
        if self.ifs_coordinates_fit_hdu is not None and isinstance(self.ifs_coordinates_fit_hdu,
                                                                   fits.hdu.image._ImageBaseHDU):
            in_fit_hdu = np.in1d(ids, self.ifs_coordinates_fit_id)
            hdu_indices = match_on_id(self.ifs_coordinates_fit_id, ids[in_fit_hdu])
            if nlayer == self.ndisp and in_fit_hdu.all():
                self.ifs_coordinates[..., 1] = self.ifs_coordinates_fit_hdu.data[:, hdu_indices]
            elif hdu_indices.size > 0:
                for i in xrange(nlayer):
                    self.ifs_coordinates[i:i + 1, in_fit_hdu, :, 1:2] = \
                        self.ifs_coordinates_fit_hdu.section[layers[i]][hdu_indices][..., np.newaxis]

        # or from the dictionary of previously performed fits
        i = 0
        wave = self.wave[layers]
        for ID in ids:
            if ID in self.ifs_coordinates_fit_functions.keys():
                xfit, yfit = self.ifs_coordinates_fit_functions[ID]
                self.ifs_coordinates[:, i, 0, 1] = xfit(wave)
                self.ifs_coordinates[:, i, 1, 1] = yfit(wave)
            i += 1
        t4 = time.time()
        logging.debug("Time required to arrange IDs: {0}s".format(t1-t0))
        logging.debug("Time required to initialize array: {0}s".format(t2-t1))
        logging.debug("Time required to load IFS coordinates: {0}s".format(t3-t2))
        logging.debug("Time required to load fits to IFS coordinates: {0}s".format(t4-t3))

    def fit_coordinate_transformation(self, index=slice(None)):
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
        assert self.x is not None, "Reference coordinates missing."
        assert self.y is not None, "Reference coordinates missing."

        if self.ifs_coordinates is None:
            logging.error("Cannot determine coordinate transformation: because no IFS coordinates are available.")
        assert self.ifs_coordinates is not None, "IFS coordinates missing."

        # check if IFS coordinates are available for at least 2 sources.
        if self.ifs_coordinates_loaded.sum() < 2:
            logging.error(
                "Cannot determine coordinate transformation because <2 sources have IFS coordinates available.")
        assert self.ifs_coordinates_loaded.sum() >= 2, \
            "Need at least 2 sources with IFS coordinates to determine transformation."

        # if integer has been provided as keyword 'index', translate it into a slice
        if isinstance(index, (int, np.int)):
            index = slice(index, index + 1, 1)

        # carry out fit
        ref_coordinates = np.vstack((self.x, self.y)).T[self.ifs_coordinates_loaded]
        self.transformation.fit_to_data(ref_coordinates, self.ifs_coordinates[index][..., 0],
                                        ids=self.ID[self.ifs_coordinates_loaded])

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

    def fitIfsCoordinates(self, ids, order, bins=1, mask=None, function="plain", commonSlope=False, statusBar=None,
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
        bins : int, optional
            The number of bins in case a step-polynomial is used as fitting
            function. IMPORTANT: Not used anymore because step polynomials
            are not used anymore.
        mask : array_like or None, optional
            An optional bad-pixel mask. If specified, must have as many pixels
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
        if bins != 1:
            logging.warning("Parameter 'bins' is not used anymore.")

        # find the indices of the source(s) to be fitted in the catalogue
        if ids is None:
            indices_to_fit = np.flatnonzero(self.status < 4)
        else:
            indices_to_fit = match_on_id(self.ID, np.atleast_1d(ids))

        # check if IFU coordinates have been loaded, otherwise do so
        if not self.ifs_coordinates_loaded[indices_to_fit].all() or self.ifs_coordinates.shape[0] < self.ndisp:
            self.load_ifs_coordinates(ids=self.ID[indices_to_fit])

        if not function.lower() in self.available.keys():
            logging.critical("Unknown fitting function provided: {0}".format(function))
        fitfunction = self.available[function.lower()]

        # create pseudo-mask if None provided
        valid = ~mask if mask is not None else np.ones((self.ndisp,), dtype=np.bool)

        # determine 'mean' coordinate if option 'commonSlope' selected
        if commonSlope:

            # sources used for creating mean slope are those with status 2 or 3
            used = ((self.status > 1) & (self.status < 4))[self.ifs_coordinates_loaded]

            # if "instrument" is known, then only use sources within FoV to determine common slope.
            if instrument is not None:
                used = used & instrument.infov(self.ifs_coordinates[..., 0].mean(axis=0))

            meancoord = np.mean(self.ifs_coordinates[:, used], axis=1)
            meanvalid = valid & ~np.isnan(meancoord)
            xfit = fitfunction.fit(self.wave[meanvalid], meancoord[meanvalid, 0, 0], order)
            yfit = fitfunction.fit(self.wave[meanvalid], meancoord[meanvalid, 1, 0], order)

            order = 0  # all individual coordinates are now "fitted" with zeroth order polynomials

# TODO        _i = self.ifs_coordinates_loaded.cumsum()[indices_to_fit] - 1
# TODO        slc = np.ix_(mask.nonzero()[0], _i)
# TODO
# TODO        t0 = time.time()
# TODO        xfit = np.polynomial.polynomial.polyfit(self.wave[mask], self.ifs_coordinates[slc][..., 0, 0], order)
# TODO        yfit = np.polynomial.polynomial.polyfit(self.wave[mask], self.ifs_coordinates[slc][..., 1, 0], order)
# TODO        t1 = time.time()
# TODO        print t1-t0, type(xfit), type(yfit)

        # loop over selected sources and get their polynomial coordinate fits
        k = 1
        for j in indices_to_fit:

            # get index of source in array of IFS coordinates
            i = self.ifs_coordinates_loaded.cumsum()[j] - 1

            valid_xi = valid & ~np.isnan(self.ifs_coordinates[:, i, 0, 0])
            valid_yi = valid & ~np.isnan(self.ifs_coordinates[:, i, 1, 0])

            if commonSlope:
                valid_xi = valid & ~np.isnan(self.ifs_coordinates[:, i, 0, 0])
                xfit_i = xfit + fitfunction.fit(
                    self.wave[valid_xi], self.ifs_coordinates[valid_xi, i, 0, 0] - xfit(self.wave[valid_xi]), order)

                yfit_i = yfit + fitfunction.fit(
                    self.wave[valid_yi], self.ifs_coordinates[valid_yi, i, 1, 0] - yfit(self.wave[valid_yi]), order)
            else:
                xfit_i = fitfunction.fit(self.wave[valid_xi], self.ifs_coordinates[valid_xi, i, 0, 0], order)
                yfit_i = fitfunction.fit(self.wave[valid_yi], self.ifs_coordinates[valid_yi, i, 1, 0], order)

            self.ifs_coordinates_fit_functions[self.ID[j]] = [xfit_i, yfit_i]

            # find index of source in (full) source catalogue
            self.ifs_coordinates_fitted[j] = True
            self.ifs_coordinates[:, i, 0, 1] = xfit_i(self.wave)
            self.ifs_coordinates[:, i, 1, 1] = yfit_i(self.wave)

            logging.debug("Polynomial fit to source #{0} [{1}/{2}]".format(self.ID[j], k, len(indices_to_fit)))
            if statusBar is not None:
                statusBar.showMessage(
                    "Polynomial fit to source #{0} [{1}/{2}]".format(self.ID[j], k, len(indices_to_fit)), 5000)
            k += 1

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
            select = np.arange(self.nsrc, dtype=np.int32)

        # sort by increasing magnitudes (if possible)
        if self._mag is not None: select = select[self._mag[select].argsort()]

        outCols = [fits.Column(name="id", format="J", array=self.ID[select])]

        formats = {"x": "E", "y": "E", "mag": "E", "ra": "D", "dec": "D", "status": "I", "specrow": "I", "primary": "J"}
        for attribute in ["x", "y", "mag", "ra", "dec", "status", "primary", "specrow"]:
            if getattr(self, "_{0}".format(attribute)) is not None:
                logging.debug("Adding column '{0}' to table...".format(attribute))
                outCols.append(fits.Column(name=attribute, format=formats[attribute],
                                           array=getattr(self, "_{0}".format(attribute))[select]))

        return fits.new_table(outCols, header=fits.Header(cards=[fits.Card(key="EXTNAME", value="SOURCES")]))

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
            components = np.arange(self.ncmp, dtype=np.int32)
        elif isinstance(components, (int, np.integer, list, np.ndarray)):
            components = np.atleast_1d(components)
        elif components == "current":
            components = np.arange(self.ncmp, dtype=np.int32)[self.spectrum_loaded]

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

        outHDU = fits.ImageHDU(self.data, name="SPECTRA")
        outHDU.header["CRPIX3"] = self.crpix
        outHDU.header["CRVAL3"] = self.crval
        outHDU.header["CDELT3"] = self.cdelt
        if self.ctype is not None: outHDU.header["CTYPE3"] = self.ctype
        if self.cunit is not None: outHDU.header["CUNIT3"] = self.cunit

        return outHDU

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

            # get the maximum polynomial degree used in the fits. Note that this would not be necessary one would use
            # the feature that FITS tables can store arrays of variable length
            # (http://docs.astropy.org/en/tmp-v0.1-wcython/io/fits/usage/unfamiliar.html)
            # Unfortunately, the implementation in astropy seems buggy and I always get an array full of zeros when
            # opening the file again. Therefore, in case different degrees have been used in the fits, the higher
            # orders of the low-order polynomials are padded with zeros.
            xmax = max(self.ifs_coordinates_fit_functions[i][0].degree() for i in ids_tosave)
            ymax = max(self.ifs_coordinates_fit_functions[i][0].degree() for i in ids_tosave)

            # prepare the arrays to hold the data
            src = np.zeros((nfit, ), dtype=np.int32)
            xfct = np.zeros((nfit, ), dtype=np.int16)
            xprm = np.zeros((nfit, xmax + 1), dtype=np.float32)
            yfct = np.zeros((nfit, ), dtype=np.int32)
            yprm = np.zeros((nfit, ymax + 1), dtype=np.float32)

            #  loop over all sources that have polynomial fits available
            i = 0
            for ID, polyfits in self.ifs_coordinates_fit_functions.iteritems():
                if ID not in ids:
                    continue
                src[i] = ID
                for fct in self.available.keys():
                    # get fit function: first check which class the instance originates from, then translate name to
                    # an integer
                    if isinstance(polyfits[0], self.available[fct]):
                        xfct[i] = self.polyident[fct]
                xprm[i, :(polyfits[0].degree() + 1)] = polyfits[0].coef
                for fct in self.available.keys():
                    if isinstance(polyfits[1], self.available[fct]):
                        yfct[i] = self.polyident[fct]
                yprm[i, :(polyfits[1].degree() + 1)] = polyfits[1].coef

                # store domain, this is needed because the domain set by the fit is the min and max of the fitted range
                # and this if different from the default one used by the __init__ methods. Since currently the start
                # and end of the domain are stored as HIERARCH-keywords, the domain must be the same for all fits.
                if i == 0:
                    domain = polyfits[0].domain
                else:
                    assert abs(polyfits[0].domain - domain).max() < 1e-3
                assert abs(polyfits[1].domain - domain).max() < 1e-3

                i += 1

            # prepare the output table
            outhdu = fits.new_table([fits.Column(name="id", format="J", array=src),
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
