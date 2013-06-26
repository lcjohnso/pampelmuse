"""
SOURCES
=======
The `Sources`-class is the PampelMuse-routine that handles the entire source
catalogue. It keeps track of the loaded sources, their grouping into the
individual components, their spectra and positions. It also provides some
methods to write the individual properties into FITS-HDUs.

Since the `CoordinateTransformation`-class of PampelMuse is provided as a 
superclass to `Sources`, its functionality can also be accessed through the
`Sources`-class.

The routine is heavily used by all the main scripts of PampelMuse.

CHANGELOG:
----------
2012/03/07, v0.1.0.0, preliminary version completed 
2012/04/20, v0.2.0.0, implemented a first version of fitting the same slope to
                      all source IFU coordinates.
2012/04/23, v0.2.1.0, bugfix: masked regions were not properly used in a
                              `commonSlope`-fit in method `fitIFUCoords`.
2012/05/03, v0.2.2.0, code crashed if the `primary` component of a source had
                      `status` -1. Now such sources are excluded as well.
2012/06/16, v0.2.3.0, bugfix that caused problems when no coordinate
                      transformation was used.
2012/06/12, v0.2.4.0, bugfix: previous update replaced an `if`-statement by
                              `elif` in property `Fit`. This resulted in a
                              fixed coord. transformation even if `posvar`>0.
2012/08/07, v0.2.5.0, bugfix: in previous versions the preparation of `specrow`
                              did not work when a source had ID 0 but was
                              excluded later on. Then specrow=0 was missing.
2012/09/03, v0.2.5.1, rewrote method `distToBrighter` with a loop because
                      previous calculation caused heavy memory usage when many
                      sources were in the list.
2012/11/10, v0.3.0.0, when creating a Sources instances, it is now possible to
                      provide an instance of the new Instrument-class to handle
                      instrument-specific characteristics.
                      changed behavior of fitIFUCoords (sources now have same
                      slope if resp. option is used), added option for fitting
                      Chebyshev-polynomial and added some documentation.
2012/11/11, v0.3.0.1, removed remaining print-command used for debugging.
2013/03/26, v0.4.0.0, modified the 'weights'-property and the 'signalToNoise'-
                      method in order to adapt them to the new way of
                      calculating the PSF profiles.
2013/03/27, v0.4.0.1, slight modification on how Instrument-instance is
                      initialized.
2013/04/05, v0.5.0.0, modified definition of the various properties derived
                      from the catalogue of provided sources; revised and de-
                      bugged the way of providing the coordinates of the
                      sources on the IFU data, in particular no instances of
                      pampelmuse.core.value_array.ValueArray are used anymore. 
2013/04/08, v0.5.1.1, bugfix: when fitting polynomials to IFU coords, x and y
                              were confused when determining the sources inside
                              the FoV (method Sources.fitIFUCoords).
                      also included some 'assert' statements when calculating
                      new coordinate transformation (method Sources.calcTrans).
2013/04/11, v0.5.2.1, bugfix: code crashed when IDs were provided as a list in
                              Sources.loadIFUCoords or matchOnID. This is solved
                              now.
2013/04/19, v0.5.2.2, included option to provide unit of wavelength scale in
                      method 'loadSpectra' and to store it in 'makeSpecHDU'.
2013/05/03, v0.5.2.3, included a check in method 'fitIFUCoords' whether the
                      provided function is valid.
2013/05/29, v0.5.2.4, some minor changes
2013/05/31, v0.5.2.5, removed spurious 'print' command used for debugging
2013/06/13, v0.6.0.0, removed methods 'edgeDist' and 'signalToNoise' as they were
                      only loosely connected to the class 'Sources'.
2013/06/26, v0.6.0.1, modified the method 'loadIFUCoords' so that instead of
                      issuing a warning when not all sources have IFU coordinates
                      available, it loads the reference coordinates for those
                      sources.
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.6.0.1"
import logging
import numpy as np, pyfits
from numpy.polynomial import Chebyshev, Legendre

try: from PyQt4 import QtCore ## check if PyQt is available 
except ImportError: pass

## required PampelMuse-packages
from pampelmuse.core.positions import CoordinateTransformation
from pampelmuse.functions.step_poly_fit import StepPolyFit
## from pampelmuse.core.value_array import ValueArray
from pampelmuse.core.instruments import Instrument


def matchOnID(parent, child):
    """
    Compare to 1dim. lists for common entries. For each element in `child`
    that is also part of the `parent` list, the index of the element in the
    `parent' list is returned.

    Parameters
    ----------
    parent : array_like, 1dim.
        The list that that is scanned
    child : array_like, 1dim.
        The list containing the entries that `parent' is scanned for.

    Returns
    -------
    indices : nd_array
        The index of every `child'-entry in the `parent' list (if found).
    """
    c = np.asarray(child, dtype=np.int32)
    p = np.asarray(parent, dtype=np.int32)
    return np.nonzero((c[:,np.newaxis]-p)==0)[1]




class Sources(CoordinateTransformation, object):
    """
    The 'Sources'-class is designed for the following purposes:
    * Initialize and maintain a coherent catalog of sources. To this aim, the
      class uses several properties that are either inherent to the sources
      (ID, x, y, mag) or assigned to each source depending on its usage in
      the analysis (status, specrow, primary)
    * Load, maintain and save the set of spectra that is associated with the
      source catalog
    * Load, maintain and save the set of IFU coordinates that is associated
      with the source catalog. To this aim, the 'CoordinateTransformation'
      -class of PampelMuse is loaded as a superclass. This way, a coordinate
      transformation that might be in use can be fully accessed.
    """

    ## Properties that can be set by the user:
    _x        = None  ## input x coordinates
    _y        = None  ## input y coordinates
    _mag      = None  ## broadband magnitudes
    _status   = None  ## status of source in analysis [-1, 4]
    _primary  = None  ## primary component of the source

    ## Properties that are calculated internally unless provided by the
    ## user
    _specrow  = None  ## row containing spectrum in the array of spectra
                      ## NOTE that although this information is usually
                      ## included in the 'SOURCES' HDU, it is calculated
                      ## internally

    ## Properties that are only calculated internally
    _indexmap = None  ## map from specrow to index
    _weights  = None  ## map from specrow to relative magnitude


    def __init__(self, data=None, instrument=None, **kwargs):
        """
        Initialize a new instance of the `Sources`-class.

        Parameters
        ----------
        data : a pyfits BinTable-HDU instance or a structured array, optional
            The source catalog that should be used to initialize the instance.
            It must contain at least a field named `id`, all other fields are
            optional and the corresponding properties might be defined later.
            If `data` is not provided, than `ids` must be given.
        instrument : instance of pampelmuse.core.instruments.Instrument-class
                     or None, optional
            If a class-instance is provided, it will be used to identify bad
            spaxels, determine the exact shape of the FoV and the location of
            sources within or outside the FoV.
            E.g., if FLAMES/ARGUS is used, the missing edges and fibres
            outside the CCD will be correctly identified. If instrument=None,
            the routine will work with a rectangular FoV of size 22x14.
        ids : array_like, optional
            A 1dim. array containing the source IDs that should be used to
            initialize the scene. Only used if 'data' is not provided or does
            not contain an column with the source IDs.

        Note
        ----
        Additional keywords can be used to provide source-properties that are
        not present in 'data'

        """
        if "hdu" in kwargs.keys():
            logging.warning("Parameter 'hdu' is not used anymore.")

        if isinstance(data, pyfits.BinTableHDU):
            logging.info("Initializing Sources instance using provided HDU...")
            data = np.array(data.data, dtype=data.data.dtype)
        else:
            logging.info("Initializing Sources instance...")

        ## Initialize properties:
        ## Note that the  properties below can be accessed in two ways:
        ## a) with an underscore: direct access, no sanity checks performed
        ## b) without underscore: check whether requested property has been
        ##                        defined and whether data used to update
        ##                        property is valid.
        
        ## Source IDs must be present
        if data != None:
            try:
                self.ID = data[["id"]].view(data["id"].dtype).reshape(-1,)
            except ValueError:
                logging.critical("No field `ID' found in provided data.")
                pass
        elif "ids" in kwargs.keys():
            self.ID = np.asarray(kwargs["ids"])
            logging.info("Catalog contains {0} sources".format(self.nsrc))
        if self.ID == None:
            logging.critical("Source IDs must be provided upon initialization.")

        ## other properties are optional and can be provided later.
        for name in ["x", "y", "mag", "status", "primary", "specrow"]:
            if data != None and name in data.dtype.names:
                setattr(self, name, 
                        data[[name]].view(data.dtype[name]).reshape(-1,))
            elif name in kwargs.keys():
                setattr(self, name, kwargs[name])


        ## "Initialize" instrument-class
        if isinstance(instrument, Instrument):
            self.instrument = instrument
##            logging.debug("Selected instrument is '{0}'.".format(instrument.name))
        else:
            logging.warning("No valid instrument specifier provided.")
            self.instrument = Instrument()

            
        ## initialize arrays that will contain important information about
        ## source coordinates:
        ## -a boolean array indicating if IFU coords have been loaded for a
        ##  source
        self.loadedIFUCoords = np.zeros((self.nsrc,), dtype=np.bool)

        ## -a boolean array indicating if a polynomial fit has been performed
        ##  to the IFU coords of a source
        self.hasPolyFit = np.zeros((self.nsrc,), dtype=np.bool)

        ## -a boolean array indicating if the IFU coords of a source are a
        ##  free fit parameter
        self.isFreePar = np.zeros((self.nsrc,), dtype=np.bool)

        ## place-holders for spectral properties
        self.wave  = None
        self.data  = None
        self.sigma = None
        self.ndisp = None
        self.crval = 1.
        self.crpix = 1
        self.cdelt = 1.
        self.ctype = None

        ## place-holders for the arrays that will contain the IFU coords:
        self.xdata = None
        self.ydata = None

        super(Sources, self).__init__(**kwargs)


###############################################################################
##                   Definitions of the various properties,                  ##
##                   and the methods to create & access them                 ##
###############################################################################

    @property
    def nsrc(self):
        """
        Returns
        -------
        nsrc : int
            The total number of sources in the catalogue (incl. background
            -components). 
        """
        return self.ID.size

## Reference x-coordinates
    def set_x(self, value):
        """
        The setter method for the 'x'-property, containing the coordinate of
        each source along the x-axis of the reference system.

        Parameters
        ----------
        value : array_like
            The x-coordinates. Must be a 1dim. list of length 'nsrc'.
        """
        value = np.asarray(value, dtype=np.float32)
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set attribute 'x', data has invalid shape.")
        else:
            self._x = value

    def get_x(self):
        """
        The getter method of the 'x'-property.

        Returns
        -------
        x : nd_array
            A 1dim. of length 'nsrc' containing the coordinate of each source
            along the x-axis of the reference system.
        """
        if self._x == None:
            logging.error("Requested attribute 'x' not provided so far.")
        return self._x

    x  = property(\
        get_x, set_x, None,
        "The source coordinates along the x-axis of the reference system.")
            

## Reference y-coordinates
    def set_y(self, value):
        """
        The setter method for the 'y'-property, containing the coordinate of
        each source along the y-axis of the reference system.

        Parameters
        ----------
        value : array_like
            The y-coordinates. Must be a 1dim. list of length 'nsrc'.
        """
        value = np.asarray(value, dtype=np.float32)
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set attribute 'y', data has invalid shape.")
        else:
            self._y = value

    def get_y(self):
        """
        The getter method of the 'y'-property.

        Returns
        -------
        y : nd_array
            A 1dim. of length 'nsrc' containing the coordinate of each source
            along the y-axis of the reference system.
        """
        if self._y == None:
            logging.error("Requested attribute 'y' not provided so far.")
        return self._y

    y = property(\
        get_y, set_y, None,
        "The source coordinates along the y-axis of the reference system.")


## Broadband magnitudes
    def set_mag(self, value):
        """
        The setter method for the 'mag'-property, containing the magnitude of
        each source in some reference filter.

        Parameters
        ----------
        value : array_like
            The magnitudes. Must be a 1dim. list of length 'nsrc'.
        """
        value = np.asarray(value, dtype=np.float32)        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set attribute 'mag', data has invalid shape.")
        else:
            self._mag = value
            
    def get_mag(self):
        """
        The getter method of the 'mag'-property.

        Returns
        -------
        mag : nd_array
            A 1dim. of length 'nsrc' containing the magnitude of each source
            in the reference filter.
        """
        if self._mag == None:
            logging.error("Requested attribute 'mag' not provided so far.")
        return self._mag

    mag = property(get_mag, set_mag,  None,
                   "The broadband magnitudes of the sources.")

    
## The status of the sources during the fit
    def set_status(self, value):
        """
        The setter method for the 'status'-property, containing the status of
        each source during the fit.

        Parameters
        ----------
        value : array_like
            Must be a 1dim. list of length 'nsrc'. The status of each source
            should be provided as an integer:
            * "-1" means source is not used,
            * "0" means source contributes to unresolved component,
            * "1" means source is fitted in a single component together with
              a brighter source,
            * "2" means the source is fitted independently,
            * "3" means the source is used as a PSF source and
            * "4" means a background component.
        """
        value = np.asarray(value, dtype=np.int32)
        if not value.size == self.nsrc and value.ndim==1:
            logging.error(\
                "Cannot set attribute 'status', data has invalid shape.")
        else:
            self._status = value
                    
    def get_status(self):
        """
        The getter method of the 'status'-property.

        Returns
        -------
        status : nd_array
            A 1dim. of length 'nsrc' containing the status of each source
            in the fit.
        """
        if self._status == None:
            logging.error("Requested attribute 'status' not provided so far.")
        return self._status

    status = property(get_status, set_status, None,
                      "The status of every source during the fitting process.")


## The primary source in each component
    def set_primary(self, value):
        """
        The setter method for the 'primary'-property, containing the index of
        the primary source for each entry in the source catalogue. This index
        is identical to the index of the source itself for sources with
        'status'>2.

        Parameters
        ----------
        value : array_like
            The indices of the primary sources. Must be a 1dim. list of
            integers of length 'nsrc'.
        """
        value = np.asarray(value, dtype=np.int32)
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set 'primary', data has invalid shape.")
        else:
            self._primary = value

    def get_primary(self):
        """
        The getter method of the 'primary'-property.

        Returns
        -------
        primary : nd_array
            A 1dim. of integers and of length 'nsrc' containing the index of
            the primary source of each entry in the source catalogue.
        """
        if self._primary == None:
            logging.error("Requested attribute 'primary' not provided so far.")
        return self._primary

    primary = property(get_primary, set_primary, None,
                       "The primary source of each available source.")


## The data row containing the spectrum of a source.
    def set_specrow(self, value):
        """
        The setter method for the 'specrow'-property, containing the index of
        the row that contains the spectrum of a source.

        Parameters
        ----------
        value : array_like
            The indices of the rows containing the spectra. Must be a 1dim.
            list of integers of length 'nsrc'.
        """
        value = np.asarray(value, dtype=np.int32)
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set 'specrow', data has invalid shape.")
        else:
            self._specrow = value


    def get_specrow(self):
        """
        The getter method of the 'specrow'-property.

        Returns
        -------
        specrow : nd_array
            A 1dim. of integers and of length 'nsrc' containing the index of
            the row that contains the spectra in the data array. Unless the
            property has been set before using its setter method, the indices
            are automatically assigned, using the information from the
            attributes 'primary' and 'status' (if possible).
        """
        if self._specrow == None:
            if self.status == None:
                logging.error("Cannot create attribute 'specrow'.")
                logging.error("Attribute 'status' is not specified.")
            else:
                if self.status.min()<2 and self.primary == None:
                    logging.error("Cannot create attribute 'specrow'.")
                    logging.error("Attribute 'primary' is not specified.")
                else:
                    self._specrow = np.zeros(self.nsrc, dtype=np.int32)

                    ## set row for independent ('primary') sources
                    self._specrow[self.status>1] \
                        = np.arange((self.status>1).sum(), dtype=np.int32)

                    unresRow = self._specrow.max()+1

                    for i in xrange(self.nsrc):
                        if self.status[i] == 1:
                            ## find spectrum of primary component; following
                            ## loop necessary because initially assigned
                            ## primary component might be secondary component
                            ## of another source.
                            secondary = []
                            j=i                  
                            while self.primary[j]!=self.ID[j]:

                                ## exclude source if primary source has
                                ## status = -1.
                                if not (self.ID==self.primary[j]).any():
                                    logging.error("No 'primary' counterpart for source #{0}".format(self.ID[i]))
                                    logging.error("Source #{0} excluded".format(self.ID[i]))
                                    self.status[i] = -1
                                    break

                                secondary.append(j)
                                j = np.where(self.ID==self.primary[j])[0][0]

                            self._specrow[secondary] = self._specrow[j]
                        elif self.status[i] == 0:
                            self._specrow[i] = unresRow

        return self._specrow

    specrow = property(\
        get_specrow, set_specrow, None,
        "The column of the data array containing the spectrum of a source.")

        
## The mapping from a spectrum to the indices of contributing sources in the
## catalogue.
    @property
    def indexmap(self):
        """
        Returns
        -------
        indexmap : dictionary
            The mapping from the 'specrow' (keys) to the index/indices (values)
            of the corresponding source IDs in the catalogue.
        """
        if self._indexmap == None:
            if self.specrow != None:
                self._indexmap = {}
                for i in xrange(self.ncmp):
                    self._indexmap[i] = np.where(self.specrow==i)[0]
            else:
                logging.error("Cannot create attribute 'indexmap'.")
                logging.error("Attribute 'specrow' is not specified.")
        return self._indexmap


## The magnitude weights of the individual contributions to a spectrum
    @property
    def weights(self):
        """
        Returns
        -------
        weights: nd_array
            A 1dim. array of the same length as the whole sourcelist containing
            the magnitude weight of each source in its component. The magnitude
            weight is defined as 10**(-0.4*df), where df is the fraction of
            flux the source contributes to the component.

        Notes
        -----
        Changed since version 0.4. Before 'weights' was a dictionary mapping
        the 'specrow' (keys) to the relative magnitudes (values) of its
        component(s).
        """
        if self._weights == None:
            if self.specrow == None:
                logging.error("Cannot calculate attribute 'weights'.")
                logging.error("Attribute 'specrow' is not specified.")
            else:
                self._weights = np.zeros((self.nsrc,), dtype=np.float64)
                if self.status.min() <= 1:

                    if self.mag == None:
                        logging.error("Cannot calculate attribute 'weights'.")
                        logging.error("Attribute 'mag' is not specified.")
                    else:
                        for i in xrange(self.nsrc):
                            same_specrow = (self.specrow==self.specrow[i])

                            self._weights[i] = self._mag[i] + 2.5*np.log10(\
                                (10**(-0.4*self._mag[same_specrow])).sum())
        return self._weights


## The number of independent components in the fit.
    @property
    def ncmp(self):
        """
        Returns
        -------
        ncmp : int
            The number of components (i.e. independent spectra) in the
            catalogue.
        """
        if self.specrow == None: return None
        else: return self.specrow.max()+1


## Return a boolean array of len(nsrc) that is true if the source is resolved. 
    @property
    def resolved(self):
        """
        Returns
        -------
        resolved : ndarray
            A boolean array of len(self.nsrc) indicating whether a source is
            resolved. If self.status is undefined, all sources are considered
            resolved.
        """
        if self.status == None: return np.ones((self.nsrc,), dtype=np.bool)
        else: return (self.status>0)&(self.status<4)


## A flag on whether a sky component is included in the catalogue. 
    @property
    def hasSky(self):
        """
        Returns
        -------
        hasSky : bool
            A flag indicating whether a sky component is included in the
            catalogue.
        """
        if self.status == None: return False
        return (self.status==4).any()
    

## The row of the sky component (if any)
    @property
    def skyRow(self):
        """
        Returns
        -------
        skyRow : int or None
            The row of the sky spectrum (if any, otherwise returns None)
        """
        if not self.hasSky or self.specrow==None: return None
        else: return self.specrow[np.where(self.status==4)[0][0]]


## A flag on whether a unresolved stellar component is included in the
## catalogue.     
    @property
    def hasUnres(self):
        """
        Returns
        -------
        hasUnres : bool
            A flag indicating whether an unresolved stellar component is
            included in the catalogue.
        """
        if self.status == None: return False
        return (self.status==0).any()


## The row of the unresolved stellar component (if any)
    @property
    def unresRow(self):
        """
        Returns
        -------
        unresRow : int or None
            The row of the spectrum of the unresolved component (if any,
            otherwise returns None)
        """
        if not self.hasUnres or self.specrow==None: return None
        else: return self.specrow[np.where(self.status==0)[0][0]]
            

## The indices of the sources used to determine the PSF (if any)
    @property
    def psfIndices(self):
        """
        Returns
        -------
        psfIndices : nd_array
            A 1dim. array containing the indices of the sources that are used
            to determine the PSF (i.e. sources whose `status` is 3).
        """
        return np.arange(self.nsrc, dtype=np.int32)[self.status==3]


###############################################################################
##  Methods providing the user with informations about the source catalogue  ##
###############################################################################
    def info(self):
        """
        Print out some information about the loaded source catalogue. Also
        check if all necessary properties have been defined.
        """
        try:
            logging.debug("Current structure of loaded Sources-instance:")
            logging.debug(" Component:   IDs (& mags):")

            for i in xrange(self.specrow.max()+1):
                logString = " {0:10d}".format(i)
   
                slc = np.where(self.specrow==i)[0]

                for j in xrange(slc.size):
                    logString += "   {0:6d} ({1:.2f})".format(\
                        int(self.ID[slc[j]]),
                        float(abs(self.weights[slc[j]])))

                    if not np.mod(j+1, 2):
                        logging.debug(logString)
                        logString = 11*" "
                        
                if np.mod(j, 2)==0: logging.debug(logString)
                
        except AttributeError:
            logging.error("Some source properties could not be set.")
            return

        if self.verifyTrans:
            CoordinateTransformation.info(self)



###############################################################################
##              Methods dealing with the spectra of the sources              ##
###############################################################################
    def loadSpectra(self, data, sigma=None, crval=1., cdelt=1.,
                    crpix=1, ctype="AWAV", cunit="Angstrom", **kwargs):
        """
        Load the spectra for all sources in the catalogue. The method might
        also be used to provide an empty array that is filled later.

        Properties
        ----------
        data : array_like or pyfits ImageHDU-instance, shape=('ncmp', k)
            The spectra to be loaded, given either as ian image-like FITS-HDU
            or a plain array.
        sigma : array_like or pyfits ImageHDU-instance, same shape as `data`
                optional
            Uncertainties associated with the spectra provided as `data`.
            Must be same type (i.e. HDU or array)  as `data`.
        hdu : boolean, optional
            Flag indicating whether `data` (and `sigma` if provided) is an
            HDU or an array.
        crval : float, optional
            The wavelength at the reference pixel. Note that the value might
            be overwritten in case `hdu` is True and the keyword `CRVAL1` is
            found in the FITS-header of `data`.
        cdelt : float, optional
            The dispersion of the spectra (per spectral pixel). Note that the
            value might be overwritten in case `hdu` is True and the keyword
            `CDELT1` is found in the FITS-header of `data`.
        crpix : int, optional
            The reference pixel. Note that the value might be overwritten in
            case `hdu` is True and the keyword `CRPIX1` is found in the FITS-
            header of `data`.
        ctype : string, optional
            The wavelength type of the spectra. Note that the value might be
            overwritten in case `hdu` is True and the keyword `CTYPE1` is
            found in the FITS-header of `data`. Default is 'AWAV'.
        cunit : string, optional
            The unit of the wavelength scale. Note that the value might be
            overwritten in case `hdu` is True and the keyword `CUNIT1` is
            found in the FITS-header of `data`. Default is 'Angstrom'.
        """
        if self.ncmp == None:
            logging.error("No assigment of sources to spectra performed.")
       
        self.crval = crval
        self.cdelt = cdelt
        self.crpix = crpix
        self.ctype = ctype
        self.cunit = cunit

        if "hdu" in kwargs.keys():
            logging.warning("Parameter 'hdu' is not used anymore.")

        if isinstance(data, (pyfits.ImageHDU,  pyfits.PrimaryHDU)):
            self.data = data.data
            if "CRVAL1" in data.header: self.crval = data.header["CRVAL1"]
            if "CDELT1" in data.header: self.cdelt = data.header["CDELT1"]
            if "CRPIX1" in data.header: self.crpix = data.header["CRPIX1"]
            if "CTYPE1" in data.header: self.ctype = data.header["CTYPE1"]
            if "CUNIT1" in data.header: self.cunit = data.header["CUNIT1"]
        else:
            self.data = data

        if self.data.shape[0] != self.ncmp:
            logging.error(\
                "No. of provided spectra [{0}] and components [{1}] differ."\
                    .format(data.shape[0], self.ncmp))
            assert self.data.shape[0] == self.ncmp, \
                "No. of spectra must match no. of components."

        if sigma != None and isinstance(sigma, pyfits.BinTableHDU):
            self.sigma = sigma.data
        else:
            self.sigma = sigma

        if self.sigma != None:
            if self.sigma.shape != self.data.shape:
                logging.error("Spectra and sigma arrays have different shapes.")
                assert self.sigma.shape == self.data.shape, \
                    "Sigma array must match shape of data array."
        else:
            self.sigma = np.zeros(self.data.shape, dtype=np.float32)
            ## create empty 'sigma'-array if none provided.
            
        self.ndisp = self.data.shape[1] ## length of dispersion axis

        ## create wavelength array
        self.wave = np.arange(self.ndisp, dtype=np.float64)
        self.wave -= (self.crpix-1)
        self.wave *= self.cdelt
        self.wave += self.crval


    def provideNames(self):
        """
        Short routine to provide the IDs/names of the independent components
        in the loaded catalogue as a QStringList to the `InspectFitData`-class
        of PampelMuse.

        Returns
        -------
        names : QStringList
           The names of the coomponents
        """
        ownSpec = np.in1d(self.status, np.arange(2,4))

        self.names   = QtCore.QStringList(self.ID[ownSpec].astype("|S6"))
        self.nameMap = dict((zip(self.ID[ownSpec].astype("|S6"),
                                 self.specrow[ownSpec])))
        if self.hasSky:
            self.names.append("Sky")
            self.nameMap["Sky"] = self.skyRow
        if self.hasUnres:
            self.names.append("Unresolved")
            self.nameMap["Unresolved"] = self.unresRow
        
        return self.names



    def viewSpectrum(self, name=None, ID=None):
        """
        Return the spectrum corresponding to a provided ID or name of the
        component.
        
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
        The requested spectrum.
        """
        if name==None and ID==None:
            logging.error("Either name or ID must be provided.")
            return

        if name != None:
            return self.data[self.nameMap[name]]
        else:
            return self.data[self.specrow[np.where(self.ID==ID)[0][0]]]



###############################################################################
##            Methods dealing with IFU coordinates of the sources            ##
###############################################################################

## Check if source coordinates or the coordinate transformation should be
## fitted
    @property
    def Fit(self):
        """
        Returns
        -------
        Fit : bool
            Flag indicating whether any of the parameters that deal with the
            source coordinates (incl. coord. transf.) are free fit parameters. 
        """
        if self.hasPolyFit[self.status<4].all():
            return False
        else:
            if self.isFreePar.any():
                return True
            elif self.verifyTrans:                
                return super(Sources, self).Fit
            else:
                return False



    def loadPositionsHDU(self, hdu):
        """
        Load a pyfits 'PrimaryHDU'- or 'ImageHDU'-instance that contains IFU
        coordinates for the sources in the catalog (or a subset of them). The
        HDU must contain a 3dim. data-cube with shape 2 x Nsources x Ndisp.

        If only a subset of all sources in the catalog have IFU coordinates
        stored in th HDU, the hierarch keywords 'PAMPELMUSE POSITIONS ID<i>'
        should be present, giving the ID of the source stored in row i.

        The method will automatically check whether the HDU contains data or
        polynomial fits, using hierarch-keyword 'PAMPELMUSE POSITIONS POLY'
        (or keyword 'ISFITTED').

        Also, the method will automatically check whether provided IFU
        coordinates are free parameters in the fit using the hierarch-keyword
        'PAMPELMUSE POSITIONS FULLFIT'
        
        Note that in order to save memory, the method will not actually load
        the data into memory. It will only provide the method
        'loadIFUCoords' with the necessary information to do so.

        Parameters
        ----------
        hdu : a pyfits 'PrimaryHDU'- or 'ImageHDU'-instance
            The HDU instance containing the coordinates.
        """

        if "EXTNAME" in hdu.header:
            logging.info("Reading IFU coords. from HDU `{0}`.".format(\
                    hdu.header["EXTNAME"]))
        else:
            logging.info("Reading IFU coords. from unnamed HDU.")

        ## check for which sources the HDU contains IFU coordinates, using
        ## header hierarch keywords 'PAMPELMUSE POSITIONS ID{i}'. If the
        ## keywords do not exist, the routine expects that coords are provided
        ## for every resolved source in the catalogue.
        ifuID = np.zeros((hdu.header["NAXIS2"],), dtype=np.int32)
        for i in xrange(hdu.header["NAXIS2"]):
            try:
                ID = hdu.header["PAMPELMUSE POSITIONS ID{0}".format(i)]
                if not ID in self.ID:
                    logging.error("Received coords for unknown ID: #{0}".format(ID))
                assert ID in self.ID, "Received coords for unknown ID: #{0}".format(ID)
                ifuID[i] = ID
            except KeyError:
                if hdu.header["NAXIS2"] != (self.nsrc-self.hasSky):
                    logging.error("Cannot infer IDs for provided coords.")
                assert hdu.header["NAXIS2"] == (self.nsrc - self.hasSky), \
                    "Shape of data in HDU not compatible with source catalogue."
                ifuID = self.ID[self.status<4]
                break

        ## NOT SUPPORTED ANYMORE!!!
        ## The following is to check whether the opened HDU contains a data
        ## structure used by early versions of PampelMuse where the division
        ## into x- & y-coords happened on the last axis 
##         if hdu.header["NAXIS1"] in [self.nsrc, self.nsrc+1]\
##                 and hdu.header["NAXIS2"] not in [self.nsrc, self.nsrc+1]:
##             logging.warning("Opened FITS-HDU has an outdated structure.")
##             logging.warning("Will try to update and use as polyfit coords.")
## 
##             if hdu.header["NAXIS1"] == self.nsrc:
##                 tmpHDU = pyfits.ImageHDU(np.rollaxis(hdu.data, 1, 3))
##             else:
##                 tmpHDU = pyfits.imageHDU(np.rollaxis(hdu.data[...,:-1],1,3))
## 
##             tmpHDU.writeto("tmpPos.fits", clobber=True)
##             self.polyFitHDU = pyfits.open("tmpPos.fits")[1]
##             self.polyFitID  = np.copy(hduID)
## 
##         else:

        ## check if HDU contains polynomial fits or measured IFU coordinates
        try: poly = hdu.header["PAMPELMUSE POSITIONS POLY"]
        except:
            try: poly = hdu.header["ISFITTED"]
            except: poly = False

        ## check if measured IFU coordinates are free parameters in the fit
        try: free = hdu.header["PAMPELMUSE POSITIONS FULLFIT"]
        except: free = False

        ## initialize properties used by the method 'loadIFUCoords'
        if not poly:
            self.dataHDU = hdu
            self.dataID  = np.copy(ifuID)
            self.isFreePar[matchOnID(self.ID, ifuID)] = free
        else:
            self.polyFitHDU = hdu
            self.polyFitID  = np.copy(ifuID)
            self.hasPolyFit[matchOnID(self.ID, ifuID)] = True



    def loadIFUCoords(self, data=None, ids=None, index=slice(None),
                      free=False):
        """
        Provide the IFU coordinates for an individual source, a subset of
        sources or the entire catalogue. The coordinates might be loaded for
        the entire cube, an individual layer or more layers. Note however
        that the latter option is currently limited to slices (see python
        documentation for 'slice').

        The method tries to obtain the IFU coordinates in different ways (and
        in the following order):
        (1) Directly, if `data` is not None
        (2) Otherwise...
         (2a) reading the coordinates from a previously provided (using the
              method 'loadPositionsHDU') FITS-ImageHDU.
         (2b) using the coordinate transformation if a valid one is available.
         (2c) directly use the reference coordinates (i.e. assuming the coord.
              transformation is A=1, B=C=D=0)
        Additionally, the method check whether polynomial fits to the IFU
        coordinates exists as a FITS-ImageHDU.

        The routine will prepare two 3dim. arrays, named 'xdata' and 'ydata',
        containing the x- and y-coordinates. The shape of the arrays will be
        (2, <no. of IDs>, <no. of layers>). The coordinates will be stored in
        the first half of the arrays, the second one will contain the
        polynomial fits.

        Parameters
        ----------
        data : array_like, shape=(2, len('ids'), len('index')), optional
            The IFU coordinates. If none are provided, the routine will try
            to use a ImageHDU-instance or a valid coordinate transformation.
        ids : array_like or integer, optional
            The ID(s) of the sources for which the IFU coordinates should be
            loaded. If none are provided, the method will use the entire
            source catalogue.
        index : integer or python 'slice' instance, optional
            The index/indices of the layer(s) for which the IFU coordinates
            should be loaded. IMPORTANT: index-lists of boolean-lists are not
            currently supported!
        free : bool, optional
            Whether the provided IFU coordinates are free fit parameters. Only
            used if 'data' is provided.
        """

        ## check if the requested layer selection is valid 
        if not isinstance(index, slice):
            if np.asarray(index).shape == (): index = slice(index, index+1, 1)
            else: logging.error("'index' is {0} but must be slice or integer.".format(type(index)))

        ## get number of layers (if possible)
        if index.step is not None: nlayer = index.step
        elif self.ndisp != None: nlayer = self.ndisp
        else:
            if data == None: logging.error("Cannot infer length of spectral axis.")
            assert data != None, "Cannot infer length of spectral axis."
            nlayer = None

        ## check the IDs for which the data should be loaded
        if ids==None:
            ids = self.ID[self.status<4]
        else:
            ids = np.asarray(ids)

        ## check if all provided IDs are actually in the source catalogue
        match = matchOnID(self.ID, ids)
        if match.size < len(ids):
            logging.error("Invalid source IDs have been provided.")
        assert match.size == len(ids), "Some provided IDs have no counterpart in source catalogue."
        nsrc = len(ids)

        ## bring the provided IDs in the correct order wrt the source catalogue
        ## (only if needed):
        if (np.diff(match)<0).any():
            sorter = np.unique(match, return_index=True)
            ## 'sorter' contains 2 arrays, one with the indices of the matched
            ## IDs in the proper order and one with the indices of those IDs
            ## in the provided list of 'ids'.

            ids = self.ID[sorter[0]]
        else:
            sorter = None

        ## update boolean arrays that keep track of which sources have IFU
        ## coordinates and for which sources these coordinates are free
        ## parameters in the fit
        self.loadedIFUCoords[:] = False
        self.loadedIFUCoords[match] = True

        if data != None:

            self.isFreePar[:] = False
            self.isFreePar[match] = free

            ## check if data array has a valid shape
            ## - first axis must have length 2 (x- & y-axis)
            if data.shape[0] != 2:
                logging.error("Provided IFU coords array has invalid shape.")
            assert data.shape[0] == 2, "First axis of array must be x-/y-coords."
            
            ## - number of sources
            if data.shape[1] != nsrc:
                logging.error("Cannot match provided IFU coords data to source IDs.")
            assert data.shape[1] == nsrc, "Lengths of IFU coords data and array of source IDs differ."

            ## - number of layers
            if data.ndim == 2:
                if nlayer == 1: pass
                elif nlayer == None: nlayer = 1
                else: logging.error("Shape of IFU coords array does not match no of layers.")
            elif data.ndim == 3:
                if nlayer  == None: nlayer = data.shape[2]
                elif data.shape[2] != nlayer: logging.error("IFU coords array does not match no. of layers.")

            assert data.ndim in [2,3], "IFU coords array must be 3D if more than 1 layer loaded."
            if data.ndim == 3: assert data.shape[2] == nlayer, "3rd axis of IFU coords array must match no. of layers."

            ## make sure coordinates are stored in proper order & fill arrays
            if sorter == None:
                self.xdata = np.resize(data[0], (2, nsrc, nlayer))
                self.ydata = np.resize(data[1], (2, nsrc, nlayer))
            else:
                self.xdata = np.resize(data[0,sorter[1]], (2, nsrc, nlayer))
                self.ydata = np.resize(data[1,sorter[1]], (2, nsrc, nlayer))
        else:
            ## prepare arrays that will contain data
            self.xdata = np.zeros((2, nsrc, nlayer), dtype=np.float32)
            self.ydata = np.zeros((2, nsrc, nlayer), dtype=np.float32)


            if "dataHDU" in self.__dict__:
                
                ## keep track of sources for which coords have been calculated
                processed = np.in1d(ids, self.dataID)
                
                i=0
                for j in matchOnID(self.dataID, ids[processed]):
                    self.xdata[0,i] = self.dataHDU.section[0,j,index]
                    self.ydata[0,i] = self.dataHDU.section[1,j,index]
                    i += 1
            else:
                processed = np.zeros((len(ids),), dtype=np.bool)

            if (processed==False).any():
                xin = self.x[self.loadedIFUCoords][processed==False]
                yin = self.y[self.loadedIFUCoords][processed==False]
                if self.verifyTrans:
                    xNew, yNew = self.__call__(x=xin, y=yin, index=index)
                    self.xdata[0,processed==False] = xNew
                    self.ydata[0,processed==False] = yNew

                else:
                    self.xdata[0, processed==False] = xin
                    self.ydata[0, processed==False] = yin

            if "polyFitHDU" in self.__dict__:
            
                inPolyFitHDU = np.in1d(ids, self.polyFitID)

                for j in matchOnID(self.polyFitID, ids[inPolyFitHDU]):
                    i = np.where(ids==self.polyFitID[j])[0][0]

                    self.xdata[1,i] = self.polyFitHDU.section[0,j,index]
                    self.ydata[1,i] = self.polyFitHDU.section[1,j,index]


    def calcTrans(self, index=slice(None)):
        """
        Calculate/Initialize a new coordinate transformation using the input
        coordinates and the IFU coordinates. For this method to work, the
        properties 'x' and 'y' must have been defined and the method
        'loadIFUCoordinates' must have been called previously. The method will
        then use those two sets of input-coordinates to create a new instance
        of the 'CoordinateTransformation'-superclass.

        Parameters
        ----------
        index : int or 'slice'-instance, optional
            The layer(s) for which the coordinate transformation should be
            calcluated. The default is to use the entire cube.
        """
        if self.x == None or self.y == None:
            logging.error("Cannot determine coordinate transformation:")
            logging.error("   No reference coordinates available.")

        if self.xdata == None or self.ydata == None:
            logging.error("Cannot determine coordinate transformation:")
            logging.error("   No IFU coordinates available.")

        if self.loadedIFUCoords.sum() < 2:
            logging.error("Cannot determine coordinate transformation:")
            logging.error(\
                "   Too few IFU sources ({0}<2).".format(\
                    self.loadedIFUCoords.sum()))

        assert self.x is not None, "Reference coordinates missing."
        assert self.y is not None, "Reference coordinates missing."
        assert self.xdata is not None, "IFU coordinates missing."
        assert self.ydata is not None, "IFU coordinates missing."
        assert self.loadedIFUCoords.sum() >= 2, \
            "Need at least 2 sources to determine transformation." 

        coordTrans = self.quickFit(\
            [self.x[self.loadedIFUCoords], self.y[self.loadedIFUCoords]],
            [self.xdata[0], self.ydata[0]],
            outSigma=None)[0]

        self.makeNewTrans()

        for i in xrange(self.npars):
            self.addParameter(name=self.parameters[i], value=coordTrans[i])


        ## Only logging from this line
        logging.info("Determined coord. transf. for {0} layer(s).".format(\
                self.xdata.shape[-1]))

        logging.info("Best-fit parameters:")
        for i in xrange(len(self.parameters)):
            name = self.parameters[i]
            logging.info(\
                "   {0} = {1:8.3f}".format(name, self.ctData[0,i].mean()))

        logging.info("Offsets of individual sources:")
        logging.info("       ID     dx     dy")
        [x, y] = self.__call__(x=self.x[self.loadedIFUCoords],
                               y=self.y[self.loadedIFUCoords])
        j = 0
        for i in np.flatnonzero(self.loadedIFUCoords):
            logging.info("   {0:6d}  {1:+5.2f}  {2:+5.2f}".format(\
                    int(self.ID[i]),
                    float((x[j]-self.xdata[0,j]).mean(axis=-1)),
                    float((y[j]-self.ydata[0,j]).mean(axis=-1))))
            j += 1


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
        posSel = np.in1d(self.status, np.arange(minStatus,4))

        return QtCore.QStringList(self.ID[posSel].astype("|S6"))

    

    def getIFUCoords(self, coord, ID):
        """
        Return the x- or y-coordinate of an individual source. If a polynomial
        fit for this source exists, it is also returned.
        Note that the method can only be used after 'loadIFUCoords' has been
        called (this is checked at the beginning).

        Parameters
        ----------
        coord : string
            The coordinate to be returned. Can be either 'x' or 'y'
        ID : int
            The ID of the source for which the coordinates should be returned.

        Returns
        -------
        data : ndarray
            The requested IFU coordinate and its polynomial fit (if it is
            available.)
        """
        ## check if IFU coords have been loaded
        if self.xdata == None or self.ydata == None:
            logging.error("Cannot continue as no IFU coords have been loaded.")
            logging.error("Please call 'loadIFUCoords' first.")
        assert self.xdata != None and self.xdata != None, \
            "Please call 'loadIFUCoords' first."

        ## check if 'coord' is valid
        if not coord in ["x", "y"]: 
            logging.error("Unknown coordinate selection: '{0}'".format(coord))
        assert coord in ["x", "y"], "Coordinate must be either 'x' or 'y'."

        ## check if provided ID is valid
        if not ID in self.ID:
            logging.error(\
                "Cannot return IFU coords - unknown ID: #{0}".format(ID))
        assert ID in self.ID, "Unknown ID provided: #{0}".format(ID)
        
        ## get index of source
        j = np.where(self.ID==ID)[0][0]

        ## check if it has IFU coords available.
        if not self.loadedIFUCoords[j]:
            logging.error("No IFU coords loaded for source #{0}.".format(ID))
        assert self.loadedIFUCoords[j], "IFU coords not available.".format(ID)

        ## get index of source in data arrays and return data
        jj = np.sum(self.loadedIFUCoords[:(j+1)])-1
        if not self.hasPolyFit[j]:
            if coord == "x": return self.xdata[0,jj], None
            else: return self.ydata[0,jj], None
        else:
            if coord == "x": return self.xdata[:,jj]
            else: return self.ydata[:,jj]


    def fitIFUCoords(self, ids, coord, order, bins=1, mask=None,
                     function="step", commonSlope=False, statusBar=None):
        """
        Method to fit the source positions as a function of wavelength. Two
        basic modes are supported currently:
        (a) fit a polynomial individually to each position (still the default)
        (b) find a polynomial fit for all resolved sources and shift it along
            the selected axis to obtain a position for each source
            (recommended)
        Note that the method can only be used after 'loadIFUCoords' has been
        called (this is checked at the beginning).

        Parameters
        ----------
        ids : array_like or None
            The ID[s] of the sources that should be fitted. Use None to fit
            all sources for which IFU coordinates have been loaded.
        order : int
            The order of the polynomial used to perform the fit
        bins : int, optional
            The number of bins in case a step-polynomial is used as fitting
            function
        mask : array_like or None, optional
            An optional bad-pixel mask. If specified, must have as many pixels
            as spectral pixels are available. Good pixels should be set to
            True, bad ones to False.
        function : string, optional
            The fitting function that should be used. Currently a
            step-polynomial (step) or a Chebyshev polynomial (chebyshev) are
            supported.
        commonSlope : boolean, optional
            Flag indicating whether a common polynomial should be used to model
            the wavelength dependency of all sources.

        Notes
        -----
        In versions <0.5, it  had to be specified which coordinate (x or y)
        should be fitted. This option was removed in v0.5 and the fit is now
        performed automatically for both coordinates.
        """
        
##         ## check the coordinate that should be fitted
##         if not coord in ["x", "y"]: return
##         else: i = ["x", "y"].index(coord)
        
        ## check if IFU coords have been loaded
        if self.xdata == None or self.ydata == None:
            logging.error("Cannot continue as no IFU coords have been loaded.")
            logging.error("Please call 'loadIFUCoords' first.")
        assert self.xdata != None and self.xdata != None, \
            "Please call 'loadIFUCoords' first."

        ## check if fit function is valid:
        available = ["step",  "plain",  "chebyshev"]
        if not function.lower() in available:
            logging.error("Unknown fitting fucntion provided: {0}".format(\
                    function))
            assert function.lower() in available, "Unknown fitting function."
        else:
            function = function.lower()

        ## find the indices of the source(s) to be fitted
        if ids==None:
            indices2Fit = np.arange(self.loadedIFUCoords.sum(), dtype=np.int32)
        else:
            indices2Fit = matchOnID(\
                self.ID[self.loadedIFUCoords], np.atleast_1d(ids))

        ##  prepare wavelength coordinate
        wCoord = np.arange(self.ndisp, dtype=np.float32)

        ## create pseudo-mask if None provided
        if mask == None:
            mask = np.ones((self.ndisp,), dtype=np.bool)
    
        ## determine 'mean' coordinate if option 'commonSlope' selected
        if commonSlope:

            ## sources used for creating mean slope are those with status 2 or 3
            used = ((self.status>1)&(self.status<4))[self.loadedIFUCoords]

            ## if "instrument" is known, then only use sources within FoV to
            ## determine common slope.
            if self.instrument!= None:
                used = used&self.instrument.infov(\
                    self.xdata[0,:].mean(axis=-1),self.ydata[0,:].mean(axis=-1))

##             if coord == "x": 
            meanXCoord = np.mean(self.xdata[0, used], axis=0)                
##             else:
            meanYCoord = np.mean(self.ydata[0, used], axis=0)

            if function in ["step", "plain"]:
                meanXCoordFit = StepPolyFit(meanXCoord, order=order, bins=bins,
                                            mask=mask)
                meanYCoordFit = StepPolyFit(meanYCoord, order=order, bins=bins,
                                            mask=mask)

            elif function == "chebyshev":
                cxfit = Chebyshev.fit(wCoord[mask], meanXCoord[mask], order)
                meanXCoordFit = cxfit(wCoord)
                cyfit = Chebyshev.fit(wCoord[mask], meanYCoord[mask], order)
                meanYCoordFit = cyfit(wCoord)


        ## loop over selected sources and get their polynomial coordinate fits
        k = 1
        for j in indices2Fit:
##             if coord == "x":
            if commonSlope:
                xoffset = np.polyfit(\
                    wCoord[mask],(self.xdata[0,j]-meanXCoordFit)[mask],deg=0)
                self.xdata[1,j] = meanXCoordFit + np.polyval(xoffset, wCoord)
                yoffset = np.polyfit(\
                    wCoord[mask],(self.ydata[0,j]-meanYCoordFit)[mask],deg=0)
                self.ydata[1,j] = meanYCoordFit + np.polyval(yoffset, wCoord)

            elif function in ["step", "plain"]:
                self.xdata[1,j] = StepPolyFit(\
                    self.xdata[0,j], order=order, bins=bins, mask=mask)
                self.ydata[1,j] = StepPolyFit(\
                    self.ydata[0,j], order=order, bins=bins, mask=mask)

            elif function == "chebyshev":
                cxfit =Chebyshev.fit(wCoord[mask],self.xdata[0,j,mask],order)
                self.xdata[1,j] = cxfit(wCoord)
                cyfit =Chebyshev.fit(wCoord[mask],self.ydata[0,j,mask],order)
                self.ydata[1,j] = cyfit(wCoord)
                    
            ## find index of source in (full) source catalogue
            jj = np.where(np.cumsum(self.loadedIFUCoords)==(j+1))[0][0]
            self.hasPolyFit[jj] = True
            
            logging.debug("Polyn. fit to source #{0} [{1}/{2}]".format(\
                    self.ID[jj], k, len(indices2Fit)))
            if statusBar != None:
                statusBar.showMessage("Polyn. fit to source #{0} [{1}/{2}]".format(self.ID[jj], k, len(indices2Fit)), 5000)
            k += 1


    
################################################################################
##              Source selection methods/properties                           ##
################################################################################

    def distToBrighter(self):
        logging.info("Determining distance to nearest brighter source.")

        x   = self.xdata[0].mean(axis=-1)
        y   = self.ydata[0].mean(axis=-1)
        mag = self._mag[self.loadedIFUCoords]

        mindist = np.zeros(x.shape, dtype=np.float32)
        nearsrc = np.zeros(x.shape, dtype=np.int32)

        for i in xrange(x.size):
            dist_i =np.sqrt((x-x[i])**2 + (y-y[i])**2)
            dist_i[(mag-mag[i])>=0] = 99.

            mindist[i] = dist_i.min()
            nearsrc[i] = self.ID[self.loadedIFUCoords][dist_i.argmin()]

        return (mindist, nearsrc)

##         dist = np.sqrt((x-x[:,np.newaxis])**2 + (y-y[:,np.newaxis])**2)
##         dist[(mag-mag[:,np.newaxis])<=0] = 99.
## 
##         return (dist.min(axis=0), self.ifuID[dist.argmin(axis=0)])



    def shoulderDrop(self, r):
        ## see: http://en.wikipedia.org/wiki/Topographic_prominence
        ## (slightly modified)
        logging.info("Determining shoulder drop of sources.")

        x   = self.xdata[0].mean(axis=-1)
        y   = self.ydata[0].mean(axis=-1)
        mag = self._mag[self.loadedIFUCoords]

        dist = np.sqrt((x-x[:,np.newaxis])**2 + (y-y[:,np.newaxis])**2)
        np.fill_diagonal(dist, 99.)

        offset = mag[:,np.newaxis]-mag
        offset[dist>r] = 99.

        return 10**(-0.4*offset.min(axis=0))


################################################################################
##                             Output methods                                 ##
################################################################################
    def toRec(self, ids=None):
        
        if ids == None:
            select = np.arange(self.nsrc, dtype=np.int32)
        else:
            select = matchOnID(self.ID, np.asarray(ids))

        names = ["ID"]
        for attribute in ["x","y","mag","status","primary","specrow"]:
            value = getattr(self, "_{0}".format(attribute))
            if value != None: names.append(attribute)
        return np.rec.fromarrays(\
            [getattr(self, name)[select] for name in names], names=names)



    def makeSourceHDU(self):
        logging.info("Saving source catalog in FITS BinTableHDU.")

        try: select = np.nonzero(self._status>=0)[0]
        except ValueError: select = np.arange(self.nsrc, dtype=np.int32)

        ## sort by increasing magnitudes (if possible)
        if self._mag != None:
            select = select[self._mag[select].argsort()]

        outCols = [pyfits.Column(name="id", format="J", array=self.ID[select])]

        formats = {"x":"E", "y":"E", "mag":"E", "status":"I", "specrow":"I",
                   "primary":"J"}
        for attribute in ["x", "y", "mag", "status", "primary", "specrow"]:
            if self.__dict__["_{0}".format(attribute)] != None:
                outCols.append(pyfits.Column(\
                        name=attribute, format=formats[attribute],
                        array=self.__dict__["_{0}".format(attribute)][select]))


        return pyfits.new_table(outCols, header=pyfits.Header(\
                cards=[pyfits.Card(key="EXTNAME", value="SOURCES")]))



    def makeSpecHDU(self, makeSigma=True):
        logging.info("Saving spectra of components in FITS ImageHDU.")

        outHDU = pyfits.ImageHDU(self.data, name="SPECTRA")
        outHDU.header.update("CRPIX1", self.crpix)
        outHDU.header.update("CRVAL1", self.crval)
        outHDU.header.update("CDELT1", self.cdelt)
        outHDU.header.update("CTYPE1", self.ctype)
        outHDU.header.update("CUNIT1", self.cunit)

        if makeSigma:
            if self.sigma != None:
                sigmaHDU = pyfits.ImageHDU(self.sigma, name="SIGMASPC")
                sigmaHDU.header.update("CRPIX1", self.crpix)
                sigmaHDU.header.update("CRVAL1", self.crval)
                sigmaHDU.header.update("CDELT1", self.cdelt)
                sigmaHDU.header.update("CTYPE1", self.ctype)
                sigmaHDU.header.update("CUNIT1", self.cunit)
                return outHDU, sigmaHDU
            else:
                logging.error("Cannot create Sigma HDU: No sigma data")
                return outHDU, None
        else:
            return outHDU



    def makeCoordinatesHDU(self, polyFit=False, name="POSITIONS", **kwargs):

        if not polyFit:
            toStore = self.isFreePar[self.loadedIFUCoords]
            i=0
        else:
            toStore = self.hasPolyFit[self.loadedIFUCoords]
            i=1
        if not toStore.any():
            logging.error("Cannot store source coordinates to HDU. No data available.")
            return None

        outHDU = pyfits.ImageHDU(np.rollaxis(\
                np.dstack((self.xdata[i,toStore], self.ydata[i,toStore])),2,0),
                                 name=name)
        outHDU.header.update("hierarch PAMPELMUSE POSITIONS POLY", polyFit)
        outHDU.header.update("hierarch PAMPELMUSE POSITIONS FULLFIT", (polyFit==False))
            
        ## create entry in header about ID of each source and the row containing its
        ## position
        k = 0
        for j in xrange(toStore.size):
            if toStore[j]:
                outHDU.header.update(\
                    "hierarch PAMPELMUSE POSITIONS ID{0}".format(k), self.ID[self.loadedIFUCoords][j])
                k += 1

        return outHDU
