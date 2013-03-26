"""
SOURCES
=======
The `Sources`-class is the PampelMuse-routine that handles the entire source
catalogue. It keeps track of the loaded sources, their grouping into the
individual components, their spectra and postions. It also provides some
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
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.4.0.0"
import logging
import numpy as np, pyfits
from numpy.polynomial import Chebyshev, Legendre

try: from PyQt4 import QtCore ## check if PyQt is available 
except ImportError: pass

## required PampelMuse-packages
from pampelmuse.core.positions import CoordinateTransformation
from pampelmuse.functions.step_poly_fit import StepPolyFit
from pampelmuse.core.value_array import ValueArray
from pampelmuse.core.instruments import Argus, Pmas


def matchOnID(parent, child):
    """
    Compare to 1dim. lists for common entries. For each element in `children`
    that is also part of the `parent`-list, the index of in the `parent'-list
    is returned.

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
    return np.nonzero((child[:,np.newaxis]-parent)==0)[1]




class Sources(CoordinateTransformation, object):
    """
    The `Sources`-class is designed for the following purposes:
    * Initialize and maintain a coherent catalog of sources. To this aim, the
      class uses several properties that are either inherent to the sources
      (ID, x, y, mag) or assigned to each source depending on its usage in
      the analysis (status, specrow, primary)
    * Load, maintain and save the set of spectra that is associated with the
      source catalog
    * Load, maintain and save the set of IFU coordinates that is associated
      with the source catalog. To this aim, the `Sources`-class uses the
      `CoordinateTransformation`-class of PampelMuse as a superclass. This
      way, it can also access any coordinate transformation that might be in
      use.
    """
    def __init__(self, data=None, hdu=False, ids=None, instrument=None):
        """
        Initialize a new instance of the `Sources`-class.

        Parameters
        ----------
        data : a pyfits BinTable-HDU instance or a structured array, optional
            The source catalog that should be used to initialize the instance.
            It must contain at least a field named `id`, all other fields are
            optional and the corresponding properties might be defined later.
            If `data` is not provided, than `ids` must be given.
        hdu : boolean, optional
            Flag indicating whether `data` is a pyfits BinTable-HDU instance 
            (if `data` is provided).
        ids : array_like
            A 1dim. array containing the source IDs that should be used to
            initialize the scene. All other properties must be defined
            afterwards.
        instrument : instance of pampelmuse.core.instruments.Instrument-class
                     or None, optional
            If a class-instance is provided, it will be used to identify bad
            spaxels, determine the exact shape of the FoV and the location of
            sources within or outside the FoV.
            E.g., if FLAMES/ARGUS is used, the missing edges and fibres
            outside the CCD will be correctly identified. If instrument=None,
            the routine will work with a rectangular FoV of size 22x14.


        Returns
        -------
        The newly created instance.
        """
        if hdu:
            logging.info("Initializing Sources instance from HDU.")
            data = np.array(data.data, dtype=data.data.dtype)
        else:
            logging.info("Initializing Sources instance.")

        ## Initialize properties:
        ## Note that the  properties below can be accessed in two ways:
        ## a) with an underscore: direct access, no sanity checks performed
        ## b) without underscore: check whether requested property has been
        ##                        defined and whether data used to update
        ##                        property is valid.
        if ids != None: self._id = np.asarray(ids)
        else: self._id = None  ## source IDs -note: prop. w\ underscore is `ID`
        self._x        = None  ## input x coordinates
        self._y        = None  ## input y coordinates
        self._mag      = None  ## broadband magnitudes
        self._status   = None  ## status of source in analysis [-1, 4]
        self._specrow  = None  ## row containing spectrum in `data`-property
        self._primary  = None  ## primary component of the source
        self._indexmap = None  ## map from specrow to index
        self._weights  = None  ## map from specrow to relative magnitude

        self.values    = None  ## for later usage of IFU coordinates

        ## IDs must be present
        if data != None:
            try:
                self._id = data[["id"]].view(data["id"].dtype).reshape(-1,)
            except ValueError:
                pass

        if self._id == None:
            logging.critical("No field `ID' found in provided data.")

        self.nsrc = self._id.size

        ## some properties are optional
        if data != None:
            for name in ["x", "y", "mag"]:
                if name in data.dtype.names:
                    self.__dict__["_{0}".format(name)]\
                        = data[[name]].view(data[name].dtype).reshape(-1,)

            ## include `specrow`, `status`, `primary` if available
            if "specrow" in data.dtype.names: self.setSpecrow(data["specrow"])
            if "status" in data.dtype.names: self.setStatus(data["status"])
            if "primary" in data.dtype.names: self.setPrimary(data["primary"])


        ## "Initialize" instrument
        if isinstance(instrument, (Argus, Pmas)):
            self.instrument = instrument
            logging.info("Selected instrument is '{0}'.".format(\
                    instrument.name))
        elif isinstance(instrument, str):
            if instrument == "pmas":
                logging.info("Selected instrument is PMAS.")
                self.instrument = Pmas()
            elif instrument == "argus":
                logging.info("Selected instrument is FLAMES/ARGUS.")
                self.instrument = Argus()
        else:
            logging.warning("No valid instrument specifier provided.")
            self.instrument = Instrument()

##         if self.instrument is None:
##             logging.warning("No information on used instrument avaialable.")


        super(Sources, self).__init__()


###############################################################################
##                   Definitions of the various properties,                  ##
##                   and the methods to create & acces them                  ##
###############################################################################

## __getitem__- and __setitem__-methods for the various properties:
    def getID(self):
        return self._id



    def setX(self, value):
        value = np.asarray(value, dtype=np.float32)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `x`, data has invalid shape.")
        else:
            self._x = value

    def getX(self):
        if self._x == None:
            logging.error("Requested attribute `x` not specified.")
        return self._x
            


    def setY(self, value):
        value = np.asarray(value, dtype=np.float32)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `y`, data has invalid shape.")
        else:
            self._y = value

    def getY(self):
        if self._y == None:
            logging.error("Requested attribute `y` not specified.")
        return self._y


    
    def setStatus(self, value):
        value = np.asarray(value, dtype=np.int16)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `y`, data has invalid shape.")
        else:
            self._status = value
            if self._specrow==None:
                if not (self._status<2).any(): self.makeSpecrow(unique=True)
                elif self._primary != None: self.makeSpecrow(unique=False)
                    
    def getStatus(self):
        if self._status == None:
            logging.error("Requested attribute `status` not specified.")
        return self._status



    def setPrimary(self, value):
        value = np.asarray(value, dtype=np.int32)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `primary`, data has invalid shape.")
        else:
            self._primary = value
            if self._status != None and self._specrow==None:
                self.makeSpecrow(unique=False)

    def getPrimary(self):
        if self._primary == None:
            logging.error("Requested attribute `primary` not specified.")
        return self._primary



    def setMag(self, value):
        value = np.asarray(value, dtype=np.float32)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `mag`, data has invalid shape.")
        else:
            self._mag = value
            
    def getMag(self):
        if self._mag == None:
            logging.error("Requested attribute `mag` not specified.")
        return self._mag

            

    def setSpecrow(self, value):
        value = np.asarray(value, dtype=np.int16)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `specrow`, data has invalid shape.")
        elif self._specrow != None:
            logging.error("Cannot set `specrow` as it has been previously set.")
        else:
            self._specrow = value
            self.makeIndexmap() ## create dictionaries to link `specrow` to
            self.makeWeights()  ## source indices and magnitudes. 

            
    def getSpecrow(self):
        if self._specrow == None:
            logging.error("Requested attribute `specrow` not specified.")
        return self._specrow



    def getIndexmap(self):
        if self._indexmap == None:
            logging.error("Requested attribute `indexmap` not specified.")
        return self._indexmap



    def getWeights(self):
        if self._weights == None:
            logging.error("Requested attribute `weights` not specified.")
        return self._weights


            
    ## the properties:
    ID       = property(getID, None, None,
                        "The source IDs")
    x        = property(getX, setX, None,
                        "The source x-coordinates")
    y        = property(getY, setY, None,
                        "The source y-coordinates")
    mag      = property(getMag, setMag,  None,
                        "The source magnitudes")
    status   = property(getStatus, setStatus, None,
                        "The status of every source")
    specrow  = property(getSpecrow, setSpecrow, None,
                        "The row with the source spectrum")
    primary  = property(getPrimary, setPrimary, None,
                       "The primary source in the component.")
    indexmap = property(getIndexmap, None, None,
                        "Mapping from `specrow`-property to index.")
    weights  = property(\
        getWeights, None, None,
        "Relative magnitudes of the sources in their components.")


    def makeSpecrow(self, unique):
        """
        Method that is wrapped around the __setitem__-method of `specrow`.
        It is designed to automatically assign a `specrow`-entry to every
        source in the catalog.

        Parameters
        ----------
        unique : boolean
            Flag indicating whether all sources have their own spectrum (T)
            or whether some sources share a spectrum (F).
        """
        if unique:
            value = np.arange(self.nsrc, dtype=np.int16)
        else:
##             logging.debug("{0}".format(str(self._id)))
##             logging.debug("{0}".format(str(self._primary)))
            value = ((self._id==self._primary)&(self._status>-1)).cumsum()-1
            for i in xrange(self._id.size):
                if self._status[i] == 1: ## Find spectrum of primary component
                    j=i                  
                    ## Following loop necessary since initially assigned
                    ## primary component might be secondary component o
                    ## another source.
                    while self._primary[j]!=self._id[j]:

                        ## exclude source if primary source has status -1.
                        if not (self._id==self.primary[j]).any():
                            logging.error(\
                                "No `primary` counterpart for source #{0}"\
                                    .format(self._id[i]))
                            logging.error("Source #{0} excluded".format(\
                                    self._id[i]))
                            self.status[i] = -1
                            break

                        j = np.where(self._id==self._primary[j])[0][0]
                    value[i] = value[j]

            value[self._status==0] = value.max()+1
##             logging.debug("{0}".format(str(value)))
        self.setSpecrow(value)


        
    def makeIndexmap(self):
        ## `indexmap` is a dictionary mapping the `specrow` (keys) to the
        ## index/indices (values) in the `sources`-table
        self._indexmap = {}
        for i in xrange(self.ncmp):
            self._indexmap[i] = np.where(self._specrow==i)[0]
##         logging.debug("{0}".format(str(self._indexmap)))



    def makeWeights(self):
        """
        CHANGED in version 0.4 - 'weights' is 1dim. array of the same length
        as the whole sourcelist and contains the magnitude weight of each
        source in its component. The magnitude weight is defined as
        10**(-0.4*df), where df is the fraction of flux the source contributes
        to the component.

        OUTDATED: 'weights' is a dictionary mapping the `specrow` (keys) to
        the relative magnitudes (values) of its component(s).
        """
        if self._mag == None:
            logging.warning("Property `mag` not set, assuming equal weights.")

        self._weights = np.zeros((self.nsrc,), dtype=np.float64)
        
        for i in xrange(self.nsrc):
            same_specrow = (self._specrow==self._specrow[i])
            if self._mag != None:
                self._weights[i] = self._mag[i] + 2.5*np.log10(\
                    (10**(-0.4*self._mag[same_specrow])).sum())
            else:
                self._weights[i] = 2.5*np.log10(same_specrow.size)


##         self._weights = {}
##         for row, ind in self._indexmap.iteritems():
##             if self._mag != None:
##                 self._weights[row] = self._mag[ind] \
##                     + 2.5*np.log10((10**(-0.4*self._mag[ind])).sum())
##             else:
##                 n = len(ind)
##                 self._weights[row] \
##                     = np.asarray(n*[2.5*np.log10(n)], dtype=np.float32)


    @property
    def ncmp(self):
        """
        The number of components (i.e. independent spectra) in the catalogue.
        """
        if self._specrow == None: return None
        else: return self._specrow.max()+1


    @property
    def hasSky(self):
        """
        Flag indicating whether a sky component is included in the catalog.
        """
        if self._status == None: return False
        return (self._status==4).any()
    
    @property
    def skyRow(self):
        """
        Returns the row of the sky spectrum (if any, otherwise returns None)
        """
        if not self.hasSky or self._specrow==None: return None
        else: return self._specrow[np.where(self.status==4)[0][0]]

    
    @property
    def hasUnres(self):
        """
        Flag indicating whether a unresolved component is included in the
        catalog.
        """
        if self._status == None: return False
        return (self._status==0).any()

    @property
    def unresRow(self):
        """
        Returns the row of the spectrum of the unresolved component (if any,
        otherwise returns None)
        """
        if not self.hasUnres or self._specrow==None: return None
        else: return self._specrow[np.where(self.status==0)[0][0]]
            

    @property
    def psfIndices(self):
        """
        Returns the indices of the sources that are used to determine the
        PSF (i.e. sources whose `status` is 3).
        """
        return np.arange(self.nsrc, dtype=np.int32)[self.status==3]


    def info(self):
        """
        Print out some information about the loaded source catalogue.
        """
        try:
            logging.debug("Current structure of loaded Sources-instance:")
            logging.debug(" Component:   IDs (& mags):")

            for i in xrange(self.specrow.max()+1):
                logString = " {0:10d}".format(i)
   
                slc = np.where(self.specrow==i)[0]

                for j in xrange(slc.size):
                    logString += "   {0:6d} ({1:.2f})".format(\
                        int(self._id[slc[j]]),
                        float(abs(self._weights[slc[j]])))

                    if not np.mod(j+1, 2):
                        logging.debug(logString)
                        logString = 11*" "
                        
                if np.mod(j, 2)==0: logging.debug(logString)
                
             

##             for row, ind in self.indexmap.iteritems():
##                 logString = " {0:10d}".format(int(row))
##                 j = 0
##             
##                 for i in ind:
##                     logString += "   {0} ({1:.2f})".format(\
##                         int(self._id[i]), float(self._weights[row][j]))
##                     j +=1
##                     if not np.mod(j, 2):
##                         logging.debug(logString)
##                         logString = 11*" "
##                 if np.mod(j, 2)>0: logging.debug(logString)
        except AttributeError:
            logging.warning("Some source properties are not set.")
            return

        if self.verifyTrans:
            CoordinateTransformation.info(self)



###############################################################################
##              Methods dealing with the spectra of the sources              ##
###############################################################################
    def loadSpectra(self, data, sigma=None, hdu=False, crval=1., cdelt=1.,
                    crpix=1, ctype="AWAV"):
        """
        Load the spectra for all sources in the catalogue. The method might
        also be used to provide an empty array that is filled later.

        Properties
        ----------
        data : array_like or pyfits ImageHDU-instance, shape=(`ncmp`, k)
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
            found in the FITS-header of `data`.
        """
        if self.ncmp == None:
            logging.error("No assigment of sources to spectra performed.")
       
        self.crval = crval
        self.cdelt = cdelt
        self.crpix = crpix
        self.ctype = ctype

        if hdu:
            self.data = data.data
            if "CRVAL1" in data.header: self.crval = data.header["CRVAL1"]
            if "CDELT1" in data.header: self.cdelt = data.header["CDELT1"]
            if "CRPIX1" in data.header: self.crpix = data.header["CRPIX1"]
            if "CTYPE1" in data.header: self.ctype = data.header["CTYPE1"]
        else:
            self.data = data

        if self.data.shape[0] != self.ncmp:
            logging.error("Number of provided spectra and components differ.")

        if hdu and sigma != None:
            self.sigma = sigma.data
        else:
            self.sigma = sigma

        if self.sigma!=None:
            if self.sigma.shape != self.data.shape:
                logging.error("Spectra and sigma arrays have different shapes.")
        else:
            self.sigma = np.zeros(self.data.shape, dtype=np.float32)
            ## create empty `sigma`-spectra if none provided.
            
        self.ndisp = self.data.shape[1] ## length of dispersion axis

        

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
    @property
    def Fit(self):
        """
        Return True or False depending on whether any of the parameters that
        deal with the source coordinates (incl. coord. transf.) are free fit
        parameters. 
        """
        if "poly" in self.__dict__ and "free" in self.__dict__:
            if self.free.any() and not self.poly.all():
                return True
            elif self.poly.all():
                return False

        if self.verifyTrans:
            return super(Sources, self).Fit

        else:
            return True


    def loadIFUCoords(self, data=None, ids=None, index=slice(None),free=False):
        """
        Provide the IFU coordinates for an individual source, a subset of
        sources or the entire catalogue.
        The coordinates might be loaded for an individual image or more images.
        Note however that this `index` selection is currently limited by the
        fact that the `ValueArray` can only handle integers or `slice`-
        instances.
        The method tries to obtain the IFU coordinates in different ways (and
        in the following order):
        (1) Directly, if `data` is not None
        (2) Using the coord. transf., if a valid one is available.
        (3) Reading the coordinates from a previously provided (using the
            `loadPositionsHDU`-method) FITS-ImageHDU.
        Additionally, the method check whether polynomial fits to the IFU
        coordinates exists as a FITS-ImageHDU.

        The routine will prepare two instance of `ValueArray`, named `xvalues`
        and `yvalues` that will contain the coordinates.

        Parameters
        ----------
        data : array_like, shape=(2, len(`ids`), len(`index`)), optional
            The IFU coordinates. If none are provided, the routine will try to
            use a valid coord. transf. or ImageHDU-instance.
        ids : array_like, 1dim., optional
            The IDs of the sources for which the IFU coordinates should be
            loaded. If none are provided, the `method` will  try to use the
            entire source catalogue.
        index : integer or `slice`-instance, optional
            The index/indices of the images for which the IFU coordinates
            should  be loaded. IMPORTANT: index-lists of boolean-lists are not
            currently supported!
        free : bool, optional
            Whether the provided IFU coordinates are free fit parameters. Only
            used if `data` is not None.
        """


        if not isinstance(index, slice):
            if np.asarray(index).shape == (): index = slice(index, index+1, 1)
            else: logging.error("`index` must be slice or integer, not {0}"\
                                    .format(type(index)))

        if data != None:

            if ids==None:
                logging.warning("No IDs for IFU coordinates were provided.")
                self.ifuID = self._id[self._id>=0]
            else:
                self.ifuID = np.asarray(ids)

            self.ifuSelect = matchOnID(self._id, self.ifuID)
            if self.ifuSelect.size < self.ifuID.size:
                logging.error("Invalid source IDs have been provided.")
                return

            data = np.atleast_3d(data)
            if data.shape[0] != 2:
                logging.error("Provided `data` has invalid shape.")
                return
            if data.shape[1] != len(self.ifuID):
                logging.error("`data`-shape incompatible with no. of IDs.")
                return

            self.xdata = np.resize(data[0], (2,)+data[0].shape)
            self.ydata = np.resize(data[1], (2,)+data[1].shape)

            self.free = np.asarray(free).resize(data.shape[1])

        elif self.verifyTrans:

            if ids==None: self.ifuID = self._id[self._id>=0]
            else: self.ifuID = np.asarray(ids)

            self.ifuSelect = matchOnID(self._id, self.ifuID)
            if self.ifuSelect.size < self.ifuID.size:
                logging.error("Invalid source IDs have been provided.")
                return

            xNew, yNew = self.__call__(x=self._x[self.ifuSelect],
                                       y=self._y[self.ifuSelect],
                                       index=index)
        
            self.xdata = np.resize(xNew, (2,)+xNew.shape)
            self.ydata = np.resize(yNew, (2,)+yNew.shape)

            self.free = np.zeros(self.ifuID.shape, dtype=np.bool)

        elif "dataHDU" in self.__dict__ or "polyFitHDU" in self.__dict__:

            if "dataID" in self.__dict__: availableID = self.dataID
            else: availableID = self.polyFitID

            if ids==None:
                self.ifuID = availableID
                slc = None
            else:
                self.ifuID = np.atleast_1d(ids)
                slc = matchOnID(availableID, self.ifuID)
                if slc.size < self.ifuID.size:
                    logging.error("Requested ID(s) do not possess IFU coords.")
                    return

            self.ifuSelect = matchOnID(self._id, self.ifuID)
            if self.ifuSelect.size < self.ifuID.size:
                logging.error("Invalid source IDs have been provided.")
                return

            if index == slice(None):
                ndisp = self.ndisp
            elif isinstance(index, slice):
                ndisp = (index.stop-index.start+index.step-1)/index.step

            self.xdata = np.zeros((2, self.ifuID.size, ndisp), dtype=np.float32)
            self.ydata = np.zeros((2, self.ifuID.size, ndisp), dtype=np.float32)

            if "dataHDU" in self.__dict__:
                if slc == None:
                    self.xdata[0] = self.dataHDU.section[0,:,index]
                    self.ydata[0] = self.dataHDU.section[1,:,index]
                else:
                    for i in xrange(slc.size):
                        self.xdata[0,i] = self.dataHDU.section[0,slc[i],index]
                        self.ydata[0,i] = self.dataHDU.section[1,slc[i],index]

            self.free  = np.ones(self.ifuID.shape, dtype=np.bool)

        else:
            logging.error("Cannot load IFU coords. Information missing.")


        self.poly = np.zeros((2,self.ifuID.size), dtype=np.bool)
        if "polyFitHDU" in self.__dict__:
            
            slc = matchOnID(self.polyFitID, self.ifuID)
            if slc.size < self.ifuID.size:
                logging.warning("Some ID(s) do not possess polyFit coords.")
            put = np.in1d(self.ifuID, self.polyFitID, assume_unique=True)

            if put.all():
                self.xdata[1] = self.polyFitHDU.section[0,:,index]
                self.ydata[1] = self.polyFitHDU.section[1,:,index]
            else:
                for i in xrange(put.size):
                    if put[i]:
                        self.xdata[1,i] =self.polyFitHDU.section[0,slc[i],index]
                        self.ydata[1,i] =self.polyFitHDU.section[1,slc[i],index]

            self.poly[:,put] = True

        self.xvalues = ValueArray(self.xdata, free=self.free, poly=self.poly[0])
        self.yvalues = ValueArray(self.ydata, free=self.free, poly=self.poly[1])



    def calcTrans(self, index=slice(None)):
        """
        Calculate/Initialize a new coordinate transformation using the input
        coordinates and the IFU coordinates. For this method to work, the
        properties `x` and `y` must have been defined and the method
        `loadIFUCoordinates` must have been called previously. The method will
        then use those two sets of input-coordinates to create a new instance
        of the `CoordinateTransformation`-superclass.

        Parameters
        ----------
        index : int or `slice`-instance, optional
            The image(s) for which the coordinate transformation should be
            calcluated. The default is to use the entire cube.
        """
        if self._x == None or self._y == None:
            logging.error("Cannot determine coordinate transformation:")
            logging.error("   No input coordinates available.")
            return
        if "xdata" not in self.__dict__ or "ydata" not in self.__dict__:
            logging.error("Cannot determine coordinate transformation:")
            logging.error("   No IFU coordinates available.")
            return
        if len(self.ifuID) < 2:
            logging.error("Cannot determine coordinate transformation:")
            logging.error(\
                "   Too few IFU sources ({0}<2).".format(len(self.ifuID)))
            return
   
        coordTrans = self.quickFit(\
            [self._x[self.ifuSelect], self._y[self.ifuSelect]],
            [self.xdata[0,:,index], self.ydata[0,:,index]],
            outSigma=None)[0]

        self.makeNewTrans()

        for i in xrange(self.npars):
            self.addParameter(name=self.parameters[i], value=coordTrans[i])


        ## Only logging from this line
        logging.info("Determined coord. transf. for {0} image(s).".format(\
                self.xdata.shape[-1]))

        logging.info("Best-fit parameters:")
        for i in xrange(len(self.parameters)):
            name = self.parameters[i]
            logging.info(\
                "   {0} = {1:8.3f}".format(name, self.ctData[0,i].mean()))

        logging.info("Offsets of individual sources:")
        logging.info("       ID     dx     dy")
        [x, y] = self.__call__(x=self._x[self.ifuSelect],
                               y=self._y[self.ifuSelect])
        for i in xrange(len(self.ifuID)):

            logging.info("   {0:6d}  {1:+5.2f}  {2:+5.2f}".format(\
                    int(self.ifuID[i]),
                    float((x[i]-self.xdata[0,i]).mean(axis=-1)),
                    float((y[i]-self.ydata[0,i]).mean(axis=-1))))



    def loadPositionsHDU(self, name):
        """
        Load a pyfits `PrimaryHDU`- or `ImageHDU`-instance that contains IFU
        coordinates for the sources in the catalog (or a subset of them). The
        HDU must contain a 3dim. data-cube with shape 2 x Nsources x Ndisp.

        If only a subset of all sources in the catalog have IFU coordinates
        stored in th HDU, the hierarch keywords `PAMPELMUSE POSITIONS ID<i>`
        should be present, giving the ID of the source stored in row i.

        The method will automatically check whether the HDU contains data or
        polynomial fits, using hierarch-keyword `PAMPELMUSE POSITIONS POLY`
        (or keyword `ISFITTED`).
        
        Note that in order to save memory, the method will not actually load
        the data into memory. It will only provide the method
        `loadIFUCoordinates` with the necessary information to do so.

        Parameters
        ----------
        name : a pyfits `PrimaryHDU`- or `ImageHDU`-instance
            The HDU instance containing the coordinates.
        """
        ## need to provide nameID and polyFitID
        ## look for hierarch-keywords "PAMPELMUSE POSITIONS ID{i}"

        if "EXTNAME" in name.header:
            logging.info("Reading IFU coords. from HDU `{0}`.".format(\
                    name.header["EXTNAME"]))
        else:
            logging.info("Reading IFU coords. from unnamed HDU.")

        hduID = np.arange(name.header["NAXIS2"], dtype=np.int32)
        for i in xrange(hduID.size):
            if "PAMPELMUSE POSITIONS ID{0}".format(i) in name.header:
                hduID[i] = name.header["PAMPELMUSE POSITIONS ID{0}".format(i)]
            else:
                hduID = self._id[self._id>=0][:name.header["NAXIS2"]]
                break

        if name.header["NAXIS1"] in [self.nsrc, self.nsrc+1]\
                and name.header["NAXIS2"] not in [self.nsrc, self.nsrc+1]:
            logging.warning("Opened FITS-HDU has an outdated structure.")
            logging.warning("Will try to update and use as polyfit coords.")

            if name.header["NAXIS1"] == self.nsrc:
                tmpHDU = pyfits.ImageHDU(np.rollaxis(name.data, 1, 3))
            else:
                tmpHDU = pyfits.imageHDU(np.rollaxis(name.data[...,:-1], 1, 3))

            tmpHDU.writeto("tmpPos.fits", clobber=True)
            self.polyFitHDU = pyfits.open("tmpPos.fits")[1]
            self.polyFitID  = np.copy(hduID)

        else:
            try: poly = name.header["PAMPELMUSE POSITIONS POLY"]
            except:
                try: poly = name.header["ISFITTED"]
                except: poly = False

            if not poly:
                self.dataHDU = name
                self.dataID  = np.copy(hduID)
            else:
                self.polyFitHDU = name
                self.polyFitID  = np.copy(hduID)



    def providePosIDs(self, minStatus=1):

        posSel = np.in1d(self.status, np.arange(minStatus,4))

        return QtCore.QStringList(self.ID[posSel].astype("|S6"))

    

    def getIFUCoords(self, coord, ID):

        if not coord in ["x", "y"]: return None, None
        else: i = ["x", "y"].index(coord)

        if not ID in self._id:
            logging.error(\
                "Cannot return IFU coords - unknown ID: {0}".format(ID))
            return None, None
        else:
            j = matchOnID(self.ifuID, np.atleast_1d(ID))[0]

        if not self.poly[i,j]:
            if coord == "x": return self.xdata[0,j], None
            else: return self.ydata[0,j], None
        else:
            if coord == "x": return self.xdata[:,j]
            else: return self.ydata[:,j]
          


    def fitIFUCoords(self, ids, coord, order, bins=1, mask=None,
                     function="step", commonSlope=False):
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
            The ID[s] of the sources that should be fitted. Use None to fit all\
            sources.
        coord : either 'x' or 'y'
            The spatial coordinate for which the fit should be performed
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
        """
        
        ## check the coordinate that should be fitted
        if not coord in ["x", "y"]: return
        else: i = ["x", "y"].index(coord)
        
        ## find the indices of the source(s) to be fitted
        if ids==None: indices2Fit = np.arange(self.ifuID.size, dtype=np.int32)
        else: indices2Fit = matchOnID(self.ifuID, np.atleast_1d(ids))

        ##  prepare wavelength coordinate
        wCoord = np.arange(self.ndisp, dtype=np.float32)

        ## create pseudo-mask if None provided
        if mask == None:
            mask = np.ones((self.ndisp,), dtype=np.bool)

        ## determine 'mean' coordinate if option 'commonSlope' selected
        if commonSlope:

            ## sources used for creating mean slope are those with status 2 or 3
            used = (self.status[self.ifuSelect]>1)\
                &(self.status[self.ifuSelect]<4)

            ## if "instrument" is known, then only use pixels within FoV to
            ## determine common slope.
            if self.instrument!= None:
                used = used&self.instrument.infov(\
                    self.ydata[0,:].mean(axis=-1),self.xdata[0,:].mean(axis=-1))
           
            if coord == "x": 
                meanCoord = np.mean(self.xdata[0, used], axis=0)                
            else:
                meanCoord = np.mean(self.ydata[0, used], axis=0)

            if function == "step":
                meanCoordFit = StepPolyFit(meanCoord, order=order, bins=bins,
                                           mask=mask)
            elif function == "chebyshev":
                cfit = Chebyshev.fit(wCoord[mask], meanCoord[mask], order)
                meanCoordFit = cfit(wCoord)

        ## loop over selected sources and get their polynomial coordinate fits
        k = 1
        for j in indices2Fit:
            if coord == "x":
                if commonSlope:
                    offset = np.polyfit(\
                        wCoord[mask],(self.xdata[0,j]-meanCoordFit)[mask],deg=0)
                    self.xdata[1,j] = meanCoordFit + np.polyval(offset, wCoord)
                elif function == "step":
                    self.xdata[1,j] = StepPolyFit(\
                        self.xdata[0,j], order=order, bins=bins, mask=mask)
                elif function == "chebyshev":
                    cfit =Chebyshev.fit(wCoord[mask],self.xdata[0,j,mask],order)
                    self.xdata[1,j] = cfit(wCoord)

            else:
                if commonSlope:
                    offset = np.polyfit(\
                        wCoord[mask],(self.ydata[0,j]-meanCoordFit)[mask],deg=0)
                    self.ydata[1,j] = meanCoordFit + np.polyval(offset, wCoord)
                elif function == "step":
                    self.ydata[1,j] = StepPolyFit(\
                        self.ydata[0,j], order=order, bins=bins, mask=mask)
                elif function == "chebyshev":
                    cfit =Chebyshev.fit(wCoord[mask],self.ydata[0,j,mask],order)
                    self.ydata[1,j] = cfit(wCoord)
                    
            self.poly[i,j] = True
            logging.debug("Polyn. fit to source #{0} [{1}/{2}]".format(\
                    self.ifuID[j], k, len(indices2Fit)))
            k += 1


    
################################################################################
##              Source selection methods/properties                           ##
################################################################################
    def edgeDist(self, fovShape=(16,16)):
        """
        Determine the distance of each source included in the scene to the
        edges of the FoV.

        Parameters
        ----------
        fovShape : array_like, optional
           The x- & y-size of the FoV. Only used if not Instrument-class
           instance is being used.

        Returns
        -------
        dist2bright : nd_array
            A 1dim. array containing the distance of each source to the edge
            of the FoV. Negative distances are used for sources outside the
            FoV
        """
        logging.info("Determining position of sources in IFU FoV.")

        if self.instrument is not None:
            return self.instrument.dist2edge(self.xdata[0].mean(axis=-1),
                                             self.ydata[0].mean(axis=-1))
        else:
            return np.min([self.xdata[0].mean(axis=-1),
                           self.ydata[0].mean(axis=-1),
                           fovShape[1]-1-self.xdata[0].mean(axis=-1),
                           fovShape[0]-1-self.ydata[0].mean(axis=-1)],
                          axis=0)


    def distToBrighter(self):
        logging.info("Determining distance to nearest brighter source.")

        x   = self.xdata[0].mean(axis=-1)
        y   = self.ydata[0].mean(axis=-1)
        mag = self._mag[self.ifuSelect]

        mindist = np.zeros(x.shape, dtype=np.float32)
        nearsrc = np.zeros(x.shape, dtype=self.ifuID.dtype)

        for i in xrange(x.size):
            dist_i =np.sqrt((x-x[i])**2 + (y-y[i])**2)
            dist_i[(mag-mag[i])>=0] = 99.

            mindist[i] = dist_i.min()
            nearsrc[i] = self.ifuID[dist_i.argmin()]

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
        mag = self._mag[self.ifuSelect]

        dist = np.sqrt((x-x[:,np.newaxis])**2 + (y-y[:,np.newaxis])**2)
        np.fill_diagonal(dist, 99.)

        offset = mag[:,np.newaxis]-mag
        offset[dist>r] = 99.

        return 10**(-0.4*offset.min(axis=0))



    def signalToNoise(self, class_psf, magZ, sigma, r=5):
        """
        Calculate the S/N ratio of all stars in the scene. This routine
        was strongly modified in version 0.4 to include the new way to
        calculate the PSF profiles of the sources.

        Parameters
        ----------
        psf : A pampelmuse.core.new_profiles-class
            An instance of this a PSF-class should be present for every
            source in the scene. This is required to determine the flux
            distribution across the FoV
        magZ : float
            The magnitude zeropoint, i.e., the magnitude of a source that
            contributes 1count/layer
        sigma : nd_array
            The average uncertainties in the individual spaxels.
        r : float, optional
            The radius out to which the PSF profile is calculated.
            
        Returns
        -------
        snRatios : nd_array
            A 1dim. array containing the S/N-ratio for each source

        """
        logging.info("Determining Signal-to-Noise of sources.")

##         x   = self.xdata[0].mean(axis=-1)
##         y   = self.ydata[0].mean(axis=-1)
##         mag = self._mag[self.ifuSelect]

        snRatios = np.zeros(self.ifuSelect.shape, dtype=np.float32)

        i=0
        for psf in class_psf:
##         for i in xrange(snRatios.size):
            
            assert psf.src_id==self.ID[i], "Source IDs do not match PSF IDs."

            src_mag = psf.mag            
            flx = 10**(-0.4*(src_mag-magZ))

            psf.mag = 0.
            spaxel, psf_contrib = psf.get_flux()

            ## get flux
            snRatios[i] = flx*np.sqrt(((psf_contrib/sigma[spaxel])**2).sum())

            ## recover initial star magnitude for PSF
            psf.mag = src_mag

            i += 1

        return snRatios



##             ##  calculate PSF profile
## ##             _profile = psf(size=sigma.shape, x_c=x[i], y_c=y[i],
## ##                            mag=0., maxrad=r, return_sparse=True,
## ##                            as_sparse_matrix=True)
##             
##             ## remove "empty"-pixels if instance of Instrument-class is
##             ## available
##             if self.instrument != None:
##                 profile = (np.asarray(_profile.todense())/sigma)[\
##                     self.instrument.fovmask]
##             else:
##                 profile = np.asarray(_profile.todense())/sigma
## 
## 
##             ## PSF array must be converted to a numpy array before calculation
##             snRatios[i] = flx*np.sqrt((profile**2).sum())
## 
##         return snRatios



################################################################################
##                             Output methods                                 ##
################################################################################
    def toRec(self, ids=None):
        
        if ids == None:
            select = np.arange(self.nsrc, dtype=np.int32)
        else:
            select = matchOnID(self._id, np.asarray(ids))

        names = []
        for attribute in ["id","x","y","mag","status","primary","specrow"]:
            if self.__dict__["_{0}".format(attribute)] != None:
                names.append(attribute)
        return np.rec.fromarrays(\
            [self.__dict__["_{0}".format(name)][select] for name in names],
            names=names)



    def makeSourceHDU(self):
        logging.info("Saving source catalog in FITS BinTableHDU.")

        try: select = np.nonzero(self._status>=0)[0]
        except ValueError: select = np.arange(self.nsrc, dtype=np.int32)

        ## sort by increasing magnitudes (if possible)
        if self._mag != None:
            select = select[self._mag[select].argsort()]

        outCols = [pyfits.Column(name="id", format="J", array=self._id[select])]

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

        if makeSigma:
            if self.sigma != None:
                sigmaHDU = pyfits.ImageHDU(self.sigma, name="SIGMASPC")
                sigmaHDU.header.update("CRPIX1", self.crpix)
                sigmaHDU.header.update("CRVAL1", self.crval)
                sigmaHDU.header.update("CDELT1", self.cdelt)
                sigmaHDU.header.update("CTYPE1", self.ctype) 
                return outHDU, sigmaHDU
            else:
                logging.error("Cannot create Sigma HDU: No sigma data")
                return outHDU, None
        else:
            return outHDU



    def makeCoordinatesHDU(self, polyFit=False):

        if not polyFit:
            toStore = np.ones(self.ifuID.size, dtype=np.bool)
            i = 0
        else:
            ## NOTE: polyFit-coords are only stored for sources for which both
            ##       x & y have been fitted.
            toStore = (self.poly[0]&self.poly[1])
            if not toStore.any(): return None
            i=1
            
        outHDU = pyfits.ImageHDU(np.rollaxis(\
                np.dstack((self.xdata[i,toStore], self.ydata[i,toStore])),2,0),
                                 name="POSITIONS")
        outHDU.header.update("hierarch PAMPELMUSE POSITIONS POLY", bool(i))

        k = 0
        for j in xrange(self.ifuID.size):
            if toStore[j]:
                outHDU.header.update(\
                    "hierarch PAMPELMUSE POSITIONS ID{0}".format(k),
                    self.ifuID[j])
                k += 1

        return outHDU
