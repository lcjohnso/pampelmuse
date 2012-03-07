"""
SOURCES
=======
The `Sources'-class is the PampelMuse-routine that handles the entire source
catalogue. It keeps track of the loaded sources, their grouping  into the
individual components, their spectra and postions. It also provides some
methods to write the individual properties into FITS-HDUs.
The routine is heavily used by all the main scripts of PampelMuse.

CHANGELOG:
----------
2012/03/07, v0.1.0.0, preliminary version completed 
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.0.0"
import logging
import numpy as np, pyfits

from pampelmuse.core.positions import CoordinateTransformation



def matchOnID(parent, child):
    """
    For each element in `children', return indix of element in `parent'.

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

    def __init__(self, data=None, hdu=False, ids=None):
        """
        Initialize an instance of the `StoreParam'-class.

        Parameters
        ----------
        data
        hdu
        """
        if hdu:
            logging.info("Initializing Sources instance from HDU.")
            data = np.array(data.data, dtype=data.data.dtype)
        else:
            logging.info("Initializing Sources instance.")

        ## initialize properties:
        self._id       = ids
        self._x        = None
        self._y        = None
        self._mag      = None
        self._status   = None
        self._specrow  = None
        self._primary  = None
        self._indexmap = None
        self._weights  = None

        if data==None:
            data = np.array([], dtype=[('empty', np.int16),])

        ## IDs must be present
        try:
            self._id = data[["id"]].view(data["id"].dtype).reshape(-1,)
        except ValueError:
            if self._id == None:
                logging.critical("No field `ID' found in provided data.")
        self.nsrc = self._id.size

        ## some properties are optional
        for name in ["x", "y", "mag"]:
            try:
                self.__dict__["_{0}".format(name)]\
                    = data[[name]].view(data[name].dtype).reshape(-1,)
            except ValueError:
                continue

        ## include specrow, status, primary
        if "specrow" in data.dtype.names: self.setSpecrow(data["specrow"])
        if "status" in data.dtype.names: self.setStatus(data["status"])
        if "primary" in data.dtype.names: self.setPrimary(data["primary"])

        super(Sources, self).__init__()



    def getID(self):
        return self._id



    def setX(self, value):
        value = np.asarray(value, dtype=np.float32)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `x', data has invalid shape.")
        else:
            self._x = value

    def getX(self):
        if self._x == None:
            logging.error("Requested attribute `x' not specified.")
        return self._x
            


    def setY(self, value):
        value = np.asarray(value, dtype=np.float32)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `y', data has invalid shape.")
        else:
            self._y = value

    def getY(self):
        if self._y == None:
            logging.error("Requested attribute `y' not specified.")
        return self._y


    
    def setStatus(self, value):
        value = np.asarray(value, dtype=np.int16)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `y', data has invalid shape.")
        else:
            self._status = value
            if self._specrow==None:
                if not (self._status==1).any(): self.makeSpecrow(unique=True)
                elif self._primary != None: self.makeSpecrow(unique=False)

    def getStatus(self):
        if self._status == None:
            logging.error("Requested attribute `status' not specified.")
        return self._status



    def setPrimary(self, value):
        value = np.asarray(value, dtype=np.int32)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `primary', data has invalid shape.")
        else:
            self._primary = value
            if self._status != None and self._specrow==None:
                self.makeSpecrow(unique=False)

    def getPrimary(self):
        if self._primary == None:
            logging.error("Requested attribute `primary' not specified.")
        return self._primary



    def setMag(self, value):
        value = np.asarray(value, dtype=np.float32)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `mag', data has invalid shape.")
        else:
            self._mag = value
            
    def getMag(self):
        if self._mag == None:
            logging.error("Requested attribute `mag' not specified.")
        return self._mag

            

    def setSpecrow(self, value):
        value = np.asarray(value, dtype=np.int16)
        
        if not value.size == self.nsrc and value.ndim==1:
            logging.error("Cannot set `specrow', data has invalid shape.")
        elif self._specrow != None:
            logging.error("Cannot set `specrow' as it has been previously set.")
        else:
            self._specrow = value
            self.makeIndexmap()
            self.makeWeights()

            
    def getSpecrow(self):
        if self._specrow == None:
            logging.error("Requested attribute `specrow' not specified.")
        return self._specrow



    def getIndexmap(self):
        if self._indexmap == None:
            logging.error("Requested attribute `indexmap' not specified.")
        return self._indexmap



    def getWeights(self):
        if self._weights == None:
            logging.error("Requested attribute `weights' not specified.")
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
                        "Mapping from `specrow'-property to index.")
    weights  = property(getWeights, None, None,
                        "Relative magnitudes of sources in a component.")


    def makeSpecrow(self, unique):

        if unique:
            value = np.arange(self.nsrc, dtype=np.int16)
        else:
            value = (self._id==self._primary).cumsum()-1
            for i in xrange(self._id.size):
                if self._status[i] == 1:
                    j=i
                    while self._primary[j]!=self._id[j]:
                        j = np.where(self._id==self._primary[j])[0][0]
                    value[i] = value[j]
            value[self._status==0] = value.max()+1

        self.setSpecrow(value)


        
    def makeIndexmap(self):
        ## `indexmap' is a dictionary mapping the `specrow' (keys) to the
        ## index/indices (values) in the `sources'-table
        self._indexmap = {}
        for i in xrange(self.ncmp):
            self._indexmap[i] = np.where(self._specrow==i)[0]



    def makeWeights(self):
        ## `weights' is a dictionary mapping the `specrow' (keys) to the
        ## relative magnitudes (values) of its component(s)
        if self._mag == None:
            logging.warning("Property `mag' not set, assuming equal weights.")

        self._weights = {}
        for row, ind in self._indexmap.iteritems():
            if self._mag != None:
                self._weights[row] = self._mag[ind] \
                    + 2.5*np.log10((10**(-0.4*self._mag[ind])).sum())
            else:
                n = len(ind)
                self._weights[row] \
                    = np.asarray(n*[2.5*np.log10(n)], dtype=np.float32)


    @property
    def ncmp(self):
        if self._specrow == None: return None
        else: return self._specrow.max()+1


    @property
    def hasSky(self):
        if self._status == None: return False
        return (self._status==4).any()
    
    @property
    def skyRow(self):
        if not self.hasSky or self._specrow==None: return None
        else: return self._specrow[np.where(self.status==4)[0][0]]

    
    @property
    def hasUnres(self):
        if self._status == None: return False
        return (self._status==0).any()

    @property
    def unresRow(self):
        if not self.hasUnres or self._specrow==None: return None
        else: return self._specrow[np.where(self.status==0)[0][0]]
            


    def info(self):
        
        try:
            logging.debug("Current structure of loaded Sources-instance:")
            logging.debug(" Component:   IDs (& mags):")

            for row, ind in self.indexmap.iteritems():
                logString = " {0:10d}".format(int(row))
                j = 0
            
                for i in ind:
                    logString += "   {0} ({1:.2f})".format(\
                        int(self._id[i]), float(self._weights[row][j]))
                    j +=1
                    if not np.mod(j, 2):
                        logging.debug(logString)
                        logString = 11*" "
                if np.mod(j, 2)>0: logging.debug(logString)
        except AttributeError:
            logging.warning("Some source properties are not set.")
            return



###############################################################################
##              Methods dealing with the spectra of the sources              ##
###############################################################################
    def loadSpectra(self, data, sigma=None, hdu=False, crval=1., cdelt=1.,
                    crpix=1, ctype="AWAV"):
             
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

        self.ndisp = self.data.shape[1]

        

    def provideNames(self):

        ownSpec = np.in1d(self.status, np.arange(2,4))

        self.names   = QtCore.QStringList(self.ID[ownSpec].astype("|S6"))
        self.nameMap = dict((zip(self.ID[ownSpec].astype("|S6"),
                                 self.specrow[ownSpec])))
        if self.hasSky:
            self.names.append("Sky")
            self.nameMap["Sky"] = self.skyRow
        if self.hasUnRes:
            self.names.append("Unresolved")
            self.nameMap["Unresolved"] = self.unResRow
        
        return self.names



    def viewSpectrum(self, name=None, ID=None):
        
        if name==None and ID==None:
            logging.error("Either name or ID must be provided.")
            return

        if name != None:
            return self.data[self.nameMap[name]]
        else:
            return self.data[self.specrow[np.where(self.ID==ID)[0][0]]]



##     def viewData(self, index, ID=None):
##         
##         index = np.atleast_1d(index)
##         
##         if ID == None:
##             return self.data[:,index]
##         else:
##             ID = np.atleast_1d(ID)
##             select = self.specrow[matchOnID(self.ID, ID)]
##             if select.size < ID.size:
##                 logging.error("Cannot return data: Invalid ID(s) provided.")
##                 return None
##             return self.data[select[:,np.newaxis], index]
## 
## 
## 
##     def addData(self, data, index, ID=None):
## 
##         index = np.atleast_1d(index)
##         
##         if ID == None:
##             self.data[:,index] += data[:,np.newaxis]
##         else:
##             ID = np.atleast_1d(ID)
##             select = self.specrow[matchOnID(self.ID, ID)]
##             if select.size < ID.size:
##                 logging.error("Cannot add to data: Invalid ID(s) provided.")
##                 return
##             self.data[select[:,np.newaxis], index] += data
## 
## 
## 
## 
##     def putSigma(self, sigma, index, ID=None):
## 
##         if self.sigma==None:
##             self.sigma = np.zeros(self.data.shape, dtype=self.data.dtype)
##             
##         index = np.atleast_1d(index)
##         
##         if ID == None:
##              self.sigma[:,index] = sigma
##         else:
##             ID = np.atleast_1d(ID)
##             select = self.specrow[matchOnID(self.ID, ID)]
##             if select.size < ID.size:
##                 logging.error("Cannot put sigma: Invalid ID(s) provided.")
##                 return
##             self.sigma[select[:,np.newaxis], index] = sigma



###############################################################################
##            Methods dealing with IFU coordinates of the sources            ##
###############################################################################
    def loadIFUCoords(self, data, polyfit=None, ids=None, free=False):

        self.free = free

        if ids==None:
            logging.warning("No IDs for IFU coordinates were provided.")
            self.ifuID = self._id[self._ID>=0]
        else:
            self.ifuID = np.asarray(ids)
        
        self.ifuSelect = matchOnID(self._id, self.ifuID)
        if self.ifuSelect.size < self.ifuID.size:
            logging.error("Invalid source IDs have been provided.")
            return

        data = np.atleast_3d(data)
        if data.shape[0] != 2:
            logging.error("Provided `data' has invalid shape.")
            return
        if data.shape[1] != len(self.ifuID):
            logging.error("`data'-shape incompatible with no. of IDs.")
            return

        self.xdata = np.resize(data[0], (2,)+data[0].shape)
        self.ydata = np.resize(data[1], (2,)+data[0].shape)

        self.poly = False
        if polyfit != None:
            polyfit = np.at_least_3d(polyfit)
            if data.shape != polyfit.shape:
                logging.error(\
                    "`polyfit' must have same shape as `data' if provided.")
                return
            self.xdata[1] = polyfit[0]
            self.ydata[1] = polyfit[1]
            self.poly = True



    def calcTrans(self):

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
            [self.xdata[0], self.ydata[0]], outSigma=None)[0]

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


    def callTrans(self):

        self.ifuID     = self._id[self._id>=0]
        self.ifuSelect = matchOnID(self._id, self.ifuID)

        xNew, yNew = self.__call__(x=self._x[self._id>0], y=self._y[self._id>0])
        
        if "xdata" in self.__dict__ and "ydata" in self.__dict__:
            if xNew.shape == self.xdata.shape[1:]:
                self.xdata[0] = xNew
                self.ydata[0] = yNew
                return

        logging.warning("Shape of IFU coord. array changed.")
        logging.warning("Lost scope of polynomial fits (if any).")

        self.xdata = np.zeros((2,)+xNew.shape)
        self.ydata = np.zeros((2,)+yNew.shape)
        self.xdata[0] = xNew

        self.ydata[0] = yNew


##         if hdu:
##             if "EXTNAME" in data.header:
##                 logging.info("Reading IFU coords. from HDU '{0}'.".format(\
##                         data.header["EXTNAME"]))
##             else:
##                 logging.info("Reading IFU coords. from unnamed HDU.")
## 
##             if data.header["NAXIS1"] in [self.nsrc, self.nsrc+1]\
##                     and data.header["NAXIS2"] not in [self.nsrc, self.nsrc+1]:
##                 logging.warning("Opened FITS-HDU has an outdated structure.")
##                 logging.warning("Will try to update and use as polyfit coords.")
## 
##                 if data.header["NAXIS1"] == self.nsrc:
##                     self.ifuPolyfit = np.rollaxis(data.data, 1, 3)
##                 else:
##                     self.ifuPolyfit = np.rollaxis(data.data[...,:-1], 1, 3)
## 
##             else:
##                 if "FREEPAR" in data.header:
##                     self.free = data.header["FREEPAR"]
##                 if "ISFITTED" in data.header:
##                     self.poly = data.header["ISFITTED"]
## 
##                 if self.poly:
##                     self.ifuPolyfit = data.data
##                 else:
##                     self.ifu = data.data
## 


################################################################################
##              Source selection methods/properties                           ##
################################################################################
    def edgeDist(self, fovShape):
        logging.info("Determining position of sources in IFU FoV.")

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

        dist = np.sqrt((x-x[:,np.newaxis])**2 + (y-y[:,np.newaxis])**2)
        dist[(mag-mag[:,np.newaxis])<=0] = 99.

        return (dist.min(axis=0), self.ifuID[dist.argmin(axis=0)])



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



    def signalToNoise(self, psf, magZ, sigma, r=5):
        logging.info("Determining Signal-to-Noise of sources.")

        x   = self.xdata[0].mean(axis=-1)
        y   = self.ydata[0].mean(axis=-1)
        mag = self._mag[self.ifuSelect]

        snRatios = np.zeros(self.ifuSelect.shape, dtype=np.float32)
        for i in xrange(snRatios.size):

            profile = psf(size=sigma.shape, x_c=x[i], y_c=y[i],
                          mag=0., maxrad=r, return_sparse=True,
                          as_sparse_matrix=True)

            ## get flux
            flx = 10**(-0.4*(mag[i]-magZ))

            ## PSF array must be converted to a numpy array before calculation
            snRatios[i] = flx*np.sqrt(\
                ((np.asarray(profile.todense())/sigma)**2).sum())

        return snRatios



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
            [self.__dict__["_{0}".format(name)] for name in names], names=names)



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
