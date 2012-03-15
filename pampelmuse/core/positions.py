"""
POSITIONS
=========
The routine is designed as an interface for the loading, modification and
storage of the source coordinates that are used by PampelMuse. It consists
of two classes: `CoordinateTransformation' is designed to work with a
transformation from input to IFU coordinates. the `Positions'-class can access
all methods provided by CoordinateTransformation and has own methods defined
that are dedicated to handle IFU coordinates diectly.

CHANGELOG
---------
2012/02/28, v0.1.0.0
2012/03/01, v0.1.0.1, modified matchOnID-function so that it returns the
                      indices of the `parent'-array in the same order the
                      provided IDs are given.
2012/03/07, v0.2.0.0, major updates to the CoordinateTransformation-class so
                      that it makes use of the new ValueArray-structure.
                      Usage of Positions-class is deprecated, required methods
                      have been transfered to Sources-class in pampelmuse.core
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.2.0.0"
import logging
import numpy as np, pyfits
from PyQt4 import QtCore

## required PampelMuse-packages:
from pampelmuse.functions.step_poly_fit import StepPolyFit
from pampelmuse.core.value_array import ValueArray


def myLstsqSigma(A, b, sigma=None):

    if not sigma: return np.linalg.lstsq(A, b)
    else:
        b_ = b/sigma
        if b_.ndim > 1:
            results = [np.zeros((4,b_.shape[1]), dtype=np.float32)]
            for i in xrange(b_.shape[1]):
                results[0][i] = np.linalg.lstsq(A/sigma[:,i][:,np.newaxis], b_)
            return results
        else:
            return np.linalg.lstsq(A/sigma[:,np.newaxis], b_)




def matchOnID(parent, child, assume_unique=False):
    """
    For each element in `children', return indix of element in `parent'.

    Parameters
    ----------
    parent : array_like, 1dim.
        The list that that is scanned
    child : array_like, 1dim.
        The list containing the entries that `parent' is scanned for.
    assume_unique : boolean
        Whether the entries of the two lists are unique.

    Returns
    -------
    indices : nd_array
        The index of every `child'-entry in the `parent' list (if found).
    """
##     indices \
##         = np.nonzero(np.in1d(parent, child, assume_unique=assume_unique))[0]
##     return indices
##
## increases speed by ~ x2:
    return np.nonzero((child[:,np.newaxis]-parent)==0)[1]



class CoordinateTransformation(object):
    
    parameters = ["A", "B", "C", "D"]

    def __init__(self):

        self.ctData   = None
        self.ctFree   = None
        self.ctPoly   = None
        self.ctValues = None
        self.npars    = len(self.parameters)
##         self.missing  = [name for name in self.parameters]



    def makeNewTrans(self, **kwargs):

        logging.info("Initializing coordinate transformation.")

        self.backup = False

        self.unsavedTrans = True

        self.initStructure()

        for name, value in kwargs.iteritems():
            self.addParameter(name, value)


        
    def calcTrans(self, inCoords, outCoords):
        
        logging.info("Determining coordinate transformation.")

        self.backup = False

        ## check if provided data is valid
        for name in ['id', 'x', 'y']:
            if not name in inCoords.dtype.names:
                logging.error(\
                    "Cannot find field '{0}' in input coords.".format(name))
                return
            if not name in outCoords.dtype.names:
                logging.error(\
                    "Cannot find field '{0}' in output coords.".format(name))
                return

        if outCoords["x"].shape != outCoords["y"].shape:
            logging.error("Output x- & y-coords. have different dimensions.")
            return
        else:
            disp = (outCoords["x"].ndim>1)
        if disp:
            ndisp = outCoords["x"].shape[1]
        else:
            ndisp = 1

        ## check if errors for out-coordinates are provided
        if "x_err" in outCoords.dtype.names \
                and "y_err" in outCoords.dtype.names:
            err = True
            logging.info("Found uncertainties of output coords.")
        else:
            err = False
            logging.info("No uncertainties of output coords. found.")


        inSel  = []
        outSel = []

        ## find matching coordinate pairs
        for i in xrange(inCoords.size):
            inID = inCoords['id'][i]
            if not inID in outCoords['id']:
                continue
            inSel.append(i)
            outSel.append(np.where(outCoords['id']==inID)[0][0])
    
        m = len(inSel) ## No of matching sources provided
        if m<2:
            logging.critical("Number of matching sources <2!")
            return
        elif m<4:
            logging.warning("Number of matching sources <4!")
        else:
            logging.info("Number of matching sources: {0}".format(m))


        ## create ndarrays for fitting
        inCoordArr  = np.zeros((2, m), dtype=np.float32)
        inCoordArr[0] = inCoords["x"][inSel]
        inCoordArr[1] = inCoords["y"][inSel]

        outCoordArr = np.zeros((2, m, ndisp), dtype=np.float32)
        outCoordArr[0] = outCoords["x"][outSel][:,np.newaxis]
        outCoordArr[1] = outCoords["y"][outSel][:,np.newaxis]

        if err:
            outSigmaArr = np.zeros(outCoordArr.shape, dtype=np.float32)
            outSigmaArr[0] = outCoords["x_err"][outSel]
            outSigmaArr[1] = outCoords["y_err"][outSel]
        else:
            outSigmaArr = None

        coordTrans = self.quickFit(inCoordArr, outCoordArr,
                                   outSigma=outSigmaArr)[0]

        self.makeNewTrans()

        for i in xrange(self.npars):
            self.addParameter(name=self.parameters[i], value=coordTrans[i])


        ## Only logging from this line
        logging.info(\
            "Determined coord. transf. for {0} image(s).".format(ndisp))

        logging.info("Best-fit parameters:")
        for name in self.parameters:
            logging.info("   {0} = {1:8.3f}".format(\
                    name, self.value[name].mean()))

        logging.info("Offsets of individual sources:")
        logging.info("       ID     dx     dy")
        [x, y] = self.__call__(x=inCoordArr[0], y=inCoordArr[1])
        for i in xrange(m):
            J = inSel[i]
            K = outSel[i]

            logging.info("   {0:6d}  {1:+5.2f}  {2:+5.2f}".format(\
                    int(inCoords['id'][J]),
                    float((x[i]-outCoordArr[0,K]).mean(axis=-1)),
                    float((y[i]-outCoordArr[1,K]).mean(axis=-1))))



    def quickFit(self, inCoords, outCoords, outSigma=None):

        m = inCoords[0].shape[0]
        M = 2*m

        A = np.ones((M,4), dtype=np.float32)
        A[:m,0] =  inCoords[0].flatten()
        A[m:,0] =  inCoords[1].flatten()
        A[:m,1] =  inCoords[1].flatten()
        A[m:,1] = -inCoords[0].flatten()
        A[m:,2] = 0.
        A[:m,3] = 0.

        sigma = None
        if outCoords[0].ndim>1:
            b = np.vstack(outCoords)
            if outSigma != None: sigma = np.vstack(outSigma)
        else:
            b = np.hstack(outCoords)
            if outSigma != None: sigma = np.hstack(outSigma)

        return myLstsqSigma(A, b, sigma=sigma)



    def readTransHDU(self, HDU, backup=False):

        if "EXTNAME" in HDU.header:
            logging.info("Initializing coord. transf. from HDU '{0}'.".format(\
                    HDU.header["EXTNAME"]))
        else:
            logging.info("Initializing coord. transf. from unnamed HDU.")


        ## check if required fields 'name' and 'value exist':
        for field in ["name", "value"]:
            if not field in HDU.data.names:
                logging.critical(\
                    "Cannot find field '{0}' in transf. HDU.".format(field))
                return

        self.backup = backup

        self.initStructure()

        ## read data from selected HDU
        freepar  = False
        isfitted = False
        polyfit  = None

        for i in xrange(HDU.data.size):
            if "freepar" in HDU.data.names:
                freepar=HDU.data.field("freepar")[i]
            if "isfitted" in HDU.data.names:
                isfitted=HDU.data.field("isfitted")[i]
            if "polyfit" in HDU.data.names:
                polyfit=HDU.data.field("polyfit")[i]
                
            self.addParameter(name=HDU.data.field("name")[i],
                              value=HDU.data.field("value")[i],
                              freepar=freepar,
                              isfitted=isfitted,
                              polyfit=polyfit)

        self.unsavedTrans = False



    def initStructure(self):
        """
        The method is used to prepare the required arrays containing the
        properties of the PSF parameters. It should be called immediately
        after a PSF has been initialized and before any parameters are
        added.
        """
        self.disp       = False
        self.lockSize   = False ## Cannot resize after call to makeParameters
        self.missing    = [name for name in self.parameters]
        self.ctData = np.zeros((2,self.npars,1),  dtype=np.float32)
        self.ctFree = np.zeros((self.npars,),  dtype=np.bool)
        self.ctPoly = np.zeros((self.npars,),  dtype=np.bool)
 

        
    def addParameter(self, name, value, freepar=False, isfitted=False,
                     polyfit=None):

        if not name in self.missing:
            logging.error("Cannot add parameter: '{0}' not missing.".format(\
                    name))
            return
        
        i = self.parameters.index(name)

        ## check if value has correct data shape, reshape self.ctData if needed 
        value = np.asarray(value)
        if value.size > self.ctData.shape[-1]:
            if self.disp:
                logging.warning("Differing data size of parameter '{0}'.")
                logging.warning("Transformation structure will be resized.")
            self.resizeStructure(size=value.size)
        elif value.size < self.ctData.shape[-1] and value.size > 1:
            logging.error("Cannot add parameter: '{0}' has invalid data size."\
                              .format(name))
            return

        ##  check if polyfit information is valid
        if isfitted and polyfit==None:
            logging.error("Cannot add parameter: '{0}'".format(name))
            logging.error(" No `polyfit' available, but `isfitted'-flag set.")
            return
        
        if polyfit != None:
            polyfit = np.asarray(polyfit)
            if polyfit.size != value.size:
                logging.error("Cannot add parameter: '{0}'".format(name))
                logging.error("Incompatible sizes of `value' and `polyfit'.")
                return
        
        self.ctData[0,i] = value
        self.ctFree[i]   = freepar
        self.ctPoly[i]   = isfitted
        if polyfit    !=  None: self.ctData[1,i]  = polyfit
##         if self.backup: self.value_init[name] = value

        self.missing.remove(name)

        self.unsavedTrans = True

        logging.info("Added parameter '{0}' to transformation.".format(name))



    @property
    def verifyTrans(self):

        if not "parameters" in dir(self):
            return False

        if self.missing != []:
##             logging.warning("Some transf. parameters have not been specified:")
##             for prm in self.missing:
##                 logging.warning("   Missing parameter '{0}'".format(prm))
            return False
        else:
            return True



    def info(self):
        
        logging.info("Summary of coordinate transformation instance:")
##         logging.info(" Valid: {0}".format(self.verifyTrans))
        logging.info(" Parameter  defined?  free?  polyfit?")

        for i in xrange(self.npars):
            name = self.parameters[i]

            logString = " *{0:8s}".format(name)
            if not name in self.missing:
                logString += "      [x]"
            else:
                logstring += "      [ ]     [ ]       [ ]"
                logging.info(logString)
                continue

            if self.ctFree[i]: logString += "     [x]"
            else: logString += "     [ ]"

            if self.ctPoly[i]: logString += "       [x]"
            else: logString += "       [ ]"

            logging.info(logString)
                


    def provideParNames(self):

        return QtCore.QStringList(self.parameters)



    def resizeStructure(self, size):
        
        if self.lockSize:
            logging.error("Cannot resize after calling `makeValues'.")
            logging.error("To resize, call `delValues' first.")
            return

        if self.ctData.shape[-1] == size:
            return

        if self.ctData.shape[-1] == 1 and size > 1:
            self.disp = True

        kMax = min(size, self.ctData.shape[-1])-1
        newData = np.zeros((self.ctData.shape)[:2]+(size,),  dtype=np.float32)
        newData[...,:kMax] = self.ctData[...,:kMax]
        if size > self.ctData.shape[-1]:
            newData[...,kMax:] = self.ctData[...,-1][:,:,np.newaxis]
        self.ctData = newData


##         if self.backup:
##             Needs to be implemented

        
        self.unsavedTrans = True



    def makeValues(self):

        if not self.verifyTrans:
            logging.error(" PSF definition is not valid.")
            return

        self.ctValues = ValueArray(\
            self.ctData, free=self.ctFree, poly=self.ctPoly)

        self.lockSize = True



    def delValues(self):
        
        if not self.lockSize:
            return

##         del self.ctValues
        self.ctValues = None
        self.lockSize = False

        logging.warning("Deleted attribute `values', unlocked data size.")

    

    @property
    def Fit(self):

        if not self.verifyTrans:
            return False
        
        return (self.ctFree&(self.ctPoly==False)).any()

            

    def __call__(self, x, y, index=Ellipsis):
        ## index can be any valid slice definition:  an integer, a slice()-
        ## object or an Ellipsis-object
        
        if not self.verifyTrans:
            logging.error("Cannot determine pixel positions of sources:")
            logging.error(" No valid coord. transf. avalable.")
            return None

        if not self.lockSize:
            self.makeValues()

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.ndim != 1:
            logging.error("Provided x- & y-coords have invalid dimensions.")
        if x.shape != y.shape:
            logging.error("Length of provided x- & y-coords differ.")
            return None
        
        A = self.ctValues[0, index] 
        B = self.ctValues[1, index]
        C = self.ctValues[2, index]
        D = self.ctValues[3, index]

        xOut = A*x[:,np.newaxis] + B*y[:,np.newaxis] + C
        yOut = A*y[:,np.newaxis] - B*x[:,np.newaxis] + D

        return [xOut, yOut]




    def makeTransHDU(self,  includePolyFit=True):
        
        if self.unsavedTrans:
            logging.info("Saving updated coord-trans. to FITS BinTable-HDU.")
        else:
            logging.warning(\
                "Saving unchanged coord-trans. to FITS BinTable-HDU.")


        if not self.disp: 
            dataFmt = "E"
        else:
            dataFmt = "{0}E".format(self.ctData.shape[-1])

        outColumns = [
            pyfits.Column(name="name", format="10A", array=self.parameters),
            pyfits.Column(name="value", format=dataFmt,
                          array=np.squeeze(self.ctData[0])),
            pyfits.Column(name="freepar", format="L", array=self.ctFree)
            ]

        if includePolyFit:
            outColumns.extend([\
                    pyfits.Column(\
                        name="isfitted", format="L", array=self.ctPoly),
                    pyfits.Column(\
                        name="polyfit", format=dataFmt,
                        array=np.squeeze(self.ctData[1]))])
            
        outHDU = pyfits.new_table(outColumns, header=pyfits.Header(\
                cards=[pyfits.Card(key="EXTNAME", value="POSPARS")]))

        outHDU.add_checksum()

        self.unsavedTrans = False

        return outHDU



class Positions(CoordinateTransformation): ## usage is deprecated

    def __init__(self, sources, hdu=False):
        logging.error("Usage os Positions-class  is deprecated.")

        if hdu:
            logging.info("Initializing source coord. instance from HDU.")
            sources = np.array(sources.data,
                               dtype=sources.data.dtype)
        else:
            logging.info("Initializing source coord. instance.")

        if not "id" in sources.dtype.names:
            logging.critical("No field named `id' found in provided list.")

        ## sources with ID<0 are considered unresolved
        rsv = (sources["id"]>=0)

        self.ID = sources[["id"]][rsv].view(sources["id"].dtype).reshape(-1,)
        
        if "x" not in sources.dtype.names or "y" not in sources.dtype.names:
            self.hst = None
            logging.warning("No input coordinates (fields `x'  and `y') found.")
        else:
            if sources["x"].dtype != sources["y"].dtype:
                logging.error("Array types of x- and y-coordinates differ.")
            
            self.hst = sources[["x","y"]][rsv].view(sources["x"].dtype)\
                .reshape(rsv.sum(), 2).swapaxes(0,1)
            logging.info("Found input coordinates for {0} sources.".format(\
                    self.hst.shape[1]))

        self.hst2ifu = None
        if self.hst != None and "posrow" in sources.dtype.names:
            self.hst2ifu = matchOnID(sources["posrow"],
                                     sources["posrow"][sources["posrow"]>=0])

        self.nSrc = self.ID.size ## No. of sources considered

        self.ifu         = None
        self.ifuPolyfit  = None
        self.ifuView     = None
        self.ifuIsfitted = None
        self.unsavedIfu  = False


    def loadIfuCoords(self, data, ids=None, hdu=False, freepar=False,
                      isfitted=False):

        self.freepar = freepar
        self.isfitted = isfitted

        if hdu:
            if "EXTNAME" in data.header:
                logging.info("Reading IFU coords. from HDU '{0}'.".format(\
                        data.header["EXTNAME"]))
            else:
                logging.info("Reading IFU coords. from unnamed HDU.")

            if data.header["NAXIS2"] > self.nSrc \
                    and data.header["NAXIS1"] in [self.nSrc, self.nSrc+1]:
                logging.warning("Opened FITS-HDU has an outdated structure.")
                logging.warning("Will try to update and use as polyfit coords.")

                if data.header["NAXIS1"] == self.nSrc:
                    self.ifuPolyfit = np.rollaxis(data.data, 1, 3)
                else:
                    self.ifuPolyfit = np.rollaxis(data.data[...,:-1], 1, 3)

            else:
                if "FREEPAR" in data.header:
                    self.freepar = data.header["FREEPAR"]
                if "ISFITTED" in data.header:
                    self.isfitted = data.header["ISFITTED"]

                if self.isfitted:
                    self.ifuPolyfit = data.data
                else:
                    self.ifu = data.data

        else:
            if self.isfitted:
                self.ifuPolyfit = data
            else:
                self.ifu = data
            self.unsavedIfu = True

        if self.ifuPolyfit != None:
            self.ifuIsfitted = np.ones(self.ifuPolyfit.shape[:-1],
                                       dtype=np.bool)

        if self.ifuPolyfit != None:
            self.ifuView = self.ifuPolyfit
        elif self.ifu != None:
            self.ifuView = self.ifu
            

        if self.ifuView != None:
            self.ndisp = self.ifuView.shape[-1]
            self.nIfu  = self.ifuView.shape[-2]


        logging.info(\
            "Got IFU pixel coordinates for {0} source(s).".format(self.nIfu))

        if ids!=None:
            self.hst2ifu = matchOnID(self.ID, ids)
            if self.hst2ifu.size < len(ids):
                logging.error("Unknown source ID(s) provided.")
            self.ifuID = self.ID[self.hst2ifu]

        if self.hst2ifu==None and self.hst!=None:
            logging.warning("No mapping from input to IFU coords provided.")
            logging.warning("Assuming coords belong to first {0} source(s)."\
                                .format(self.nIfu))
            self.hst2ifu = np.arange(self.nIfu, dtype=np.int32)
            self.ifuID = self.ID[self.hst2ifu]



    @property
    def Fit(self):
        
        if self.ifuIsfitted != None:
            return (self.ifuIsfitted.all()==False)

        elif self.ifu != None:
            return self.freepar
        
        else:
            return self.toUpdate.any()

    @property
    def unsavedTrans(self):
        if not self.verifyTrans:
            return False
        else:
            return CoordinateTransformation.unsavedTrans

        
    def provideIDs(self):

        if self.ifuView == None:
            return QtCore.QStringList(self.ID.astype("|S6"))
        else:
            return QtCore.QStringList(self.ifuID.astype("|S6"))



    def putIfuCoords(self, values, index=None, ids=None):

        if self.ifu == None:
            return

        values = np.asarray(values)
        if index != None:
            index = np.asarray(index)

        if values.shape[0] != 2:
            logging.error("Cannot update IFU coords using `values':")
            logging.error(" First axis must have len=2, not {0}.".format(\
                    values.shape[0]))
            return

        ## match provided IDs to IFU coordinates.
        if ids == None:
            ids = self.ifuID
        else:
            ids = np.asarray(ids)
        IDmatch = matchOnID(parent=self.ifuID, child=ids, assume_unique=True)

        if IDmatch.size < ids.size:
            logging.error("Cannot update IFU coords using `values':")
            logging.error(" Invalid source ID(s) have been provided.")

        if not values.shape[1] == ids.size:
            logging.error("Cannot update IFU coords using `values':")
            logging.error(" Array size does not match # of IDs to update.")
            return
        
        ## check if dispersion indices are useful
        if index!=None:
            if index.min()<0 or index.max()>=self.ndisp:
                logging.error("Cannot update IFU coords using `values':")
                logging.error(" Index/indices out of valid range [0,{0})."\
                                  .format(self.ndisp))
                return
        elif values.ndim>2:
            if values.shape[2] != self.ifu.shape[-1]:
                logging.error("Cannot update IFU coords using `values':")
                logging.error(" Lengths of dispersion axes do not match.")
                return
        else:
            if self.ifuView.shape[-1]>1:
                logging.error("Cannot update IFU coords using `values':")
                logging.error(" Lengths of dispersion axes do not match.")
                return
            
        if index == None:
            self.ifu[:,IDmatchID] = values
        else:
            self.ifu[:,IDmatch,index] = values
        self.unsavedIfu = True



    def getIfuCoords(self, index=None, ids=None, forceTrans=True):
            
        if self.ifuView==None and not self.verifyTrans:
            logging.error("Cannot return IFU coords for requested sources:")
            logging.error(" Neither IFU coords nor coord. trans. provided.")
            return None
        elif self.ifuView==None and not forceTrans:
            forceTrans=True

        if index!=None:
            index = np.atleast_1d(np.asarray(index))
            ndisp = index.size
        else:
            ndisp = self.ndisp

        ## match provided IDs to input coordinates.
        if ids == None:
            ids = self.ID
        else:
            ids = np.atleast_1d(np.asarray(ids))
        IDmatch = matchOnID(parent=self.ID, child=ids, assume_unique=True)

        if IDmatch.size < ids.size:
            logging.error("Cannot return IFU coords for requested sources:")
            logging.error(" Invalid source ID(s) have been provided.")
            return None

        if self.ifuView != None:
            ifuIDmatch = matchOnID(\
                parent=self.ifuID, child=ids, assume_unique=True)

        if self.verifyTrans:
            outIFU = self.__call__(self.hst[0,IDmatch], self.hst[1,IDmatch],
                                   index=index)
            if forceTrans: return outIFU

        else:
            outIFU = np.zeros((2, ifuIDmatch.size, ndisp), dtype=np.float32)


        if not self.verifyTrans and ifuIDmatch.size < ids.size:
            logging.error("Cannot return IFU coords for requested sources:")
            logging.error(" Source(s) without IFU coords have been provided.")
            return None

        if ifuIDmatch.size>0:
            sel = np.in1d(ids, self.ifuID)
            if index == None:
                outIFU[:,sel] = self.ifuView[:,ifuIDmatch]
            else:
                outIFU[:,sel] = self.ifuView[:,ifuIDmatch[:,np.newaxis],index]

        ## about slicing of n-dim. arrays using index arrays (e.g. [0,1,4]):
        ####################################################################
        ## *if multiple index arrays are used, like in x[...,[1,2,3],[2,3,4],
        ##  numpy will extract all entries whose last two indices are (1,2),
        ##  (2,3) or (3,4) from x, hence the dimension of x is reduced by 1.
        ## *Using x[:,[1,2,3], [4]] will result in broadcasting of the second
        ##  index array: x[...,[1,2,3], [1,1,1]] and extraction of entries
        ##  whose last 2 indices are (1,1), (2,1), (3,1), again reducing the
        ##  array dimension by one.
        ## *To extract without reducing the dimension of x, use
        ##  x[...,[[1]],[2],[3]], [1]] or  x[...,[[1]],[2],[3]], [2,3,4]]

        return outIFU



    def viewIfu(self, ID, coord, forceTrans=True):

        if not coord in ["x", "y"]:
            return None
        elif coord == "x":
            index = 0
        else:
            index = 1

        if not forceTrans and self.ifuPolyfit != None and self.ifu != None:
            ## need to change focus of ifuView-pointer to data, otherwise
            ## polyfit is returned twice.
            self.ifuView = self.ifu

        outData = self.getIfuCoords(ids=ID, forceTrans=forceTrans)

        if not forceTrans and self.ifuPolyfit != None and self.ifu != None:
            self.ifuView = self.ifuPolyfit


        if self.ifuPolyfit == None:
            return outData[index,0], None
        else:
            if self.verifyTrans:
                i = np.squeeze(matchOnID(parent=self.ID,
                                         child=np.atleast_1d(ID),
                                         assume_unique=True))
            else:
                i = np.squeeze(matchOnID(parent=self.ifuID,
                                         child=np.atleast_1d(ID),
                                         assume_unique=True))

            if self.ifuIsfitted[index, i]:
                return outData[index,0], self.ifuPolyfit[index, i]
            else:
                return outData[index,0], None
        


    def fitIfuCoordinates(self, ids, coord, order, bins, mask=None,
                          forceTrans=True, commonSlope=False):

        if commonSlope:
            logging.warning("Flag `commonSlope' not yet supported.")

        if not coord in ["x", "y"]:
            return None
        elif coord == "x":
            index = 0
        else:
            index = 1


        if self.ifuPolyfit == None:
            if not self.verifyTrans:
                self.ifuPolyfit = np.zeros(self.ifu.shape,  dtype=np.float32)
            else:
                self.ifuPolyfit = np.zeros((2, self.ID.size, self.ndisp),
                                           dtype=np.float32)
            self.ifuIsfitted = np.zeros(self.ifuPolyfit.shape[:-1],
                                        dtype=np.bool)

        if ids != None:
            ids = np.atleast_1d(np.asarray(ids))
        elif not self.verifyTrans:
            ids = self.ifuID
        else:
            ids = self.ID
        

        if not self.verifyTrans:
            indices = matchOnID(parent=self.ifuID, child=ids,
                                assume_unique=True)
        else:
            indices = matchOnID(parent=self.ID, child=ids,
                                assume_unique=True)

        if not forceTrans and self.ifuPolyfit != None and self.ifu != None:
            ## need to change focus of ifuView-pointer to data, otherwise
            ## polyfit is returned twice.
            self.ifuView = self.ifu

        for i in xrange(indices.size):

            self.ifuPolyfit[index, indices[i]] = StepPolyFit(\
                self.getIfuCoords(ids=ids[i], forceTrans=forceTrans)[index,0],
                order=order, bins=bins, mask=mask)
            self.ifuIsfitted[index] = True
        
        if not forceTrans and self.ifuPolyfit != None and self.ifu != None:
            self.ifuView = self.ifuPolyfit

        self.unsavedIfu = True
        


    def updateTransformation(self, ids=None, index=None):
        
        if not self.verifyTrans:
            logging.error("Cannot update coordinate transformation:")
            logging.error(" No valid transformation defined.")
            return


        if ids == None:
            ids = self.ifuID
        else:
            ids = np.asarray(ids)
        selected = np.in1d(self.ifuID, ids, assume_unique=True)


        if index == None:
            newCT = self.quickFit(self.hst[self.hst2ifu][selected],
                                  self.ifu[selected])
        else:
            newCT = self.quickFit(self.hst[self.hst2ifu][selected],
                                  self.ifu[selected][index])
            
        self.updateValues(newCT, index=index)



    def makeIfuHDU(self):

        if self.ifuPolyfit != None:

            ifuPolyfitHDU = pyfits.ImageHDU(self.ifuPolyfit, name="POSITIONS")
            ifuPolyfitHDU.header.update("ISFITTED", True)
            ifuPolyfitHDU.add_checksum()

            if self.ifu != None:
                ifuHDU = pyfits.ImageHDU(self.ifu, name="INPOSITIONS")
                ifuHDU.header.update("FREEPAR", self.freepar)
                ifuHDU.header.update("ISFITTED", False)
                ifuHDU.add_checksum()
            else:
                ifuHDU = None


        elif self.ifu != None:
            
            ifuPolyfitHDU = None

            ifuHDU = pyfits.ImageHDU(self.ifu, name="POSITIONS")
            ifuHDU.header.update("FREEPAR", self.freepar)
            ifuHDU.header.update("ISFITTED",  False)
            ifuHDU.add_checksum()
        else:
            ifuPolyfitHDU = None
            ifuHDU        = None

        self.unsavedIfu = False

        return ifuHDU, ifuPolyfitHDU
