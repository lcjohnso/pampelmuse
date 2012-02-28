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
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.0.0"
import logging
import numpy as np, pyfits
from PyQt4 import QtCore

## required PampelMuse-packages:
from pampelmuse.functions.step_poly_fit import StepPolyFit


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
    indices = np.nonzero(np.in1d(parent, child, assume_unique=assume_unique))[0]
    
    return indices




class CoordinateTransformation:

    def makeNewTrans(self, **kwargs):

        logging.info("Initializing coordinate transformation.")

        self.backup = False

        self.initStructure()

        for name, value in kwargs.iteritems():
            self.addParameter(name, value)

        self.unsavedTrans = True


        
    def calcTrans(self, inCoords, outCoords):
        
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
        if not quiet:
            logging.info(\
                "Determining coord. transf. using {0} sources.".format(m))
        if m<4:
            logging.error("Number of sources <4!")
            return


        ## create ndarrays for fitting
        inCoordArr  = np.zeros((2, m), dtype=np.float32)
        inCoordArr[0] = inCoords["x"][inSel]
        inCoordArr[1] = inCoords["y"][inSel]

        outCoordArr = np.zeros((2, m, ndisp), dtype=np.float32)
        outCoordArr[0] = outCoords["x"][inSel]
        outCoordArr[1] = outCoords["y"][inSel]

        if err:
            outSigmaArr = np.zeros(outCoordArr.shape, dtype=np.float32)
            outCoordArr[0] = outCoords["x_err"][inSel]
            outCoordArr[1] = outCoords["y_err"][inSel]
        else:
            outSigmaArr = None

        coordTrans = quickFit(inCoordArr, outCoordArr, outSigma=outSigmaArr)[0]

        self.initStructure()

        for i in xrange(self.npars):
            self.addParameter(name=self.parameters[i], value=coordTrans[i])


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
                    int(incoords['id'][J]),
                    float(x-outCoordArr[0,K].mean(axis=-1)),
                    float(y-outCoordArr[1,K].mean(axis=-1))))



    def readTransHDU(self, HDU, backup=False):

        logging.info(\
            "Reading coord. transf. from HDU '{0}'.".format(\
                HDU.header["EXTNAME"]))


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

        self.parameters = ["A", "B", "C", "D"]
        self.npars      = len(self.parameters)
        self.disp       = False
        self.lockSize   = False ## Cannot resize after call to makeParameters

        self.value      = np.zeros(1, {"names":self.parameters,
                                       "formats":self.npars*[np.float32]})
        self.freepar    = np.zeros(1, {"names":self.parameters,
                                       "formats":self.npars*[np.bool]})
        self.isfitted   = np.zeros(1, {"names":self.parameters,
                                       "formats":self.npars*[np.bool]})
        self.polyfit    = np.zeros(1, {"names":self.parameters,
                                       "formats":self.npars*[np.float32]})
        if self.backup:
            self.value_init = np.zeros(1, {"names":self.parameters,
                                           "formats":self.npars*[np.float32]})
        self.missing    = [name for name in self.parameters]
 

        
    @property
    def ndisp(self):
        return self.value.size



    def addParameter(self, name, value, freepar=False, isfitted=False,
                     polyfit=None):

        if not name in self.missing:
            logging.error(\
                "Cannot add parameter '{0}', it is not missing.".format(name))
            return

        value = np.asarray(value)
        if isfitted:
            if polyfit==None:
                logging.error(\
                    "No `polyfit' available, but `isfitted'-flag set.")
                return
            polyfit = np.asarray(polyfit)
            if polyfit.size != value.size:
                logging.error("Incompatible sizes of `value' and `polyfit'.")
                return
        
        if value.size > self.value.size:
            if self.disp:
                logging.warning("Differing data size of parameter '{0}'.")
                logging.warning("Transf. structure will be resized.")
            self.resizeStructure(size=value.size)
        elif value.size < self.value.size and value.size > 1:
            logging.error(\
                "Cannot add parameter '{0}', it has invalid data size.")
            return

        self.value[name] = value
        if freepar  != None: self.freepar[name]  = freepar
        if isfitted != None: self.isfitted[name] = isfitted
        if polyfit  != None: self.polyfit[name]  = polyfit
        if self.backup: self.value_init[name] = value
        self.missing.remove(name)

        logging.info(\
            "Added parameter '{0}' to transf. definition.".format(name))

        self.unsavedTrans = True



    @property
    def verifyTrans(self):

        if not "parameters" in dir(self):
            return False

        if self.missing != []:
            logging.warning("Some transf. parameters have not been specified:")
            for prm in self.missing:
                logging.warning(\
                    "   Missing parameter '{0}'".format(", ".join(prm)))
                return False
        else:
            return True



    def info(self):
        
        logging.info("Summary of coordinate transformation instance:")
        logging.info(" Valid: {0}".format(self.verifyTrans))
        logging.info(" Parameter  defined?  free?  polyfit?")

        for i in xrange(self.npars):
            name = self.parameters[i]

            logString = " *{0:8s}".format(name)
            if not name in self.missing:
                logString += "      [x]"
            else:
                logstring += "      [ ]     [ ]       [ ]\n"
                logging.info(logString)
                continue

            if self.freepar[name]: logString += "     [x]"
            else: logString += "     [ ]"

            if self.isfitted[name]: logString += "       [x]"
            else: logString += "       [ ]"

            logging.info(logString)
                


    def provideParNames(self):

        return QtCore.QStringList(self.parameters)



    def resizeStructure(self, size):
        
        if self.lockSize:
            logging.error("Cannot resize after calling or modifying instance.")
            return

        if self.value.size == size:
            return

        oldsize = self.value.size
        if oldsize == 1 and size > 1:
            self.disp = True

        self.value.resize(size)
        for prm in self.value.dtype.names:
            self.value[prm][oldsize:] = self.value[prm][oldsize-1]

        self.polyfit.resize(size)

        if self.backup:
            self.value_init.resize(size)
            for prm in self.value_init.dtype.names:
                self.value_init[prm][oldsize:] = self.value_init[prm][oldsize-1]

        self.unsavedTrans = True


        
    def makeParameters(self):

        if not self.verifyTrans:
            logging.error(" Definition of coord. trans. is not valid.")
            return

        ## value array
        self.valueView = self.value.view(self.value["A"].dtype)\
            .reshape(len(self.value), -1)

        ## freepar & isfitted
        self.freeparView = self.freepar.view(self.freepar["A"].dtype)\
            .reshape(-1,)
        self.isfittedView = self.isfitted.view(self.isfitted["A"].dtype)\
            .reshape(-1,)

        ## Parameter A
        if self.isfitted["A"]:
            self.A = self.polyfit["A"].view(self.polyfit["A"].dtype).reshape(-1)
        else:
            self.A = self.value["A"].view(self.value["A"].dtype).reshape(-1)

        ## Parameter B
        if self.isfitted["B"]:
            self.B = self.polyfit["B"].view(self.polyfit["B"].dtype).reshape(-1)
        else:
            self.B = self.value["B"].view(self.value["B"].dtype).reshape(-1)

        ## Parameter C
        if self.isfitted["C"]:
            self.C = self.polyfit["C"].view(self.polyfit["C"].dtype).reshape(-1)
        else:
            self.C = self.value["C"].view(self.value["C"].dtype).reshape(-1)

        ## Parameter D
        if self.isfitted["D"]:
            self.D = self.polyfit["D"].view(self.polyfit["D"].dtype).reshape(-1)
        else:
            self.D = self.value["D"].view(self.value["D"].dtype).reshape(-1)
            
        self.lockSize = True
        
        
    
    @property    
    def toUpdate(self):
        return self.freeparView&(self.isfittedView==False)



    @property
    def Fit(self):

        if not self.verifyTrans:
            return False
        
        return self.toUpdate.any()



    def quickFit(self, inCoords, outCoords, outSigma=None):
        
        m = inCoords.shape[1]
        M = 2*m

        A = np.ones((M,4), dtype=np.float32)
        A[:m,0] =  inCoords[0].flatten()
        A[m:,0] =  inCoords[1].flatten()
        A[:m,1] =  inCoords[1].flatten()
        A[m:,1] = -inCoords[0].flatten()
        A[m:,2] = 0.
        A[:m,3] = 0.

        sigma = None
        if outCoords.ndim == 2 or outCoords.shape[-1] == 1:
            b = outCoords.flatten()
            if outSigma != None:
                sigma = outSigma.flatten()
        else:
            b = outCoords.reshape(-1, outCoords.shape[-1])
            if outSigma != None:
                sigma = outSigma.reshape(-1, outCoords.shape[-1])

        return myLstsqSigma(A, b, sigma=sigma)



    def viewParameter(self, name):

        if not self.lockSize:
            self.makeParameters()

        if not name in self.parameters:
            logging.error("Unknown transf. parameter: {0}".format(name))
            return None

        i = self.parameters.index(name)

        if not self.isfittedView[i]:
            return self.valueView[:,i], None
        else:
            return self.valueView[:,i], self.polyfitView[:,i]



    def viewValue(self, index=None):
        
        if not self.lockSize:
            self.makeParameters()

        if index != None:
            index = np.asarray(index)
            if index.min()<0 or index.max()>=self.valueView.shape[0]:
                logging.error("Cannot view coordinate transformation:")
                logging.error(" Index/indices out of valid range [0,{0})."\
                                  .format(self.value.size))
            
            return self.valueView[index]
        else:
            return self.valueView



    def putValue(self, data, index=None):

        if not self.lockSize:
            self.makeParameters()

        if np.atleast_2d(data).shape[1] != self.valueView.shape[1]:
            logging.error("Cannot update coordinate transformation:")
            logging.error(" Data size does not match number of parameters.")
            return
        
        if index != None:
            index = np.asarray(index)
            if index.min()<0 or index.max()>=self.valueView.shape[0]:
                index = np.asarray(values)
                logging.error("Cannot update coordinate transformation:")
                logging.error(" Index/indices out of valid range [0,{0})."\
                                  .format(self.value.size))
                return
            if index.size != np.atleast_2d(data).shape[0]:
                logging.error("Cannot update coordinate transformation:")
                logging.error(" Data size does not match number of indices")
                return

            self.valueView[index,self.toUpdate] = data[self.toUpdate]

        else:
            if not data.shape[0] == self.valueView.shape[0]:
                logging.error("Cannot update coordinate transformation:")
                logging.error(" Data size does not match size of parameters.")
                return

            self.valueView[:,self.toUpdate] = data[self.toUpdate]

        self.unsavedTrans = True

            

    def __call__(self, x, y, index=None):

        if not self.verifyTrans:
            return None

        if not self.lockSize:
            self.makeParameters()

        x = np.asarray(x)
        y = np.asarray(y)
        if index != None:
            index = np.asarray(index)

        if x.size != y.size:
            logging.error("Length of provided x- & y-coords differ.")
            return None
        
        if index != None:
            if index.min()<0 or index.max()>=self.value.size: 
                logging.error("Cannot call coordinate transformation:")
                logging.error(" Index/indices out of valid range [0,{0})."\
                                  .format(self.ndisp))
                return None

        if x.size > 1:
            if index == None:
                xOut = self.A*x[:,np.newaxis] +self.B*y[:,np.newaxis] +self.C
                yOut = self.A*y[:,np.newaxis] -self.B*x[:,np.newaxis] +self.D
            else:
                xOut = self.A[index]*x +self.B[index]*y + self.C[index]
                yOut = self.A[index]*y -self.B[index]*x + self.D[index]

            return np.dstack((xOut.T, yOut.T)).T
        else:
            if index == None:
                xOut = self.A*x + self.B*y + self.C
                yOut = self.A*y - self.B*x + self.D
            else:
                xOut = self.A[index]*x + self.B*y + self.C[index]
                yOut = self.A[index]*y - self.B*x + self.D[index]

            return np.rollaxis(np.dstack((xOut, yOut)), 2, 0)



    def makeTransHDU(self,  includePolyFit=True):
        
        if self.unsavedTrans:
            logging.info("Saving updated coord. trans. to FITS BinTable-HDU.")
        else:
            logging.warning(\
                "Saving unchanged coord. trans. to FITS BinTable-HDU.")


        if not self.disp: 
            dataFmt = "E"
        else:
            dataFmt = "{0}E".format(self.value.size)

        outColumns = [
            pyfits.Column(name="name", format="1A", array=self.parameters),
            pyfits.Column(name="value", format=dataFmt),
            pyfits.Column(name="freepar", format="L")]

        if includePolyFit:
            outColumns.extend([
                    pyfits.Column(name="isfitted", format="L"),
                    pyfits.Column(name="polyfit", format=dataFmt)])
            
        if self.backup:
            outColumns.append(
                pyfits.Column(name="value_init", format=dataFmt))

        outHDU = pyfits.new_table(outColumns, header=pyfits.Header(\
                cards=[pyfits.Card(key="EXTNAME", value="POSPARS")]))

        for i in xrange(self.npars):
            name = outHDU.data.field("name")[i]

            outHDU.data.field("value")[i] = self.value[name]
            outHDU.data.field("freepar")[i] = self.freepar[name]
            if includePolyFit:
                outHDU.data.field("isfitted")[i] = self.isfitted[name]
                outHDU.data.field("polyfit")[i] = self.polyfit[name]
            if self.backup:
                outHDU.data.field("value_init")[i] = self.value_init[name] 

        outHDU.add_checksum()

        self.unsavedTrans = False

        return outHDU  



class Positions(CoordinateTransformation):

    def __init__(self, sources, hdu=False):

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
