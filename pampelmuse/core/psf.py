"""
PSF
===
The routine is designed as an interface to laod, modify and store a PSF that
is used by PampelMuse. The `PSF'-class provides methods to open an existing
PSF (from a FITS-HDU), create a new PSF, view and modify the PSF parameters,
create a caller for the PSF and store an updated/created PSF in a FITS-HDU.

CHANGELOG
---------
2012/02/28, v0.1.0.0
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.0.0"

import logging, os
import numpy as np, pyfits
from PyQt4 import QtCore

## required PampelMuse-packages
from pampelmuse.core.my_profiles import Gaussian, Moffat
from pampelmuse.functions.step_poly_fit import StepPolyFit


PROFILES  = {"moffat": ["beta", "fwhm", "e", "theta"],
             "gauss":  [        "fwhm", "e", "theta"]}


class PSF:

    def makeNewPSF(self, profile, **kwargs):

        logging.info("Initializing PSF of type '{0}'.".format(profile))

        if not profile in PROFILES.keys():
            logging.critical("Invalid PSF type: {0}".format(profile))
            logging.critical("Supported types are: {0}".format(\
                    ", ".join([psftype for psftype in PROFILES.keys()])))
            raise IOError, "PSF type not understood."

        self.psftype = profile

        self.backup  = False
        
        self.unsavedPSF = True

        self.initStructure()
        self.defineFunction()

        for name, value in kwargs.iteritems():
            self.addParameter(name, value)



    def readFitsHDU(self, HDU, backup=False):

        if "EXTNAME" in HDU.header:
            logging.info("Initializing PSF from HDU '{0}'.".format(\
                    HDU.header["EXTNAME"]))
        else:
            logging.info("Initializing PSF from unnamed HDU.")


        ## determine PSF-type
        if not "PSFTYPE" in HDU.header:
            logging.critical("Cannot determine type of PSF to use.")
            logging.critical("Keyword 'PSFTYPE' missing in PSF file.")
            return

        self.psftype = HDU.header["PSFTYPE"]
        logging.info("PSF is of type '{0}'.".format(self.psftype))

        ## check if required fields 'name' and 'value exist':
        for field in ["name", "value"]:
            if not field in HDU.data.names:
                logging.critical(\
                    "Cannot find field '{0}' in PSF HDU.".format(field))
                return

        self.backup = backup

        self.initStructure()
        self.defineFunction()

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

        self.unsavedPSF = False



    def initStructure(self):

        if not "psftype" in dir(self):
            logging.error("No PSF profile has been loaded/initialized.")
            return False

        self.parameters = PROFILES[self.psftype]
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
        self.missing    = [name for name in PROFILES[self.psftype]]



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
                logging.warning("PSF structure will be resized.")
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

        self.unsavedPSF = True

        logging.info("Added parameter '{0}' to PSF definition.".format(name))



    @property
    def verifyPSF(self):

        if not "psftype" in dir(self):
            logging.error("No PSF profile has been loaded/initialized.")
            return False

        if self.missing != []:
            logging.warning("Some PSF parameters have not been specified:")
            for prm in self.missing:
                logging.warning(\
                    "   Missing parameter '{0}'".format(", ".join(prm)))
                return False
        else:
            return True



    def info(self):
        
        logging.info("Summary of PSF instance:")
        logging.info(" Type: '{0}'".format(self.psftype))
        logging.info(" Valid: {0}".format(self.verifyPSF))
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
        
        self.unsavedPSF = True
                

    
    def makeParameters(self):

        if not self.verifyPSF:
            logging.error(" PSF definition is not valid.")
            return

        ## value array
        self.valueView = self.value.view(\
            self.value[self.parameters[0]].dtype).reshape(len(self.value),-1)


        ## value array
        self.polyfitView = self.polyfit.view(\
            self.polyfit[self.parameters[0]].dtype).reshape(len(self.value),-1)
        

        ## freepar & isfitted
        self.freeparView = self.freepar.view(\
            self.freepar[self.parameters[0]].dtype).reshape(-1,)
        self.isfittedView = self.isfitted.view(\
            self.isfitted[self.parameters[0]].dtype).reshape(-1,)

        ## parameters
        if not self.isfittedView[0]: self.par0View = self.valueView[:,0]
        else: self.par0View = self.polyfitView[:,0]

        if not self.isfittedView[1]: self.par1View = self.valueView[:,1]
        else: self.par1View = self.polyfitView[:,1]

        if not self.isfittedView[2]: self.par2View = self.valueView[:,2]
        else: self.par2View = self.polyfitView[:,2]

        if len(self.parameters)>3:
            if not self.isfittedView[3]: self.par3View = self.valueView[:,3]
            else: self.par3View = self.polyfitView[:,3]

        self.lockSize = True



    @property    
    def toUpdate(self):
        return self.freeparView&(self.isfittedView==False)



    @property
    def Fit(self):

        if not self.verifyPSF:
            return False

        return self.toUpdate.any()




    def defineFunction(self):

        if self.psftype == "moffat":
            self.function = lambda index : Moffat(\
                beta =self.par0View[index],
                fwhm =self.par1View[index],
                e    =self.par2View[index],
                theta=self.par3View[index])

        elif self.psftype == "gauss":
            self.function = lambda index : Gaussian(\
                fwhm =self.par0View[index],
                e    =self.par1View[index],
                theta=self.par2View[index])




    def viewValue(self, index=None):

        if not self.lockSize:
            self.makeParameters()

        if index != None:
            index = np.asarray(index)
            if index.min()<0 or index.max()>=self.valueView.shape[0]:
                logging.error("Cannot view PSF:")
                logging.error(" Index/indices out of valid range [0,{0})."\
                                  .format(self.value.size))
            
            return self.valueView[index]
        else:
            return self.valueView



    def viewParameter(self, name):
        
        if not self.lockSize:
            self.makeParameters()

        if not name in self.parameters:
            logging.error("Unknown PSF parameter: {0}".format(name))
            return None

        i = self.parameters.index(name)

        if not self.isfittedView[i]:
            return self.valueView[:,i], None
        else:
            return self.valueView[:,i], self.polyfitView[:,i]



    def putValue(self, data, index=None):

        if not self.lockSize:
            self.makeParameters()

        if not np.atleast_2d(data).shape[1] == self.valueView.shape[1]:
            logging.error("Cannot update PSF parameters:")
            logging.error(" Data size does not match number of parameters.")
            return


        if index != None:
            index = np.asarray(index)
            if index.min()<0 or index.max()>=self.valueView.shape[0]:
                index = np.asarray(values)
                logging.error("Cannot update PSF parameters:")
                logging.error(" Index/indices out of valid range [0,{0})."\
                                  .format(self.value.size))
                return
            if index.size != np.atleast_2d(data).shape[0]:
                logging.error("Cannot update PSF parameters:")
                logging.error(" Data size does not match number of indices")
                return

            self.valueView[index,self.toUpdate] = data[self.toUpdate]

        else:
            if not data.shape[0] == self.valueView.shape[0]:
                logging.error("Cannot update PSF parameters:")
                logging.error(" Data size does not match size of parameters.")
                return

            self.valueView[:,self.toUpdate] = data[self.toUpdate]

        self.unsavedPSF = True
            


    def polyFitParameter(self, name, order, bins, mask=None):

        if not self.lockSize:
            self.makeParameters()

        if not name in self.parameters:
            logging.error("Unknown PSF parameter: {0}".format(name))
            return None
        
        i = self.parameters.index(name)

        self.polyfitView[:,i] = StepPolyFit(self.valueView[:,i], order=order,
                                            bins=bins, mask=mask)
        self.isfittedView[i]  = True
        
        if i==0: self.par0View = self.polyfitView[:,0]
        elif i == 1: self.par1View = self.polyfitView[:,1]
        elif i == 2: self.par2View = self.polyfitView[:,2]
        elif i == 3: self.par3View = self.polyfitView[:,3]

        self.unsavedPSF = True



    def  __call__(self, index=0):

        if not self.lockSize:
            self.makeParameters()

        if not self.verifyPSF:
            logging.error("Cannot call PSF instance:")
            logging.error(" No valid PSF profile avalable.")
            return None

        if index > self.value.size:
            logging.error("Cannot call PSF instance:")
            logging.error(" Index out of valid range [0,{1}).".format(\
                    self.value.size))
            return None

        return self.function(index=index)
        


    def makeFitsHDU(self,  includePolyFit=True):
        
        if self.unsavedPSF:
            logging.info("Saving updated PSF profile to FITS BinTable-HDU.")
        else:
            logging.warning(\
                "Saving unchanged PSF profile to FITS BinTable-HDU.")

        if not self.disp:
            dataFmt = "E"
        else:
            dataFmt = "{0}E".format(self.value.size)

        outColumns = [
            pyfits.Column(name="name", format="10A", array=self.parameters),
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
                cards=[pyfits.Card(key="EXTNAME", value="PSFPARS")]))

        for i in xrange(self.npars):
            name = outHDU.data.field("name")[i]

            outHDU.data.field("value")[i] = self.value[name]
            outHDU.data.field("freepar")[i] = self.freepar[name]
            if includePolyFit:
                outHDU.data.field("isfitted")[i] = self.isfitted[name]
                outHDU.data.field("polyfit")[i] = self.polyfit[name]
            if self.backup:
                outHDU.data.field("value_init")[i] = self.value_init[name] 

        outHDU.header.update("PSFTYPE", self.psftype)
        outHDU.add_checksum()

        self.unsavedPSF = False

        return outHDU            
