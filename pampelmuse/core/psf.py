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
2012/03/06, v0.2.0.0, signigicantly updated the `PSF'-class by implementing
                      the new `ValueArray'-class that allows for a strait-
                      forward access of the PSF parameters taking into account
                      their individual properties (i.e whether they are free
                      fit parameters or have been replaced by a polynomial
                      fit.)
2012/03/08, v0.2.1.0, some debugging wrt. polynomial fitting of parameters
2012/03/20, v0.3.0.0, replaced lambda-functions to initialize PSF since those
                      do not work with multithreading.
2012/06/19, v0.4.0.0, modified the code so that it is compatible with the
                      changed definition of the PSF profiles.
2012/07/03, v0.4.1.0, bugfix: renaming of old `fwhm`-parameter did not work
                              in previous versions.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.4.0.0"

import logging, os
import numpy as np, pyfits
from PyQt4 import QtCore

## required PampelMuse-packages
from pampelmuse.core.my_profiles import Gaussian, Moffat, Lorentz, GauMoff
from pampelmuse.functions.step_poly_fit import StepPolyFit
from pampelmuse.core.value_array import ValueArray

PROFILES  = {"moffat":  [         "beta", "Mfwhm",          "e", "theta"],
             "gauss":   ["Gfwhm",                           "e", "theta"],
             "lorentz": ["Lfwhm",                           "e", "theta"],
             "gaumoff": ["Gfwhm", "beta", "Mfwhm", "Mfrac", "e", "theta"]}


class PSF(object):

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
        """
        The method is used to prepare the required arrays containing the
        properties of the PSF parameters. It should be called immediately
        after a PSF has been initialized and before any parameters are
        added.
        """
        if not "psftype" in dir(self):
            logging.error("No PSF profile has been loaded/initialized.")
            return False

        self.parameters = PROFILES[self.psftype]
        self.npars      = len(self.parameters)
        self.disp       = False
        self.lockSize   = False ## Cannot resize after call to makeParameters

        self.data = np.zeros((2,self.npars,1),  dtype=np.float32)
        self.free = np.zeros((self.npars,),  dtype=np.bool)
        self.poly = np.zeros((self.npars,),  dtype=np.bool)
        
        self.missing    = [name for name in PROFILES[self.psftype]]



    def addParameter(self, name, value, freepar=False, isfitted=False,
                     polyfit=None):

        if name == "fwhm": ## adapt new parameter naming convention
            name = "{0}fwhm".format(str.upper(self.psftype[0]))

        if not name in self.missing:
            logging.error("Cannot add parameter: '{0}' not missing.".format(\
                    name))
            return

        i = self.parameters.index(name)

        ## check if value has correct data shape, reshape self.data if needed 
        value = np.asarray(value)
        if value.size > self.data.shape[-1]:
            if self.disp:
                logging.warning("Differing data size of parameter '{0}'.")
                logging.warning("PSF structure will be resized.")
            self.resizeStructure(ndisp=value.size)
        elif value.size < self.data.shape[-1] and value.size > 1:
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
        
        self.data[0,i] = value
        self.free[i]   = freepar
        self.poly[i]   = isfitted
        if polyfit    != None: self.data[1,i]  = polyfit
##         if self.backup: self.value_init[name] = value

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
                logging.warning("   Missing parameter '{0}'".format(prm))
            return False
        else:
            return True



    def info(self):
        
        logging.info("Summary of PSF instance:")
        logging.info(" Type: '{0}'".format(self.psftype))
##         logging.info(" Valid: {0}".format(self.verifyPSF))
        logging.info(" Parameter  defined?  free?  polyfit?")

        for i in xrange(self.npars):
            name = self.parameters[i]

            logString = " *{0:8s}".format(name)
            if not name in self.missing:
                logString += "      [x]"
            else:
                logString += "      [ ]     [ ]       [ ]"
                logging.info(logString)
                continue

            if self.free[i]: logString += "     [x]"
            else: logString += "     [ ]"

            if self.poly[i]: logString += "       [x]"
            else: logString += "       [ ]"

            logging.info(logString)
                


    def provideParNames(self):

        return QtCore.QStringList(self.parameters)



    def resizeStructure(self, ndisp):
        
        if self.lockSize:
            logging.error("Cannot resize after calling `makeValues'.")
            logging.error("To resize, call `delValues' first.")
            return

        if self.data.shape[-1] == ndisp:
            return

        if self.data.shape[-1] == 1 and ndisp > 1:
            self.disp = True

        kMax = min(ndisp, self.data.shape[-1])-1
        newData = np.zeros((self.data.shape)[:2]+(ndisp,),  dtype=np.float32)
        newData[...,:kMax] = self.data[...,:kMax]
        if ndisp > self.data.shape[-1]:
            newData[...,kMax:] = self.data[...,-1][:,:,np.newaxis]
        self.data = newData


##         if self.backup:
##             Needs to be implemented

        
        self.unsavedPSF = True
                

    
    def makeValues(self):

        if not self.verifyPSF:
            logging.error(" PSF definition is not valid.")
            return

        self.values = ValueArray(self.data, free=self.free, poly=self.poly)

        self.lockSize = True


        
    def delValues(self):
        
        if not self.lockSize:
            return

        del self.values
        self.lockSize = False

        logging.warning("Deleted attribute `values', unlocked data size.")



    @property
    def Fit(self):

        if not self.verifyPSF:
            return False

        return (self.free&(self.poly==False)).any()



    def Moffat(self, index, osfactor=5, osradius=2):

        return Moffat(beta =self.values[0,index],
                      Mfwhm=self.values[1,index],
                      e    =self.values[2,index],
                      theta=self.values[3,index],
                      osfactor=osfactor,
                      osradius=osradius)



    def Gaussian(self, index, osfactor=5, osradius=2):

        return Gaussian(Gfwhm=self.values[0,index],
                        e    =self.values[1,index],
                        theta=self.values[2,index],
                        osfactor=osfactor,
                        osradius=osradius)



    def Lorentz(self, index, osfactor=5, osradius=2):

        return Lorentz(Lfwhm=self.values[0,index],
                       e    =self.values[1,index],
                       theta=self.values[2,index],
                       osfactor=osfactor,
                       osradius=osradius)



    def GauMoff(self, index, osfactor=5, osradius=2):

        return GauMoff(Gfwhm=self.values[0, index],
                       beta =self.values[1, index],
                       Mfwhm=self.values[2, index],
                       Mfrac=self.values[3, index],
                       e    =self.values[4, index],
                       theta=self.values[5, index],
                       osfactor=osfactor,
                       osradius=osradius)


    def  __call__(self, index=0, osradius=2, osfactor=5):

        if not self.verifyPSF:
            logging.error("Cannot call PSF instance:")
            logging.error(" No valid PSF profile avalable.")
            return None

        if not self.lockSize:
            self.makeValues()


        if index > self.data.shape[-1]:
            logging.error("Cannot call PSF instance:")
            logging.error(" Index out of valid range [0,{1}).".format(\
                    self.values.size))
            return None

        if self.psftype == "moffat": return self.Moffat(\
            index=index,osradius=osradius,osfactor=osfactor)
        elif self.psftype == "lorentz": return self.Lorentz(\
            index=index,osradius=osradius,osfactor=osfactor)
        elif self.psftype == "gauss": return self.Gaussian(\
            index=index,osradius=osradius,osfactor=osfactor)
        elif self.psftype == "gaumoff": return self.GauMoff(\
            index=index,osradius=osradius,osfactor=osfactor)




    def polyFitParameter(self, name, order, bins, mask=None):

        if not self.lockSize:
            self.makeValues()

        if not name in self.parameters:
            logging.error("Unknown PSF parameter: {0}".format(name))
            return None
        
        i = self.parameters.index(name)

        self.data[1,i] = StepPolyFit(self.data[0,i,:], order=order, bins=bins,
                                     mask=mask)
        self.poly[i]   = True
        
        self.unsavedPSF = True
        


    def makePSFHDU(self,  includePolyFit=True):
        
        if self.unsavedPSF:
            logging.info("Saving updated PSF profile to FITS BinTable-HDU.")
        else:
            logging.warning(\
                "Saving unchanged PSF profile to FITS BinTable-HDU.")

        if not self.disp:
            dataFmt = "E"
        else:
            dataFmt = "{0}E".format(self.data.shape[-1])

        outColumns = [
            pyfits.Column(name="name", format="10A", array=self.parameters),
            pyfits.Column(name="value", format=dataFmt,
                          array=np.squeeze(self.data[0])),
            pyfits.Column(name="freepar", format="L", array=self.free)
            ]

        if includePolyFit:
            outColumns.extend([\
                    pyfits.Column(\
                        name="isfitted", format="L", array=self.poly),
                    pyfits.Column(\
                        name="polyfit", format=dataFmt,
                        array=np.squeeze(self.data[1]))])
            
        outHDU = pyfits.new_table(outColumns, header=pyfits.Header(\
                cards=[pyfits.Card(key="EXTNAME", value="PSFPARS")]))

        outHDU.header.update("PSFTYPE", self.psftype)
        outHDU.add_checksum()

        self.unsavedPSF = False

        return outHDU            
