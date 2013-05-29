"""
Two classes are defined that are used to carry the configuration
parameters of PampelMuse through the analysis.
Namespace is, as the name suggests, a pure namespace, suited to store,
update and read attributes. For each subroutine of PampelMuse, an
individual instance of Namespace will be created.
PampelMuseConfig is also designed as a pure namspace. However, it defines
additional methods 'readHeader' and 'makeHeader' that can be used to read a
configuration from a pyfits HDU and to store the current configuration of
PampelMuse again in the header of a pyfits HDU.
"""
__version__ = "0.1" ## 2013/05/26
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"

import datetime, re
import pyfits


class Namespace(): pass


class PampelMuseConfig(Namespace):

    def __init__(self):
        """
        Initialize an instance of class 'PampelMuseConfig'
        """
        
        self.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
        
        
    def readHeader(self, hdu):
        """
        Read a configuration from the header of a pyfits HDU.
        
        Parameters
        ----------
        hdu : a pyfits HDU instance
            The HDU from which the configuration should be read. The routine
            will scan the header of the HDU for hierarch keywords starting
            with PAMPELMUSE.
        """
        self.configFinder = re.compile(r"PAMPELMUSE\s([A-Z]+)\s?([A-Z]+)?$")
        
        for entry in hdu.header.ascardlist():   
            match = self.configFinder.match(entry.key)
            if match != None:
                if match.group(2) == None: setattr(self, match.group(1).lower(), entry.value)
                else:
                    if not hasattr(self, match.group(1).lower()): ## create sub-namespace if it does not exist
                        setattr(self, match.group(1).lower(), Namespace())
                    setattr(self.__dict__[match.group(1).lower()],  match.group(2).lower(), entry.value) ## add parameter to sub-namespace
        
        
    def makeHeader(self, hdu=pyfits.PrimaryHDU()):
        """
        Save the current configuration in a pyfits.Header instance.
        
        Parameters
        ----------
        hdu : a pyfits HDU instance, optional
            The HDU used to store the configuration, if none is provided,
            an empty HDU will be created.
        """
        for entry in ["version", "utc", "ifu", "start", "stop", "smallifu", "globsky", "passband", "reslim", "updaterl", "confdens",
                      "snrcut", "binrad", "useunres", "unresbin", "psfrad", "maxiter", "cpucount", "jumpstart",  "subunres", "subsky",
                      "verbose", "useqt", "aperrad", "apernois"]:
           
            if hasattr(self, entry):
                hdu.header.update("hierarch PAMPELMUSE {0}".format(entry.upper()), getattr(self, entry))
         
        if hasattr(self, "psf"):
            if hasattr(self.psf, "profile"):
                hdu.header.update("hierarch PAMPELMUSE PSF PROFILE",  getattr(self.psf, "profile"))
         
            i = 0
            while hasattr(self.psf, "parname{0}".format(i)):
                hdu.header.update("hierarch PAMPELMUSE PSF PARNAME{0}".format(i), getattr(self.psf, "parname{0}".format(i)))
                hdu.header.update("hierarch PAMPELMUSE PSF PARVALUE{0}".format(i), getattr(self.psf, "parvalue{0}".format(i)))
                hdu.header.update("hierarch PAMPELMUSE PSF FREEPAR{0}".format(i), getattr(self.psf, "freepar{0}".format(i)))
                i += 1


        if hasattr(self, "initfit"):
            for entry in ["version", "utc", "reslim", "photzero"]:
                if hasattr(self.initfit, entry):
                    hdu.header.update("hierarch PAMPELMUSE INITFIT {0}".format(entry.upper()), getattr(self.initfit, entry))

        if hasattr(self, "cubefit"):
            for entry in ["version", "utc", "status", "lastlayer"]:
                if hasattr(self.cubefit, entry):
                    hdu.header.update("hierarch PAMPELMUSE CUBEFIT {0}".format(entry.upper()), getattr(self.cubefit, entry))
                    
        if hasattr(self, "polyfit"):
            for entry in ["version", "utc"]:
                if hasattr(self.polyfit, entry):
                    hdu.header.update("hierarch PAMPELMUSE POLYFIT {0}".format(entry.upper()), getattr(self.polyfit, entry))

        return hdu
