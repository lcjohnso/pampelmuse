"""
pampelmuse.config
=================

Provides
--------
class pampelmuse.config.Namespace
class pampelmuse.config.PampelMuseConfig

Purpose
-------
The module implements two classes that are used to carry the configuration
parameters of PampelMuse through the analysis.
The class 'Namespace' is, as the name suggests, a pure namespace, suited
to store, update and read attributes. For each subroutine of PampelMuse, an
individual instance of Namespace will be created.
The class 'PampelMuseConfig' is also designed as a pure namspace. However,
it defines two additional methods PampelMuseConfig.readHeader() and
PampelMuseConfig.makeHeade() that can be used to read a configuration from
a FITS HDU and to store the current configuration of PampelMuse again in
the header of a FITS HDU.

Latest SVN revision
-------------------
231, 2015/01/22
"""
__author__   = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 231


import datetime, os, re
from astropy.io import fits

# required PampelMuse routines
from xml_interact import ModifyConfig, transform_string


class Namespace(): pass


class PampelMuseConfig(Namespace):

    def __init__(self):
        """
        Initialize an instance of class 'PampelMuseConfig'
        """
        
        self.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
        
        # the following list contains all the keywords that can currently be changed during the analysis, i.e. when
        # calling a routine different from INITFIT or SINGLESRC:
        self.changeable = ["verbose", "interact", "cpucount", "maxiter", "polytype", "polydeg", "subunres", "subsky", "usesigma"]
        
        
    def read_defaults(self):
        """
        This method reads the default configuration from the XML file that is
        part of PampelMuse and stored in the data directory of this package.
        """
        
        __currentdir__ = os.path.abspath(os.path.dirname(__file__))
        
        config_reader = ModifyConfig(os.path.join(__currentdir__, "data", "PMparams.xml"), os.path.join(__currentdir__, "data", "PMschema.xsd"))
        
        for key, value in config_reader.get_all().iteritems():
            setattr(self, key.lower(), value)
        
        
    def readHeader(self, hdu):
        """
        Read a configuration from the header of a FITS HDU.
        
        Parameters
        ----------
        hdu : an astropy.io.fits HDU instance
            The HDU from which the configuration should be read. The routine
            will scan the header of the HDU for hierarch keywords starting
            with PAMPELMUSE.
        """
        self.configFinder = re.compile(r"PAMPELMUSE\s([A-Z]+)\s?([A-Z]\w+)?$")
        
        for entry in hdu.header.cards:   
            match = self.configFinder.match(entry.keyword)
            if match is not None:
                if match.group(2) is None: setattr(self, match.group(1).lower(), entry.value)
                else:
                    if not hasattr(self, match.group(1).lower()): # create sub-namespace if it does not exist
                        setattr(self, match.group(1).lower(), Namespace())
                    setattr(self.__dict__[match.group(1).lower()],  match.group(2).lower(), entry.value) # add parameter to sub-namespace
        
        
    def makeHeader(self, hdu=fits.PrimaryHDU()):
        """
        Save the current configuration in a astropy.io.fits.Header instance.
        
        Parameters
        ----------
        hdu : an astropy.io.fits HDU instance, optional
            The HDU used to store the configuration, if none is provided,
            an empty HDU will be created.
        """
        for entry in ["revision", "utc", "verbose", "ifu", "interact", "useqt", "cpucount", "smallifu", "passband", "ccmaxoff", "aperrad", "apernois",
                      "reslim", "updaterl", "confdens", "snrcut", "binrad", "useunres", "unresbin", "globsky", "start", "stop", "psfrad",
                      "jumpstart", "usesigma", "maxiter", "polytype", "polydeg", "subunres", "subsky"
                      ]:
           
            if hasattr(self, entry):
                hdu.header["hierarch PAMPELMUSE {0}".format(entry.upper())] = getattr(self, entry)
         
        if hasattr(self, "ifs_data"):
            if hasattr(self.ifs_data, "instrument"):
                hdu.header["hierarch PAMPELMUSE IFU"] = getattr(self.ifs_data.instrument, "name")
         
        if hasattr(self, "psf"):
            if hasattr(self.psf, "profile"):
                hdu.header["hierarch PAMPELMUSE PSF PROFILE"] = getattr(self.psf, "profile")
         
            i = 0
            while hasattr(self.psf, "parname{0}".format(i)):
                hdu.header["hierarch PAMPELMUSE PSF PARNAME{0}".format(i)] = getattr(self.psf, "parname{0}".format(i))
                hdu.header["hierarch PAMPELMUSE PSF PARVALUE{0}".format(i)] = getattr(self.psf, "parvalue{0}".format(i))
                hdu.header["hierarch PAMPELMUSE PSF FREEPAR{0}".format(i)] = getattr(self.psf, "freepar{0}".format(i))
                i += 1


        if hasattr(self, "initfit"):
            for entry in ["revision", "utc", "ccxoff", "ccyoff", "reslim", "photzero"]:
                if hasattr(self.initfit, entry):
                    hdu.header["hierarch PAMPELMUSE INITFIT {0}".format(entry.upper())] = getattr(self.initfit, entry)

        if hasattr(self, "cubefit"):
            for entry in ["revision", "utc", "status", "lastlayer"]:
                if hasattr(self.cubefit, entry):
                    hdu.header["hierarch PAMPELMUSE CUBEFIT {0}".format(entry.upper())] = getattr(self.cubefit, entry)
                    
        if hasattr(self, "polyfit"):
            for entry in ["revision", "utc"]:
                if hasattr(self.polyfit, entry):
                    hdu.header["hierarch PAMPELMUSE POLYFIT {0}".format(entry.upper())] = getattr(self.polyfit, entry)

        if hasattr(self, "subtres"):
            for entry in ["revision", "utc"]:
                if hasattr(self.subtres, entry):
                    hdu.header["hierarch PAMPELMUSE SUBTRES {0}".format(entry.upper())] = getattr(self.subtres, entry)

        if hasattr(self, "shotnoi"):
            for entry in ["revision", "utc"]:
                if hasattr(self.shotnoi, entry):
                    hdu.header["hierarch PAMPELMUSE SUBTRES {0}".format(entry.upper())]  = getattr(self.shotnoi, entry)

        if hasattr(self, "addwcs"):
            for entry in ["revision", "utc", "RADECSYS", "EQUINOX", "WCSDIM", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2",
                          "CD2_1", "CD2_2"]:
                if hasattr(self.addwcs, entry.lower()):
                    hdu.header["hierarch PAMPELMUSE ADDWCS {0}".format(entry.upper())] = getattr(self.addwcs, entry.lower())
                    
        return hdu
