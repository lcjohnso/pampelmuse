"""
variables.py
============

Provides
--------
The class pampelmuse.core.psf.variables.Variables

Purpose
-------
The module handles the parameters of analytical PSF profile that is used by
PampelMuse. To this aim, it provides the class 'Variables' as an interface
to load, update, provide, and store the PSF parameters. 'Variables' inherits
from the class pampelmuse.core.parameters.Parameters that defines a more
general framework for dealing with parameters within PampelMuse.

Latest SVN revision
-------------------
169, 2014/04/29
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 169

import logging, os, re
import numpy as np
from numpy.polynomial import Chebyshev, Legendre
from astropy.io import fits

## required PampelMuse-packages
from pampelmuse.core.parameters import Parameters



class Variables(Parameters):
    """
    The class Variables serves as an interface to the parameters that define
    the PSF profile. The core of the class is an array that contains the value
    of each PSF parameter as a function of wavelength; and potentially an
    analytical fit to it. Several methods are defined to read in the PSF
    parameters, access them, update them, and write them out to a FITS HDU.

    The analytical PSF that are currently supported are 'moffat' for a Moffat-
    shaped PSF and 'gauss' for a Gaussian-shaped PSF.
    
    The class inherits most of its functionality from the super-class
    pampelmuse.core.parameters.Parameters. However, some functionality that is
    specific to the PSF parameters is defined here.
    """

    ## define a dictionary containing the parameter names for each PSF profile supported by PampelMuse
    PROFILES  = {"moffat": [        "beta", "fwhm", "e", "theta"],
                 "gauss":  ["fwhm",                 "e", "theta"],
                }
                
    LABELS = {"fwhm":r"$\mathrm{FWHM}\ [\mathrm{spaxel}]$", "beta":r"$\beta$", "e":r"$e$", "theta":r"$\theta\ [\mathrm{rad}]$"}

    def __init__(self, profile=None, hdu=None, **kwargs):
        """
        Initialize an instance of pampelmuse.core.psf.attributes.Parameters
        
        Parameters
        ----------
        profile : string, optional
            The name of the profile to be initialized. Currently "moffat"
            and "gauss" are supported. Note that if 'profile' is None, the
            argument 'hdu' must not be None and provide the name of the 
            profile instead.
        hdu : an astropy.io.fits.BinTableHDU instance
            The FITS-Table HDU that contains the PSF Parameters. Unless 'profile'
            is not None, a keyword 'PSFTYPE' must be available in the header and
            its value must be the name of a supported analytical PSF profile. The
            table must contain fields named 'name' and 'value', providing the
            names and vaues, respectively, of the individual parameters that
            define the profile. Additional fields that are used when available
            are 'isfreepar', indicating whether the parameter should be varied
            when the PSF is fitted, 'isfitted', indicating whether the parameter
            has been fitted as a function of wavelength, and 'polyfit', containing
            the fit (if any).
        kwargs
            Any additional keyword arguments are passed on to the call  of the
            initialization method of the super-class. They might be used to provide
            the names and values of the parameters of the profile.
        """
        logging.info("Initializing PSF parameters...")
        
        ##  if a profile but no hdu is provided, required PSF parameters that are not provided as keyword arguments are initialized to NaN
        if profile != None and hdu == None and profile in self.PROFILES:
            kwargs["order"] = self.PROFILES[profile] ## make sure PSF variables are in correct order
            for name in self.PROFILES[profile]:
                if not name in kwargs: kwargs[name] = np.nan
        
        ## call initialization method of super-class.
        super(Variables, self).__init__(hdu=hdu, **kwargs)
    
        ##  check if PSF profile has been defined (HDU header key supersedes keyword argument 'profile')
        self.profile = profile
        if self.hdu != None and "PSFTYPE" in self.hdu.header:
            self.profile = self.hdu.header["PSFTYPE"]
        if self.profile is None:
            if self.hdu == None or "PSFTYPE" not in self.hdu.header:
                logging.critical("Cannot infer PSF type: Parameter 'profile' not set.")
            
        ## check if PSF profile is supported
        if self.profile in self.PROFILES:
            logging.info("PSF is of type '{0}'".format(self.profile.upper()))
        else:
            logging.critical("Invalid PSF type: '{0}'".format(profile))
            logging.critical("Supported types are: '{0}'".format("', '".join([psftype for psftype in self.PROFILES.keys()])))
        assert self.profile in self.PROFILES, "PSF of type '{0}' not supported.".format(self.profile)

        ## check if all required parameters are available
        for name in self.PROFILES[self.profile]:
            if not name in self.names:
                ## in earlier versions (<0.9?) different variable names were used for the FWHM of a Moffat (Mfwhm) or a Gaussian (Gfwhm) profile,
                ## so the presence of one of these names is checked.  
                if name == "fwhm" and np.in1d(self.names, ["Gfwhm", "Mfwhm"]).any(): 
                    if "Gfwhm" in self.names: i=self.names.index("Gfwhm")
                    elif "Mfwhm" in self.names: i = self.names.index("Mfwhm")
                    logging.warning("Cannot find PSF parameter 'fwhm', using '{0}' instead.".format(self.names[i]))
                    self.names[i] = "fwhm"
                    continue
                logging.critical("Missing required PSF parameter '{0}'.".format(name))
            assert name in self.names, "Cannot find '{0}' among available PSF parameters.".format(name)



    def info(self):
        """
        Print out some information about the current status of the parameters,
        i.e. whether they have been defined, what is their (mean) value, are
        they free parameters, and have the been fitted with a smooth function.
        """
        
        ## print out some information about the PSF profile in use
        logging.info("Summary of current status of PSF parameters:")
        logging.info(" Type: '{0}'".format(self.profile))

        ## call info method of super-class. Provide parameter names to get order right.
        super(Variables, self).info(self.PROFILES[self.profile])


    def make_hdu(self, **kwargs):
        """
        Write the current values of the PSF parameter to an fits.BinTableHDU
        instance. The method is basically a wrapper of the'make_hdu' instance
        defined in the super-class and adds some header keywords that provide
        information about the PSF in use.
        
        Parameters
        ----------
        kwargs
            Any keyword parameters parameters passed to this method will
            be directed to the call of the 'make_hdu' method of the super-
            class.
        """
        kwargs["extname"] = "PSFPARS"
        kwargs["psftype"] = self.profile

        logging.info("Saving PSF variables in BinTableHDU '{0}'...".format(kwargs["extname"]))

        return super(Variables, self).make_hdu(**kwargs)
