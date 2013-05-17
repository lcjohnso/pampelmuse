"""
This file provides the class 'Cube' that is thought to facilitate the 
opening of FITS cubes and checking them for validity.

CHANGELOG
2013/05/03 v0.1.0.0, initial version
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.1.0.0"
import logging
import numpy as np, pyfits

## required PampelMuse routines
from pampelmuse.core.instruments import Instrument, Argus, Pmas

class Cube:
    """
    Open a FITS datacube and check whether it is valid for the usage
    in PampelMuse.
    """
    
    def __init__(self, name, instrument=None, checks={}):
        """
        Initialize a new instance of the 'Cube' class.
        
        Parameters
        ----------
        name : string
            The path to the FITS cube that should be loaded.
        instrument : pampelmuse.core.instruments.Instrument instance or None, optional
            Can be used to provide informations about the instrument used to observe
            the data. Note that if not instrument information is provided, only 3dim.
            data is accepted.
        checks : dictionary, optional
            An optional list of (keyword, value)-pairs used to check the consistency
            of the datacube with previously loaded data.
        """
    
        self.name = name

        self._wave = None

        ## open datacube:
        self.file = pyfits.open(name)
    
        ## if no instrument instance provided, create empty one
        if instrument == None:
            
            ## try to get from FITS header:
            try:
                instrume = self.file[0].header["INSTRUME"]
            except KeyError:
                instrume = ""
            
            if instrume == "GIRAFFE":
                self.instrument = Argus()
            else:
                self.instrument = Instrument()
                
        else:
            self.instrument  = instrument
    


        ## check if calibration HDU for ARGUS exists
        if self.instrument.name == "argus":
            try:
                setup_index = self.file.index_of("FIBER_SETUP")
                self.instrument.read_fiber_setup(hdu=self.file[setup_index])
            except KeyError:
                logging.warning("No HDU with fibre setup found for ARGUS data.")
    
        ## open data HDU
        self.data_hdu = self.file[0]

        ## check dimensionality of data: must be 3D if transformation from spaxels
        ## to x- & y-coords unknown, otherwise 2dim. data is accepted
        self.ndim = self.data_hdu.header["NAXIS"]

        if self.instrument.transform == None and self.ndim!=3:
            logging.error(\
                "Data must be 3dim. unless mapping from spaxels to x&y specified.")
            assert self.ndim==3, "Expected 3dim. datacube, got {0}dim".format(self.ndim)

        ## Reading information about wavelength coverage
        if self.ndim == 3:
            if self.instrument.shape == None: ## set spatial shape if needed
                xdim = self.data_hdu.header["NAXIS1"]
                ydim = self.data_hdu.header["NAXIS2"]

                self.instrument.shape = (ydim, xdim)

                ## for 3dim. data, initialize transformation assuming rectangular array
                x = np.mod(np.arange(xdim*ydim, dtype=np.int16), xdim)
                y = np.arange(xdim*ydim, dtype=np.int16)//xdim
                self.instrument.transform = np.vstack((y,x)).T

            self.spectral_axis = 3
        else:
            self.spectral_axis = self.instrument.spectral_axis

        self.cdelt = self.data_hdu.header['CDELT{0}'.format(self.spectral_axis)]
        self.crval = self.data_hdu.header['CRVAL{0}'.format(self.spectral_axis)]

        try:
            self.ctype = self.data_hdu.header["CTYPE{0}".format(self.spectral_axis)]
        except KeyError:
            self.ctype = "AWAV"
        try:
            self.cunit = self.data_hdu.header["CUNIT{0}".format(self.spectral_axis)]
        except KeyError:
            self.cunit = self.ctype

        self.ndisp = self.data_hdu.header['NAXIS{0}'.format(self.spectral_axis)]

        for key, value in checks.iteritems():
            try:
                if self.__dict__[key] != value:
                    logging.error("Invalid value encountered for '{0}' in <{1}>.".format(key,  self.name))
                    assert self.__dict__[key] == value, "Keyword '{0}' should  be {1} but is {2}"\
                        .format(key, value,  self.__dict__[key])
            except KeyError:
                logging.error("Cannot check validity of key '{0}' in <{1}> - not found.".format(key,  self.name))

        logging.info("Checking okay: {0}".format(self.name))


    @property
    def wave(self):

        if self._wave == None:
            self._wave = np.arange(self.ndisp, dtype=np.float64)
            self._wave *= self.cdelt
            self._wave += self.crval
        return self._wave


    @property
    def unit(self):

        if self.cunit == "Angstrom":
            return r"\mathrm{{\AA}}"
        else:
            return r"\mathrm{{{0}}}".format(self.cunit)
