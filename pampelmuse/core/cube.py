"""
This file provides the class 'Cube' that is thought to facilitate the 
opening of FITS cubes and checking them for validity.

CHANGELOG
2013/05/03, v0.1.0.0, initial version
2013/05/29, v0.2.0.0, added methods to return a layer from a datacube,
                      also included more documentation
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
    Note that for the ARGUS IFU, the routine will try to read the fiber
    configuration from an extension named FIBER_SETUP if that extension
    is available.
    """
    
    def __init__(self, name, sigma=None, mask=None, instrument=None, checks={}):
        """
        Initialize a new instance of the 'Cube' class.
        
        Parameters
        ----------
        name : string
            The path to the FITS cube that should be loaded. The data should be
            either in the first HDU or in an extension named 'DATA'.
        sigma : string, optional
            The path to the uncertainties of the data. If none is provided, the
            routine will check whether the data file contains an extension 'ERROR'
        mask : string, optional
            The path to the bad pixel mask of the cube. If none is provided, the
            routine will check whether the data file contains an extension 'MASK'.
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

        self.data_hdu  = None
        self.sigma_hdu = None
        self.mask_hdu  = None

        ## open datacube:
        self.file = pyfits.open(name)
    
        ## create instrument instance if none provided
        if isinstance(instrument, Instrument):
            self.instrument = instrument
            
        elif isinstance(instrument, str):
            if instrument == "argus": self.instrument = Argus()
            elif instrument == "pmas": self.instrument = Pmas()
            else: logging.error("Unknown instrument specifier: '{0}'".format(instrument))

        else:
            ## if no instrument name is provided, try to get from FITS header:
            try:
                instrume = self.file[0].header["INSTRUME"]
            except KeyError:
                instrume = ""
            
            if instrume == "GIRAFFE":
                self.instrument = Argus()
            else: ## unfortunately PMAS does not use the INSTRUME key
                self.instrument = Instrument()
        
        ## check if calibration HDU for ARGUS exists, extension (FIBER_HDU)
        if self.instrument.name == "argus":
            try:
                setup_index = self.file.index_of("FIBER_SETUP")
                self.instrument.read_fiber_setup(hdu=self.file[setup_index])
            except KeyError:
                logging.warning("No HDU with fibre setup found for ARGUS data.")
    
        ## open data HDU - check if HDU "DATA" exists, otherwise use first HDU
        try: self.data_hdu = self.file["DATA"]
        except KeyError: self.data_hdu = self.file[0]

        ## check dimensionality of data: must be 3D if transformation from spaxels
        ## to x- & y-coords unknown, otherwise 2dim. data is accepted
        self.ndim = self.data_hdu.header["NAXIS"]

        if self.instrument.transform == None and self.ndim!=3:
            logging.error("Data must be 3dim. unless mapping from spaxels to x&y specified.")
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

        ## perform sanity checks if requested
        for key, value in checks.iteritems():
            try:
                if self.__dict__[key] != value:
                    logging.error("Invalid value encountered for '{0}' in <{1}>.".format(key,  self.name))
                    assert self.__dict__[key] == value, "Keyword '{0}' should  be {1} but is {2}"\
                        .format(key, value,  self.__dict__[key])
            except KeyError:
                logging.error("Cannot check validity of key '{0}' in <{1}> - not found.".format(key,  self.name))

        logging.info("Opening data HDU successful.")
    
        ## check if file with uncertainties is available, otherwise check data file:
        self.openSigma(sigma)

        ## check if file with masked pixels is available, otherwise check data file:
        self.openMask(mask)


    @property
    def wave(self):
        """
        The wavelength array of the loaded data.
        """

        if self._wave == None:
            self._wave = np.arange(self.ndisp, dtype=np.float64)
            self._wave *= self.cdelt
            self._wave += self.crval
        return self._wave


    @property
    def unit(self):
        """
        The data units of the wavelength array. Currently Angstrom and
        nm are supported.
        """
        if self.cunit == "Angstrom":
            return r"\mathrm{{\AA}}"
        else:
            return r"\mathrm{{{0}}}".format(self.cunit)

    
    def openSigma(self, filename=None):
        """
        Open the uncertainties of the data and check their validity. The
        uncertainties should be either in the first extension of a file or
        in an extension named 'ERROR'. Not that if no file is provided, the
        routine will check the data file for the information.
        
        Parameters
        ----------
        filename : string, optional
            The filename of the FITS file that contains the uncertainties.
        """
        if filename != None:
            hdulist = pyfits.open(filename)
        else:
            hdulist = self.file
            
        try:
            self.sigma_hdu = hdulist["ERROR"]
        except KeyError:
            if filename != None: self.sigma_hdu = hdulist[0]
            else:
                logging.error("No HDU containing uncertainties detected.")
                return
                
        ##  consistency checks: must have same dimensions as data
        if self.sigma_hdu.header["NAXIS"] != self.ndim:
            logging.error("Incompatible dimensionalities of uncertainties and science data.")
            self.sigma_hdu = None
            return
            
        for key in ["NAXIS{0}".format(i) for i in xrange(1, self.ndim+1)]:
            if self.sigma_hdu.header[key] != self.data_hdu.header[key]:
                logging.error("Different values for header key {0} in uncertainties and science data.".format(key))
                self.sigma_hdu = None
                return
                
        logging.info("Opening uncertaities HDU successful.")
        
    
    def openMask(self, filename=None):
        """
        Open the bad pixel mask of the data and check its validity. The
        uncertainties should be either in the first extension of a file or
        in an extension named 'MASK'. Not that if no file is provided, the
        routine will check the data file for the information.
        
        Parameters
        ----------
        filename : string, optional
            The filename of the FITS file that contains the mask.
        """
        if filename != None:
            hdulist = pyfits.open(filename)
        else:
            hdulist = self.file
            
        try:
            self.mask_hdu = hdulist["MASK"]
        except KeyError:
            if filename != None: self.mask_hdu = hdulist[0]
            else:
                logging.warning("No HDU containing bad pixel mask detected.")
                return
                
        ##  consistency checks: must have same dimensions as data
        if self.mask_hdu.header["NAXIS"] != self.ndim:
            logging.error("Incompatible dimensionalities of mask and science data.")
            self.mask_hdu = None
            return
            
        for key in ["NAXIS{0}".format(i) for i in xrange(1, self.ndim+1)]:
            if self.mask_hdu.header[key] != self.data_hdu.header[key]:
                logging.error("Different values for header key {0} in mask and science data.".format(key))
                self.mask_hdu = None
                return
                
        logging.info("Opening mask HDU successful.")
        
    
    def getMaskForLayer(self, index):
        """
        Return the bad pixel mask for a single layer.
       
        Parameters
        ----------
        index : int
            The index of the layer for which the mask should be returned.
            
        Returns
        -------
        mask or None, depending on whether a mask is available
        """
        if self.mask_hdu == None:  return None
        else: return self._getLayer(self.mask_hdu, index)
        
        
    def getLayer(self, index, apply_mask=True, return_sigma=True):
        """
        Return the data (and uncertainties if requested) in a layer. Note that
        the data is flattened if necessary and the bad pixel mask is applied
        (if available) if requested. If the loaded data is 2dimensional and the 
        instrument contains calibration/sky fibers, only the science fibers are
        returned.
        
        Parameters
        ----------
        index : int
            The index of the layer to be loaded.
        apply_mask : bool, optional
            Flag indicating whether the bad-pixel mask should be applied before the
            data is returned.
        return_sigma : bool, optional
            Flag indicating whether besides the data the uncertainties should be
            returned.
            
        Returns
        -------
        data_i : nd_array
            The data stored in the requested layer.
        sigma_i : nd_array, optional
            The uncertainties for the returned data (only if requested).
        """
        if apply_mask and self.mask_hdu != None:
            valid = self.getMaskForLayer(index)==0
        
            if return_sigma:
                return self._getLayer(self.data_hdu, index)[valid], self._getLayer(self.sigma_hdu, index)[valid]
            else:
                return self._getLayer(self.data_hdu, index)[valid]
        else:
            if return_sigma:
                return self._getLayer(self.data_hdu, index), self._getLayer(self.sigma_hdu, index)
            else:
                return self._getLayer(self.data_hdu, index)
        
        
    def _getLayer(self, _hdu, index):
        """
        Extract a layer from a provided HDU. This routines is mainly
        used internally and is called by 'getLayer' and 'getMaskForLayer'
        methods. It checks the dimensionality of the data and the 
        properties of the instrument to determine how to slice the data
        to obtain a layer.
        
        Parameters
        ----------
        _hdu : a pyfits.PrimaryHDU  or pyfits.ImageHDU instance
            The HDU from which a layer should be returned.
        index : int
            The index of the layer to be returned.
            
        Returns
        -------
        layer : nd_array
            The extracted lyer. Note that only science fibers are
            returned if the instrument contains calibration or
            science fibers.
        """
        
        if self.ndim == 3:
            ## Note: np.ndarray.flatten -> returns copy
            return _hdu.section[index].flatten()
        elif self.instrument.spectral_axis==1:
            return self.instrument.get_science(_hdu.section[:,index])
        else:
            return self.instrument.get_science(_hdu.section[index])
        
        
