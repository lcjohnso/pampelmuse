"""
cube.py
=======

Provides
--------
class pampelmuse.core.cube.Cube

Purpose
-------
This module implements the class 'Cube' that facilitates the working
with IFS data within PampelMuse. An instance of the class can be
initialized using either 2dim. or 3dim. data. Note however, that for
2dim. data informations about the instrument used to observe the data
must be available. This information can either be provided via a key-
word argument to the initialization method or obtained from the FITS
header of the data, via standard keyword INSTRUME. The method will also
check whether the IFS data contains HDUs providing either uncertainties
(extension ERROR) or variances (extension STAT) and a bad-pixel mask
(extension MASK). These can also provided as dedciated FITS-files.

Latest SVN revision
-------------------
173, 2014/06/11

"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 173

import logging, types
import numpy as np
from astropy import wcs
from astropy.io import fits

## required PampelMuse routines
from pampelmuse.core.instruments import *
from pampelmuse.core.instruments import available_instruments

class Cube:
    """
    Open a FITS datacube and check whether it is valid for the usage
    in PampelMuse.
    Note that for the ARGUS IFU, the routine will try to read the fiber
    configuration from an extension named FIBER_SETUP if that extension
    is available.
    """
    
    def __init__(self, name, sigma=None, mask=None, residuals=None, instrument=None, mode='readonly', checks={}):
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
            or 'SIGMA'. If an HDU named 'STAT' is detected, ithe routine assumes
            it contains variances.
        mask : string, optional
            The path to the bad pixel mask of the cube. If none is provided, the
            routine will check whether the data file contains an extension 'MASK'.
        residuals : string, optional
            The path to a file containing the fit residuals of the cube. If none is
            provided, the routine will check whether the data file contains an
            extension 'RESIDUAL'.
        instrument : pampelmuse.core.instruments.Instrument instance, string or None,
            optional
            Can be used to provide informations about the instrument used to observe
            the data. Note that if not instrument information is provided, only 3dim.
            data is accepted.
        mode :  string, optional
            Open mode, 'readonly' (default), 'update', 'append', 'denywrite', or
            'ostream'. This parameter is passed to astropy.io.fits.open().
        checks : dictionary, optional
            An optional list of (keyword, value)-pairs used to check the consistency
            of the datacube with previously loaded data.
        """
    
        self.name = name
        self.mode = mode

        self._wave = None

        self.data = None
        self.sigma = None
        self.mask = None
        self.residuals = None
        self.setup = None

        ## property to check whether uncertainties or variances are loaded
        self.is_var = False

        ## open datacube:
        logging.info("Reading IFS data from file {0}...".format(self.name))
        self.file = fits.open(name, mode=self.mode)
    
        ## if no instrument instance is provided, first option is to look for header keyword 'INSTRUME'
        ## NOTE: unfortunately PMAS does not use the INSTRUME key (even though it is a standard FITS key, Martin!)
        if not isinstance(instrument, Instrument):
            instrume = None
            for hdu in self.file:
                try:
                    instrume = hdu.header["INSTRUME"]
                    break
                except KeyError: pass
                
            ## if no 'INSTRUME' keyword is found, assume instrument name was provided as a string in 'instrument' argument
            if instrume == None:
                instrume = instrument

            if instrume != None:
                if instrume.lower() in available_instruments:
                    self.instrument = available_instruments[instrume.lower()](header=hdu.header, pixtable=False)
                else:
                    logging.warning("Unknown instrument specifier: '{0}'".format(instrume))
                    self.instrument = Instrument()
            else:
                logging.warning("Cannot determine IFU in use.")
                logging.warning("Some functions might not work properly.")
                self.instrument = Instrument()
        else:
            self.instrument = instrument
        
        ## check if calibration HDU for ARGUS exists, extension (FIBER_HDU)
        if self.instrument.name == "argus":
            try:
                setup_index = self.file.index_of("FIBER_SETUP")
                self.setup  = self.file[setup_index]
                self.instrument.read_fiber_setup(hdu=self.setup)
            except KeyError:
                logging.warning("No HDU with fibre setup found for ARGUS data.")
    
        ## for a more convenient access to spectra and layers, we add additional methods
        ## to the fits.hdu.image._ImageBaseHDU class.  This is done by following the tutorial
        ## at http://www.ianlewis.org/en/dynamically-adding-method-classes-or-class-instanc
        ## All instances of this class created later can access these methods.
        fits.hdu.image._ImageBaseHDU.spectrum = types.MethodType(self.instrument.get_spectrum, None, fits.hdu.image._ImageBaseHDU)
        fits.hdu.image._ImageBaseHDU.layer = types.MethodType(self.instrument.get_layer, None, fits.hdu.image._ImageBaseHDU)
    
        ## open data HDU - check if HDU "DATA" exists, otherwise use first HDU
        try: self.data = self.file["DATA"]
        except KeyError: self.data = self.file[0]

        ## check dimensionality of data: must be 3D if transformation from spaxels
        ## to x- & y-coords unknown, otherwise 2dim. data is accepted
        self.ndim = self.data.header["NAXIS"]

        if self.ndim != 3:
            if self.instrument.transform == None:
                logging.error("Data must be 3dim. unless mapping from spaxels to x&y specified.")
                assert self.ndim==3, "Expected 3dim. datacube, got {0}dim".format(self.ndim)
            else:
                self.spectral_index = 1 if self.instrument.spectral_axis==1 else 2
        else:
            if self.instrument.shape == None: ## set spatial shape if needed
                xdim = self.data.header["NAXIS1"]
                ydim = self.data.header["NAXIS2"]

                self.instrument.shape = (ydim, xdim)

                ## for 3dim. data, initialize transformation assuming rectangular array
                x = np.mod(np.arange(xdim*ydim, dtype=np.int32), xdim)
                y = np.arange(xdim*ydim, dtype=np.int32)//xdim
                self.instrument.transform = np.vstack((y,x)).T

            self.spectral_index = 3

        ## check if data compatible with coordinate transformation of the instrument
        self.nvoxel = reduce(lambda x,y: x*y, self.data.data.shape)
        self.nlayer = self.data.header["NAXIS{0}".format(self.spectral_index)]


        ## Reading information about wavelength coverage
        self.crpix = self.data.header['CRPIX{0}'.format(self.spectral_index)]
        self.crval = self.data.header['CRVAL{0}'.format(self.spectral_index)]
        
        try: self.cdelt = self.data.header['CDELT{0}'.format(self.spectral_index)]
        except KeyError:
            try: self.cdelt = self.data.header['CD{0}_{0}'.format(self.spectral_index)]
            except KeyError:
                logging.error("Cannot find dispersion keyword ('CDELT{0}' or 'CD{0}_{0}')".format(self.spectral_index))
                self.cdelt = None

        try:
            self.ctype = self.data.header["CTYPE{0}".format(self.spectral_index)]
        except KeyError:
            self.ctype = "AWAV"
        try:
            self.cunit = self.data.header["CUNIT{0}".format(self.spectral_index)]
        except KeyError:
            self.cunit = "Angstrom"

        self.ndisp = self.data.header['NAXIS{0}'.format(self.spectral_index)]

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
        i = self.load_auxiliary_hdu("sigma", filename=sigma, hduname=["SIGMA", "ERROR", "STAT"])
        if self.sigma==None: logging.error("Unable to load uncertainties for IFS data.")
        else: logging.info("Loaded uncertainties from HDU {0} of file {1}.".format(self.sigma.name if self.sigma.name!='' else i, self.sigma._file.name))

        ## check if file with masked pixels is available, otherwise check data file:
        i = self.load_auxiliary_hdu("mask", filename=mask, hduname="MASK")
        if self.mask==None: logging.warning("Unable to load bad-pixel mask for IFS data.")
        else: logging.info("Loaded bad-pixel mask from HDU {0} of file {1}.".format(self.mask.name if self.mask.name!='' else i, self.mask._file.name))
        
        ## check if file or HDU with fit residuals  is available
        i = self.load_auxiliary_hdu("residuals", filename=residuals, hduname="RESIDUAL")
        if self.residuals!=None:
            logging.info("Loaded residuals from HDU {0} of file {1}.".format(self.residuals.name if self.residuals.name!='' else i, self.residuals._file.name))

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

    
    def load_auxiliary_hdu(self, attribute, filename=None, hduname=None):
        """
        Loads an auxiliary HDU of the one containing the IFS data and adds
        it to the instance by setting the provided attribute.
        
        Parameters
        ----------
        attribute : string
            The name of the attribute that should be set to the HDU using the
            setattr-method.
        hdunames : string or iterable
            One or more names to identify the HDU in the file (if no filename
            is provided, the list of HDUs of the file containing the IFS data
            is searched.
        filename, string, optional
            The name of the file containing the HDU.

        Returns
        -------
        i : int
            The index of the HDU containing the requested data. If no matching
            HDU is found, None is returned.
        """
        if hduname!=None and not hasattr(hduname, "__iter__"):
            hduname = [hduname] ##  make sure hduname is iterable

        if filename != None:
            hdulist = fits.open(filename, mode=self.mode)
        else:
            hdulist = self.file
    
        available_names = [hdu.name for hdu in hdulist]
    
        i = None ## will contain the index of the HDU to be loaded
        
        ## check if one of the provided names is available in the list of HDUs
        for name in hduname:
            try:
                i = available_names.index(name)
                if attribute=="sigma" and available_names[i] == "STAT": self.is_var=True
                break
            except ValueError:
                pass
    
        if i == None:
            if filename != None: i=0
            else: return None

        ##  consistency checks: must have same dimensions as data
        if hdulist[i].header["NAXIS"] != self.ndim:
            logging.error("Loaded HDU has different dimensionality than IFS data.")
            return
            
        ## lengths of individual axes can be different, e.g. residuals are stored without calibration/sky fibers and only for the wavelength range
        ## that is analysed
        for key in ["NAXIS{0}".format(j) for j in xrange(1, self.ndim+1)]:
            if hdulist[i].header[key] != self.data.header[key]:
                logging.warning("Loaded HDU has different axis dimensions than IFS data.".format(key))
##                 return None

        setattr(self, attribute, hdulist[i])
        
        return i
    

        
    def get_key(self, name):
        """
        Read a keyword from the header of the data HDU. It is possible
        to provide more than one name under whiich the keyword may be
        referenced.
        
        Parameters
        ----------
        name : string or iterable
            The name(s) of the keyword. Note that HIERARCH-keys should be
            referenced without the HIERARCH-prefix.
            
        Returns
        -------
        value : int, float, or string
            The value of the requested keyword if it exists, otherwise
            None is returned.
        """
        if not hasattr(name, "__iter__"): name = [name]
        
        for n in name:
            if n in self.data.header:
                return self.data.header[n]
                
        if len(name)==1: logging.error("Key '{0}' missing in FITS header.".format(name[0]))
        else: logging.error("None of keys '{0}' present in FITS header.".format("','".join(name)))
        return None
        
        
    def make_wcs(self):
        """
        Method to construct a WCS instance from the instrument properties
        (rotation angle, spaxel scale) and the RA and DEC information
        available in the header.
        """
        
        self.wcs = wcs.WCS(naxis=2)
        
        if self.ndim == 3: ## if 3dim. data are provided, check whether full WCS information is already available. 
            inwcs = wcs.WCS(self.data.header)
            if inwcs.wcs.has_cd:
                self.wcs.wcs.cd = inwcs.wcs.cd[:2, :2]
                self.wcs.wcs.ctype = [inwcs.wcs.ctype[0], inwcs.wcs.ctype[1]]
                self.wcs.wcs.cunit = [inwcs.wcs.cunit[0], inwcs.wcs.cunit[1]]
                self.wcs.wcs.crpix = inwcs.wcs.crpix[:2]
                self.wcs.wcs.crval = inwcs.wcs.crval[:2]
                logging.info("Copied WCS information from header of data HDU.")
                
        if self.wcs.wcs.is_unity():
                
            self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            self.wcs.wcs.cunit = ["deg", "deg"]

            self.wcs.wcs.crpix = [self.instrument.shape[1]/2., self.instrument.shape[0]/2.]
                
            if self.instrument.rotation_angle != None and self.instrument.spaxel_scale != None:
                theta = np.radians(self.instrument.rotation_angle)
                self.wcs.wcs.cd = np.array([[np.cos(theta), -np.sin(theta)], [-np.sin(theta), -np.cos(theta)]] , dtype=np.float32)*(self.instrument.spaxel_scale/3600.)
                ## Note that the rotation matrix is et up such that north is up and east is left (standard FITS convention)
            else:
                logging.error("Missing IFS rotation angle and/or spaxel scale. WCS information will be incomplete.")

            if "RA" in self.data.header and "DEC" in self.data.header:
                self.wcs.wcs.crval = [self.data.header["RA"], self.data.header["DEC"]]
            else:
                logging.error("Missing RA and DEC information. WCS information will be incomplete.")
        
        
    def make_cube(self, asPrimary=False, returnSigma=False, returnMask=False, **kwargs):
        """
        Create an actual datacube, i.e. a 3dim. data structure that can be
        opened with programs like QFitsView or ds9.
        
        Parameters
        ----------
        asPrimary : boolean, optional
            If this flag is true, the data is contained in a fits.PrimaryHDU
            instance. Otherwise, it is contained in a fits.ImageHDU instance.
        kwargs
            Any additional keyword argument is passed on to the call of the
            method pampelmuse.core.instrument.make_grid that is used to
            convert the 2dim. into 3dim. data.
            
        Returns
        -------
        hdu : a fits ImageHDU-instance
            The HDU that contains the 3dim. data. The header will be created
            from the header of the loaded IFS data by changing the axes
            information (CDELT?, CRVAL?, ...) to match the structure of the
            cube.
        """
        if self.ndim == 3:
            
            if isinstance(self.data, fits.PrimaryHDU) and not asPrimary:
                hdus = [fits.ImageHDU(self.data.data, header=self.data.header), ]
            elif not isinstance(self.data, fits.PrimaryHDU) and asPrimary:
                hdus = [fits.PrimaryHDU(self.data.data, header=self.data.header), ]
            else:
                hdus = [self.data,]
                
            if returnSigma:
                if isinstance(self.sigma, fits.PrimaryHDU) and not asPrimary:
                    hdus.append(fits.ImageHDU(self.sigma.data, header=self.sigma.header))
                elif not isinstance(self.sigma, fits.PrimaryHDU) and asPrimary:
                    hdus.append(fits.PrimaryHDU(self.sigma.data, header=self.sigma.header))
                else:
                    hdus.append(self.sigma)
                    
            if returnMask:
                if isinstance(self.mask, fits.PrimaryHDU) and not asPrimary:
                    hdus.append(fits.ImageHDU(self.mask.data, header=self.mask.header))
                elif not isinstance(self.mask, fits.PrimaryHDU) and asPrimary:
                    hdus.append(fits.PrimaryHDU(self.mask.data, header=self.mask.header))
                else:
                    hdus.append(self.mask)
            
            if returnSigma|returnMask: return hdus
            else: return hdus[0]
                    
        else:
            data = [self.instrument.make_grid(self.data.data), ]
            if returnSigma: data.append(self.instrument.make_grid(self.sigma.data))
            if returnMask: data.append(self.instrument.make_grid(self.mask.data))
            
            logging.info("Updating header information...")
            out_headers = [self.data.header, ]
            if returnSigma: out_headers.append(self.sigma.header)
            if returnMask: out_headers.append(self.mask.header)
            
            for key in ["crpix", "crval", "cdelt", "ctype", "cunit"]:
                for header in out_headers:
                    try:
                        header.update("{0}3".format(key.upper()), getattr(self, key), after="{0}2".format(key.upper()))
                    except KeyError:
                        header.update("{0}3".format(key.upper()), getattr(self, key))

                    ## keywords for "old" dispersion axis are not valid anymore, so they are deleted (if present)
                    if "{0}{1}".format(key.upper(), 1 if self.instrument.spectral_axis==1 else 2) in header:
                        del header["{0}{1}".format(key.upper(), 1 if self.instrument.spectral_axis==1 else 2)]
    
            if asPrimary:
                if returnSigma|returnMask: return [fits.PrimaryHDU(data[i], header=out_headers[i]) for i in xrange(len(data))]
                else: return fits.PrimaryHDU(data[0], header=out_headers[0])
            else: 
                if returnSigma|returnMask: return [fits.ImageHDU(data[i], header=out_headers[i]) for i in xrange(len(data))]
                return fits.ImageHDU(data[0], header=out_headers[0])


    def __del__(self):

        ## close FITS files that were opened by instance 
        self.file.close()
        if self.sigma!=None: self.sigma._file.close()
        if self.mask!=None: self.mask._file.close()
        
