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
109, 2013/08/20

"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 109

import logging
import numpy as np, pyfits

## required PampelMuse routines
from pampelmuse.core.instruments import *


class Cube:
    """
    Open a FITS datacube and check whether it is valid for the usage
    in PampelMuse.
    Note that for the ARGUS IFU, the routine will try to read the fiber
    configuration from an extension named FIBER_SETUP if that extension
    is available.
    """
    
    def __init__(self, name, sigma=None, mask=None, instrument=None, mode='readonly', checks={}):
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
        instrument : pampelmuse.core.instruments.Instrument instance, string or None,
            optional
            Can be used to provide informations about the instrument used to observe
            the data. Note that if not instrument information is provided, only 3dim.
            data is accepted.
        mode :  string, optional
            Open mode, 'readonly' (default), 'update', 'append', 'denywrite', or
            'ostream'. This parameter is passed to pyfits.open().
        checks : dictionary, optional
            An optional list of (keyword, value)-pairs used to check the consistency
            of the datacube with previously loaded data.
        """
    
        self.name = name
        self.mode = mode

        self._wave = None

        self.data_hdu  = None
        self.sigma_hdu = None
        self.mask_hdu  = None
        self.calibration_hdu = None

        ## property to check whether uncertainties or variances are loaded
        self.is_var = False

        ## open datacube:
        logging.info("Reading IFS data from file {0}...".format(self.name))
        self.file = pyfits.open(name, mode=self.mode)
    
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
                if instrume.upper() in ["GIRAFFE", "ARGUS"]:
                    self.instrument = Argus()
                elif instrume.upper() == "PMAS":
                    self.instrument = Pmas()
#            elif instrume.upper() == "MUSE":
#                self.instrument = Muse()
                else:
                    logging.error("Unknown instrument specifier: '{0}'".format(instrume))
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
                self.calibration_hdu = self.file[setup_index]
                self.instrument.read_fiber_setup(hdu=self.calibration_hdu)
            except KeyError:
                logging.warning("No HDU with fibre setup found for ARGUS data.")
    
        ## open data HDU - check if HDU "DATA" exists, otherwise use first HDU
        try: self.data_hdu = self.file["DATA"]
        except KeyError: self.data_hdu = self.file[0]

        ## check dimensionality of data: must be 3D if transformation from spaxels
        ## to x- & y-coords unknown, otherwise 2dim. data is accepted
        self.ndim = self.data_hdu.header["NAXIS"]

        if self.ndim != 3:
            if self.instrument.transform == None:
                logging.error("Data must be 3dim. unless mapping from spaxels to x&y specified.")
                assert self.ndim==3, "Expected 3dim. datacube, got {0}dim".format(self.ndim)
            else:
                self.spectral_axis = self.instrument.spectral_axis
        else:
            if self.instrument.shape == None: ## set spatial shape if needed
                xdim = self.data_hdu.header["NAXIS1"]
                ydim = self.data_hdu.header["NAXIS2"]

                self.instrument.shape = (ydim, xdim)

                ## for 3dim. data, initialize transformation assuming rectangular array
                x = np.mod(np.arange(xdim*ydim, dtype=np.int32), xdim)
                y = np.arange(xdim*ydim, dtype=np.int32)//xdim
                self.instrument.transform = np.vstack((y,x)).T

            self.spectral_axis = 3

        ## check if data compatible with coordinate transformation of the instrument
        self.nvoxel = reduce(lambda x,y: x*y, self.data_hdu.data.shape)
        self.nlayer = self.data_hdu.header["NAXIS{0}".format(self.spectral_axis)]


        ## Reading information about wavelength coverage
        self.crpix = self.data_hdu.header['CRPIX{0}'.format(self.spectral_axis)]
        self.crval = self.data_hdu.header['CRVAL{0}'.format(self.spectral_axis)]
        
        try: self.cdelt = self.data_hdu.header['CDELT{0}'.format(self.spectral_axis)]
        except KeyError:
            try: self.cdelt = self.data_hdu.header['CD{0}_{0}'.format(self.spectral_axis)]
            except KeyError:
                logging.error("Cannot find dispersion keyword ('CDELT{0}' or 'CD{0}_{0}')".format(self.spectral_axis))
                self.cdelt = None

        try:
            self.ctype = self.data_hdu.header["CTYPE{0}".format(self.spectral_axis)]
        except KeyError:
            self.ctype = "AWAV"
        try:
            self.cunit = self.data_hdu.header["CUNIT{0}".format(self.spectral_axis)]
        except KeyError:
            self.cunit = "Angstrom"

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
        self.open_sigma(sigma)

        ## check if file with masked pixels is available, otherwise check data file:
        self.open_mask(mask)


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

    
    def open_sigma(self, filename=None):
        """
        Open the uncertainties or variances of the data and check their
        validity. Uncertainties should be either in the first extension of
        a file or in an extension named 'ERROR' or 'SIGMA'. Variance should
        be in an extension named 'STAT'. Note that if no file is provided,
        the routine will check the data file for the information.
        
        Parameters
        ----------
        filename : string, optional
            The filename of the FITS file that contains the uncertainties or
            variances.
        """
        if filename != None:
            self.sigma_file = pyfits.open(filename, mode=self.mode)
            hdulist = self.sigma_file
        else:
            hdulist = self.file
        
        ## check for uncertainties
        for name in ["ERROR", "SIGMA"]:
            try:
                self.sigma_hdu = hdulist[name]
                logging.info("Detected extension '{0}', assuming it contains uncertainties.".format(name))
                break
            except KeyError:
                pass

        ## check for variances
        for name in ["STAT", ]:
            try:
                self.sigma_hdu = hdulist[name]
                self.is_var = True
                logging.info("Detected extension '{0}', assuming it contains variances.".format(name))
            except KeyError:
                pass
        
        if self.sigma_hdu == None:
            if filename != None:
                self.sigma_hdu = hdulist[0]
            else:
                logging.error("No HDU containing uncertainties or variances detected.")
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
                
        logging.info("Opening uncertaities/variances HDU successful.")
        
    
    def open_mask(self, filename=None):
        """
        Open the bad pixel mask of the data and check its validity. The
        mask should be either in the first extension of a file or in an
        extension named 'MASK'. Not that if no file is provided, the
        routine will check the data file for the information.
        
        Parameters
        ----------
        filename : string, optional
            The filename of the FITS file that contains the mask.
        """
        if filename != None:
            self.mask_file = pyfits.open(filename, mode=self.mode)
            hdulist = self.mask_file
        else:
            hdulist = self.file
            
        for name in ["MASK", "BADPIX"]:
            try:
                self.mask_hdu = hdulist[name]
                logging.info("Detected extension '{0}', assuming it contains the bad voxel mask.".format(name))
            except KeyError:
                pass
                
        if self.mask_hdu == None:
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
        
    
    def get_mask_for_layer(self, index):
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
        if self.mask_hdu == None:  return np.zeros(self.instrument.nspaxel, dtype=np.int16)
        else: return self._get_layer(self.mask_hdu, index)
        
        
    def get_layer(self, index, apply_mask=True, return_sigma=True):
        """
        Return the data (and uncertainties if requested) in a layer. Note that
        the data is flattened if necessary and the bad pixel mask is applied
        if (available and) requested. If the loaded data is 2dimensional and the 
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
            valid = self.get_mask_for_layer(index)==0
        
            if return_sigma:
                if not self.is_var: return self._get_layer(self.data_hdu, index)[valid], self._get_layer(self.sigma_hdu, index)[valid]
                else: return self._get_layer(self.data_hdu, index)[valid], np.sqrt(self._get_layer(self.sigma_hdu, index)[valid])
            else:
                return self._get_layer(self.data_hdu, index)[valid]
        else:
            if return_sigma:
                if not self.is_var: return self._get_layer(self.data_hdu, index), self._get_layer(self.sigma_hdu, index)
                else: return self._get_layer(self.data_hdu, index), np.sqrt(self._get_layer(self.sigma_hdu, index))
            else:
                return self._get_layer(self.data_hdu, index)
        
        
    def _get_layer(self, _hdu, index):
        """
        Extract a layer from a provided HDU. This routines is mainly used
        internally and is called by Cube.get_layer() and Cube.get_mask_for_layer()
        methods. It checks the dimensionality of the data and the  properties of
        the instrument to determine how to slice the data to obtain a layer.
        
        Parameters
        ----------
        _hdu : a pyfits.PrimaryHDU  or pyfits.ImageHDU instance
            The HDU from which a layer should be returned.
        index : int
            The index of the layer to be returned.
            
        Returns
        -------
        layer : nd_array
            The extracted layer. Note that only science fibers are
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
        
        
    def get_spectrum(self, index):
        """
        Return the spectrum from spaxel 'index'. Note that 'index' may refer
        to a calibration or a sky fiber if the instrument includes such fibers.
        
        Parameters
        ----------
        index : int
            The index of the spaxel for which the spectrum should be returned.
            
        Returns
        -------
        spectrum : nd_array
            The spectrum in spaxel 'index'.
        """
        ## check if provided index is valid
        if index>=self.instrument.nfiber or index<-self.instrument.nfiber:
            logging.error("Fiber index ({0}) out of range. Must be <{1}.".format(index, self.instrument.nfiber))
        assert index<self.instrument.nfiber and index>=-self.instrument.nfiber, "Fiber index out of range."
        
        if self.ndim == 3:
            ## for 3dim. data, there are no calibration or sky fibers, thus nfiber=len(transform)
            j, k = self.instrument.transform[index]
            return self.data_hdu.section[:, j, k]
        elif self.instrument.spectral_axis==1:
            return self.data_hdu.section[index]
        else:
            return self.data_hdu.section[:, index]
        
        
    def make_cube(self, asPrimary=False, returnSigma=False, returnMask=False, **kwargs):
        """
        Create an actual datacube, i.e. a 3dim. data structure that can be
        opened with programs like QFitsView or ds9.
        
        Parameters
        ----------
        asPrimary : boolean, optional
            If this flag is true, the data is contained in a pyfits.PrimaryHDU
            instance. Otherwise, it is contained in a pyfits.ImageHDU instance.
        kwargs
            Any additional keyword argument is passed on to the call of the
            method pampelmuse.core.instrument.make_grid that is used to
            convert the 2dim. into 3dim. data.
            
        Returns
        -------
        hdu : a pyfits ImageHDU-instance
            The HDU that contains the 3dim. data. The header will be created
            from the header of the loaded IFS data by changing the axes
            information (CDELT?, CRVAL?, ...) to match the structure of the
            cube.
        """
        if self.ndim == 3:
            
            if isinstance(self.data_hdu, pyfits.PrimaryHDU) and not asPrimary:
                hdus = [pyfits.ImageHDU(self.data_hdu.data, header=self.data_hdu.header), ]
            elif not isinstance(self.data_hdu, pyfits.PrimaryHDU) and asPrimary:
                hdus = [pyfits.PrimaryHDU(self.data_hdu.data, header=self.data_hdu.header), ]
            else:
                hdus = [self.data_hdu,]
                
            if returnSigma:
                if isinstance(self.sigma_hdu, pyfits.PrimaryHDU) and not asPrimary:
                    hdus.append(pyfits.ImageHDU(self.sigma_hdu.data, header=self.sigma_hdu.header))
                elif not isinstance(self.sigma_hdu, pyfits.PrimaryHDU) and asPrimary:
                    hdus.append(pyfits.PrimaryHDU(self.sigma_hdu.data, header=self.sigma_hdu.header))
                else:
                    hdus.append(self.sigma_hdu)
                    
            if returnMask:
                if isinstance(self.mask_hdu, pyfits.PrimaryHDU) and not asPrimary:
                    hdus.append(pyfits.ImageHDU(self.mask_hdu.data, header=self.mask_hdu.header))
                elif not isinstance(self.mask_hdu, pyfits.PrimaryHDU) and asPrimary:
                    hdus.append(pyfits.PrimaryHDU(self.mask_hdu.data, header=self.mask_hdu.header))
                else:
                    hdus.append(self.mask_hdu)
            
            if returnSigma|returnMask: return hdus
            else: return hdus[0]
                    
        else:
            data = [self.instrument.make_grid(self.data_hdu.data), ]
            if returnSigma: data.append(self.instrument.make_grid(self.sigma_hdu.data))
            if returnMask: data.append(self.instrument.make_grid(self.mask_hdu.data))
            
            logging.info("Updating header information...")
            out_headers = [self.data_hdu.header, ]
            if returnSigma: out_headers.append(self.sigma_hdu.header)
            if returnMask: out_headers.append(self.mask_hdu.header)
            
            for key in ["crpix", "crval", "cdelt", "ctype", "cunit"]:
                for header in out_headers:
                    try:
                        header.update("{0}3".format(key.upper()), getattr(self, key), after="{0}2".format(key.upper()))
                    except KeyError:
                        header.update("{0}3".format(key.upper()), getattr(self, key))

                    ## keywords for "old" dispersion axis are not valid anymore, so they are deleted (if present)
                    if "{0}{1}".format(key.upper(), self.instrument.spectral_axis) in header:
                        del header["{0}{1}".format(key.upper(), self.instrument.spectral_axis)]
    
            if asPrimary:
                if returnSigma|returnMask: return [pyfits.PrimaryHDU(data[i], header=out_headers[i]) for i in xrange(len(data))]
                else: return pyfits.PrimaryHDU(data[0], header=out_headers[0])
            else: 
                if returnSigma|returnMask: return [pyfits.ImageHDU(data[i], header=out_headers[i]) for i in xrange(len(data))]
                return pyfits.ImageHDU(data[0], header=out_headers[0])


    def __del__(self):

        ## close FITS files that were  opened by instance 
        self.file.close()
        if hasattr(self, "sigma_file"): self.sigma_file.close()
        if hasattr(self, "mask_file"): self.mask_file.close()
