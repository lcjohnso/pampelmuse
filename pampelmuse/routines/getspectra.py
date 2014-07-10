"""
getspectra.py
=============

Provides
--------
function pampelmuse.routines.getspectra

Purpose
-------
The function getspectra is called when Pampelmuse is started with the
GETSPECTRA command. Its purpose is to read all the spectra that have
been extracted from a datacube and store them as individual FITS files.
The structure of each created FITS file is such that the primary exten-
sion contains the spectrum and the second extension (named 'SIGMA') con-
tains the uncertainty for the measured spectra as estimated by the CUBE-
FIT routine of PampelMuse.

Added
-----
2014/06/26, version 0.15.0.0

Latest SVN revision
-------------------
198 # 2014/07/10
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 198

import datetime, logging, os
import numpy as np
from astropy.io import fits

## required PampelMuse-packages
from pampelmuse.core import Sources
from pampelmuse.core.coordinates import Transformation
from pampelmuse.core.psf import Variables
from pampelmuse.functions.statistics import DER_SNR




def getspectra(pampelmuse):

    logging.info("Executing requested routine GETSPECTRA [Revision {0:d}].".format(__revision__))

    if not os.path.exists(pampelmuse.getspectra.folder):
        logging.info("Creating folder '{0}' to store spectra...".format(pampelmuse.getspectra.folder))
        os.mkdir(pampelmuse.getspectra.folder)

    ## Collect information about sources/create Sources-instance:
    logging.info("Obtaining information on sources in the field...")
    
    ## - Source catalogue
    sources = Sources(data=pampelmuse.hdulist["SOURCES"], instrument=pampelmuse.ifs_data.instrument)

    
    ## - Spectra
    logging.info("Obtaining source spectra...")
    try:
        i = pampelmuse.hdulist.index_of("SPECTRA")
        try:
            j = pampelmuse.hdulist.index_of("SIGMASPC")
            sources.open_spectra_hdu(pampelmuse.hdulist[i], sigma_hdu=pampelmuse.hdulist[j])
        except KeyError:
            sources.open_spectra_hdu(pampelmuse.hdulist[i])
    except KeyError:
        logging.critical("No spectra (HDU 'SPECTRA') have been provided.")
    
    
    ## - Positions
    logging.info("Obtaining information about source coordinates...")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.open_coordinates_hdu(pampelmuse.hdulist[i])
    except KeyError:
        logging.warning("Cannot find IFS coordinates (HDU 'POSITIONS').")
        
    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.transformation = Transformation(hdu=pampelmuse.hdulist[i])
    except KeyError:
        logging.warning("No coordinate transformation (HDU 'POSPARS') detected.")

    sources.info()

    
    ## Initializing PSF variables
    logging.info("Obtaining PSF properties...")
    try:
        i = pampelmuse.hdulist.index_of("PSFPARS")
        psf_attributes = Variables(hdu=pampelmuse.hdulist[i])
    except KeyError: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.warning("No information about PSF profile (HDU 'PSFPARS') found.")
        psf_attributes = None

    
    ## initialize an empty header that will contain the header cards that are the
    ## same for all spectra collected from an observation
    header = fits.Header()
    

    ## add dispersion information to header
    header["CRPIX1"] = sources.crpix
    header["CRVAL1"] = sources.crval
    header["CDELT1"] = sources.cdelt
    header["CTYPE1"] = sources.ctype
    header["CUNIT1"] = sources.cunit
    header["WSTART"] = (sources.crval-(sources.crpix-1)*sources.cdelt)
    header["WSTEP"] = sources.cdelt
#    header["DC-FLAG"] = 0


    header["OBJNAME"] = "NAME" ## place holder, will be adapted for each spectrum

    for key in ["OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS", "MJD-OBS", "JUL-DATE", "EXPTIME", "RA", "DEC", "EPOCH"]:
        value = pampelmuse.ifs_data.get_key(key)
        if value != None:
            header[key] = value


    ## add JUL-DATE keyword if only MJD-OBS is provided
    if "MJD-OBS" in header and "JUL-DATE" not in header:
        header["JUL-DATE"] = header["MJD-OBS"]+2400000.5  
    
    if "JUL-DATE" in header: juldate = header["JUL-DATE"]
    else: juldate = 0
    
    ## add keywords describing observing conditions
    for ambikey in ["AIRM START", "AIRM END", "AMBI FWHM START", "AMBI FWHM END", "AMBI PRES START", "AMBI PRESS END", 
                    "AMBI RHUM", "AMBI TAU0", "AMBI TEMP", "AMBI WINDDIR", "AMBI WINDSP"]:
        inkey = "ESO TEL {0}".format(ambikey)
        value = pampelmuse.ifs_data.get_key(inkey)
        if value != None:
            header["HIERARCH OBSERVATION {0}".format(ambikey)] = value

#        ## if both start and end keywords exist, we add a keyword giving the mean of the two
#        if "OBSERVATION {0} END".format(prop) in header and "OBSERVATION {0} START".format(prop) in header:
#            header["HIERARCH OBSERVATION {0} MEAN".format(prop)] = (header["OBSERVATION {0} END".format(prop)] + header["OBSERVATION {0} START".format(prop)])/2.

    


    ## add possibly relevant PampelMuse keywords to the FITS header
    for key in ["version", "revision", "filename", "utc", "passband"]:
        if hasattr(pampelmuse, key):
            if isinstance(getattr(pampelmuse, key), str) and len(getattr(pampelmuse, key))>(55-len(key)): ## check that entry not longer than 80 characters
                header["hierarch PAMPELMUSE {0}".format(key.upper())] = getattr(pampelmuse, key)[:55-len(key)]
            else:
                header["hierarch PAMPELMUSE {0}".format(key.upper())] = getattr(pampelmuse, key)

    ## add estimate for seeing on detector using the fitted FWHM of the PSF profile
    if psf_attributes != None:
        i = list(psf_attributes.names).index("fwhm")
        header["hierarch PAMPELMUSE SEEING"] = (psf_attributes.data[:,i,0].mean(), "Fitted PSF FWHM in spaxels")


    ## loop over all components, i.e. all spectra
    for i in xrange(sources.ncmp): ## collect spectrum and properties of every requested spectrum in observation

        j  = sources.indexmap[i][0] ## The index of the primary (brightest) component contributing to the spectrum
        k    = sources.specrow[j] ## The index of the spectrum in the data array

        ## sanity check: sources with status 1 (=near brighter source) or -1 (=not used) should have no spectrum
        if sources.status[j] in [1,-1]:
            logging.error("Found spectrum for source #{0} with status {1}".format(sources.ID[j], sources.status[j]))
            continue

        flag = 0
        ## each spectum is assigned a flag which is the sum of the following numbers:
        ##  1 - More than one star contribute to the spectrum
        ##  2 - The estimated S/N of the spectrum is <10
        ##  4 - The mean flux of the spectrum is <0
        ##  8 - The centroid of the star is outside the field of view

        ## load spectrum for current component into memory
        sources.load_spectra(components=[i,])
        primary_hdu = fits.PrimaryHDU(sources.data[:,0,0], header=header)

        sigma_hdu = fits.ImageHDU(sources.data[:,0,1])
        sigma_hdu.header["EXTNAME"] = "SIGMA"
        
        if sources.status[j] in [2,3]:
            
            ID = sources.ID[j]
            
            primary_hdu.header["hierarch STAR ID"] = ID

            ## update RA & Dec values if provided in source catalog
            if not np.isnan(sources.ra[j]) and not np.isnan(sources.dec[j]):
                primary_hdu.header["RA"] = sources.ra[j]
                primary_hdu.header["DEC"] = sources.dec[j]
                primary_hdu.header["hierarch STAR RA"] = sources.ra[j]
                primary_hdu.header["hierarch STAR DEC"] = sources.dec[j]

            ## add magnitude to header
            primary_hdu.header["hierarch STAR MAG"] = sources.mag[j]

            ##  Check if more than one source contribute to the spectrum
            cmt = "Whether multiple stars contribute"
            if len(sources.indexmap[i]) > 1:
                flag += 1
                for l in xrange(1, len(sources.indexmap[i])): ## if so, add RA, DEC, and magnitude information for all secondary stars
                    m = sources.indexmap[i][l]
                    if not np.isnan(sources.ra[m]) and not np.isnan(sources.dec[m]):
                        primary_hdu.header["hierarch STAR{0} RA".format(l+1)] = sources.ra[m]
                        primary_hdu.header["hierarch STAR{0} DEC".format(l+1)] = sources.dec[m]
                    primary_hdu.header["hierarch STAR{0} MAG".format(l+1)] = sources.mag[m]
                primary_hdu.header["HIERARCH SPECTRUM MULTISTAR"] = (True, cmt)
                primary_hdu.header["OBJNAME"] = "{0} {1} {2}+".format(pampelmuse.getspectra.prefix.upper(), pampelmuse.ifu.upper(), ID)
            else:
                primary_hdu.header["HIERARCH SPECTRUM MULTISTAR"] = (False, cmt)
                primary_hdu.header["OBJNAME"] = "{0} {1} {2}".format(pampelmuse.getspectra.prefix.upper(), pampelmuse.ifu.upper(), ID)

            ## load IFS coordinates for the star and check if it is inside the field of view
            sources.load_ifs_coordinates(ids=ID)
            xy_j = np.mean(sources.ifs_coordinates[:,0,:,0], axis=0)
            
            infield = pampelmuse.ifs_data.instrument.infov(xy_j)
            if not infield: flag += 8
            primary_hdu.header["HIERARCH SPECTRUM INFIELD"] =  (infield, "Whether centroid inside FOV")

            edgedistance = pampelmuse.ifs_data.instrument.distance_to_edge([xy_j])
            primary_hdu.header["HIERARCH SPECTRUM EDGEDIST"] = (edgedistance[0], "Distance to FOV edge")


        ## calculate S/N of the spectrum
        SNR = DER_SNR(sources.data[:,0,0])
        if SNR < 10: flag += 2
        primary_hdu.header["HIERARCH SPECTRUM SNRATIO"] = (SNR, "Estimated signal-to-noise")

        if np.mean(sources.data[:,0,0]) < 0: flag += 4

        primary_hdu.header["HIERARCH SPECTRUM QLTFLAG"] = (flag, "Quality flag of source")


        if sources.status[j] == 0:
            name = "unresolved"
            primary_hdu.header["hierarch SPECTRUM TYPE"] = name
            primary_hdu.header["OBJNAME"] = "{0} {1} UNRES".format(pampelmuse.getspectra.prefix.upper(), pampelmuse.ifu.upper())
        elif sources.status[j] == 4:
            name = "sky"
            primary_hdu.header["hierarch SPECTRUM TYPE"] = name
            primary_hdu.header["OBJNAME"] = "{0} {1} SKY".format(pampelmuse.getspectra.prefix.upper(), pampelmuse.ifu.upper())
        else:
            name = "id{0:09d}".format(ID)
            primary_hdu.header["hierarch SPECTRUM TYPE"] = "star"


        outname = "{0}{1}jd{2:07d}p{3:04d}f{4:03d}".format(pampelmuse.getspectra.prefix, name, int(juldate), int(10000*(juldate-int(juldate))), int(flag))
        
        utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
        primary_hdu.header["DATE-GEN"] = (utc, "UTC this file was generated.")
        
        outfile = fits.HDUList(hdus=[primary_hdu, sigma_hdu])
        outfile.writeto(os.path.join(pampelmuse.getspectra.folder,"{0}.fits".format(outname)), clobber=True)
        
        logging.info("Wrote spectrum '{0}' [{1}/{2}]".format(os.path.join(pampelmuse.getspectra.folder,"{0}.fits".format(outname)), i+1, sources.ncmp))
        
