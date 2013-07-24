"""
subtres.py
==========

Provides
--------
function pampelmuse.routines.subtres.subtres

Purpose
-------
The method implements the function 'subtres' that is designed to subtract th
contribution of the resolved sources from the IFS data. To this aim, a valid
PampelMuse PRM-file must exist along with the data.  This function is called
if PampelMuse is started with the 'SUBTRES' option.

Latest SVN revision
-------------------
103, 2013/07/23
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 103
import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core import Sources
from pampelmuse.core.coordinates import Transformation
from pampelmuse.core.psf import *


def subtres(pampelmuse):

    logging.info("Executing requested routine SUBTRES [Revision {0:d}].".format(__revision__))

    ## Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    ## Source catalogue
    sources = Sources(data=pampelmuse.hdulist["SOURCES"])


    ## Check if PRM-file contains results of successful fit
    if hasattr(pampelmuse, "cubefit") and hasattr(pampelmuse.cubefit, "status"):
        fitStatus = pampelmuse.cubefit.status
    else:
        fitStatus = "Unknown"

    if fitStatus != "SUCCESS":
        logging.error("PRM-file does not contain results of successful fit.")


    ## Collect source spectra
    logging.info("Obtaining source spectra...")
    try:
        i = pampelmuse.hdulist.index_of("SPECTRA")
        sources.open_spectra_hdu(pampelmuse.hdulist[i])
    except KeyError:
        logging.critical("Cannot find spectra (HDU `SPECTRA') in provided PRM-file.")


    ## Collect source positions
    logging.info("Obtaining information about source coordinates...")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        try:
            i = pampelmuse.hdulist.index_of("INPOSITIONS")
            sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        except KeyError:
            pass
    except KeyError:
        pass

    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.transformation = Transformation(hdu=pampelmuse.hdulist[i])
        
    except KeyError:
        logging.warning("No coordinate transformation (HDU 'POSPARS') detected.")


    ## Initializing PSF variables
    logging.info("Obtaining PSF properties...")
    try:
        i = pampelmuse.hdulist.index_of("PSFPARS")
        psf_attributes = Variables(hdu=pampelmuse.hdulist[i])
    except KeyError: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.warning("No HDU 'PSFPARS' found. Using default PSF.")
        
        kwargs = {"profile":pampelmuse.pfsftype}
        free = []
        i=0
        while hasattr(pampelmuse.psf, "parname{0}".format(i)):
            kwargs[getattr(pampelmuse.psf, "parname{0}".format(i))] = getattr(pampelmuse.psf, "parvalue{0}".format(i))
            free.append(getattr(pampelmuse.psf, "freepar{0}".format(i)))
        psf_attributes = Variables(**kwargs)
        psf_attributes.free = np.atleast_1d(free)

    psf_attributes.info()


    ## Initialize PSF profiles of the sources:
    if psf_attributes.profile == "gauss":
        PSF = Gaussian
    elif psf_attributes.profile == "moffat":
        PSF = Moffat
    else:
        logging.critical("Invalid PSF type: '{0}'".format(psf_attributes.psftype))
        raise IOError, "PSF type not understood."
    PSF.set_transform(pampelmuse.ifs_data.instrument.transform)
    for i in xrange(sources.nsrc):
        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=sources.weights[i], maxrad=pampelmuse.psfrad, osfactor=5, osradius=2)


    ## load full(!) cube into memory.
    logging.info("Creating an array to hold the results...")
    if pampelmuse.ifs_data.instrument.spectral_axis == 2:
        data = np.zeros((sources.ndisp, pampelmuse.ifs_data.instrument.nspaxel), dtype=np.float32)
    else:
        data = np.zeros((pampelmuse.ifs_data.instrument.nspaxel, sources.ndisp), dtype=np.float32)

    for kk in xrange(pampelmuse.start, pampelmuse.stop):
        logging.info("Handling layer #{0} [{1:.1f}%].".format(kk, (100.*(kk-pampelmuse.start))/(pampelmuse.stop-pampelmuse.start+1)))
        
        ## account for offset between layers of datacube and PRM file if not the whole cube is analyzed 
        jj = kk - pampelmuse.start

        ## get current layer 
        layer_kk = pampelmuse.ifs_data.get_layer(index=kk, apply_mask=False, return_sigma=False)

        ## update PSF instances of the sources with the variables fitted for current layer
        i = 0 
        for prm in psf_attributes.names:
            getattr(PSF, "set_{0}".format(prm))(psf_attributes.data[jj,i,int(psf_attributes.fitted[i])])
            i += 1

        ## load fluxes and IFS coordinates for current layer 
        sources.load_spectra(layers=jj)
        sources.load_ifs_coordinates(layers=jj)

        i  = -1
        ii =  0
        for psf in PSF:
            i += 1
            j  = sources.specrow[i]
            
            if sources.ifs_coordinates_loaded[i]:
                psf.set_xc(sources.ifs_coordinates[0,ii,0,int(sources.ifs_coordinates_fitted[i])])
                psf.set_yc(sources.ifs_coordinates[0,ii,1,int(sources.ifs_coordinates_fitted[i])])
                ii += 1

            ## subtract contribution of the source
            if j != sources.specrow_sky:
                if j != sources.specrow_unresolved or pampelmuse.subunres:

                    layer_kk[psf.used] -= sources.data[0,j,0]*psf.f

            elif pampelmuse.subsky:
                layer_kk -= sources.data[jj,j,0]

        if pampelmuse.ifs_data.instrument.spectral_axis == 2:
            data[jj] = layer_kk
        else:
            data[:,jj] = layer_kk

    pampelmuse.hdulist.close() ## finished reading source catalog.

    pampelmuse.subtres.revision = __revision__
    pampelmuse.subtres.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")

    ## create HDU for output
    logging.info("Creating HDU from processed data...")
    if pampelmuse.ifs_data.ndim==3: ## make output data 3D if input was 3D
        data = pampelmuse.ifs_data.instrument.make_grid(data)
    outHDU = pyfits.PrimaryHDU(data, header=pampelmuse.ifs_data.data_hdu.header)
    
    ## add PampelMuse configuration to HDU just created:
    outHDU = pampelmuse.makeHeader(hdu=outHDU)
  
    ## delete any existing residuals file
    logging.info("Writing results to file {0}.res.fits...".format(pampelmuse.prefix))
    if os.path.exists("{0}.res.fits".format(pampelmuse.prefix)):
        os.remove("{0}.res.fits".format(pampelmuse.prefix))
        logging.warning("Existing residual cube has been overwritten.")

    ## save output (including HDU with fiber setup if available)
    if pampelmuse.ifs_data.calibration_hdu != None:
        HDUList = pyfits.HDUList([outHDU, pampelmuse.ifs_data.calibration_hdu])
        HDUList.writeto("{0}.res.fits".format(pampelmuse.prefix))
    else:
        outHDU.writeto("{0}.res.fits".format(pampelmuse.prefix))

