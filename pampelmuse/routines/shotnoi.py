"""
shotnoi.py
==========

Provides
--------
function pampelmuse.routines.shotnoi.shotnoi

Purpose
-------
The module implements th function 'shotnoi' that is called if PampelMuse is
called with the SHOTNOI option. The function estimates the strength of shot
noise in every spaxel of an IFS dataset. As a measure for the shot noise, the
function uses the fractional flux that the brightest star contributes to the
integrated light in a spaxel.

The common prefix of the names of the files associated to the datacube under
investigation must be provided. Optionally, a photometry file can be provided.
It none is provided, the source list included in the PRM-file is used. The
former option is useful if only the brghter stars are included in the PRM-file.

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
from pampelmuse.functions.io import read_asciifile



def shotnoi(pampelmuse):

    logging.info("Executing requested routine SHOTNOI [Revision {0:d}].".format(__revision__))


    ## Check if PRM-file contains results of successful fit
    if hasattr(pampelmuse, "cubefit") and hasattr(pampelmuse.cubefit, "status"):
        fitStatus = pampelmuse.cubefit.status
    else:
        fitStatus = "Unknown"

    if fitStatus != "SUCCESS":
        logging.error("PRM-file does not contain results of successful fit.")


    ## Check if dedicated photometry file provided
    if pampelmuse.shotnoi.sourcelist != None:
        logging.info("Additional photometry provided: {0}".format(pampelmuse.shotnoi.sourcelist))
        cfp_results = read_asciifile(pampelmuse.shotnoi.sourcelist, usecols=pampelmuse.cfpcols, usenames="id,x,y,{0}".format(pampelmuse.passband), 
                                     aliases={"F606W":["f606w", "Vvega", "V"], "F814W":["f814w", "Ivega", "I"]})
        add_id = np.asarray(cfp_results["id"], dtype=np.int32)
        add_x = np.asarray(cfp_results["x"], dtype=np.float32)
        add_y = np.asarray(cfp_results["y"], dtype=np.float32)
        add_mag = np.asarray(cfp_results[pampelmuse.passband], dtype=np.float32)

        ## it is assumed that the reference coordinates in the PRM-file are from the same data, i.e. they only differ from the ones in the
        ## photometry by a constant offset in x & y.

        ## find common entries in both lists
        logging.info("Matching additional photometry to sources available in PRM-file...")
        match  = (add_id[:,np.newaxis]==pampelmuse.hdulist["SOURCES"].data.field("id"))
        m1 = match.argmax(axis=0)[match.any(axis=0)]
        m2 = match.any(axis=0)
        logging.info("Found {0} common sources in both catalogues.".format(m1.size))
    
        ## determine mean offset of matched sources in x and y and modify positions in additional photometry
        x_off = np.mean(add_x[m1]-pampelmuse.hdulist["SOURCES"].data.field("x")[m2])
        y_off = np.mean(add_y[m1]-pampelmuse.hdulist["SOURCES"].data.field("y")[m2])

        logging.info("Calculated offset of additional photometry in (x,y) is ({0:.2f},{1:.2f}).".format(x_off, y_off))
        
        add_x -= x_off
        add_y -= y_off
        
        ## only select sources in footprint covered by sources in PRM file & only select sources that are not yet in the PRM file
        select = (add_x>pampelmuse.hdulist["SOURCES"].data.field("x")[m2].min())&(add_x<pampelmuse.hdulist["SOURCES"].data.field("x")[m2].max())\
            &(add_y>pampelmuse.hdulist["SOURCES"].data.field("y")[m2].min())&(add_y<pampelmuse.hdulist["SOURCES"].data.field("y")[m2].max())\
            &(np.in1d(np.arange(add_id.size, dtype=np.int32), m1)==False)

        ## create combined table
        logging.info("Creating combined source catalogue...")
        _nrows = pampelmuse.hdulist["SOURCES"].data.shape[0]
        hdu = pyfits.new_table(pampelmuse.hdulist["SOURCES"].columns, nrows=_nrows+select.sum())
        hdu.data.field("id")[_nrows:]  = add_id[select]
        hdu.data.field("x")[_nrows:]   = add_x[select]
        hdu.data.field("y")[_nrows:]   = add_y[select]
        hdu.data.field("mag")[_nrows:] = add_mag[select]
        hdu.data.field("status")[_nrows:]  = -1
        hdu.data.field("primary")[_nrows:] = -2    
        hdu.data.field("specrow")[_nrows:] = -1 ## pampelmuse.hdulist["SOURCES"].data.field("specrow").max() + 1

        sources = Sources(data=hdu)
    else:
        sources = Sources(data=pampelmuse.hdulist["SOURCES"])


    ## Collect source spectra
    logging.info("Obtaining source spectra...")
    try:
        i = pampelmuse.hdulist.index_of("SPECTRA")
        sources.open_spectra_hdu(pampelmuse.hdulist[i])
    except KeyError:
        logging.error("Cannot find spectra (HDU `SPECTRA') in provided PRM-file.")


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
        if pampelmuse.shotnoi.sourcelist != None: logging.error("Cannot determine IFS coordinates for sources in additional photometry.")


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
        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=0, maxrad=pampelmuse.psfrad, osfactor=5, osradius=2)
        ## Note that the weights given to the individual sources that constitute a component are implemented
        ## below when calclulating fluxes. Each PSF profile has an integrated flux of 1.


    ## initialize output array: the first half of it contains the flux distributions of all stars, the second half
    ## contains only the contributions from the unresolved stars
    data_flux = np.zeros((2, sources.nsrc, pampelmuse.ifs_data.instrument.nspaxel), dtype=np.float32)

    ## determine integrated fluxes of resolved and unresolved stars
    rsv_ifu = 0. ## total flux of resolved stars from IFS data
    rsv_cat = 0. ## total flux of resolved stars from photometric data
    unr_ifu = 0. ## flux from unresolved stars, calculated from IFS data
    unr_cat = 0. ## flux from unresolved stars, calculated from photometric data

    sources.info()
    ## loop over layers; currently the central 20 layers of the covered wavelength range of the IFS data are used
    for kk in xrange(pampelmuse.ifs_data.nlayer//2-10, pampelmuse.ifs_data.nlayer//2+10):

        jj = kk - pampelmuse.start ## account for possibility that analysis was not started at first layer of IFS data

        ## update PSF instances of the sources with the variables fitted for current layer
        i = 0 
        for prm in psf_attributes.names:
            getattr(PSF, "set_{0}".format(prm))(psf_attributes.data[jj,i,int(psf_attributes.fitted[i])])
            i += 1

        ## load fluxes and IFS coordinates for current layer 
        sources.load_spectra(layers=jj)
        sources.load_ifs_coordinates(layers=jj)
        
        ## loop over sources:
        i=0
        ii=0
        for psf in PSF:
       
            if sources.ifs_coordinates_loaded[i]:
                psf.set_xc(sources.ifs_coordinates[0,ii,0,int(sources.ifs_coordinates_fitted[i])])
                psf.set_yc(sources.ifs_coordinates[0,ii,1,int(sources.ifs_coordinates_fitted[i])])
                ii += 1
       
            j = sources.specrow[i]

            if j == sources.specrow_sky:
                unr_ifu += sources.data[0,j,0].sum()
            else:
                data_flux[0,i,psf.used] += 10**(-0.4*(sources.mag[i]-25.))*psf.f

                if sources.status[i] <= 0:
                    data_flux[1,i] = data_flux[0,i]
                    unr_cat += psf.f.sum()*10**(-0.4*(sources.mag[i]-25.))
                    if sources.status[i]==0:
                        unr_ifu += psf.f.sum()*sources.data[0,j,0]*10**(-0.4*sources.weights[i])
                else:
                    rsv_cat += psf.f.sum()*10**(-0.4*(sources.mag[i]-25.))
                    rsv_ifu += psf.f.sum()*sources.data[0,j,0]*10**(-0.4*sources.weights[i])
            i += 1

    ## Print out fraction of unresolved light using calculations based on IFS analysis or photometric data
    logging.info("Contribution of unresolved to total light:")
    logging.info("  Photometric catalogue: {0:.1f}%".format(float(100.*unr_cat/(rsv_cat+unr_cat))))
    logging.info("  PampelMuse analysis: {0:.1f}%".format(float(100.*unr_ifu/(rsv_ifu+unr_ifu))))


    pampelmuse.hdulist.close() ## finished reading source catalog.

    ## create two arrays, each representing the FoV of the IFS and showing for each spaxel the fractional flux of the 
    ## source with the largest contribution. In the first array, all sources are considered, in the second array, only
    ## the unresolved sources
    
    ## sort entries per spaxel by flux 
    data_flux.sort(axis=1)

    ## get cumulative sum for each spaxel
    data_flux_summed  = data_flux.cumsum(axis=1)

    ## divide cumulative sums by total flux in each spaxel to get fractional fluxes
    data_flux_summed[0] /= data_flux_summed[0,-1]
    data_flux_summed[1] /= data_flux_summed[1,-1]

    ## the contribution of the brightest source is the difference between the last and second-to-last entry of the cumulative sums
    shotnoise = data_flux_summed[:,-1]-data_flux_summed[:,-2]

    ## save results in a FITS file
    outHDU = pyfits.PrimaryHDU(pampelmuse.ifs_data.instrument.make_grid(shotnoise))

    pampelmuse.shotnoi.revision = __revision__
    pampelmuse.shotnoi.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")

    ## add PampelMuse configuration to HDU just created:
    outHDU = pampelmuse.makeHeader(hdu=outHDU)
  
    if os.path.exists("{0}.shot.fits".format(pampelmuse.prefix)):
        os.remove("{0}.shot.fits".format(pampelmuse.prefix))
        logging.warning("Existing shot noise cube has been overwritten.")
    outHDU.writeto("{0}.shot.fits".format(pampelmuse.prefix))

