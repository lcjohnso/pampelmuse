"""
SUBTRES
=======
The routine is designed to subtract the contribution of the resolved sources
from an IFU datacube. To this aim, a valid PampelMuse PRM-file must exist along
with the datacube

CHANGELOG
---------
2012/03/01, v0.1.0.0
2012/03/02, v0.1.1.0, fixed bug that code crashed if header key PMFITIND
                      was missing.
2012/04/17, v0.2.0.0, added `SUBSKY` and `SUBUNRES`-parameters to PampelMuse
                      configuration.
2012/04/22, v0.2.1.0, bugfix: replaced `ndisp` with `size` when calling
                              `resizeStructure`
2012/11/20, v0.3.0.0, included 'Instruments'-class so that dead or non-existent
                      spaxels are excluded from the computation
2013/03/27, v0.4.0.0, modified code to implement new way of creating PSF
                      profiles for the sources. probably still needs to be
                      debugged.
2013/03/28, v0.4.1.0, some debugging done, especially fixed bug that code
                      did not subtract anything because the loop over all
                      sources got stuck at the first source. Also now saving
                      fiber setup of Argus in output FITS-file if available.
2013/04/10, v0.4.2.0, further debugging: HDU with coordinate transformation
                      was not always loaded but always required.
2013/05/31, v0.5.0.0, modified overall structure of code so that it is
                      compatible with the new way the PampelMuse configuration
                      is handled.
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.4.2.0"
import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core.sources import Sources
from pampelmuse.core.psf import PsfAttributes
from pampelmuse.core.new_profiles import Gaussian, Moffat




def subtres(pampelmuse):

    print ""
    print 21* " "+"***  SubtRes, version {0}  ***".format(__version__)+21*" "
    print ""

    ## Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    ## Source catalogue
    sources = Sources(data=pampelmuse.hdulist["SOURCES"], instrument=pampelmuse.ifs_data.instrument)


    ## Check if PRM-file contains results of successful fit
    if hasattr(pampelmuse, "cubefit") and hasattr(pampelmuse.cubefit, "status"):
        fitStatus = pampelmuse.cubefit.status
    else:
        fitStatus = "Unknown"

    if fitStatus != "SUCCESS":
        logging.error("PRM-file does not contain results of successful fit.")


    ## Collect source spectra
    logging.info("Obtaining source spectra.")
    try:
        i = pampelmuse.hdulist.index_of("SPECTRA")
        sources.loadSpectra(data=pampelmuse.hdulist[i])
    except KeyError:
        logging.critical("Cannot find spectra (HDU `SPECTRA') in provided PRM-file.")


    ## Collect source positions
    logging.info("Obtaining information about source coordinates.")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.loadPositionsHDU(pampelmuse.hdulist[i])
        try:
            i = pampelmuse.hdulist.index_of("INPOSITIONS")
            sources.loadPositionsHDU(pampelmuse.hdulist[i])
        except KeyError:
            pass
    except KeyError:
        pass

    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.readTransHDU(pampelmuse.hdulist[i], backup=False)
        sources.resizeStructure(size=sources.ndisp)
        sources.makeValues()
        sources.info()
        
    except KeyError:
        logging.warning("No coord. transf. (HDU 'POSPARS') detected.")


    ## Initializing PSF properties
    psf_attributes = PsfAttributes()
    try:
        i = pampelmuse.hdulist.index_of("PSFPARS")
        psf_attributes.readFitsHDU(pampelmuse.hdulist[i])
    except KeyError: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.warning("No HDU 'PSFPARS' found. Using default PSF.")
        psf_attributes.makeNewPSF(profile=pampelmuse.psftype)

        i=0
        while hasattr(pampelmuse.psf, "parname{0}".format(i)):
            psf_attributes.addParameter(name=getattr(pampelmuse.psf, "parname{0}".format(i)),
                                        value=getattr(pampelmuse.psf, "parvalue{0}".format(i)), 
                                        freepar=getattr(pampelmuse.psf, "freepar{0}".format(i)))

    psf_attributes.resizeStructure(ndisp=sources.ndisp)
    psf_attributes.makeValues()
    psf_attributes.info()

    ## Initialize PSF profiles of the sources:
    if psf_attributes.psftype == "gauss":
        PSF = Gaussian
    elif psf_attributes.psftype == "moffat":
        PSF = Moffat
    else:
        logging.critical("Invalid PSF type: '{0}'".format(psf_attributes.psftype))
        raise IOError, "PSF type not understood."
    PSF.set_transform(pampelmuse.ifs_data.instrument.transform)
    for i in xrange(sources.nsrc):
        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=sources.weights[i], maxrad=pampelmuse.psfrad, osfactor=5, osradius=2)


    ## load full(!) cube into memory.
    logging.info("Reading full cube into memory...")
    data = pampelmuse.ifs_data.data_hdu.data


    for kk in xrange(pampelmuse.start, pampelmuse.stop):
        logging.info("Handling layer #{0} [{1:.1f}%].".format(kk, (100.*(kk-pampelmuse.start))/(pampelmuse.stop-pampelmuse.start+1)))
        
        ## account for offset between layers of datacube and PRM file if not the whole cube is analyzed 
        jj = kk - pampelmuse.start

        ## update PSF instances
        ## -with current parameter fits
        i = 0 
        for prm in psf_attributes.parameters:
            getattr(PSF, "set_{0}".format(prm))(psf_attributes.values[i, jj])
            i += 1

        ## -with parameters of coordinate transformation (if available)
        if sources.verifyTrans:
            i = 0
            for prm in sources.parameters:
                getattr(PSF, "set_{0}".format(prm))(sources.ctValues[i, jj])
                i += 1

        ## -with direct positions if required
        i  = -1
        ii =  0
        
        sources.loadIFUCoords(index=jj)

        for psf in PSF:
            i += 1
            j  = sources.specrow[i]
            
            if sources.loadedIFUCoords[i]:
                if sources.hasPolyFit[i]:
                    psf.set_xc(sources.xdata[1,ii,0])
                    psf.set_yc(sources.ydata[1,ii,0])
                elif sources.isFreePar[i]:
                    psf.set_xc(sources.xdata[0,ii,0])
                    psf.set_yc(sources.ydata[0,ii,0])
                ii += 1


            if j != sources.skyRow:
                if j != sources.unresRow or pampelmuse.subunres: 

                    flux = np.zeros((pampelmuse.ifs_data.instrument.transform.shape[0],), dtype=np.float32)

                    flux[psf.used] += sources.data[j,jj]*psf.f

                    if pampelmuse.ifs_data.ndim == 3:
                        data[kk] -= pampelmuse.ifs_data.instrument.make_grid(flux)
                    elif pampelmuse.ifs_data.instrument.spatial_axis==1:
                        data[kk, pampelmuse.ifs_data.instrument.isscience] -= flux
                    else:
                        data[pampelmuse.ifs_data.instrument.isscience, kk] -= flux

            elif pampelmuse.subsky:
                if ndim == 3:
                    data[kk] -= sources.data[j,jj]
                elif pampelmuse.ifs_data.instrument.spatial_axis==1:
                    data[kk, pampelmuse.ifs_data.instrument.isscience] -= sources.data[j,jj]
                else:
                    data[pampelmuse.ifs_data.instrument.isscience, kk] -= sources.data[j,jj]

    pampelmuse.hdulist.close() ## finished reading source catalog.

    pampelmuse.subtres.version = __version__
    pampelmuse.subtres.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")

    ## create HDU for output
    outHDU = pyfits.PrimaryHDU(data, header=pampelmuse.ifs_data.data_hdu.header)
    
    ## add PampelMuse configuration to HDU just created:
    outHDU = pampelmuse.makeHeader(hdu=outHDU)
  
    ## delete any existing residuals file
    if os.path.exists("{0}.res.fits".format(pampelmuse.prefix)):
        os.remove("{0}.res.fits".format(pampelmuse.prefix))
        logging.warning("Existing residual cube has been overwritten.")

    ## save output (including HDU with fiber setup if available)
    if pampelmuse.ifs_data.calibration_hdu != None:
        HDUList = pyfits.HDUList([outHDU, pampelmuse.ifs_data.calibration_hdu])
        HDUList.writeto("{0}.res.fits".format(pampelmuse.prefix))
    else:
        outHDU.writeto("{0}.res.fits".format(pampelmuse.prefix))

