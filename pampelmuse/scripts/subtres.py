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
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.4.1.0"
import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core.sources import Sources
from pampelmuse.core.psf import PsfAttributes
from pampelmuse.core.new_profiles import Gaussian, Moffat




def subtres(arguments):

    print ""
    print 21* " "+"***  SubtRes, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    psfRad   = arguments.config["PSFRAD"]                 ##
    subtSky  = arguments.config["SUBSKY"]                 ##
    subtUnRes= arguments.config["SUBUNRES"]               ##
    copyHead = True
    ########################################################


    ## open PRM-file
    logging.info("Reading PampelMuse-file: {0}.prm.fits".format(prefix))
    prmFile  = pyfits.open("{0}.prm.fits".format(prefix))
    extNames = [prmFile[i].name for i in xrange(len(prmFile))]

    ## Check if PRM-file contains results of successful fit
    try: fitStatus = prmFile[0].header['PAMPELMUSE CUBEFIT STATUS']
    except KeyError:
        try: fitStatus = prmFile[0].header['PMFITIND']
        except KeyError: fitStatus = None

    if fitStatus != "SUCCESS":
        logging.warning("PRM-file does not contain results of successful fit.")


    ## Check if information on used instrument available
    if "PAMPELMUSE IFU" in prmFile[0].header.keys():
        instrument = prmFile[0].header["PAMPELMUSE IFU"]
    else:
        instrument = None


    ## Collect information about sources
    ## -Spectra
    logging.info("Obtaining source spectra.")
    sources = Sources(data=prmFile["SOURCES"],  hdu=True, instrument=instrument)
    sources.loadSpectra(data=prmFile["SPECTRA"], hdu=True)
                 
    ## -Positions
    logging.info("Obtaining source coordinates")
    if "POSITIONS" in extNames:
        logging.info("Reading 'POSITIONS' HDU.")
        sources.loadPositionsHDU(prmFile["POSITIONS"])

    elif 'POSPARS' in extNames:
        logging.info("Reading 'POSPARS' HDU.")
        sources.readTransHDU(prmFile["POSPARS"], backup=False)
        sources.resizeStructure(size=sources.ndisp)
        sources.makeValues()
        sources.info()

    ## -PSF attributes
    psf_attributes = PsfAttributes()
    if "PSFPARS" in extNames: ##  check if HDU exists
        logging.info("Reading 'PSFPARS' HDU.")
        psf_attributes.readFitsHDU(prmFile["PSFPARS"])
    else: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.error("No HDU 'PSFPARS' found. Assuming Gaussian PSF.")
        psf_attributes.makeNewPSF(profile="gauss", fwhm=3.0, e=0., theta=0.)
    psf_attributes.info()
    psf_attributes.resizeStructure(ndisp=sources.ndisp)
    psf_attributes.makeValues()

    
    ## open datacubes & check consistency
    logging.info("Checking data cube: {0}.dat.fits".format(prefix))
    dataFits  = pyfits.open('{0}.dat.fits'.format(prefix))

    ## check if calibration HDU for ARGUS exists
    fiber_calib = False
    if sources.instrument.name == "argus":
        try:
            setup_index = dataFits.index_of("FIBER_SETUP")
            sources.instrument.read_fiber_setup(hdu=dataFits[setup_index])
            fiber_calib = True
        except KeyError:
            logging.warning("No HDU with fibre setup found for ARGUS data.")

    ## Check format of datacube
    ndim     = dataFits[0].header["NAXIS"]

    if sources.instrument.transform == None:
        if ndim!=3:
            logging.error(\
                "Data must be 3D unless mapping from spaxels to x&y specified.")
            assert ndim==3, "Expected 3dim. datacube, got {0}dim".format(ndim)

        ## initialize transformation for 'Instrument'-class
        xdim = dataFits[0].header['NAXIS1']
        ydim = dataFits[0].header['NAXIS2']

        x = np.mod(np.arange(xdim*ydim, dtype=np.int16), xdim)
        y = np.arange(xdim*ydim, dtype=np.int16)//xdim

        sources.instrument.transform = np.vstack((y,x)).T
        
    if ndim == 3:
        ndisp = dataFits[0].header['NAXIS3']
    else:
        ndisp = dataFits[0].header[\
            'NAXIS{0}'.format(sources.instrument.spectral_axis)]

    ## Check the range of layers to be fitted 
    try: start = prmFile[0].header['PAMPELMUSE START']
    except KeyError: start = 0
    try: stop = prmFile[0].header['PAMPELMUSE STOP']
    except KeyError: stop = ndisp-1
    
    if (stop-start+1) != sources.ndisp:
        logging.critical("Cube and PRM-file have incompatible sizes.")


    data = dataFits[0].data ## load full(!) cube into memory.

    ## Initialize PSF profiles of the sources:
    if psf_attributes.psftype == "gauss":
        PSF = Gaussian
    elif psf_attributes.psftype == "moffat":
        PSF = Moffat
    else:
        logging.critical("Invalid PSF type: '{0}'".format(\
                psf_attributes.psftype))
        raise IOError, "PSF type not understood."
    PSF.set_transform(sources.instrument.transform)
    for i in xrange(sources.nsrc):
        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=sources.weights[i],
                  maxrad=psfRad, osfactor=5, osradius=2)


    for kk in xrange(start, stop):
        logging.info("Handling layer #{0} [{1:.1f}%].".format(\
                kk, (100.*(kk-start))/(stop-start+1)))
        
        jj = kk - start

        ## update PSF instances with current parameter fits
        i = 0 
        for prm in psf_attributes.parameters:
            getattr(PSF, "set_{0}".format(prm))(psf_attributes.values[i, jj])
            i += 1
        
        i = 0
        for prm in sources.parameters:
            getattr(PSF, "set_{0}".format(prm))(sources.ctValues[i, jj])
            i += 1


        ## loop over components:
        i = 0
        for psf in PSF:
            
            j = sources.specrow[i]

            if j != sources.skyRow:
                if j != sources.unresRow or subtUnRes: 

                    flux = np.zeros((sources.instrument.nspaxel,),
                                    dtype=np.float32)

                    flux[psf.used] += sources.data[j,jj]*psf.f

                    if ndim == 3:
                        data[kk] -= sources.instrument.make_grid(flux)
                    elif sources.instrument.spatial_axis==1:
                        data[kk, sources.instrument.isscience] -= flux
                    else:
                        data[sources.instrument.isscience, kk] -= flux

            elif subtSky:
                if ndim == 3:
                    data[kk] -= sources.data[j,jj]
                elif sources.instrument.spatial_axis==1:
                    data[kk, sources.instrument.isscience] -= sources.data[j,jj]
                else:
                    data[sources.instrument.isscience, kk] -= sources.data[j,jj]
            i += 1

    prmFile.close() ## finished reading source catalog.


    outHDU = pyfits.PrimaryHDU(data)
    if copyHead:
        outHDU.header = dataFits[0].header
    outHDU.header.update(\
        "hierarch PAMPELMUSE SUBTRES UTC",
        datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S"))
    outHDU.header.update("hierarch PAMPELMUSE SUBTRES VERSION", __version__)
    outHDU.header.update("hierarch PAMPELMUSE SUBTRES PSFRAD", psfRad)
    outHDU.header.update("hierarch PAMPELMUSE SUBTRES SUBTSKY", subtSky)
    outHDU.header.update("hierarch PAMPELMUSE SUBTRES SUBTUNRES", subtUnRes)
  
    if os.path.exists("{0}.res.fits".format(prefix)):
        os.remove("{0}.res.fits".format(prefix))
        logging.warning("Existing residual cube has been overwritten.")

    if fiber_calib:
        HDUList = pyfits.HDUList([outHDU, dataFits[setup_index]])
        HDUList.writeto("{0}.res.fits".format(prefix))
    else:
        outHDU.writeto("{0}.res.fits".format(prefix))

