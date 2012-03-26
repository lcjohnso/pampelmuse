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
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.1.0"
import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core.sources import Sources
from pampelmuse.core.psf import PSF




def subtres(arguments):

    print ""
    print 21* " "+"***  SubtRes, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    psfRad   = arguments.config["PSFRAD"]                 ##
    subtSky  = False
    subtUnRes= False
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


    ## CREATE ARRAYS THAT CONTAIN PARAMETERS USED IN THE FIT
    ## Collect information about sources

    ## Spectra
    logging.info("Obtaining source spectra.")
    sources = Sources(data=prmFile["SOURCES"],  hdu=True)
    sources.loadSpectra(data=prmFile["SPECTRA"], hdu=True)
                 
    ## Positions
    logging.info("Obtaining source coordinates")
    if "POSITIONS" in extNames:
        logging.info("Reading 'POSITIONS' HDU.")
        sources.loadPositionsHDU(prmFile["POSITIONS"])

    elif 'POSPARS' in extNames:
        logging.info("Reading 'POSPARS' HDU.")
        sources.readTransHDU(prmFile["POSPARS"], backup=False)
        sources.resizeStructure(ndisp=sources.ndisp)
        sources.makeValues()
        sources.info()

    ## PSF
    psf = PSF()
    if "PSFPARS" in extNames: ##  check if HDU exists
        logging.info("Reading 'PSFPARS' HDU.")
        psf.readFitsHDU(prmFile["PSFPARS"])
    else: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.error("No HDU 'PSFPARS' found. Assuming Gaussian PSF.")
        psf.makeNewPSF(profile="gauss", fwhm=3.0, e=0., theta=0.)
    psf.info()
    psf.resizeStructure(ndisp=sources.ndisp)
    psf.makeValues()


    ## open datacubes & check consistency
    logging.info("Checking data cube: {0}.dat.fits".format(prefix))
    dataFits  = pyfits.open('{0}.dat.fits'.format(prefix))
    cubeShape = (dataFits[0].header['NAXIS3'],
                 dataFits[0].header['NAXIS2'],
                 dataFits[0].header['NAXIS1'])
    if dataFits[0].header['NAXIS3'] != sources.ndisp:
        logging.critical("Cube and spectra have incompatible dimensions.")

    data = dataFits[0].data ## load full(!) cube into memory.

    for i in xrange(dataFits[0].header['NAXIS3']):
        logging.info(\
            "Handling slice #{0} [{1:.1f}%].".format(i, (100.*i)/sources.ndisp))
        
        ## initialize PSF
        psf_i = psf(index=i)

        ## load current IFU coords:
        sources.loadIFUCoords(index=i)

        ## loop over components:
        for j in xrange(sources.ncmp):

            if j != sources.skyRow:
                if j != sources.unresRow or subtUnRes: 

                    ##  subtract from cube
                    for k in xrange(sources.indexmap[j].size):
  
                        l = np.where(\
                            sources.ifuSelect==sources.indexmap[j][k])[0][0]

                        x_k    = sources.xvalues[l,0]
                        y_k    = sources.yvalues[l,0]
                        flux_k = sources.data[j,i]
                        mag_k  = sources.weights[j][k]

                        data[i] -= flux_k*psf_i(\
                            size=cubeShape[1:], x_c=x_k, y_c=y_k, mag=mag_k,
                            maxrad=psfRad)

            elif subtSky:
                data[i] -= sources.data[j,i]

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
    outHDU.writeto("{0}.res.fits".format(prefix))

