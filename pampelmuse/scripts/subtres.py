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
from pampelmuse.core.psf import PSF
from pampelmuse.core.positions import Positions
from pampelmuse.core.spectra import Spectra


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
    spectra = Spectra(prmFile["SOURCES"],  hdu=True)
    spectra.loadSpectra(data=prmFile["SPECTRA"], hdu=True)
                 
    ## Positions
    logging.info("Obtaining source coordinates")
    positions = Positions(prmFile["SOURCES"], hdu=True)
    if "POSITIONS" in extNames:
        logging.info("Reading 'POSITIONS' HDU.")
        positions.loadIfuCoords(prmFile["POSITIONS"], hdu=True)

    elif 'POSPARS' in extNames:
        logging.info("Reading 'POSPARS' HDU.")
        positions.readTransHDU(prmFile["POSPARS"], backup=False)
        positions.resizeStructure(size=spectra.ndisp)
        positions.makeParameters()
        positions.info()

    ## PSF
    psf = PSF()
    if "PSFPARS" in extNames: ##  check if HDU exists
        logging.info("Reading 'PSFPARS' HDU.")
        psf.readFitsHDU(prmFile["PSFPARS"])
    else: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.error("No HDU 'PSFPARS' found. Assuming Gaussian PSF.")
        psf.makeNewPSF(profile="gauss", fwhm=3.0, e=0., theta=0.)
    psf.info()
    psf.resizeStructure(size=spectra.ndisp)
    psf.makeParameters()

    prmFile.close() ## finished reading source catalog.


    ## open datacubes & check consistency
    logging.info("Checking data cube: {0}.dat.fits".format(prefix))
    dataFits  = pyfits.open('{0}.dat.fits'.format(prefix))
    cubeShape = (dataFits[0].header['NAXIS3'],
                 dataFits[0].header['NAXIS2'],
                 dataFits[0].header['NAXIS1'])
    if dataFits[0].header['NAXIS3'] != spectra.ndisp:
        logging.critical("Cube and spectra have incompatible dimensions.")

    data = dataFits[0].data ## load full(!) cube into memory.

    for i in xrange(dataFits[0].header['NAXIS3']):
        logging.info(\
            "Handling slice #{0} [{1:.1f}%].".format(i, (100.*i)/spectra.ndisp))
        
        ## initialize PSF
        psf_i = psf(index=i)

        ## loop over components:
        for j in xrange(spectra.ncmp):

            if j != spectra.skyRow:
                if j != spectra.unResRow or subtUnRes: 
                    IDs     = spectra.ID[spectra.map[j]] 
                    relMags = spectra.weights[j]

                    ## - get position(s) and flux(es)
                    x_ij, y_ij = positions.getIfuCoords(index=i, ids=IDs)
                    flux_ij    = spectra.data[j,i]

                    ##  subtract from cube
                    for k in xrange(spectra.map[j].size):
                        data[i] -= flux_ij*psf_i(\
                            size=cubeShape[1:], x_c=x_ij[k], y_c=y_ij[k],
                            mag=relMags[k], maxrad=psfRad)

            elif subtSky:
                data[i] -= spectra.data[j,i]


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

