"""
CUBEFIT
=======
The routine is a build around the main component of PampelMuse, namely
the fit_cube-routine. It is called automatically if PampelMuse is started
with the `FITCUBE' option. As input, it takes a dictionary containing the
keywords and their values after PampelMuse was started.

CHANGELOG
---------
2011/09/08+, v0.0.0.0
2011/10/13, v0.0.1.0, updated how PampelMuse-packages are imported
2011/10/27, v0.1.0.0, added possibility to abort stop fit with or
                      without saving current results
2011/12/08, v0.1.0.1, added option to  save cube with residuals of the fit.
                      Also some modifications wrt. to backup of input data.
2012/01/16, v0.1.0.2, modified how images with fit residuals are (temporarily)
                      stored to allow for starting multiple fits from same
                      folder.
2012/01/17, v0.1.0.3, modified code to handle new option of providing IFU
                      pixel coordinates for every source.
2012/01/30, v0.2.0.0, major modifications to make code compliant with updated
                      structure of PRM-files.
2012/01/31, v0.2.0.1, slight changes wrt. logging. 
2012/01/31, v0.2.1.1, fixed bug that code crashed when trying to overwrite an
                      existing residual cube.
2012/02/28, v0.3.0.0, significantly restructured/shortened the routine by
                      introducing dedicated classes to handle PSF and
                      source posistions [usage of class `Spectra' TBD]
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.3.0.0"
import logging, os, sys
import numpy as np, pyfits
from PyQt4 import QtCore, QtGui

## required PampelMuse-packages
from pampelmuse.core.fit_array import StarFit
from pampelmuse.core.psf import PSF
from pampelmuse.core.positions import Positions
## from pampemuse.core.spectra import Spectra

def cubefit(arguments):

    print ""
    print 21* " "+"***  CubeFit, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    fitRad   = arguments.config["FITRAD"]                 ##
    psfRad   = arguments.config["PSFRAD"]                 ##
    smallIFU = arguments.config["SMALLIFU"]               ##
    dobackup = arguments.config["DOBACKUP"]               ##
    storeRes = arguments.config["STORERES"]               ##
    jumpStart= arguments.config["JUMPSTART"]              ##
    ########################################################


    ## open PRM-file, get no. of sources
    logging.info("Reading PampelMuse-file: {0}.prm.fits".format(prefix))
    prmFile  = pyfits.open("{0}.prm.fits".format(prefix))
    extNames = [prmFile[i].name for i in xrange(len(prmFile))]

    Nsrc = prmFile['SOURCES'].header['NAXIS2'] ## no of sources
    Ndsp = prmFile['SPECTRA'].header['NAXIS1'] ## no of spectral pixels
    Nspc = prmFile['SPECTRA'].header['NAXIS2'] ## no of resolved components

    ## Check for previous PampelMuse-results
    try:
        lastFitIndex = prmFile[0].header['PMFITIND']
        if lastFitIndex == Ndsp or lastFitIndex == "SUCCESS":
            lastFitIndex = None
    except KeyError:
        lastFitIndex = None
    if lastFitIndex !=  None:
        logging.info("Continuing previously paused fit, current image: {0}"\
                         .format(lastFitIndex))


    ## CREATE ARRAYS THAT CONTAIN PARAMETERS USED IN THE FIT
    ## Collect information about sources
    logging.info("Reading 'SOURCES' HDU.")
    srcList   = np.array(prmFile['SOURCES'].data,
                         dtype=prmFile['SOURCES'].data.dtype)
    positions = Positions(prmFile["SOURCES"], hdu=True)

    if "POSITIONS" in extNames:
        logging.info("Reading 'POSITIONS' HDU.")

        positions.loadIfuCoords(prmFile["POSITIONS"], hdu=True)

    elif (srcList['status']==3).any():
        logging.info("Preparing IFU coordinates for PSF sources.")

        isPSFsource = srcList['status']==3
        ifuData = np.zeros((2, isPSFsource.sum(), Ndsp), dtype=np.float32)
        ifuData[0] = srcList["x"][isPSFsource]
        ifuData[1] = srcList["y"][isPSFsource]

        positions.loadIfuCoords(data=ifuData, ids=srcList['id'][isPSFsource],
                                freepar=True, isfitted=False)


    ## collect the  spectra
    logging.info("Reading 'SPECTRA' HDU.")
    spcList = prmFile['SPECTRA'].data
    

    ## check if uncertainties associated with spectra are available:
    if "SIGMASPC" in extNames:
        logging.info("Reading 'SIGMASPC' HDU.")
        spcSigmaList = prmFile['SIGMASPC'].data
    else:
        spcSigmaList = None


    if 'POSPARS' in extNames:
        logging.info("Reading 'POSPARS' HDU.")
        positions.readTransHDU(prmFile["POSPARS"], backup=dobackup)
        positions.resizeStructure(size=Ndsp)
        positions.makeParameters()
        positions.info()

    else:
        logging.warning("No coord. transf. (HDU 'POSPARS') detected.")


    ## Collect information about PSF
    psf = PSF()
    if "PSFPARS" in extNames: ##  check if HDU exists
        logging.info("Reading 'PSFPARS' HDU.")
        psf.readFitsHDU(prmFile["PSFPARS"])

    else: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.warning("No HDU 'PSFPARS' found. Assuming Gaussian PSF.")

        psf.makeNewPSF(profile="gauss", fwhm=3.0, e=0., theta=0.)

    psf.info()
    psf.resizeStructure(size=Ndsp)
    psf.makeParameters()


    prmFile.close() ## finished reading source catalog.


    ## open datacubes & check consistency
    logging.info("Checking data cubes.")
    dataFits  = pyfits.open('{0}.dat.fits'.format(prefix))
    cubeShape = (dataFits[0].header['NAXIS3'],
                 dataFits[0].header['NAXIS2'],
                 dataFits[0].header['NAXIS1'])
    if dataFits[0].header['NAXIS3'] != Ndsp:
        logging.critical("Data-cube and PRM-file have incompatible dimensions.")

    sigmaFits = pyfits.open('{0}.err.fits'.format(prefix))
    if sigmaFits[0].header['NAXIS3'] != Ndsp:
        logging.critical(\
            "Sigma-cube and PRM-file have incompatible dimensions.")

    if os.path.exists("{0}.msk.fits".format(prefix)):
        maskFits = pyfits.open('{0}.msk.fits'.format(prefix))
        if maskFits[0].header['NAXIS3'] != Ndsp:
            logging.critical(\
                "Mask-cube and PRM-file have incompatible dimensions.")
    else:
        maskFits = [None]


    ## start fitting process:
    logging.info("Starting fit process.")
    app  = QtGui.QApplication(sys.argv)

    fitter = StarFit(dataFits=dataFits[0], sigmaFits=sigmaFits[0],
                     maskFits=maskFits[0])

    fitter.provideSources(sources=srcList, spectra=spcList, positions=positions,
                          specSigma=spcSigmaList)

    fitter.providePSF(psf=psf, psfRad=psfRad)
        
    fitter(fitRad=fitRad, lastFitIndex=lastFitIndex, asSmallIFU=smallIFU,
           returnResiduals=storeRes, prefix=prefix, jumpStart=jumpStart)

    app.exec_()


    if not fitter.save:
        sys.exit()
        
    logging.info("Saving results.")

    ## collect fitting results
    prmFile  = pyfits.open("{0}.prm.fits".format(prefix), mode='update')
    extNames = [prmFile[i].name for i in xrange(len(prmFile))]

    ## save status of fit process in header of primary HDU
    if fitter.i != Ndsp:
        prmFile[0].header.update('PMFITIND', fitter.i,
                                 "Status of PampelMuse fitting")
    else:
        prmFile[0].header.update('PMFITIND', 'SUCCESS',
                                 "Status of PampelMuse fitting")

    ## store spectra
    crval1 = prmFile['SPECTRA'].header['CRVAL1'] 
    cdelt1 = prmFile['SPECTRA'].header['CDELT1'] 
    crpix1 = prmFile['SPECTRA'].header['CRPIX1'] 

    newSpecHDU = pyfits.ImageHDU(fitter.spectra) 
    newSpecHDU.header.update('EXTNAME', 'SPECTRA')
    newSpecHDU.header.update('CRPIX1', crpix1)
    newSpecHDU.header.update('CRVAL1', crval1)
    newSpecHDU.header.update('CDELT1', cdelt1)

    if dobackup and lastFitIndex == None:
        if 'INSPECTRA' in extNames:
            prmFile.remove(prmFile['INSPECTRA'])
        prmFile['SPECTRA'].header.update('EXTNAME', 'INSPECTRA')
    else:
        prmFile.remove(prmFile['SPECTRA'])
    prmFile.insert(2, newSpecHDU)

    
    ## store uncertainties associated with spectra
    newSigmaHDU = pyfits.ImageHDU(fitter.specSigma) 
    newSigmaHDU.header.update('EXTNAME', 'SIGMASPC')
    newSigmaHDU.header.update('CRPIX1', crpix1)
    newSigmaHDU.header.update('CRVAL1', crval1)
    newSigmaHDU.header.update('CDELT1', cdelt1)

    if dobackup and lastFitIndex == None:
        if 'INSIGMASPC' in extNames:
            prmFile.remove(prmFile['INSIGMASPC'])
        if 'SIGMASPC' in extNames:
                prmFile['POSPARS'].header.update('EXTNAME', 'INSIGMASPC')
    else:
        if 'SIGMASPC' in extNames:
            prmFile.remove(prmFile['SIGMASPC'])
    prmFile.insert(3, newSigmaHDU)


    ## Store IFU coordinates (only if they have been fitted)
    if positions.unsavedIfu:
        if "POSITIONS" in extNames:
            prmFile["POSITIONS"] = positions.makeIfuHDU()[0]
        else:
            prmFile.insert(4, positions.makeIfuHDU()[0])


    ## store coordinate transformation (only if it has been fitted)
    if positions.unsavedTrans:
        if "POSPARS" in extNames:
            prmFile["POSPARS"] = positions.makeTransHDU()
        else:
            prmFile.insert(5, positions.maskTransHDU()[0])


    ## store PSF (only if it as been fitted)
    if psf.unsavedPSF:
        if "PSFPARS" in extNames:
            prmFile["PSFPARS"] = psf.makeFitsHDU()
        else:
            prmFile.insert(5, psf.makeFitsHDU())

    prmFile.close()



    if storeRes and fitter.i == Ndsp:

        resCube = np.zeros(cubeShape, dtype="float32")

        for i in xrange(Ndsp):
            
            jj = (Ndsp>>1) - 1 - int(i/2) + i*np.mod(i, 2) + np.mod(Ndsp, 2)

            resCube[jj] = pyfits.getdata("{0}.res{1}.fits".format(prefix, i))
            os.system("rm {0}.res{1}.fits".format(prefix, i))

        resHDU = pyfits.PrimaryHDU(resCube)
        if os.path.exists("{0}.res.fits".format(prefix)):
            os.remove("{0}.res.fits".format(prefix))
            logging.warning("Existing residual cube has been overwritten.")
        resHDU.writeto("{0}.res.fits".format(prefix))

