"""
CUBEFIT
=======
The routine is a build around the main component of PampelMuse, namely the
fit_array.py-routine. It is called automatically if PampelMuse is started
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
2012/02/28, v0.3.1.0, bugfix: HDU `INSIGMASPC' was refered to as `POSPARS'
                              when saving fit results.
2012/03/01, v0.3.1.1, Implemented usage of `Spectra'-class. Some HIERARCH-keys
                      are now added to  the header of the PRM-file. Removed
                      possibility to create residual cube, instead skript
                      SUBTRES of PampelMuse can be used to do that.
2012/03/23, v0.4.0.0, coompleted implementation of `Sources`-class. Added
                      parameters for parallelization and disabling Qt-usage.
2012/07/24, v0.5.0.0, adapted the code so that it now runs without issues if
                      PyQt4 and/or matplotlib ar not available.
2013/03/26, v0.6.0.0, modified the code in order to implement the new way of
                      calculating the PSF profiles.
2013/03/27, v0.6.0.1, some minor modifications
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.6.0.1"
import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core.fit_array import StarFit
from pampelmuse.core.sources import Sources
from pampelmuse.core.psf import PsfAttributes

## IMPORTANT:
## PyQt4 and QStarFit are loaded later and only if requested and dependencies
## fulfilled


def cubefit(arguments):

    print ""
    print 21* " "+"***  CubeFit, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    fitRad   = arguments.config["FITRAD"]                 ##
    maxIter  = arguments.config["MAXITER"]                ##
    psfRad   = arguments.config["PSFRAD"]                 ##
    smallIFU = arguments.config["SMALLIFU"]               ##
    dobackup = arguments.config["DOBACKUP"]               ##
    jumpStart= arguments.config["JUMPSTART"]              ##
    useQt    = arguments.config["USEQT"]                  ##
    cpuCount = arguments.config["CPUCOUNT"]               ##
    ########################################################


    ## open PRM-file
    logging.info("Reading PampelMuse-file: {0}.prm.fits".format(prefix))
    prmFile  = pyfits.open("{0}.prm.fits".format(prefix), mode="update")
    extNames = [prmFile[i].name for i in xrange(len(prmFile))]

    ## Check for previous PampelMuse-results
    try: fitStatus = prmFile[0].header['PAMPELMUSE CUBEFIT STATUS']
    except KeyError: fitStatus = "NOT STARTED"
    
    if fitStatus == "STOPPED":
        try: lastFitIndex = prmFile[0].header['PAMPELMUSE CUBEFIT LASTLAYER']
        except KeyError: lastFitIndex = None
    else:
        lastFitIndex = None

    if lastFitIndex != None:
        logging.info("Continuing previously paused fit, current layer: {0}"\
                         .format(lastFitIndex))

###### CREATE ARRAYS THAT CONTAIN PARAMETERS USED IN THE FIT ##################
    ## Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field.")

    ## check which instrument has been used
    try:
        instrument = prmFile[0].header["PAMPELMUSE IFU"]
    except KeyError:
        instrument = None

    sources = Sources(data=prmFile["SOURCES"], hdu=True, instrument=instrument)

    ## Spectra
    logging.info("Obtaining source spectra.")
    if "SPECTRA" in extNames:
        if "SIGMASPC" in extNames:
            sources.loadSpectra(prmFile["SPECTRA"], hdu=True,
                                sigma=prmFile["SIGMASPC"])
        else:
            sources.loadSpectra(data=prmFile["SPECTRA"], hdu=True, sigma=None)
    else:
        logging.critical("No spectra (HDU `SPECTRA') have been provided.")


    ## Positions
    logging.info("Obtaining information about source coordinates.")
    fullPosFit=False

    if "POSITIONS" in extNames:
        logging.info("Reading 'POSITIONS' HDU.")
        sources.loadPositionsHDU(prmFile["POSITIONS"])

    if 'POSPARS' in extNames:
        logging.info("Reading 'POSPARS' HDU.")
        sources.readTransHDU(prmFile["POSPARS"], backup=dobackup)
        sources.resizeStructure(size=sources.ndisp)
        sources.makeValues()
        sources.info()
        
    else:
        logging.warning("No coord. transf. (HDU 'POSPARS') detected.")


    ## PSF
    psf_attributes = PsfAttributes()
    if "PSFPARS" in extNames: ##  check if HDU exists
        logging.info("Reading 'PSFPARS' HDU.")
        psf_attributes.readFitsHDU(prmFile["PSFPARS"])

    else: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.warning("No HDU 'PSFPARS' found. Assuming Gaussian PSF.")

        psf_attributes.makeNewPSF(profile="gauss", fwhm=3.0, e=0., theta=0.)

    psf_attributes.resizeStructure(ndisp=sources.ndisp)
    psf_attributes.makeValues()
    psf_attributes.info()


    ## open datacubes & check consistency
    logging.info("Checking data cubes...")
    dataFits  = pyfits.open('{0}.dat.fits'.format(prefix))

    ## check if calibration HDU for ARGUS exists
    if sources.instrument.name == "argus":
        try:
            setup_index = dataFits.index_of("FIBER_SETUP")
            sources.instrument.read_fiber_setup(hdu=dataFits[setup_index])
        except KeyError:
            logging.warning("No HDU with fibre setup found for ARGUS data.")
 
    ndim     = dataFits[0].header["NAXIS"]

    if sources.instrument.transform == None:
        if ndim!=3:
            logging.error(\
                "Data must be 3D unless mapping from spaxels to x&y specified.")
            assert ndim==3, "Expected 3dim. datacube, got {0}dim".format(ndim)

        else:
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
    except KeyError: stop = ndisp
    
    if (stop-start+1) != sources.ndisp:
        logging.critical("Datacube and PRM-file have incompatible dimensions.")

    sigmaFits = pyfits.open('{0}.err.fits'.format(prefix))
    for key in ["NAXIS"]+["NAXIS{0}".format(i) for i in xrange(1, ndim)]:
        if sigmaFits[0].header[key] != dataFits[0].header[key]:
            logging.critical(\
                "Uncertainties and data have incompatible dimensions.")
            assert sigmaFits[0].header[key]==dataFits[0].header[key], \
                "Uncertainties and data have incompatible dimensions."

    if os.path.exists("{0}.msk.fits".format(prefix)):
        maskFits = pyfits.open('{0}.msk.fits'.format(prefix))
        for key in ["NAXIS"]+["NAXIS{0}".format(i) for i in xrange(1, ndim)]:
            if maskFits[0].header[key] != dataFits[0].header[key]:
                logging.critical(\
                    "Mask and data have incompatible dimensions.")
                assert sigmaFits[0].header[key]==dataFits[0].header[key], \
                    "Mask and data have incompatible dimensions."


    else:
        maskFits = [None]


    ## start fitting process:
    logging.info("Starting fit process.")

    if useQt:
        from PyQt4 import QtCore, QtGui
        from pampelmuse.core.qfit_array import QStarFit
        app    = QtGui.QApplication(sys.argv)
        fitter = QStarFit(dataFits=dataFits[0], sigmaFits=sigmaFits[0],
                          maskFits=maskFits[0], sources=sources,
                          psf_attributes=psf_attributes, maxrad=psfRad,
                          osradius=2, osfactor=5)
    else:
        fitter = StarFit(dataFits=dataFits[0], sigmaFits=sigmaFits[0],
                         maskFits=maskFits[0], sources=sources,
                         psf_attributes=psf_attributes, maxrad=psfRad,
                         osradius=2, osfactor=5)

    fitter(layerRange=(start,stop), fitRad=fitRad, lastFitIndex=lastFitIndex,
           asSmallIFU=smallIFU, jumpStart=jumpStart, ncpu=cpuCount,
           maxiter=maxIter)

    if useQt:
        app.exec_()
        if not fitter.save:
            sys.exit()
        
    logging.info("Saving results.")


    ## store spectra
    dataHDU, sigmaHDU = sources.makeSpecHDU(makeSigma=True)

    if dobackup and lastFitIndex == None:
        if 'INSPECTRA' in extNames:
            prmFile.remove(prmFile['INSPECTRA'])
        prmFile['SPECTRA'].name = 'INSPECTRA'
    else:
        prmFile.remove(prmFile['SPECTRA'])
    prmFile.insert(2, dataHDU)

    if dobackup and lastFitIndex == None:
        if 'INSIGMASPC' in extNames:
            prmFile.remove(prmFile['INSIGMASPC'])
        if 'SIGMASPC' in extNames:
                prmFile['SIGMASPC'].name = 'INSIGMASPC'
    else:
        if 'SIGMASPC' in extNames:
            prmFile.remove(prmFile['SIGMASPC'])
    prmFile.insert(3, sigmaHDU)


    ## Store IFU coordinates (only if they have been fitted)
    if fullPosFit:
        if "POSITIONS" in extNames:
            prmFile["POSITIONS"] = sources.makeCoordinatesHDU()
        else:
            prmFile.insert(4, sources.makeCoordinatesHDU())


    ## store coordinate transformation (only if it has been fitted)
    elif sources.Fit:
        if "POSPARS" in extNames:
            prmFile["POSPARS"] = sources.makeTransHDU()
        else:
            prmFile.insert(5, sources.makeTransHDU())


    ## store PSF (only if it as been fitted)
    if psf_attributes.Fit:
        if "PSFPARS" in extNames:
            prmFile["PSFPARS"] = psf_attributes.makePSFHDU()
        else:
            prmFile.insert(5, psf_attributes.makePSFHDU())

    ## write/update header of primary HDU
    prmFile[0].header.update(\
        "hierarch PAMPELMUSE CUBEFIT UTC",
        datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S"))
    prmFile[0].header.update("hierarch PAMPELMUSE CUBEFIT VERSION", __version__)
    if fitter.i != sources.ndisp:
        prmFile[0].header.update("hierarch PAMPELMUSE CUBEFIT STATUS","STOPPED")
    else:
        prmFile[0].header.update("hierarch PAMPELMUSE CUBEFIT STATUS","SUCCESS")
    prmFile[0].header.update("hierarch PAMPELMUSE CUBEFIT LASTLAYER", fitter.i)
    prmFile[0].header.update("hierarch PAMPELMUSE CUBEFIT SMALLIFU", smallIFU)
    prmFile[0].header.update("hierarch PAMPELMUSE CUBEFIT PSFRAD", psfRad)

    prmFile.flush()

