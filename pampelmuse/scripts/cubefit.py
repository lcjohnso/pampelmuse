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
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.2.1.1"
import logging, os, sys
import numpy as np, pyfits
from PyQt4 import QtCore, QtGui

## required PampelMuse-packages
from pampelmuse.core.fit_array import StarFit

def cubefit(arguments):

    print ""
    print 21* " "+"***  CubeFit, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    fitRad   = arguments.config["FITRAD"]                 ##
    psfRad   = arguments.config["PSFRAD"]                 ##
##     posfit   = arguments.config["POSFIT"]                 ##
##     psfType  = arguments.config["PSFTYPE"]                ##
##     psfInit  = arguments.config["PSFPARS"]                ##
##     psfFit   = arguments.config["PSFFIT"]                 ##
    smallIFU = arguments.config["SMALLIFU"]               ##
    dobackup = arguments.config["DOBACKUP"]               ##
    storeRes = arguments.config["STORERES"]               ##
    ########################################################


    ## open PRM-file, get no. of sources
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
    srcList = np.array(prmFile['SOURCES'].data,
                       dtype=prmFile['SOURCES'].data.dtype)

    if "POSITIONS" in extNames:
        logging.info("Using IFU pixel coordinates from 'POSITIONS' HDU.")
        logging.info("Ignoring coordinate transformation (HDU 'POSPARS').")
        ifuCoords = prmFile["Positions"].data
    else:
        logging.info("PRM-file does not contain HDU named 'POSITIONS'.")
        logging.info("Using coordinate transformation from 'POSPARS' HDU.")
        ifuCoords = None

    ## collect the  spectra
    spcList = prmFile['SPECTRA'].data
    
    ## check if uncertainties associated with spectra are available:
    if "SIGMASPC" in extNames:
        spcSigmaList = prmFile['SIGMASPC'].data
    else:
        spcSigmaList = None


    ## Collect information about coordinate transformation    
    ## Required parameters for the coordinate transformation ######
##     requiredPars = ['A','B','C','D']                             ##
    ###############################################################

##     isfitted = np.array(4*[False])
##     if posfit in [1,2]: isfitted[2:] = True
##     if posfit == 2:     isfitted[:2] = True

    if 'POSPARS' in extNames and ifuCoords == None: ## check if HDU exists

        posPars = np.zeros(prmFile['POSPARS'].data.size,
                           dtype=[('name', '|S8'),
                                  ('value', np.float32, (Ndsp,)),
                                  ('fit', np.bool)])

        ## Perform validation of provided data
        if posPars['value'].shape[1] != Ndsp:
            logging.critical("Invalid shape of data in HDU 'PSFPARS'")

        ## check which parameters to fit and if polynomial fits to
        ## parameters are available
        logging.info("Parameter  free?  polyfit?")
        for i in xrange(posPars.size):

            posPars['name'][i] = prmFile["POSPARS"].data.field("name")[i]
            logString = " {0:8s}".format(posPars["name"][i])

            if prmFile["POSPARS"].data.field("freepar")[i]:
                logString += "   [x]"
                if prmFile["POSPARS"].data.field("isfitted")[i]:
                    logString += "       [x]"
                    posPars["value"][i] \
                        = prmFile["POSPARS"].data.field("polyfit")[i]
                else:
                    logString += "       [ ]"
                    posPars["value"][i] \
                        = prmFile["POSPARS"].data.field("value")[i]
                    posPars["fit"][i] = True
            else:
                logString += "   [ ]       [ ]"
                posPars["value"][i] = prmFile["POSPARS"].data.field("value")[i]

            logging.info(logString)

    else:
        posPars = None
        if ifuCoords == None:
            logging.critical("No coord. transf. (HDU 'POSPARS') detected.")


    ## Collect information about PSF
    ## PSF parameter definitions: ###################################
##     psfProfiles = {'gauss':('fwhm', 'e', 'theta'),                 ##
##                    'moffat':('beta', 'fwhm', 'e', 'theta')}        ##
    #################################################################
##     reqPars = psfProfiles[psfType]

    ## check which PSF parameters should be fitted
##     isfitted = np.zeros((len(reqPars),), dtype='bool')
##     expTwo   = 2**np.arange(1, isfitted.size+1)
##     for i in np.arange(isfitted.size+1,0,-1):
##         if psfFit >= (expTwo[isfitted].sum()+2**i):
##             isfitted[i-1] = True

    if "PSFPARS" in extNames: ##  check if HDU exists

        psfType = prmFile["PSFPARS"].header["PSFTYPE"]
        logging.info("Detected HDU 'PSFPARS'. Used PSF profile: {0}".format(\
                psfType))

        psfPars = np.zeros(prmFile['PSFPARS'].data.size,
                           dtype=[('name', '|S8'),
                                  ('value', np.float32, (Ndsp,)),
                                  ('fit', np.bool)])

        ## Perform validation of provided data
        if psfPars['value'].shape[1] != Ndsp:
            logging.critical("Invalid shape of data in HDU 'PSFPARS'")

        ## check which parameters to fit and if polynomial fits to
        ## parameters are available
        logging.info("Parameter  free?  polyfit?")
        for i in xrange(psfPars.size):

            psfPars['name'][i] = prmFile["PSFPARS"].data.field("name")[i]
            logString = " {0:8s}".format(psfPars["name"][i])

            if prmFile["PSFPARS"].data.field("freepar")[i]:
                logString += "   [x]"
                if prmFile["PSFPARS"].data.field("isfitted")[i]:
                    logString += "       [x]"
                    psfPars["value"][i] \
                        = prmFile["PSFPARS"].data.field("polyfit")[i]
                else:
                    logString += "       [ ]"
                    psfPars["value"][i] \
                        = prmFile["PSFPARS"].data.field("value")[i]
                    psfPars["fit"][i] = True
            else:
                logString += "   [ ]       [ ]"
                psfPars["value"][i] = prmFile["PSFPARS"].data.field("value")[i]

            logging.info(logString)

    else: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.warning("No HDU 'PSFPARS' found. Assuming Gaussian PSF.")

        psfType = "gauss"

        posPars = np.array([('fwhm', 3., True),('e', 0., False),
                            ('theta', 0., False)],
                           dtype=[('name', 'a1'),
                                  ('value', 'f8', (Ndsp)),
                                  ('freepar', 'b1')])


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

    fitter.provideSources(sources=srcList, spectra=spcList, pixCoords=ifuCoords,
                          coordTrans=posPars, specSigma=spcSigmaList)

    fitter.providePSF(psfType=psfType, psfParameters=psfPars,
                      psfRad=psfRad)
        
    fitter(fitRad=fitRad, lastFitIndex=lastFitIndex, asSmallIFU=smallIFU,
           returnResiduals=storeRes, prefix=prefix)

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


    ## store coordinate transformation (only if it has been fitted)
    if "POSPARS" in extNames and ifuCoords == None:
        if fitter.coordTrans["fit"].any():

            ## make back-up of existing values (if requested)
            if dobackup and not "value_init" in prmFile["POSPARS"].data.names:
                newCol = pyfits.Column(\
                    name="value_init", format="{0}E".format(int(Ndsp)),
                    array=np.zeros((prmFile["POSPARS"].data.size, Ndsp)))
                newTable = pyfits.new_table([newCol])
                fullTable= pyfits.new_table(\
                    prmFile["POSPARS"].columns + newTable.columns)
                fullTable.header.update("EXTNAME", "POSPARS")
                prmFile["POSPARS"] = fullTable

            ## store fit results
            for i in xrange(fitter.coordTrans.size):
                if fitter.coordTrans["fit"][i]:
                    j = np.where(prmFile["POSPARS"].data.field("name")\
                                     ==fitter.coordTrans["name"][i])[0][0]
                    prmFile["POSPARS"].data.field("value")[j] \
                        = fitter.coordTrans["value"][i]
                    if dobackup:
                        prmFile["POSPARS"].data.field("value_init")[j] \
                            = np.copy(prmFile["POSPARS"].data.field("value")[i])
            

    ## store PSF (only if it as been fitted)
    if "PSFPARS" in extNames and fitter.psfPars["fit"].any():

        ## make back-up of existing values (if requested)
        if dobackup and not "value_init" in prmFile["PSFPARS"].data.names:
            newCol = pyfits.Column(\
                name="value_init", format="{0}E".format(int(Ndsp)),
                array=np.zeros((prmFile["PSFPARS"].data.size, Ndsp)))
            newTable = pyfits.new_table([newCol])
            fullTable= pyfits.new_table(\
                prmFile["PSFPARS"].columns + newTable.columns)
            fullTable.header.update("EXTNAME", "PSFPARS")
            prmFile["PSFPARS"] = fullTable

        for i in xrange(fitter.psfPars.size):
            if fitter.coordTrans["fit"][i]:
                j = np.where(prmFile["PSFPARS"].data.field("name")\
                                 ==fitter.psfPars["name"][i])[0][0]
                prmFile["PSFPARS"].data.field("value")[j] \
                    = fitter.psfPars["value"][i]
                if dobackup:
                    prmFile["PSFPARS"].data.field("value_init")[j] \
                        = np.copy(prmFile["PSFPARS"].data.field("value")[i])


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

