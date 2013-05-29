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
2013/04/10, v0.6.0.2, removed option 'DOBACKUP' as it had no useful effect;
                      as Sources.loadSpectra did not copy the data, all the
                      changes to the spectra were made in-place and the backup
                      HDU containined the same spectra as the current HDU
2013/05/24, v0.7.0.0, modified structure of the code to match the new way that
                      PampelMuse is initialized.
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.7.0.0"
import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core.psf import PsfAttributes
from pampelmuse.core.fit_array import StarFit
from pampelmuse.core.sources import Sources


## IMPORTANT:
## PyQt4 and QStarFit are loaded later and only if requested and dependencies
## fulfilled


def cubefit(pampelmuse):

    print ""
    print 21* " "+"***  CubeFit, version {0}  ***".format(__version__)+21*" "
    print ""


    ## Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    ## Source catalogue
    sources = Sources(data=pampelmuse.hdulist["SOURCES"], instrument=pampelmuse.ifs_data.instrument)

    ## Spectra
    logging.info("Obtaining source spectra.")
    try:
        i= pampelmuse.hdulist.index_of("SPECTRA")
        try:
            j = pampelmuse.hdulist.index_of("SIGMASPC")
            sources.loadSpectra(pampelmuse.hdulist[i], sigma=pampelmuse.hdulist[j])
        except KeyError:
            sources.loadSpectra(pampelmuse.hdulist[i], sigma=None)
    except KeyError:
        logging.critical("No spectra (HDU `SPECTRA') have been provided.")


    ## Positions
    logging.info("Obtaining information about source coordinates.")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.loadPositionsHDU(pampelmuse.hdulist[i])
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
        sources.isFreePar[sources.ID>=0] = True


    ## Initializing PSF properties
    logging.info("Obtaining PSF properties...")
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


    ## Check for previous PampelMuse-results
    try: fitStatus = pampelmuse.cubefit.status
    except AttributeError: fitStatus = "NOT STARTED"
    
    if fitStatus == "STOPPED":
        try: lastFitIndex = pampelmuse.cubefit.lastlayer
        except AttributeError: lastFitIndex = None
    else: lastFitIndex = None

    if lastFitIndex != None:
        logging.info("Continuing previously paused fit, current layer: {0}".format(lastFitIndex))

    ## Check the range of layers to be fitted 
    try: start = pampelmuse.start
    except AttributeError: start = 0
    try: stop = pampelmuse.stop
    except AttributeError: stop = pampelmuse.ifs_data.ndisp-1
    
    if (stop-start+1) != sources.ndisp:
        logging.critical("Spectral dimension of PRM file does not match requested layer range to be analysed.")

    ## start fitting process:
    logging.info("Starting fit process...")

    if pampelmuse.useqt:
        from PyQt4 import QtCore, QtGui
        from pampelmuse.core.qfit_array import QStarFit
        app    = QtGui.QApplication(sys.argv)
        fitter = QStarFit(ifs_data=pampelmuse.ifs_data, sources=sources, psf_attributes=psf_attributes, maxrad=pampelmuse.psfrad, osradius=2, osfactor=5)
    else:
        fitter = StarFit(ifs_data=pampelmuse.ifs_data, sources=sources, psf_attributes=psf_attributes, maxrad=pampelmuse.psfrad, osradius=2, osfactor=5)

    fitter(layerRange=(start,stop), fitRad=pampelmuse.fitrad, lastFitIndex=lastFitIndex, asSmallIFU=pampelmuse.smallifu, jumpStart=pampelmuse.jumpstart,
           ncpu=pampelmuse.cpucount, maxiter=pampelmuse.maxiter)

    if pampelmuse.useqt:
        app.exec_()
        if not fitter.save:
            sys.exit()
        
    logging.info("Saving results...")


    ## storing spectra - not needed as Sources.loadSpectra does not copy the data,
    ##                   so changes are made in-place


    ## Store IFU coordinates (only if they have been fitted)
    if (sources.isFreePar&(sources.hasPolyFit==False)).any():
        try:
            pampelmuse.hdulist["POSITIONS"] = sources.makeCoordinatesHDU()
        except KeyError:
            pampelmuse.hdulist.append(sources.makeCoordinatesHDU())


    ## store coordinate transformation (only if it has been fitted)
    elif sources.Fit:
        try:
            pampelmuse.hdulist["POSPARS"] = sources.makeTransHDU()
        except KeyError:
            pampelmuse.hdulist.append(5, sources.makeTransHDU())


    ## store PSF (only if it as been fitted)
    if psf_attributes.Fit:
        try:
            pampelmuse.hdulist["PSFPARS"] = psf_attributes.makePSFHDU()
        except KeyError:
            pampelmuse.hdulist.append(psf_attributes.makePSFHDU())

    ## log status of analysis and create header of Primary HDU
    pampelmuse.cubefit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.cubefit.version = __version__
    if fitter.i != sources.ndisp: pampelmuse.cubefit.status = "STOPPED"
    else: pampelmuse.cubefit.status = "SUCCESS"
    pampelmuse.cubefit.lastlayer = fitter.i
    pampelmuse.makeHeader(hdu=pampelmuse.hdulist[0])
    
    ## Save the file
    pampelmuse.hdulist.flush()
