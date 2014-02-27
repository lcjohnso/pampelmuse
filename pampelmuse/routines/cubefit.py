"""
cubefit.py
==========

Provides
--------
function pampelmuse.routines.cubefit.cubefit

Purpose
-------
This module implements the function 'cubefit' that is called if the
requested PampelMuse subroutine is CUBEFIT. The function is a build
around the main component of PampelMuse, namely the class StarFit,
defined in the pampelmuse.core.fit_array module. As input, it takes
a namespace containing the configuration of PampelMuse.
There is also a version of StarFit that implements a GUI to display
the results of the analysis. This class, QStarFit, is defined in the
pampelmuse.core.qfit_array module. It is called by the function 'cube-
fit' if the configuration parameter USEQT of PampelMuse is enabled.

Latest SVN revision
-------------------
145, 2014/02/27
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 145

import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core import Sources
from pampelmuse.core.coordinates import Transformation
from pampelmuse.core.psf import Variables
from pampelmuse.core.fit_array import StarFit



## IMPORTANT:
## PyQt4 and QStarFit are loaded later and only if requested and dependencies
## fulfilled


def cubefit(pampelmuse):

    logging.info("Executing requested routine CUBEFIT [Revision {0:d}].".format(__revision__))

    ## Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    ## Source catalogue
    sources = Sources(data=pampelmuse.hdulist["SOURCES"])

    ## Spectra
    try:
        i= pampelmuse.hdulist.index_of("SPECTRA")
        try:
            j = pampelmuse.hdulist.index_of("SIGMASPC")
            sources.open_spectra_hdu(pampelmuse.hdulist[i], sigma_hdu=pampelmuse.hdulist[j])
        except KeyError:
            sources.open_spectra_hdu(pampelmuse.hdulist[i], sigma_hdu=None)
    except KeyError:
        logging.critical("No spectra (HDU 'SPECTRA') have been provided.")


    ## Positions
    logging.info("Obtaining information about source coordinates...")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.open_coordinates_hdu(pampelmuse.hdulist[i])
    except KeyError:
        pass

    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.transformation = Transformation(hdu=pampelmuse.hdulist[i])
        sources.info()
        
    except KeyError:
        logging.warning("No coord. transf. (HDU 'POSPARS') detected.")
        sources.ifs_coordinates_free[sources.status<4] = True


    ## Initializing PSF properties
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
        if not sources.ndisp == 1:
            logging.error("Length of dispersion axis of PRM file does not match no of layers in requested range.")
        sources.ndisp = (stop-start+1)
    

    ## start fitting process:
    logging.info("Starting fit process...")

    if pampelmuse.useqt:
        from PyQt4 import QtCore, QtGui
        from pampelmuse.core.qfit_array import QStarFit
        app    = QtGui.QApplication(sys.argv)
        fitter = QStarFit(ifs_data=pampelmuse.ifs_data, sources=sources, psf_attributes=psf_attributes, maxrad=pampelmuse.psfrad, osradius=2, osfactor=5)
    else:
        fitter = StarFit(ifs_data=pampelmuse.ifs_data, sources=sources, psf_attributes=psf_attributes, maxrad=pampelmuse.psfrad, osradius=2, osfactor=5)

    fitter(layerRange=(start,stop), lastFitIndex=lastFitIndex, asSmallIFU=pampelmuse.smallifu, jumpStart=pampelmuse.jumpstart,
           ncpu=pampelmuse.cpucount, maxiter=pampelmuse.maxiter)

    if pampelmuse.useqt:
        app.exec_()
        if not fitter.save:
            sys.exit()
        
    logging.info("Saving results...")


    ## storing spectra
    pampelmuse.hdulist["SPECTRA"] = sources.make_spectra_hdu()

    ## Store IFU coordinates (only if they have been fitted)
    if (sources.ifs_coordinates_free&(sources.ifs_coordinates_fitted==False)).any():
        try:
            pampelmuse.hdulist["POSITIONS"] = sources.make_ifs_coordinates_hdu()
        except KeyError:
            pampelmuse.hdulist.append(sources.make_ifs_coordinates_hdu())


    ## store coordinate transformation (only if it has been fitted)
    elif sources.transformation.contains_free_parameters:
        try:
            pampelmuse.hdulist["POSPARS"] = sources.transformation.make_hdu()
        except KeyError:
            pampelmuse.hdulist.append(5, sources.transformation.make_hdu())


    ## store PSF (only if it as been fitted)
    if psf_attributes.contains_free_parameters:
        try:
            pampelmuse.hdulist["PSFPARS"] = psf_attributes.make_hdu()
        except KeyError:
            pampelmuse.hdulist.append(psf_attributes.make_hdu())

    ## log status of analysis and create header of Primary HDU
    pampelmuse.cubefit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.cubefit.revision = __revision__
    if fitter.i != sources.ndisp: pampelmuse.cubefit.status = "STOPPED"
    else: pampelmuse.cubefit.status = "SUCCESS"
    pampelmuse.cubefit.lastlayer = fitter.i
    pampelmuse.makeHeader(hdu=pampelmuse.hdulist[0])
    
    ## Save the file
    pampelmuse.hdulist.flush()
