"""
polyfit.py
==========

Provides
--------
function pampelmuse.routines.polyfit.polyfit

Purpose
-------
The module implements the function 'polyfit' that is called if PampelMuse is
called with the POLYFIT option. The function is designed to inspect the fit
results of PampelMuse. It can also be used to perform polynomial fits to the
IFS coordinates of the sources or the variables of the PSF. To this aim, it
uses the PampelMuse GUI. As the GUI is designer using Qt, the function can
only be executed if PyQt4 is available on the system.
NOTE that the GUI itself is not defined here but implemented via the module
pampelmuse.graphics.pampelmuse_gui.

Latest SVN revision
-------------------
185, 2014/7/02
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 185


import datetime, logging, sys
from PyQt4 import QtCore, QtGui

##  required PampelMuse-packages
from pampelmuse.core import Sources
from pampelmuse.core.coordinates import Transformation
from pampelmuse.core.psf import Variables
from pampelmuse.graphics import PampelMuseGui


def polyfit(pampelmuse):

    logging.info("Executing requested routine POLYFIT [Revision {0:d}].".format(__revision__))

    ## Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    ## Source catalogue
    sources = Sources(data=pampelmuse.hdulist["SOURCES"], instrument=pampelmuse.ifs_data.instrument)

    ## Prepare GUI
    app = QtGui.QApplication(sys.argv)
    ui = PampelMuseGui(sources=sources, ifsData=pampelmuse.ifs_data, refImage=pampelmuse.polyfit.reference, 
                       namespace=pampelmuse)
    ui.outfileName = pampelmuse.filename

    ## Spectra
    logging.info("Obtaining source spectra.")
    try:
        i = pampelmuse.hdulist.index_of("SPECTRA")
        try:
            j = pampelmuse.hdulist.index_of("SIGMASPC")
            sources.open_spectra_hdu(pampelmuse.hdulist[i], sigma_hdu=pampelmuse.hdulist[j])
        except KeyError:
            sources.open_spectra_hdu(pampelmuse.hdulist[i])
    except KeyError:
        logging.critical("No spectra (HDU `SPECTRA') have been provided.")
    ui.loadSpectra()

    ## Positions
    logging.info("Obtaining information about source coordinates.")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        try:
            i = pampelmuse.hdulist.index_of("INPOSITIONS")
            sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        except KeyError:
            pass
        ui.loadPositions()
    except KeyError:
        pass

    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.transformation = Transformation(hdu=pampelmuse.hdulist[i])
        ui.loadCoordTrans()
        
    except KeyError:
        logging.warning("No coordinate transformation (HDU 'POSPARS') detected.")

    sources.info()

    ## Initializing PSF properties
    try:
        i = pampelmuse.hdulist.index_of("PSFPARS")
        ui.loadPsfParameters(pampelmuse.hdulist[i])
        ui.psf.info()
    except KeyError:
        logging.warning("No information available on PSF that was used (HDU 'PSFPARS').")

    ## add information about caller routine to PampelMuse configuration
    pampelmuse.polyfit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.polyfit.revision = __revision__

    ## update type of polynomial and its degree
    polytype = pampelmuse.polytype[0].upper()+pampelmuse.polytype[1:] ## need to make first letter uppercase
    index = ui.spectrumFitTypeComboBox.findText(polytype) ## find index of requested polynomial if it exists
    if index>-1:
        logging.info("Using {0} polynomials to fit wavelength dependencies.".format(polytype))
        ui.spectrumFitTypeComboBox.setCurrentIndex(index)
    else:
        logging.error("Unknown type of polynomial: {0}".format(polytype))

    ui.spectrumFitOrderSpinBox.setValue(pampelmuse.polydeg)
    logging.info("Order of fit polynomials set to {0}.".format(pampelmuse.polydeg))

    if pampelmuse.interact:
        ui.show()
    
        sys.exit(app.exec_())

    else: ## automatically fit PSF paramaters and source positions using the predefined polynomial type and degree
        
        for i in xrange(ui.psf.npars):
            if ui.psf.free[i]:
                logging.info("Polynomial fit to PSF parameter '{0}'...".format(ui.psf.names[i]))
                ui.psf.polynomial_fit(name=ui.psf.names[i], order=pampelmuse.polydeg, mask=None, function=pampelmuse.polytype)
        
        ## all sources are fitted simultaeously and with a common slope currently
        logging.info("Polynomial fit to source coordinates...")
        ui.sources.fitIfsCoordinates(ids=None, order=pampelmuse.polydeg, mask=None, function=pampelmuse.polytype, commonSlope=True, instrument=ui.instrument)
        
        ## save changes to the currently opened file
        ui.on_actionSave_triggered()

