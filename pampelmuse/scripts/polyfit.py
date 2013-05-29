"""
POLYFIT
=======
The routine is designed to inspect the fit results of PampelMuse. It can also
be used to fit a step-polynomial to a subset of the fit parameters (e.g PSF
parameters or source positions). The changes performed can be stored in the
PRM-file.
The graphical backbone of this routine is the PampelMuse GUI. The code describing
the GUI is located in the pampelmuse.graphics folder.

CHANGELOG
---------
... 2012/01/30, <v.0.2.0.0
2012/01/30, v0.2.0.0, major updates to the InspectFitData-class so that
                      the routine can handle the updated structure of the
                      PRM-files. This class is now located in a separate file
                      in the `graphics'-folder.
2012/01/30, v0.3.0.0, again major updates in InspectFitData-class which was
                      modified so that it now uses the dedicated classes `PSF',
                      `Positions' and `Spectra'. Also included usage of the
                      DOBACKUP-parameter.
2013/05/15, v0.4.0.0, replaced existing GUI with the new PampelMuse GUI. The
                      new GUI also displays the IFU data and allows one to load
                      a reference image.
2013/05/24, v0.5.0.0, modified the file structure to match the new initialization
                      of PampelMuse.
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.5.0.0"


import datetime, logging, sys
from PyQt4 import QtCore, QtGui

##  required PampelMuse-packages
from pampelmuse.core.psf import PsfAttributes
from pampelmuse.core.sources import Sources
from pampelmuse.graphics.pampelmuse_gui import PampelMuseGui


def polyfit(pampelmuse):

    print ""
    print 21* " "+"***  PolyFit, version {0}  ***".format(__version__)+21*" "
    print ""
    
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
            sources.loadSpectra(pampelmuse.hdulist[i], sigma=pampelmuse.hdulist[j])
        except KeyError:
            sources.loadSpectra(pampelmuse.hdulist[i], sigma=None)
    except KeyError:
        logging.critical("No spectra (HDU `SPECTRA') have been provided.")
    ui.loadSpectra()

    ## Positions
    logging.info("Obtaining information about source coordinates.")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.loadPositionsHDU(pampelmuse.hdulist[i])
        try:
            i = pampelmuse.hdulist.index_of("INPOSITIONS")
            sources.loadPositionsHDU(pampelmuse.hdulist[i])
        except KeyError:
            pass
        ui.loadPositions()
    except KeyError:
        pass

    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.readTransHDU(pampelmuse.hdulist[i], backup=False)
        sources.resizeStructure(size=sources.ndisp)
        sources.makeValues()
        sources.info()
        ui.loadCoordTrans()
        
    except KeyError:
        logging.warning("No coord. transf. (HDU 'POSPARS') detected.")


    ## Initializing PSF properties
    try:
        i = pampelmuse.hdulist.index_of("PSFPARS")
        ui.loadPsfParameters(pampelmuse.hdulist[i])
        ui.psf.info()
    except KeyError:
        logging.warning("No information available on PSF that was used (HDU 'PSFPARS').")

    ## add information about caller routine to PampelMuse configuration
    pampelmuse.polyfit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.polyfit.version = __version__

    ui.show()
    
    sys.exit(app.exec_())

