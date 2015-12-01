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
291, 2015/12/01
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 291


import datetime
import logging
import sys
import numpy as np
# note: PyQt4 modules import below only if interaction enabled

# required PampelMuse-packages
from pampelmuse.core import Sources
from pampelmuse.core.coordinates import Transformation
from pampelmuse.core.psf import Variables
# note: pampelmuse.graphics.PampelMuseGui imported below only if interaction enabled


def polyfit(pampelmuse):

    logging.info("Executing requested routine POLYFIT [Revision {0:d}].".format(__revision__))

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    # Get source catalogue from PRM-file
    sources = Sources(data=pampelmuse.hdulist["SOURCES"], instrument=pampelmuse.ifs_data.instrument)

    # Open spectra HDU from PRM-file
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

    # get IFS coordinates from PRM-file
    logging.info("Obtaining information about source coordinates.")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        try:
            i = pampelmuse.hdulist.index_of("INPOSITIONS")
            sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        except KeyError:
            pass
    except KeyError:
        pass

    # get coordinate transformation
    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.transformation = Transformation(hdu=pampelmuse.hdulist[i])

    except KeyError:
        logging.warning("No coordinate transformation (HDU 'POSPARS') detected.")

    sources.info()

    # Initialize PSF properties
    try:
        i = pampelmuse.hdulist.index_of("PSFPARS")
        psf = Variables(hdu=pampelmuse.hdulist[i])
        psf.info()
    except KeyError:
        logging.warning("No information available on PSF that was used (HDU 'PSFPARS').")
        psf = None

    # add information about caller routine to PampelMuse configuration
    pampelmuse.polyfit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.polyfit.revision = __revision__
    
    # Prepare GUI if interactive mode enabled
    if pampelmuse.interact:
        
        from PyQt4 import QtGui
        from pampelmuse.graphics import PampelMuseGui
        
        app = QtGui.QApplication(sys.argv)
        
        ui = PampelMuseGui(sources=sources, ifsData=pampelmuse.ifs_data, refImage=pampelmuse.polyfit.reference,
                           namespace=pampelmuse)
        ui.outfileName = pampelmuse.filename
        
        ui.loadSpectra()  # add information about spectra
        ui.loadPositions()  # add information about IFS coordinates
        ui.loadCoordTrans()  # add information about coordinate transformation
        ui.loadPsfParameters(psf)  # add information about PSF parameters
    
        # for type of polynomial, need to make first letter uppercase for identification in GUI
        polytype = pampelmuse.polytype[0].upper()+pampelmuse.polytype[1:]
        
        # check if type of polynomial is valid
        index = ui.spectrumFitTypeComboBox.findText(polytype)  # find index of requested polynomial if it exists
        if index > -1:
            ui.spectrumFitTypeComboBox.setCurrentIndex(index)
        else:
            logging.error("Unknown type of polynomial: {0}".format(polytype))

        # add degree of polynomial to GUI
        if pampelmuse.interact:
            ui.spectrumFitOrderSpinBox.setValue(pampelmuse.polydeg)

        # check if common slope should be used for the IFS coordinates
        if pampelmuse.interact:
            ui.actionForce_common_slope.setChecked(pampelmuse.polyfit.common)

        # exclude requested wavelength ranges (if any)
        if pampelmuse.polyfit.exclude is not None:
            for wmin, wmax in pampelmuse.polyfit.exclude:
                ui.modifyMask((wmin, wmax), True)
                logging.info("Added wavelength range to mask: {0}--{1}".format(wmin, wmax))

        ui.show()
    
        sys.exit(app.exec_())

    else:  # automatically fit PSF parameters and source positions using the predefined polynomial type and degree
        
        # prepare mask
        if pampelmuse.polyfit.exclude is not None:
            logging.info("Masked wavelength ranges: {0}".format(pampelmuse.polyfit.exclude))
            _mask = [(sources.wave >= wmin) & (sources.wave <= wmax) for wmin, wmax in pampelmuse.polyfit.exclude]
            mask = np.any(_mask, axis=0)

        # fit PSF parameters
        for i in xrange(psf.npars):
            if psf.free[i]:
                logging.info("Polynomial fit (function={0}, order={1}) to PSF parameter {2}...".format(
                    pampelmuse.polytype, pampelmuse.polydeg, psf.names[i]))
                psf.polynomial_fit(name=psf.names[i], order=pampelmuse.polydeg, mask=mask, function=pampelmuse.polytype)
        
        # fit IFS coordinates
        logging.info("Polynomial fit (function={0}, order={1}) to source coordinates...".format(
            pampelmuse.polytype, pampelmuse.polydeg))
        sources.fitIfsCoordinates(ids=None, order=pampelmuse.polydeg, mask=mask, function=pampelmuse.polytype,
                                  commonSlope=pampelmuse.polyfit.common, instrument=pampelmuse.ifs_data.instrument)
        
        # save changes to the currently opened file
        # store PSF (only if it as been fitted)
        if psf.free.any():
            pampelmuse.hdulist["PSFPARS"] = psf.make_hdu()  # extension should be there
        
        # store IFS coordinates: write "POSITIONS" HDU containing polynomial fits to source positions.
        # This is only needed when polynomial fits exist for at least one source
        if len(sources.ifs_coordinates_fit_functions) > 0:
            try:
                i = pampelmuse.hdulist.index_of("POSITIONS")
                pampelmuse.hdulist[i] = sources.make_ifs_coordinates_hdu(fit=True)
            except KeyError:
                pampelmuse.hdulist.append(sources.make_ifs_coordinates_hdu(fit=True))
        
            # write HDU containing directly fitted (i.e fitted using CUBEFIT) coordinates if they exist
            if sources.ifs_coordinates_free.any():
                try:
                    i = pampelmuse.hdulist.index_of("INPOSITIONS")
                    pampelmuse.hdulist[i] = sources.make_ifs_coordinates_hdu(fit=False, name="INPOSITIONS")
                except KeyError:
                    pampelmuse.hdulist.append(sources.make_ifs_coordinates_hdu(fit=False, name="INPOSITIONS"))

        # update the primary header information
        pampelmuse.makeHeader(hdu=pampelmuse.hdulist[0])
    
        # save the file
        pampelmuse.hdulist.flush()
