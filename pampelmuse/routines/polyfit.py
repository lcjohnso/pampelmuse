"""
polyfit.py
==========
Copyright 2013-2016 Sebastian Kamann

This file is part of PampelMuse.

PampelMuse is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

PampelMuse is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PampelMuse. If not, see <http://www.gnu.org/licenses/>.


+++
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
377, 2016/12/14
"""
import datetime
import logging
import sys
import numpy as np
from ..utils.fits import open_prm, save_prm
# note: PyQt4 modules import below only if interaction enabled
# note: pampelmuse.graphics.PampelMuseGui imported below only if interaction enabled


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 377


def add_parser(parser):
    p = parser.add_parser("POLYFIT", help="""
    This sub-routine loads the PampelMuse GUI and is designed for a visualization
    of the results from the PSF fit on the one hand and for an interactive modelling
    of the PSF variables and source coordinates on the other. The idea of modelling
    these parameters is that they should vary smoothly with wavelength, so by combi-
    ning the results from the indiviual layers one can create a model beyond the
    accuracy of an individual layer. To this aim, polynomials fits of the parameters
    can be performed. In a subsequent analysis with the sub-routine CUBEFIT, the
    routine will then use the parameters obtain in the fits and leave them unchanged.""")
    p.add_argument("reference", type=str, nargs="?", help="""
    An optional FITS image to be displayed. The idea here is to load the data
    from which the reference coordinates were extracted. In that case, the image
    is displayed in the GUI and the locations of the extracted sources are
    overplotted.""")
    p.add_argument("-x", "--exclude", type=float, nargs=2, action="append", help="""
    To exclude a wavelength range from the polynomial fits, provide the min. and max.
    wavelength of it. The provided wavelengths must be given in the same units as the
    data. Note that this option can be used multiple times to exclude more than one
    range.""")
    p.add_argument("--common", action="store_true",  default=False, help="""
    If this flag is set, then the fits to the IFS coordinates of the sources are
    constrained such that all sources have the same slope with wavelength.""")
    p.set_defaults(func=polyfit)


def polyfit(pampelmuse):

    logging.info("Executing requested routine POLYFIT [Revision {0:d}].".format(__revision__))

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    # Get source catalogue from PRM-file
    sources, psf = open_prm(pampelmuse.filename, ndisp=pampelmuse.ifs_data.wave.size)

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
        polytype = pampelmuse.polytype[0].upper() + pampelmuse.polytype[1:]
        
        # check if type of polynomial is valid
        index = ui.spectrumFitTypeComboBox.findText(polytype)  # find index of requested polynomial if it exists
        if index > -1:
            ui.spectrumFitTypeComboBox.setCurrentIndex(index)
        else:
            logging.error("Unknown type of polynomial: {0}".format(polytype))

        # add degree of polynomial to GUI
        ui.spectrumFitOrderSpinBox.setValue(pampelmuse.polydeg)

        # check if common slope should be used for the IFS coordinates
        ui.actionForce_common_slope.setChecked(pampelmuse.polyfit.common)

        # exclude requested wavelength ranges (if any)
        if pampelmuse.polyfit.exclude is not None:
            for wmin, wmax in pampelmuse.polyfit.exclude:
                ui.modifyMask((wmin, wmax), True)
                logging.info("Added wavelength range to spectrumMask: {0}--{1}".format(wmin, wmax))

        ui.show()
    
        sys.exit(app.exec_())

    else:  # automatically fit PSF parameters and source positions using the predefined polynomial type and degree
        
        # prepare spectrumMask
        if pampelmuse.polyfit.exclude is not None:
            logging.info("Masked wavelength ranges: {0}".format(pampelmuse.polyfit.exclude))
            _mask = [(sources.wave >= wmin) & (sources.wave <= wmax) for wmin, wmax in pampelmuse.polyfit.exclude]
            mask = np.any(_mask, axis=0)
        else:
            mask = None

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
                                  commonSlope=pampelmuse.polyfit.common, instrument=pampelmuse.ifs_data)
        
        # save changes to prm file
        save_prm(sources=sources, psf_attributes=psf, filename=pampelmuse.filename, header=pampelmuse.makeHeader())
