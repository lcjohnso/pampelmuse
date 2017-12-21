"""
polyfit.py
==========
Copyright 2013-2017 Sebastian Kamann

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
function pampelmuse.routines.polyfit.add_parser
function pampelmuse.routines.polyfit.polyfit

Purpose
-------
The module implements the function 'polyfit' that is called if PampelMuse is
called with the POLYFIT option. The function is designed to inspect the fit
results of PampelMuse. It can also be used to perform polynomial fits to the
IFS coordinates of the sources or the variables of the PSF. To this aim, it
uses the PampelMuse GUI. As the GUI is written in PyQt, the function can only
be run in interactive mode if PyQt4 is available on the system. Otherwise, the
routine will automatically fit the coordinates and PSF parameters.

Latest Git revision
-------------------
2017/12/21
"""
import datetime
import logging
import sys
import numpy as np
from ..utils.fits import make_header_from_config, open_ifs_data, open_prm, save_prm, get_history
# note: PyQt4 modules import below only if interaction enabled
# note: pampelmuse.graphics.PampelMuseGui imported below only if interaction enabled


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 20171221


def add_parser(parser):
    p = parser.add_parser("POLYFIT", help="""
    This sub-routine loads the PampelMuse GUI and is designed for a visualization
    of the results from the PSF fit on the one hand and for an interactive modelling
    of the PSF variables and source coordinates on the other. The idea of modelling
    these parameters is that they should vary smoothly with wavelength, so by combi-
    ning the results from the individual layers one can create a model beyond the
    accuracy of an individual layer. To this aim, polynomials fits of the parameters
    can be performed. In a subsequent analysis with the sub-routine CUBEFIT, the
    routine will then use the parameters obtain in the fits and leave them unchanged.""")
    p.add_argument("-x", "--exclude", type=float, nargs=2, action="append", help="""
    To exclude a wavelength range from the polynomial fits, provide the min. and max.
    wavelength of it. The provided wavelengths must be given in the same units as the
    data. Note that this option can be used multiple times to exclude more than one
    range.""")
    p.set_defaults(func=polyfit)


def polyfit(config):
    """
    This function performs the initialisation stage of the fit. It is called
    when the INITFIT routine of PampelMuse is called.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of PampelMuse.
    """

    logging.info("Executing requested routine POLYFIT [Revision {0:d}].".format(__revision__))

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")

    # get prefix of IFS data - note that the calling routine makes sure that the entry exists in the configuration
    prefix = config['global']['prefix']

    # open IFS data
    ifs_data = open_ifs_data(prefix=prefix)

    # Get source catalogue from PRM-file
    sources, psf = open_prm('{0}.prm.fits'.format(prefix), ndisp=ifs_data.wave.size)

    # get history of PRM file
    history = get_history(filename='{0}.prm.fits'.format(prefix))

    # for type of polynomial, need to make first letter uppercase for identification in GUI
    polytype = config['polynom']['type']
    polytype = polytype[0].upper() + polytype[1:]

    # get degree of polynomial
    polydeg = config['polynom']['degree']

    # if common slope used for all coordinates along one axis
    common = config['polynom']['common']

    # prepare mask to exclude requested wavelength ranges (if any)
    mask = config['polynom']['mask']
    # mask must be iterable and have even length
    if mask is not None:
        if not hasattr(mask, '__iter__') or np.mod(len(mask), 2):
            logging.error('Provided mask has invalid shape.')
            mask = None

    # Prepare GUI if interactive mode enabled
    if config['global']['interact']:

        # load modules required to display GUI
        from PyQt5 import QtWidgets
        from pampelmuse.graphics import PampelMuseGui

        # initialize GUI
        app = QtWidgets.QApplication(sys.argv)
        ui = PampelMuseGui(sources=sources, ifsData=ifs_data, refImage=config['polyfit'].get('reference', None))
        ui.history = history

        # for saving results, provide name of PRM-file and the current configuration
        ui.outfileName = '{0}.prm.fits'.format(prefix)
        ui.config = config

        ui.loadSpectra()  # add information about spectra
        ui.loadPositions()  # add information about IFS coordinates
        ui.loadCoordTrans()  # add information about coordinate transformation
        ui.loadPsfParameters(psf)  # add information about PSF parameters
        
        # check if type of polynomial is valid
        index = ui.spectrumFitTypeComboBox.findText(polytype)  # find index of requested polynomial if it exists
        if index > -1:
            ui.spectrumFitTypeComboBox.setCurrentIndex(index)
        else:
            logging.error("Unknown type of polynomial: {0}".format(polytype))

        # add degree of polynomial to GUI
        ui.spectrumFitOrderSpinBox.setValue(polydeg)

        # check if common slope should be used for the IFS coordinates
        ui.actionForce_common_slope.setChecked(common)

        # apply mask (if any)
        if mask is not None:
            # loop over areas to mask, assume odd/even list entries contain start/end of masked range
            for i in range(0, len(mask), 2):
                ui.modifyMask(mask[i], mask[i+1], True)
                logging.info("Added wavelength range to mask: {0}-{1}AA".format(mask[i], mask[i+1]))
        ui.show()
    
        sys.exit(app.exec_())

    else:  # automatically fit PSF parameters and source positions using the predefined polynomial type and degree
        
        # prepare spectrumMask
        if mask is not None:
            logging.info("Masked wavelength ranges: {0}".format([mask[i:i+2] for i in range(0, len(mask), 2)]))
            _mask = [(sources.wave >= mask[i]) & (sources.wave <= mask[i+1]) for i in range(0, len(mask), 2)]
            mask = np.any(_mask, axis=0)

        # fit PSF parameters
        for i in range(psf.n_parameters):
            if psf.free[i]:
                logging.info("Polynomial fit (function={0}, order={1}) to PSF parameter {2}...".format(
                    polytype, polydeg, psf.names[i]))
                psf.polynomial_fit(name=psf.names[i], order=polydeg, mask=mask, polynom=polytype)
        
        # fit IFS coordinates
        logging.info("Polynomial fit (function={0}, order={1}) to source coordinates...".format(
            polytype, polydeg))
        sources.fitIfsCoordinates(ids=None, order=polydeg, mask=mask, polynom=polytype, common_slope=common,
                                  instrument=ifs_data)

        # create header containing configuration
        header = make_header_from_config(config=config)
        # add history to header
        for entry in history:
            header['HISTORY'] = entry
        now = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
        header['HISTORY'] = 'POLYFIT [r{0}] finished at {1}'.format(__revision__, now)

        # save changes to prm file
        save_prm(sources=sources, psf_attributes=psf, filename='{0}.prm.fits'.format(prefix), header=header)
