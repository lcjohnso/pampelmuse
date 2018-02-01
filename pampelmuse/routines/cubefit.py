"""
cubefit.py
==========
Copyright 2013-2018 Sebastian Kamann

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
function pampelmuse.cubefit.add_parser
function pampelmuse.routines.cubefit.cubefit

Purpose
-------
This module implements the function 'cubefit' that is called if the requested
PampelMuse subroutine is CUBEFIT. As all other routines, it uses the
configuration of PampelMuse as only input argument. The configuration must be
provided as an instance of configparser.ConfigParser. The cubefit-function
reads the configuration and passes it to the pampelmuse.core.FitCube-class
which performs the actual analysis.

Latest SVN revision
-------------------
2018/02/01
"""
import datetime
import logging
from ..core.fit_cube import FitCube
from ..utils.fits import make_header_from_config, open_ifs_data, open_prm, save_prm, get_history


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 20180201


def add_parser(parser):

    p = parser.add_parser("CUBEFIT", help="""
    This sub-routine performs the actual PSF-fitting process, it is the workhorse
    of PampelMuse and its execution may take several  hours to complete. It expects
    a useful PRM-file to be available (created e.g using INITFIT). The analysis of
    the data proceeds in a layer-by-layer manner. In each layer, up to three analyis
    steps are performed, namely a fit of the fluxes, a fit of the PSF variables, and
    a fit of the coordinates. The former step  is performed via linear least squares
    optimization, the latter two steps use a Levenberg-Marquard algorithm. The three
    steps are iterated until convergence. Once the analysis has completed, the results
    are written to the PRM-file.""")
    p.set_defaults(func=cubefit)


def cubefit(config):
    """
    This function performs the optimization stage of the fit. It is called
    when the CUBEFIT routine of PampelMuse is called.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of PampelMuse.
    """

    logging.info("Executing requested routine CUBEFIT [Revision {0:d}].".format(__revision__))

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")

    # get prefix of IFS data - note that the calling routine makes sure that the entry exists in the configuration
    prefix = config['global']['prefix']

    # open IFS data
    ifs_data = open_ifs_data(prefix=prefix)

    # adjust numbers of first and last layer to be analysed
    first_layer = config['layers']['first']
    if first_layer < 0:
        first_layer = ifs_data.nlayer + first_layer
    last_layer = config['layers']['last']
    if last_layer < 0:
        last_layer = ifs_data.nlayer + last_layer + 1

    # Source catalogue
    sources, psf_attributes = open_prm(filename='{0}.prm.fits'.format(prefix), ndisp=ifs_data.wave.size)

    # get history of PRM file
    history = get_history(filename='{0}.prm.fits'.format(prefix))

    # start fitting process:
    logging.info("Starting fit process...")

    fitter = FitCube(ifs_data=ifs_data, sources=sources, psf_attributes=psf_attributes,
                     psf_radius=config['psf'].get('radius', 15.))

    psf_data = fitter(layer_range=(first_layer, last_layer), n_cpu=config['global'].get('cpucount', 1),
                      max_iterations=config['cubefit'].get('iterations', 5),
                      use_variances=config['cubefit'].get('usesigma', True),
                      n_bin=config['layers'].get('binning', 1), analyse_psf=config['psf'].get('inspect', False))

    logging.info("Saving results...")

    # save PSF data if requested
    if psf_data is not None:
        psf_data.writeto('{0}.psf.fits'.format(prefix), overwrite=True)

    # create header containing configuration
    header = make_header_from_config(config=config)

    # add history to header
    for entry in history:
        header['HISTORY'] = entry
    now = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    header['HISTORY'] = 'CUBEFIT [r{0}] finished at {1}'.format(__revision__, now)

    save_prm(sources, psf_attributes, filename='{0}.prm.fits'.format(prefix), header=header)
