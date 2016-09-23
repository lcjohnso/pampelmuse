"""
cubefit.py
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
374, 2016/09/23
"""
import datetime
import logging
from ..core.fit_cube import FitCube
from ..utils.fits import open_prm, save_prm


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 374


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


def cubefit(pampelmuse):

    logging.info("Executing requested routine CUBEFIT [Revision {0:d}].".format(__revision__))

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    # Source catalogue
    sources, psf_attributes = open_prm(filename=pampelmuse.filename, ndisp=pampelmuse.ifs_data.wave.size)

    # Check the range of layers to be fitted 
    try:
        start = pampelmuse.start
    except AttributeError:
        start = 0
    try:
        stop = pampelmuse.stop
    except AttributeError:
        stop = pampelmuse.ifs_data.ndisp-1

    # start fitting process:
    logging.info("Starting fit process...")

    fitter = FitCube(ifs_data=pampelmuse.ifs_data, sources=sources, psf_attributes=psf_attributes,
                     maxrad=pampelmuse.psfrad)

    fitter(layer_range=(start, stop), n_cpu=pampelmuse.cpucount, maxiter=pampelmuse.maxiter,
           use_variances=pampelmuse.usesigma, n_bin=pampelmuse.layerbin)

    logging.info("Saving results...")

    # log status of analysis
    pampelmuse.cubefit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.cubefit.revision = __revision__
    pampelmuse.cubefit.status = "SUCCESS"

    save_prm(sources, psf_attributes, filename=pampelmuse.filename, header=pampelmuse.makeHeader())
