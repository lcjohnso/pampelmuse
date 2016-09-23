"""
subtres.py
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
function pampelmuse.routines.subtres.subtres

Purpose
-------
The method implements the function 'subtres' that is designed to subtract th
contribution of the resolved sources from the IFS data. To this aim, a valid
PampelMuse PRM-file must exist along with the data.  This function is called
if PampelMuse is started with the 'SUBTRES' option.

Latest SVN revision
-------------------
373, 2016/09/23
"""
import datetime
import logging
import os
from astropy.io import fits
from ..core.make_residual_cube import MakeResidualsCube
from ..utils.fits import open_prm


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 373


def add_parser(parser):
    p = parser.add_parser("SUBTRES", help="""
    The sub-routine is designed to subtract the contribution of the (resolved) stars
    from the IFS data. A PRM-file containing the results from a successful analysis
    with CUBEFIt should be available. The resulting data is stored as a Fit-file under
    <prefix>.res.fits.""")
    p.set_defaults(func=subtres)


def subtres(pampelmuse):

    logging.info("Executing requested routine SUBTRES [Revision {0:d}].".format(__revision__))

    # Check if PRM-file contains results of successful fit
    if hasattr(pampelmuse, "cubefit") and hasattr(pampelmuse.cubefit, "status"):
        fitstatus = pampelmuse.cubefit.status
    else:
        fitstatus = "Unknown"

    if fitstatus not in ["SUCCESS", "Unknown"]:
        logging.error("PRM-file does not contain results of successful fit.")

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    sources, psf_attributes = open_prm(pampelmuse.filename, ndisp=pampelmuse.ifs_data.wave.size)

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

    fitter = MakeResidualsCube(ifs_data=pampelmuse.ifs_data, sources=sources, psf_attributes=psf_attributes,
                               maxrad=pampelmuse.psfrad)

    residuals = fitter(layer_range=(start, stop), n_cpu=pampelmuse.cpucount, subtract_background=pampelmuse.subsky)

    logging.info("Saving results...")

    pampelmuse.subtres.revision = __revision__
    pampelmuse.subtres.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")

    # create HDU for output
    logging.info("Creating HDU from processed data...")

    out_hdu = fits.PrimaryHDU(residuals, header=pampelmuse.ifs_data.wcs.to_header())

    # delete any existing residuals file
    logging.info("Writing results to file {0}.res.fits...".format(pampelmuse.prefix))
    if os.path.exists("{0}.res.fits".format(pampelmuse.prefix)):
        os.remove("{0}.res.fits".format(pampelmuse.prefix))
        logging.warning("Existing residual cube has been overwritten.")

    out_hdu.writeto("{0}.res.fits".format(pampelmuse.prefix))

