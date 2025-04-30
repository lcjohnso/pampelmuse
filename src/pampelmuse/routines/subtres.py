"""
subtres.py
==========
Copyright 2013-2025 Sebastian Kamann

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
function pampelmuse.routines.subtres.add_parser
function pampelmuse.routines.subtres.subtres

Purpose
-------
The method implements the function 'subtres' that is designed to subtract th
contribution of the resolved sources from the IFS data. To this aim, a valid
PampelMuse PRM-file must exist along with the data.  This function is called
if PampelMuse is started with the 'SUBTRES' option.

Latest Git revision
-------------------
2025/04/30
"""
import datetime
import logging
import os
from astropy.io import fits
from ..core.make_residual_cube import MakeResidualsCube
from ..utils.fits import open_prm, open_ifs_data


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20250429


def add_parser(parser):
    p = parser.add_parser("SUBTRES", help="""
    The sub-routine is designed to subtract the contribution of the (resolved) stars
    from the IFS data. A PRM-file containing the results from a successful analysis
    with CUBEFIt should be available. The resulting data is stored as a Fit-file under
    <prefix>.res.fits.""")
    p.set_defaults(func=subtres)


def subtres(config):
    """
    This function creates a residuals cube from the results of the previous
    analysis. It is called when PampelMuse is called with the SUBTRES option.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of PampelMuse.
    """
    logging.info("Executing requested routine SUBTRES [Revision {0:d}].".format(__revision__))

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")

    # get prefix of IFS data - note that the calling routine makes sure that the entry exists in the configuration
    prefix = config['global']['prefix']

    # open IFS data
    ifs_data = open_ifs_data(prefix=prefix)

    # Check if PRM-file contains results of successful fit
    fitstatus = config['cubefit'].get('status', 'Unknown')
    if fitstatus not in ["SUCCESS", "Unknown"]:
        logging.error("PRM-file does not contain results of successful fit.")

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    sources, psf_attributes = open_prm('{0}.prm.fits'.format(prefix), ndisp=ifs_data.wave.size)

    # adjust numbers of first and last layer to be analysed
    first_layer = config['layers']['first']
    if first_layer < 0:
        first_layer = ifs_data.nlayer + first_layer
    last_layer = config['layers']['last']
    if last_layer < 0:
        last_layer = ifs_data.nlayer + last_layer + 1

    # check if only a subset of sources should be subtracted or some sources should NOT be subtracted
    exclude = config['subtres']['exclude']
    include = config['subtres']['include']
    if include is not None and exclude is not None:
        logging.error('Only subtres.include or subtres.exclude may be specified.')
        return
    
    # check if name of file with source to be subtracted was specified
    filename = config['subtres']['filename']
    if filename is not None:
        logging.info('Reading source IDs to subtract from file {0}.'.format(filename))
        if include is None or exclude is None:
            logging.warning('Ignoring include/exclude options, as filename is specified.')
        with open(filename) as file:
            include = [int(line.rstrip()) for line in file]

    # start subtraction process:
    logging.info("Starting source subtraction...")

    fitter = MakeResidualsCube(ifs_data=ifs_data, sources=sources, psf_attributes=psf_attributes,
                               psf_radius=config['psf']['radius'], psf_radius_scaling=config['psf']['scaling'])

    residuals = fitter(layer_range=(first_layer, last_layer),
                       n_cpu=config['global']['cpucount'],
                       subtract_background=config['sky']['subtract'],
                       include=include, exclude=exclude)

    logging.info("Saving results...")

    config['subtres']['revision'] = __revision__
    config['subtres']['utc'] = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")

    # create HDU for output
    logging.info("Creating HDU from processed data...")

    # write WCS to header of residuals cube
    out_hdu = fits.PrimaryHDU(residuals, header=ifs_data.wcs.to_header())

    # delete any existing residuals file
    logging.info("Writing results to file {0}.res.fits...".format(prefix))
    if os.path.exists("{0}.res.fits".format(prefix)):
        os.remove("{0}.res.fits".format(prefix))
        logging.warning("Existing residual cube has been overwritten.")

    out_hdu.writeto("{0}.res.fits".format(prefix))
