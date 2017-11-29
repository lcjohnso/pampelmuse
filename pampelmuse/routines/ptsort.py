"""
ptsort.py
=========
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
Purpose
-------
This routine is designed to convert the pixtables that are produced
by the MUSE pipeline recipe scipost into a format that is supported
by PampelMuse.

Added
-----
rev. 334, 2016/07/26

Latest GIT revision
-------------------
2017/11/29
"""
import os
import logging
import numpy as np
from astropy.io import fits


__author__ = "Sebastian Kamann (skamann@ljmu.ac.uk)"
__revision__ = 20171129


logger = logging.getLogger(__name__)


def add_parser(parser):

    p = parser.add_parser("PTSORT", help="""
    The purpose of this routine is to convert a pixtable produced by the MUSE
    pipeline recipe 'scipost' to a format that is supported by PampelMuse. The
    main step of the conversion is the sorting of the table entries by
    increasing wavelength. As this requires loading the full data into memory,
    this routine can be quite demanding in this respect.""")
    p.set_defaults(func=ptsort)


def ptsort(config):
    """
    The function is designed to sort a PIXTABLE produced by the MUSE pipeline
    (recipe product PIXTABLE_REDUCED) by increasing wavelength and write the
    result to a new FITS-file.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of PampelMuse.
    """
    logger.info("Executing requested routine PTSORT [Revision {0:d}].".format(__revision__))

    # get prefix of IFS data - note that the calling routine makes sure that the entry exists in the configuration
    prefix = config['global']['prefix']

    logger.info('Opening input pixtable {0}.fits ...'.format(prefix))
    pt = fits.open("{0}.fits".format(prefix), memmap=False)

    # check if additional file exists containing IFU and slice numbers exist
    if os.path.exists('{0}.slices.fits'.format(prefix)):
        slices = fits.open('{0}.slices.fits'.format(prefix))
    else:
        slices = None

    logger.info('Sorting pixtable data by increasing wavelength ...')
    wave = pt['lambda'].data[:, 0]
    indices = np.argsort(wave)

    for i in range(1, len(pt)):
        extname = pt[i].header['EXTNAME'] if 'EXTNAME' in pt[i].header else i
        logger.info('... handling extension {0} ...'.format(extname))
        try:
            pt[i].data = pt[i].data[indices, :]
        except IndexError:
            logger.warning('... shape is invalid, extension skipped.')

    if slices is not None:
        for i in range(1, len(slices)):
            extname = slices[i].header['EXTNAME'] if 'EXTNAME' in slices[i].header else i
            logger.info('... handling extension {0} ...'.format(extname))
            pt.append(slices[i])
            pt[-1].data = pt[-1].data[indices, :]

    outfilename = config['ptsort']['outfilename']
    if outfilename is None:
        outfilename = '{0}_sorted.fits'.format(prefix)
    logger.info('Saving converted pixtable as {0} ...'.format(outfilename))

    pt.writeto(outfilename, clobber=True)
