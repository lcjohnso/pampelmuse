"""
getspectra.py
=============
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
function pampelmuse.routines.getspectra.getspectra
function pampelmuse.routines.getspectra.add_parser

Purpose
-------
The function getspectra is called when Pampelmuse is started with the
GETSPECTRA command. Its purpose is to read all the spectra that have
been extracted from a datacube and store them as individual FITS files.
The structure of each created FITS file is such that the primary exten-
sion contains the spectrum and the second extension (named 'SIGMA') con-
tains the uncertainty for the measured spectra as estimated by the CUBE-
FIT routine of PampelMuse.

Added
-----
2014/06/26

Latest Git revision
-------------------
2018/12/12
"""
import logging
import sys
from ..core.spectrum_writer import SpectrumWriter
from ..utils.fits import open_prm, open_ifs_data


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20181212


def add_parser(parser):
    p = parser.add_parser("GETSPECTRA", help="""
    Store all spectra extracted from a datacube as individual FITS files.""")
    p.set_defaults(func=getspectra)


def getspectra(config):
    """
    This function writes the spectra from the PRM-file to individual FITS-
    files. It is called when PampelMuse is run with the GETSPECTRA option.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of PampelMuse.
    """
    logging.info("Executing requested routine GETSPECTRA [Revision {0:d}].".format(__revision__))

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")

    # get prefix of IFS data - note that the calling routine makes sure that the entry exists in the configuration
    prefix = config['global']['prefix']

    # open IFS data
    ifs_data = open_ifs_data(prefix=prefix)

    # Get source catalogue from PRM-file
    sources, psf = open_prm('{0}.prm.fits'.format(prefix), ndisp=ifs_data.wave.size)

    # check if a prefix for the output spectra has been defined
    if not config['getspectra']['prefix']:
        logging.error('No prefix for output spectra defined. Please set the parameter getspectra|prefix.')
        sys.exit(1)

    # If no filters are provided for calculating magnitudes, just use the one from
    # the source selection.
    if not config['getspectra']['passbands']:
        config['getspectra']['passbands'] = [config['catalog']['passband'], ]

    # prepare instance for writing spectra to disk
    writer = SpectrumWriter(sources=sources, ifs_data=ifs_data, psf_attributes=psf)

    # identify possibly contaminated spectra (note that this may also flag variable stars)
    _, _ = writer.identify_contaminated(passband=config['catalog'].get('passband'))

    # add relevant PampelMuse keywords to the FITS header
    for key in ["version", "revision", "filename", "utc", "passband"]:
        if hasattr(config["global"], key):
            value = config['global'][key]
            # check that entry not longer than 80 characters
            if isinstance(value, str) and len(value) > (55-len(key)):
                value = value[-55 + len(key):]
            writer.add_header_key("HIERARCH PAMPELMUSE {0}".format(key.upper()), value)

    # perform spectrum writing
    writer(prefix=config['getspectra']['prefix'],
           folder=config['getspectra']['folder'],
           filter_names=config['getspectra']['passbands'],
           poststamps=config['getspectra']['poststamps'],
           add_flag_to_filename=not config['getspectra']['skipflag'])
