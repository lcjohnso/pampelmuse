"""
getspectra.py
=============
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

Latest SVN revision
-------------------
407 # 2017/04/02
"""
import logging
from ..core.spectrum_writer import SpectrumWriter
from ..utils.fits import open_prm, open_ifs_data


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 407


def add_parser(parser):
    p = parser.add_parser("GETSPECTRA", help="""
    Store all spectra extracted from a datacube as individual FITS files.""")
    # p.add_argument("specprefix", type=str, help="""
    # The common prefix of all generated FITS filenames, typically the name of
    # the target that has been observed.""")
    # p.add_argument("--folder", type=str, default="spectra",  help="""
    # The folder in which the spectra will be stored. Note that the folder will
    # be created if it does not exist.""")
    # p.add_argument("--passbands", type=str, nargs='+', help="""
    # A list of filter names for which magnitudes are estimated from the
    # spectra.""")
    # p.add_argument("--poststamps", action='store_true', default=False, help="""
    # Set this flag to store an image showing the region of the IFS data from
    # which it was extracted. Note that a separate FITS-image is stored for
    # each filter provided via --passbands.""")
    # p.add_argument("--skipflag", action='store_true', default=False, help="""
    # If this flag is used, the quality flags assigned to the individual spectra
    # are not included in the names used to store the FITS-files.""")
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
    prefix = config.get('general', 'prefix')

    # open IFS data
    ifs_data = open_ifs_data(prefix=prefix)

    sources, psf_attributes = open_prm(filename='{0}.prm.fits'.format(prefix), ndisp=ifs_data.wave.size)

    # If no filters are provided for calculating magnitudes, just use the one from
    # the source selection.
    if not config['getspectra'].get('passbands', default=None):
        config['getspectra']['passbands'] = config['catalog'].get('passband')

    # prepare instance for writing spectra to disk
    writer = SpectrumWriter(sources=sources, ifs_data=ifs_data, psf_attributes=psf_attributes)

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
