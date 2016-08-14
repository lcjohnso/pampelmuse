"""
singlesrc.py
============
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
function pampelmuse.routines.singlesrc.singlesrc

Purpose
-------
The module contains the function 'singlesrc' that is used to prepare a data
cube for the analysis for which no photometric or astrometric information is
available. It prepares a PRM-file directly from a set of provided IFS coordi-
nates and statuses. For each of the sources with status=3, aperture spectro-
photometry is performed and the initial guesses for the IFS coordinates are im-
proved using the first order moments determined in the calculation.
 
Latest SVN revision
-------------------
360, 2016/08/12
"""
import datetime
import logging
import re
import tempfile
from ..core.source_selector import SourceSelector
from ..utils.fits import save_prm


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-gooettingen.de)"
__revision__ = 360


logger = logging.getLogger(__name__)


def add_parser(parser):
    p = parser.add_parser("SINGLESRC", help="""
    This sub-routine is used to prepare the fitting of a data cube for which no
    photometric and astrometric information is available. It should not be used
    with crowded field data but is useful in combination with datacubes that only
    contain few stars (such as the standard star cubes).
    For each star that should be analysed, its status and an initial guess for the
    IFS coordinates must be provided. If the PSF should be optimized in the fit, at
    least one source with status=3 must be provided. For each PSF source, the
    initial guess for the IFS coordinates is improved via aperture spectrophoto-
    metry and moment calculation.""")
    p.add_argument("source", nargs="+", type=str, help="""
    The IFS coordinates and the status of a source, provided as <x>,<y>,<status>.""")
    p.set_defaults(func=singlesrc)


def singlesrc(pampelmuse):

    logging.info("Executing requested routine SINGLESRC [Revision {0:d}].".format(__revision__))
    
    # get x, y, and status from user-provided strings
    split_input = re.compile(r"(\-?[0-9\.]+),(\-?[0-9\.]+),([1-3])")
    
    statuses = []
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write('id x y {0}\n'.format(pampelmuse.passband))
        for entry in pampelmuse.singlesrc.source:
            source = split_input.match(entry)
            if source is None:
                logger.critical("Cannot grab source information from string '{0}'.".format(entry))
                logger.critical("Source information must be provided as '<x>,<y>,<status>'.")
                assert source is not None, "Invalid input for source: {0}".format(entry)

            statuses.append(int(source.group(3)))
            f.write('{0} {1:.2f} {2:.2f} 0.0\n'.format(len(statuses), float(source.group(1)), float(source.group(2))))

            logger.info("Added source at ({0},{1}) with status {2} and ID {3}.".format(
                source.group(1),  source.group(2), source.group(3), len(statuses)))

    selector = SourceSelector(photometry_file=f.name, ifs_data=pampelmuse.ifs_data, passband=pampelmuse.passband)

    # If requested, add background component(s) to source catalogue
    if pampelmuse.globsky:
        n_background = selector.background_setup(skysize=pampelmuse.skysize)
        statuses.extend([4]*n_background)

    selector.make_sources_instance()
    selector.sources.status = statuses
    selector.sources.info()

    # Initialize PSF to be used
    psf_parameters = []
    i = 0
    while hasattr(pampelmuse.psf, "parname{0}".format(i)):
        psf_parameters.append({'name': getattr(pampelmuse.psf, "parname{0}".format(i)),
                               'value': getattr(pampelmuse.psf, "parvalue{0}".format(i)),
                               'free': getattr(pampelmuse.psf, "freepar{0}".format(i))})
        i += 1
    selector.prepare_psf(profile=pampelmuse.psf.profile, parameters=psf_parameters)

    # perform aperture spectrophotometry for PSF sources
    # need to load complete array for IFS coordinates before, otherwise it is only loaded
    # for the sources with status=3 and the results are overwritten when loading full array
    # for storing results.
    selector.sources.load_ifs_coordinates()
    selector.run_aperture_photometry(ids=selector.sources.ID[selector.sources.status != 4],
                                     start=pampelmuse.start, stop=pampelmuse.stop, radius=pampelmuse.aperrad,
                                     n_cpu=pampelmuse.cpucount, n_bin=pampelmuse.layerbin)
    selector.sources.ifs_coordinates_free[:] = True

    # write results to file
    logger.info("Saving source selection in file {0}.".format(pampelmuse.filename))

    # add information about routine to namespace
    pampelmuse.singlesrc.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.singlesrc.revision = __revision__

    header = pampelmuse.makeHeader()  # create header from namespace
    if selector.use_wcs:
        wcsheader = pampelmuse.ifs_data.wcs.to_header()
        for key in wcsheader:
            header["hierarch PAMPELMUSE WCS {0}".format(key)] = wcsheader[key]
    save_prm(sources=selector.sources, psf_attributes=selector.psf_attributes,
             filename=pampelmuse.filename, header=header)
