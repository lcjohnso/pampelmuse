"""
singlesrc.py
============
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
function pampelmuse.routines.singlesrc.add_parser
function pampelmuse.routines.singlesrc.singlesrc

Purpose
-------
The module contains the function 'singlesrc' that is used to prepare a data
cube for the analysis for which no photometric or astrometric information is
available. It prepares a PRM-file directly from a set of provided IFS coordi-
nates and statuses. For each of the sources with status=3, aperture spectro-
photometry is performed and the initial guesses for the IFS coordinates are
improved using the first order moments determined in the calculation.
 
Latest SVN revision
-------------------
407, 2017/04/02
"""
import datetime
import logging
from ..core.source_selector import SourceSelector
from ..psf import Variables
from ..utils.fits import open_ifs_data, make_header_from_config, save_prm


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-gooettingen.de)"
__revision__ = 407


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
    p.set_defaults(func=singlesrc)


def singlesrc(config):
    """
    The function prepares the analysis for a field without an astrometry
    reference catalogue. It is called when the SINGLESRC routine of
    PampelMuse is called.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of PampelMuse
    """
    logging.info("Executing requested routine SINGLESRC [Revision {0:d}].".format(__revision__))

    # get prefix of IFS data - note that the calling routine makes sure that the entry exists in the configuration
    prefix = config['general']['prefix']

    # open IFS data
    ifs_data = open_ifs_data(prefix=prefix)

    # adjust numbers of first and last layer to be analysed
    first_layer = config['layers'].get('first', default=0)
    if first_layer < 0:
        first_layer = ifs_data.nlayer - first_layer
    last_layer = config['layers'].get('last', default=-1)
    if last_layer < 0:
        last_layer = ifs_data.nlayer - last_layer + 1

    # a catalog containing IDs, initial guesses for x/y, and statuses of the sources must be provided
    try:
        photometry_file = config['catalog']['name']
    except KeyError:
        logger.critical('Cannot find catalogue containing sources in configuration.')
        logger.critical('Configuration parameter "catalog.name" must exist in user configuration file.')
        return 1
    else:
        if not photometry_file:
            logger.critical('Configuration parameter "catalog.name" cannot be null.')
            return 1

    selector = SourceSelector(photometry_file=photometry_file, ifs_data=ifs_data,
                              passband=config['catalog'].get('passband', default='F814W'))

    statuses = list(selector.cfp_results['status'])

    # If requested, add background component(s) to source catalogue
    if config['sky'].get('use', default=True):
        n_background = selector.background_setup(skysize=config['sky'].get('binsize', default=-1))
        statuses.extend([4]*n_background)

    selector.make_sources_instance()
    selector.sources.status = statuses
    selector.sources.info()

    # Initialize PSF to be used
    # check if valid profile was provided
    profile = config['psf'].get('profile', default='moffat')
    if profile not in Variables.PROFILES.keys():
        logger.error('Unknown PSF profile: {0}'.format(profile))
        logger.error('Supported PSF profiles: {0}'. format(', '.join(Variables.PROFILES.keys())))

    # get information about free parameters
    psffit = config['psf'].get('fit', default=6)

    # get initial values for PSF parameters
    psf_parameters = []
    # loop over parameters for chosen profile, get values and see if the are free or fixed
    for parameter in Variables.PROFILES[profile][::-1]:
        i = Variables.PROFILES[profile].index(parameter)
        value = config['psf'].get(parameter)
        if psffit >= 2**i:
            free = True
            psffit -= 2**i
        else:
            free = False
        psf_parameters.append({'name': parameter, 'value': value, 'free': free})
    selector.prepare_psf(profile=profile, parameters=psf_parameters)

    # perform aperture spectrophotometry for PSF sources
    # need to load complete array for IFS coordinates before, otherwise it is only loaded
    # for the sources with status=3 and the results are overwritten when loading full array
    # for storing results.
    selector.sources.load_ifs_coordinates()
    selector.run_aperture_photometry(ids=selector.sources.ID[selector.sources.status != 4],
                                     start=first_layer, stop=last_layer,
                                     radius=config['initfit'].get('aperrad', default=3),
                                     n_cpu=config['global'].get('cpucount', default=1),
                                     n_bin=config['layers'].get('binning', default=1))
    selector.sources.ifs_coordinates_free[:] = True

    # write results to file
    logger.info("Saving source properties in file {0}.prm.fits.".format(prefix))

    # add information about routine to configuration so that they are copied to the header of the PRM-file
    if 'singlesrc' not in config.keys():
        config['signlesrc'] = {}
    config['singlesrc']['utc'] = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    config['singlesrc']['revision'] = __revision__

    # create header containing configuration
    header = make_header_from_config(config=config)
    if selector.use_wcs:
        wcs_header = ifs_data.wcs.to_header()
        for key in wcs_header:
            header["hierarch PAMPELMUSE WCS {0}".format(key)] = wcs_header[key]

    # save PRM file
    save_prm(sources=selector.sources, psf_attributes=selector.psf_attributes, filename='{0}.prm.fits'.format(prefix),
             header=header)
