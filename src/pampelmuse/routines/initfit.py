"""
initfit.py
==========
Copyright 2013-2021 Sebastian Kamann

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
function pampelmuse.routines.initfit.add_parser
function pampelmuse.routines.initfit.initfit

Purpose
-------
The function add_parser provides the parser that is used to execute the
INITFIT process from the command line.

The function 'initfit' is executed when the PampelMuse subroutine INITFIT is
called. The function is designed to create a sophisticated list of sources to
be analysed with the PampelMuse subroutine CUBEFIT. It is called with a single
parameter - the configuration of PampelMuse, provided as an instance of a
configparser.ConfigParser class. It must contain information about the data
cube that should  be analysed and the reference catalog of sources. The data
are used to initialize the catalogue of sources and to create a mock image
from the catalogue at the spatial resolution of the IFS data. This image is
displayed together with the IFS data using the PampelMuse GUI. The spatial
position of at least 4 sources from the catalogue must be interactively
selected. This information is used afterwards to locate the position of all
sources from the catalogue in (or outside) the IFS data, estimate their S/N,
and perform the final selection of the sources that are used in the analysis
carried out by CUBEFIT. All the information required for this analysis,
including an initial estimate of the PSF and the coordinate transformation,
are saved in the PRM file that is written to disk.
"""
import sys
import datetime
import logging
import numpy as np
from ..core.source_selector import SourceSelector
from ..psf import Variables
from ..utils.fits import open_ifs_data, make_header_from_config, save_prm


__author__ = "Sebastian Kamann (skamann@ljmu.ac.uk)"
__revision__ = 20210622


logger = logging.getLogger(__name__)


def add_parser(parser):

    p = parser.add_parser("INITFIT", help="""
    The sub-routine is used to assemble a source list that is used in the PSF
    fitting process. Recall that PampelMuse assumes that an inventory of the
    sources in the field is available. From this inventory, the subset of
    sources that are accessible to an analysis with PampelMuse are identified
    and written to a dedicated FITS-file, the PRM-file, that is used through-
    out the remaining analysis. The different steps to obtain the PRM-file are
    a) Interactive identification of bright sources in the field based on the
    IFS data and a simulated image created from the provided sourcelist at
    the spatial resolution of the datacube.
    b) Determination of all sources in the inventory relative to the observe
    field of view. This information is used to estimate the photometric zero-
    point in the IFS data and to estimate the S/N of the individual sources.
    c) Assignment of the individual sources to components that are used in the
    analysis. A component is defined as an entity of one or more sources for
    which the spectrum is extracted independently from otther sources during
    the PSF fitting. The assignment is performed based on the estimated S/N
    of the sources and their relative distances.
    d) If possible, perform aperture spectrophotometry for the bright sources to
    obtain more accurate initial guesses for the parameters of the coordinate
    transformation and the PSF variables.""")
    p.set_defaults(func=initfit)


def initfit(config):
    """
    This function performs the initialisation stage of the fit. It is called
    when the INITFIT routine of PampelMuse is called.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration of PampelMuse.
    """
    logger.info("Executing requested routine INITFIT.")

    # Check if version for small or large FoV should be used
    smallifu = config['global']['smallifu']
    logger.info("Using version designed for {0} IFUs.".format("small" if smallifu else "large"))

    # get prefix of IFS data - note that the calling routine makes sure that the entry exists in the configuration
    prefix = config['global']['prefix']

    # open IFS data
    ifs_data = open_ifs_data(prefix=prefix)

    # adjust numbers of first and last layer to be analysed
    first_layer = config['layers'].get('first', 0)
    if first_layer < 0:
        first_layer = ifs_data.nlayer + first_layer
    last_layer = config['layers'].get('last', -1)
    if last_layer < 0:
        last_layer = ifs_data.nlayer + last_layer + 1

    # Initialize instance of SourceSelector class
    selector = SourceSelector(photometry_file=config['catalog']['name'],
                              ifs_data=ifs_data,
                              passband=config['catalog']['passband'],
                              zeropoint=config['initfit']['zeromag'],
                              position_fit=config['initfit']['posfit'])

    # Create image from IFS data in provided passband
    # This is done first because for some instruments (e.g. MUSE cubes) the image is needed to determine the edge of
    # the field of view.
    selector.create_image_from_ifs(apply_variances=config['initfit']['usesigma'])

    # if it is possible to match IFS data and catalogue via WCS, remove all
    # sources from catalogue that have a distance to the field of view which
    # is larger than the maximum allowed offset provided via the configuration
    # parameter CCMAXOFF
    if selector.use_wcs:
        # default value is 0.5x length of major IFS data axis
        ccmaxoff = config['initfit'].get('ccmaxoff', -1)
        if ccmaxoff < 0:
            ccmaxoff = 0.5*max(selector.ifs_data.shape)
        xy = np.dstack((selector.cfp_results['x'], selector.cfp_results['y'])).astype('<f4')
        distance_to_edge = selector.ifs_data.distance_to_edge(xy[0])
        selector.cfp_results['valid'] &= (distance_to_edge > -ccmaxoff)
        logger.info('Excluded sources with large distances to IFS footprint, {0} sources remain.'.format(
            selector.cfp_results['valid'].sum()))
        if not selector.cfp_results['valid'].any():
            logger.error('Reference catalogue does not contain sources near IFS field of view.')
            sys.exit(1)
    else:
        ccmaxoff = None

    # If requested, add background component(s) to source catalogue
    if config['sky'].get('use', True):
        selector.background_setup(skysize=config['sky']['binsize'], slices=config['sky']['slices'],
                                  oversampling=config['sky']['oversampling'])

    # Initialize PSF to be used
    # check if valid profile was provided
    profile = config['psf'].get('profile', 'moffat')
    if profile not in Variables.PROFILES.keys():
        logger.error('Unknown PSF profile: {0}'.format(profile))
        logger.error('Supported PSF profiles: {0}'. format(', '.join(Variables.PROFILES.keys())))

    # get information about free parameters
    psffit = config['psf'].get('fit', 6)

    # get initial values for PSF parameters
    psf_parameters = []
    # loop over parameters for chosen profile, get values and see if the are free or fixed
    for parameter in Variables.PROFILES[profile][::-1]:
        i = Variables.PROFILES[profile].index(parameter) + 1
        try:
            value = config['psf'][parameter]
        except KeyError:
            raise IOError('Missing initial value for PSF parameter "{0}"'.format(parameter))
        if psffit >= 2**i:
            free = True
            psffit -= 2**i
        else:
            free = False
        psf_parameters.append({'name': parameter, 'value': value, 'free': free})
    selector.prepare_psf(profile=profile, parameters=psf_parameters, use_lut=config['psf']['lut1d'],
                         lut_radius=config['psf']['radius'])

    # Create mock image from photometry catalog
    n_cpu = config['global']['cpucount']
    selector.create_mock_image(n_cpu=n_cpu)

    # if WCS information is available for both the IFS data and the catalog, estimate position in sky via
    # cross correlation
    if selector.use_wcs:
        selector.cross_correlate_images(ccmaxoff=ccmaxoff)

    # load GUI for visual check/identification
    if config['global'].get('interact', False):
        selector.identify_footprint()

    # Perform photometric calibration and S/N estimation:
    selector.calibrate_photometry(n_cpu=n_cpu, update_zeropoint=not config['initfit']['zerofix'])

    selector.select(reslim=None,
                    sncut=config['initfit'].get('snrcut', 3),
                    confusion_density=config['initfit'].get('confdens', 0.4),
                    binrad=config['initfit'].get('binrad', 0.3),
                    use_unres=config['unresolved'].get('use', False),
                    unres_range=config['unresolved'].get('binsize', 2))

    if not smallifu:
        # find suitable PSF stars

        # check if saturated stars exist that should be excluded from PSF search
        saturated = config['catalog'].get('saturated', None)

        # maximum allowed magnitude of PSF star is 3mag brighter than faintest resolved star unless specified in
        # user configuration file
        if config['initfit']['psfmag'] is not None:
            max_mag = config['initfit']['psfmag']
        else:
            max_mag = selector.sources.mag[selector.sources.status == 2].max() - 3.

        selector.search_psf_stars(saturated=saturated,
                                  max_mag=max_mag,
                                  psf_radius=config['psf'].get('radius', 15),
                                  psf_radius_scaling=config['psf']['scaling'],
                                  aperture_noise=config['initfit'].get('apernois', 0.1))

        # perform aperture spectrophotometry for PSF stars
        selector.run_aperture_photometry(start=first_layer, stop=last_layer, n_cpu=n_cpu,
                                         radius=config['initfit'].get('aperrad', 3),
                                         n_bin=config['layers'].get('binning', 1))
    else:
        selector.sources.ndisp = selector.ifs_data.nlayer

    # write results to file
    logger.info("Saving source selection in file {0}.prm.fits.".format(prefix))

    # create header containing configuration
    header = make_header_from_config(config=config)
    if selector.use_wcs:
        wcs_header = ifs_data.wcs.to_header()
        for key in wcs_header:
            header["HIERARCH PAMPELMUSE WCS {0}".format(key)] = wcs_header[key]

    # add history entry for INITFIT run
    now = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    header['HISTORY'] = 'INITFIT finished at {0}'.format(now)

    # save PRM file
    save_prm(sources=selector.sources, psf_attributes=selector.psf_attributes, filename='{0}.prm.fits'.format(prefix),
             header=header)
