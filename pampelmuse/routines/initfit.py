"""
initfit.py
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
class pampelmuse.routines.initfit.ConfigurationError
function pampelmuse.routines.initfit.calculate_noise
function pampelmuse.routines.initfit.estimate_snr
function pampelmuse.routines.initfit.initfit

Purpose
-------
The module contains the function 'initfit' that is called when the PampelMuse
subroutine INITFIT is called. The function is designed to create a sophisticated
list of sources to be analysed with the PampelMuse subroutine CUBEFIT. It is
called with a single parameter that is a namespace containing the configuration
of PampelMuse. INITFIT must be called with the mandatory argument 'sourcelist'
to provide an ascii-file containing the crowded field photometry. This data are
used to initialize the catalogue of sources and to create a mock image from the
catalogue at the spatial resolution of the IFS data. This image is displayed to-
gether with the IFS data using the PampelMuse GUI. The spatial position of at
least 2 sources from the catalogue must be interactively selected. This information
is used afterwards to locate the position of all sources from the catalogue in (or
outside) the IFS data, estimate their S/N, and perform the final selection of the
sources that are used in the analysis carried out by CUBEFIT. All the information
required for this analysis, including an initial estimate of the PSF and the coor-
dinate transformation, are saved in the PRM file that is written to disk.

The class 'ConfigurationError' is introduced to handle cases in which the confi-
guration of PampelMuse does not allow for an analysis of the provided IFS data.
For example, a ConfigurationError is raised if no sources that pass the criteria
for the selection of PSF sources are detected.

The function 'estimate_snr' is designed to estimate the S/N of all sources in a
catalogue and is called by the function 'initfit'.

Latest SVN Revision
-------------------
377, 2016/12/14
"""
import datetime
import logging
import numpy as np
from ..core.source_selector import SourceSelector
from ..utils.fits import save_prm


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 357


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
    p.add_argument("sourcelist", type=str, nargs="?", help="""
    ASCII-file containing the results from crowded field photometry. Empty lines
    or lines starting with '#' are considered comments. The first columns may
    contain the column names, in which case the routine will look for columns
    named 'id', 'x', 'y', and <FNAME>, where <FNAME> is the name of the photo-
    metric filter used in the source selection process. If no header is available,
    the routine will use the first four columns.""")
    p.set_defaults(func=initfit)


def initfit(pampelmuse):

    logger.info("Executing requested routine INITFIT [Revision {0:d}].".format(__revision__))
    logger.info("Using version designed for {0} IFUs.".format("small" if pampelmuse.smallifu else "large"))

    # Initialize instance of SourceSelector class
    selector = SourceSelector(photometry_file=pampelmuse.initfit.sourcelist, ifs_data=pampelmuse.ifs_data,
                              passband=pampelmuse.passband,
                              zeropoint=pampelmuse.zeromag if pampelmuse.zeromag < 99 else None)

    # if it is possible to match IFS data and catalogue via WCS, remove all
    # sources from catalogue that have a distance to the field of view which
    # is larger than the maximum allowed offset provided via the configuration
    # parameter CCMAXOFF
    if selector.use_wcs:
        ccmaxoff = pampelmuse.ccmaxoff if pampelmuse.ccmaxoff > 0 else 0.5*max(selector.ifs_data.shape)
        xy = np.dstack((selector.cfp_results['x'], selector.cfp_results['y'])).astype('<f4')
        distance_to_edge = selector.ifs_data.distance_to_edge(xy[0])
        selector.valid &= (distance_to_edge > -ccmaxoff)
        logger.info('Excluded sources with large distances to IFS footprint, {0} sources remain.'.format(
            selector.valid.sum()))
    else:
        ccmaxoff = None

    # If requested, add background component(s) to source catalogue
    if pampelmuse.globsky:
        selector.background_setup(skysize=pampelmuse.skysize)

    # Initialize PSF to be used
    psf_parameters = []
    i = 0
    while hasattr(pampelmuse.psf, "parname{0}".format(i)):
        psf_parameters.append({'name': getattr(pampelmuse.psf, "parname{0}".format(i)),
                               'value': getattr(pampelmuse.psf, "parvalue{0}".format(i)),
                               'free': getattr(pampelmuse.psf, "freepar{0}".format(i))})
        i += 1
    selector.prepare_psf(profile=pampelmuse.psf.profile, parameters=psf_parameters)

    # Create mock image from photometry catalog
    selector.create_mock_image(n_cpu=pampelmuse.cpucount)

    # Create image from IFS data in provided passband
    selector.create_image_from_ifs()

    # if WCS information is available for both the IFS data and the catalog, estimate position in sky via
    # cross correlation
    if selector.use_wcs:
        selector.cross_correlate_images(ccmaxoff=ccmaxoff)

    # load GUI for visual check/identification
    if pampelmuse.interact:
        selector.identify_footprint()

    # Perform photometric calibration (unless provided) and S/N estimation:
    selector.calibrate_photometry(n_cpu=pampelmuse.cpucount, update_zeropoint=pampelmuse.updaterl)

    selector.select(reslim=pampelmuse.reslim if not pampelmuse.updaterl else None, sncut=pampelmuse.snrcut,
                    confusion_density=pampelmuse.confdens, binrad=pampelmuse.binrad,
                    use_unres=pampelmuse.useunres, unres_range=pampelmuse.unresbin)

    if not pampelmuse.smallifu:
        # find suitable PSF stars
        # maximum allowed magnitude of PSF star is 3mag brighter than faintest resolved star
        max_mag = selector.sources.mag[selector.sources.status == 2].max() - 3.
        selector.search_psf_stars(saturated=pampelmuse.saturated, max_mag=max_mag, psf_radius=pampelmuse.psfrad,
                                  aperture_noise=pampelmuse.apernois)

        # perform aperture spectrophotometry for PSF stars
        selector.run_aperture_photometry(start=pampelmuse.start, stop=pampelmuse.stop, radius=pampelmuse.aperrad,
                                         n_cpu=pampelmuse.cpucount, n_bin=pampelmuse.layerbin)
    else:
        selector.sources.ndisp = selector.ifs_data.nlayer

    # write results to file
    logger.info("Saving source selection in file {0}.".format(pampelmuse.filename))

    # add information about routine to namespace
    pampelmuse.initfit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.initfit.revision = __revision__

    header = pampelmuse.makeHeader()  # create header from namespace
    if selector.use_wcs:
        wcsheader = pampelmuse.ifs_data.wcs.to_header()
        for key in wcsheader:
            header["hierarch PAMPELMUSE WCS {0}".format(key)] = wcsheader[key]
    save_prm(sources=selector.sources, psf_attributes=selector.psf_attributes,
             filename=pampelmuse.filename, header=header)
