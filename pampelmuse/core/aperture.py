"""
aperture.py
===========
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
class pampelmuse.core.aperture.AperPhoto

Purpose
-------
This module implements  the class 'AperPhoto' that is designed to perform
aperture photometry on a 2dim. image or aperture spectrophotometry a 3dim.
datacube. An instance of the class can be initialized from IFS data and a
cources catalogue. To this aim, a valid instance of pampelmuse.core.cube.Cube
and a valid instance of pampelmuse.core.sources.Sources must be provided.
The aperture photometry is performed for each layer of the IFS data indepen-
dently. A local background around every source is estimated and subtracted
before its flux and moments are determined. If requested, the routine also
estimates the global background in each layer of the data.

Provides
--------
function pampelmuse.core.aperture.run_ap

Latest SVN revision
-------------------
356, 2016/08/10
"""
import logging
from multiprocessing import Pool
import numpy as np
from scipy import spatial
from ..utils.statistics import clip_sigma
from ..instruments import Instrument, MusePixtable


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 356


logger = logging.getLogger(__name__)


def init_worker(centroids, background, r_aperture, calc_moments, threshold, *args):
    """
    This function is used to set the parameters for the parallel execution of
    the aperture photometry that do not change from one layer to the next.
    Therefore, they need to be provided only once at the beginning of the
    analysis.

    Parameters
    ----------
    centroids : nd_array
        The initial guesses for the centroids of the sources as an array of
        shape (n, 2) for n sources.
    background : basestring
        How the background is modeled: should be 'local' or 'global'.
    r_aperture : float
        The aperture radius in units of pixels.
    calc_moments : boolean
        Flag indicating if the moments should also be measured besides the
        fluxes.
    threshold : float
        Threshold for including a pixel in the calculation, given in units of
        the background noise.
    """
    global g_centroids
    global g_background
    global g_r_aperture
    global g_calc_moments
    global g_threshold

    g_centroids = centroids
    g_background = background
    g_r_aperture = r_aperture
    g_calc_moments = calc_moments
    g_threshold = threshold


def run_ap(input_data):
    """
    This function performs aperture photometry on one or more images around
    a set of input coordinates. Each image must be a 1dim. array, and the
    positions of the individual spaxels must be provided with the data.
    For the purpose of parallelizing the code, this function is defined out-
    side of the AperPhoto class. The parallelization via multi-threading
    seems not to work if a method of the same class is called.
    
    Parameters
    ----------
    input_data : dictionary
        The dictionary must contain the following entries:

        transformation : array_like, shape (M,2)
            The x- and y-positions of the M individual pixels in the 'data'
            array
        data : array_like, shape (M,) or (P,M)
            The data on which to perform aperture photometry. It should be
            provided with the spatial dimensions collapsed, i.e. a single
            image should be provided as a 1D array and P images should be
            provided as a 2D array with the first axis counting the images.
            Note that the data can have NaN values for invalid pixels. They
            are masked out before the calculation.

    Returns
    -------
    results : nd_array, shape (P,N,1) or (P,N,6)
        The calculated values are returned as a 3D array: first axis counts images,
        the second counts centroids, and the third counts the measured properties.
        If calc_moments is set to False, the only property will be the measured
        flux, otherwise, the flux will be returned together with the first and
        second order spatial moments.
    background : ndarray, shape (P,), optional
        The spectrum of the global background is only returned if the parameter
        background is set to 'global'.
    """
    global g_centroids
    global g_background
    global g_r_aperture
    global g_threshold
    global g_calc_moments

    # data can either be an individual layer (1D) or a sequence of layers (2D). In the latter case, the first axis of
    # the array must denote the layer number. The code performs a loop over the first axis, so it must be ensured that
    # data is always 2D.
    data = np.atleast_2d(input_data['data'])

    # prepare output array
    results = np.zeros((data.shape[0], g_centroids.shape[0], 1 + 5 * g_calc_moments), dtype=np.float32)

    # calculate distance of centroid of each source to each spaxel. Note that the shape of the returned array is
    # (N_spaxels, N_sources)
    offsets = spatial.distance.cdist(input_data['transformation'], g_centroids)

    # in the next steps, we define properties that do not change from one layer to another
    # 1) global background pixels, defined as having distances >r_aper to all sources
    if g_background == "global":
        backpix = (offsets > g_r_aperture).all(axis=1)
        if backpix.sum() < 20:
            logger.warning("Using < 20 pixels to estimate global background.")

        backvalues = np.zeros((data.shape[0],), dtype=np.float32)  # also create empty array for background values

    # 2) local background pixels, defined as having distances >2*r_aper and <4*r_aper to one source
    else:
        backpix = (offsets > 2 * g_r_aperture) & (offsets <= 4 * g_r_aperture)
        backvalues = None

    # 3) flux pixels, defined a having distances <=r_aper to one source.
    fluxpix = offsets <= g_r_aperture

    # estimate global background value if requested. For this, all spaxels within a distance r_aper to any source
    # centroid are masked out
    for i in range(data.shape[0]):

        valid = ~np.isnan(data[i])
        data[i, ~valid] = 0

        if not valid.any():
            logger.error("Layer contains only masked pixels.")
            results[i] = np.nan
            continue

        if g_background == "global":
            logger.debug("Determining global background...")

            # perform kappa-sigma clipping to determine background level and noise
            backdata = clip_sigma(data[i, backpix & valid], axis=None)
            backval = backdata.mean()
            backstd = backdata.std()

            backvalues[i] = backval
            logger.debug("...global background is {0:.2g}/pixel".format(backval))

        # loop over all centroids
        for j in range(g_centroids.shape[0]):

            logger.debug("Processing source {0}/{1}...".format(j + 1, g_centroids.shape[0]))

            # estimate local background in annulus between 2*r_aper and 4*r_aper (if requested, otherwise global
            # background is used)
            if g_background == "local":
                backdata = clip_sigma(data[i, backpix[:, j] & valid], axis=None)
                backval = backdata.mean()
                backstd = backdata.std()
                logger.debug("...local background is {0:.2g}+/-{1:0.2g}".format(backval, backstd))

            # Get array for calculations & subtract background estimate
            fluxdata = data[i, fluxpix[:, j] & valid] - backval
            results[i, j, 0] = fluxdata.sum()
            logger.debug("...flux inside aperture is {0:.3g}".format(results[i, j, 0]))

            # Moment calculation, only if requested:
            if g_calc_moments:
                mompix = fluxpix[:, j] & valid & (data[i] > (backval + g_threshold * backstd))
                momdata = data[i, mompix]
                logger.debug("...{0} pixel(s) available for moment calculation".format(mompix.sum()))
                if mompix.sum() > 3:
                    X = input_data['transformation'][mompix, 1]
                    Y = input_data['transformation'][mompix, 0]

                    total = float(momdata.sum())
                    x = (X * momdata).sum() / total
                    y = (Y * momdata).sum() / total
                    x2 = (X * X * momdata).sum() / total - x ** 2
                    y2 = (Y * Y * momdata).sum() / total - y ** 2
                    xy = (X * Y * momdata).sum() / total - x * y

                    # if unphysical moments measured or centroid outside measurement area, spectrumMask out all
                    # results for this source
                    if x2 <= 0 or y2 <= 0 or xy ** 2 >= (x2 * y2) or x < X.min() or x > X.max() or y < Y.min() \
                            or y > Y.max():
                        x = np.nan
                        y = np.nan
                        x2 = np.nan
                        y2 = np.nan
                        xy = np.nan
                    logger.debug(
                        "...calculated moments: x={0:.1f}, y={1:.1f}, x2={2:.1f}, y2={3:.1f}, xy={4:.1f}".format(
                            x, y, x2, y2, xy))
                else:
                    logger.debug("...too few pixels - skipping moment calculation.")
                    x = np.nan
                    y = np.nan
                    x2 = np.nan
                    y2 = np.nan
                    xy = np.nan
                results[i, j, 1:6] = [x, y, x2, y2, xy]

        return [results, backvalues] if g_background == "global" else results


class AperPhoto:
    """
    Perform aperture (spectro)photometry on 2 or 3dimensional data using
    a provided list of sources. The calculation of the first and second order
    moments for the provided sources  is optional. Prior to any flux or moment
    measurement, a background (global or local if  requested) is determined
    and subtracted. 
    """

    def __init__(self, cube, centroids, psf_instance=None):
        """
        Initialize an instance of AperPhoto-class

        Parameters
        ----------
        cube : an instance of a pampelmuse.instruments class
            All the information about the data that should  be analysed should
            be provided here. For further information on how to do this, check
            the documentation included in the class 'Cube'.
        centroids : array_like
            Initial guesses for the centroids of the sources, provided as an
            array of shape (n, 2) for n sources.
            ------------------------------------------------------------------
            IMPORTANT: According to the numpy convention, coordinates must be
                       provided in reverse order, i.e (y, x) instead of (x, y)
            ------------------------------------------------------------------
        psf_instance : an instances of pampelmuse.psf.profiles.Profile
            To estimate the aperture correction, i.e. the fraction of flux
            outside the aperture, provide a model of the PSF of the sources.
            Warning: the aperture correction is not implemented yet.
        """
        logger.info("Initializing aperture photometry, validating input...")

        # Checking 'cube'
        assert isinstance(cube, Instrument), \
            "Parameter 'cube' must be an instance of a pampelmuse.instrument class, not{0}.".format(cube.__class__)
        self.cube = cube

        self.centroids = np.atleast_2d(centroids)
        assert self.centroids.shape[1] == 2, 'Last axis of centroids parameter must have length 2.'

        self.n_sources = self.centroids.shape[0]
        logger.info('Centroids for {0} source(s) provided.'.format(self.n_sources))

        if psf_instance is not None:
            raise(NotImplementedError, 'Aperture corrections are not implemented yet.')
        self.psf_instance = None

        self.layers_to_analyse = None

    def __call__(self, r_aper=3., layer_range=(0, -1), calc_moments=True, thresh=3., background="global", n_cpu=1,
                 n_bin=1):
        """
        Method performs the actual aperture (spectro)photometry (and
        measures the moments if requested). An estimate of the global
        background is determined using a sigma-clipping approach.
        Optionally, a local background around every source can be
        determined.

        Note that the aperture (spectro)photometry is always performed
        around the input coordinates, NOT around the first order moments
        (even if they are calculated).

        Parameters
        ----------
        r_aper : float, optional
            Measurement radius; only pixels with distances <= r_aper to
            the source center are taken into account
        layer_range : array_like
            The indices of the first and the last layer of the data cube
            that should be included in the calculation. The default is to
            use the entire cube. 
        calc_moments : bool, optional
            Boolean flag that tells AperPhoto whether moments should be
            calculated
        thresh : float, optional
            Only pixels with values larger than <thresh> times the
           estimated background noise are considered in the moment
            calculation.
        background : str, optional
            Can be either set to 'global' or to 'local'. In the former case,
            a global background for each layer is obtained, while in the
            latter case, the background is calculated locally in an annulus
            around each source.
        n_cpu : integer, optional
            The number of CPUs used in the calculation.
        n_bin : int, optional
            The number of layers of the IFS data that are combined into a
            layer that is analysed.

        Returns
        -------
        results : numpy_record_array
            For every source, the flux/spectrum is provided in field
            'spectrum' and moments are provided in fields 'x', 'y', 'x2',
            'y2' and 'xy'.
            Note that if the parameter 'background' is set to 'global' the
            spectrum of the background will be stored in the 'spectrum'
            field of a hypothetical source with ID=-1.
        """
        self.r_aper = r_aper

        start, stop = layer_range
        if start < 0:
            start = self.cube.wave.size + start
        if stop < 0:
            stop = self.cube.wave.size + (stop + 1)
        self.nbin = n_bin
        self.layers_to_analyse = np.arange(start + self.nbin//2, stop, self.nbin, dtype=np.int16)

        self.thresh = thresh
        self.calc_moments = calc_moments
        self.n_cpu = n_cpu

        if background not in ["local", "global"]:
            logger.critical("Unknown value '{0}' for parameter 'background'. Must be 'local' or 'global'.".format(
                background))
        assert(background in ["local", "global"]),\
            "Parameter 'background' must be  either 'local' or 'global', not '{0}'.".format(background)

        logger.info("Starting calculation with the following setup:")
        logger.info("   aperture radius [pixel]: {0:.1f}".format(r_aper))
        logger.info("   range of considered layers: {0} - {1}, ".format(start, stop))
        logger.info("   binning factor of input data: {0}".format(self.nbin))
        logger.info("   pixel threshold [sigma]: {0}".format(thresh))
        logger.info("   calculating moments?: {0}".format(calc_moments))
        logger.info("   background estimation: {0}".format(background))
        logger.info("   number of CPUs: {0}".format(n_cpu))

        # prepare record array that will contains output.
        logger.info("   preparing array to hold results...")
        dtype = [("id", np.int32), ("flux", np.float32, (self.cube.wave.size,))]
        if calc_moments:
            for name in ["x", "y", "x2", "y2", "xy"]:
                dtype.append((name, np.float32, (self.cube.wave.size,)))
        self.results = np.zeros(self.n_sources + (background == "global"), dtype=dtype)
        if self.results.size == 1:
            self.results.view("<f4")[:] = np.nan
        else:
            self.results[:] = np.nan

        # use ID=-1 for a global background if it is used
        if background == "global":
            self.results["id"][-1] = -1
            for name in ["x", "y", "x2", "y2", "xy"]:
                self.results[name][-1] = np.nan

        # prepare list to store intermediate results
        _results = []

        # initialize parallelisation
        initargs = (self.centroids, background, r_aper, calc_moments, thresh)
        if self.n_cpu > 1:
            pool = Pool(self.n_cpu, initializer=init_worker, initargs=initargs)
        else:
            pool = None
            init_worker(*initargs)

        # loop over layers, collect input data.
        # every time number of layers is equal to number of CPUs, trigger parallel analysis
        input_data = []
        for i, layer_index in enumerate(self.layers_to_analyse):
            logger.info("Processing layer {0}/{1}".format(i + 1, self.layers_to_analyse.size))

            # load data - note that layer binning is currently not performed for pixtables.
            if n_bin == 1 or isinstance(self.cube, MusePixtable):
                data = self.cube.get_layer(i=layer_index)[0]
            else:
                if layer_index != self.layers_to_analyse[-1]:
                    data = self.cube.integrate(layer_index - n_bin // 2, layer_index + n_bin // 2)[0]
                else:
                    data = self.cube.integrate(layer_index - n_bin // 2, stop)[0]

            input_data.append({'data': data, 'transformation': self.cube.transform})

            if not i % self.n_cpu or layer_index == self.layers_to_analyse[-1]:
                if n_cpu > 1:
                    mapper = pool.map_async(run_ap, input_data)
                    _results.extend(mapper.get())
                else:
                    _results.append(run_ap(input_data[0]))
                input_data = []

        if background == "global":
            results = np.vstack([r[0] for r in _results])
            self.results["flux"][-1, self.layers_to_analyse] = np.hstack((r[1] for r in _results))
        else:
            results = np.vstack(_results)

        self.results["flux"][:results.shape[1], self.layers_to_analyse] = results[..., 0].T
        if calc_moments:
            i = 1
            for name in ["x", "y", "x2", "y2", "xy"]:
                self.results[name][:results.shape[1], self.layers_to_analyse] = results[..., i].T
                i += 1

        return self.results
