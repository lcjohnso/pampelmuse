"""
aperture.py
===========

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
183, 2014/06/30
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 183

import itertools
import logging
import sys
from multiprocessing import cpu_count, Pool
from scipy import spatial

import numpy as np

# required PampelMuse-packages
from pampelmuse.core import Cube
from pampelmuse.core.psf import Profile
from pampelmuse.functions.statistics import clip_sigma


def run_ap(centroids, pixpos, data, background="global", r_aper=3., calc_moments=False, thresh=3.):
    """
    This function performs aperture photometry on one or more images around
    a set of input coordinates. Each image must be a 1dim. array, and the
    positions of the individual spaxels must be provided with the data.
    For the purpose of parallelizing the code, this function is defined out-
    side of the AperPhoto class. The parallelization via multithreading
    seems not to work if a method of the same class is called.
    
    Parameters
    ----------
    centroids : array_like, shape (N,2)
        The N coordinates around which aperture photometry should be performed
    pixpos : array_like, shape (M,2)
        The x- and y-positions of the M individual pixels in the 'data' array
    data : array_like, shape (M,) or (P,M)
        The data on which to perform aperture photometry. It should be provided
        with the spatial dimensions collapsed, i.e. a single image should be
        provided as a 1D array and P images should be provided as a 2D array
        with the first axis counting the images. Note that the data can have NaN
        values for invalid pixels. They are masked out before the calculation.
    background : str, optional
        Can be either set to 'global' or to 'local'. In the former case, a global
        background for each layer is obtained, while in the latter case, the
        background is calculated locally in an annulus around each source.
    r_aper : float, optional
        Measurement radius; only pixels with distances <= r_aper to a centroid
        are taken into account.
    calc_moments : bool, optional
        Boolean flag that tells AperPhoto whether moments should be calculated.
    thresh : float, optional
        Only pixels with values larger than <thresh> times the estimated back-
        ground noise are considered in the moment calculation.
    
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
    # data can either be an individual layer (1D) or a sequence of layers (2D). In the latter case, the first axis of
    # the array must denote the layer number. The code performs a loop over the first axis, so it must be ensured that
    # data is always 2D.
    data = np.atleast_2d(data)

    # prepare output array
    results = np.zeros((data.shape[0], centroids.shape[0], 1 + 5 * calc_moments), dtype=np.float32)

    # calculate distance of centroid of each source to each spaxel. Note that the shape of the returned array is
    # (N_spaxels, N_sources)
    offsets = spatial.distance.cdist(pixpos, centroids)

    # in the next steps, we define properties that do not change from one layer to another
    # 1) global background pixels, defined as having distances >r_aper to all sources
    if background == "global":
        backpix = (offsets > r_aper).all(axis=1)
        if backpix.sum() < 20:
            logging.warning("Using < 20 pixels to estimate global background.")

        backvalues = np.zeros((data.shape[0],), dtype=np.float32)  # also create empty array for background values

    # 2) local background pixels, defined as having distances >2*r_aper and <4*r_aper to one source
    else:
        backpix = (offsets > 2 * r_aper) & (offsets <= 4 * r_aper)

    # 3) flux pixels, defined a having distances <=r_aper to one source.
    fluxpix = offsets <= r_aper

    # estimate global background value if requested. For this, all spaxels within a distance r_aper to any source
    # centroid are masked out
    for i in xrange(data.shape[0]):

        valid = ~np.isnan(data[i])
        data[i, ~valid] = 0

        if not valid.any():
            logging.error("Layer contains only masked pixels.")
            results[i] = np.nan
            continue

        if background == "global":
            logging.debug("Determining global background...")

            # perform kappa-sigma clipping to determine background level and noise
            backdata = clip_sigma(data[i, backpix & valid], axis=None)
            backval = backdata.mean()
            backstd = backdata.std()

            backvalues[i] = backval
            logging.debug("...global background is {0:.2g}/pixel".format(backval))

        # loop over all centroids
        for j in xrange(centroids.shape[0]):

            logging.debug(
                "Processing source [{0:{1}d}/{2}]...".format(i + 1, len(str(centroids.shape[0])), centroids.shape[0]))

            # estimate local background in annulus between 2*r_aper and 4*r_aper (if requested, otherwise global
            # background is used)
            if background == "local":
                backdata = clip_sigma(data[i, backpix[:, j] & valid], axis=None)
                backval = backdata.mean()
                backstd = backdata.std()
                logging.debug("...local background is {0:.2g}+/-{1:0.2g}".format(backval, backstd))

            # Get array for calculations & subtract background estimate
            fluxdata = data[i, fluxpix[:, j] & valid] - backval
            results[i, j, 0] = fluxdata.sum()
            logging.debug("...flux inside aperture is {0:.3g}".format(results[i, j, 0]))

            # Moment calculation, only if requested:
            if calc_moments:
                mompix = fluxpix[:, j] & valid & (data[i] > (backval + thresh * backstd))
                momdata = data[i, mompix]
                logging.debug("...{0} pixel(s) available for moment calculation".format(mompix.sum()))
                if mompix.sum() > 3:
                    X = pixpos[mompix, 1]
                    Y = pixpos[mompix, 0]

                    total = float(momdata.sum())
                    x = (X * momdata).sum() / total
                    y = (Y * momdata).sum() / total
                    x2 = (X * X * momdata).sum() / total - x ** 2
                    y2 = (Y * Y * momdata).sum() / total - y ** 2
                    xy = (X * Y * momdata).sum() / total - x * y

                    # if unphysical moments measured or centroid outside measurement area, mask out all results for
                    # this source
                    if x2 <= 0 or y2 <= 0 or xy ** 2 >= (x2 * y2) or x < X.min() or x > X.max() or y < Y.min() \
                            or y > Y.max():
                        x = np.nan
                        y = np.nan
                        x2 = np.nan
                        y2 = np.nan
                        xy = np.nan
                    logging.debug(
                        "...calculated moments: x={0:.1f}, y={1:.1f}, x2={2:.1f}, y2={3:.1f}, xy={4:.1f}".format(
                            x, y, x2, y2, xy))
                else:
                    logging.debug("...too few pixels - skipping moment calculation.")
                    x = np.nan
                    y = np.nan
                    x2 = np.nan
                    y2 = np.nan
                    xy = np.nan
                results[i, j, 1:6] = [x, y, x2, y2, xy]

        return [results, backvalues] if background == "global" else results


class AperPhoto:
    """
    Perform aperture (spectro)photometry on 2 or 3dimensional data using
    a provided list of sources. The calculation of the first and second order
    moments for the provided sources  is optional. Prior to any flux or moment
    measurement, a background (global or local if  requested) is determined
    and subtracted. 
    """

    def __init__(self, cube, psf_instances):
        """
        Initialize an instance of AperPhoto-class

        Parameters
        ----------
        cube : an instance of pampelmuse.core.cube.Cube
            All the information about the data that should  be analysed should
            be provided here. For further information on how to do this, check
            the documentation included in the class 'Cube'.
        psf_instances : a list of instances of pampelmuse.psf.profiles.Profile
            The PSF instances should contain the information about the sources
            for which aperture photometry should be performed. At least the
            properties 'xc' and 'yc' must be available for each instance and set
            to the central coordinates of the source on the cube.
        """
        logging.info("Initializing aperture photometry, validating input...")

        # Checking 'cube'
        assert isinstance(cube, Cube),\
            "Parameter 'cube' must be an instance of pampelmuse.core.cube.Cube, not{0}.".format(cube.__class__)
        self.cube = cube

        # Checking PSF instances
        if not hasattr(psf_instances, "__iter__"):
            psf_instances = [psf_instances]
        for psf in psf_instances:
            assert isinstance(psf, Profile),\
                "Entries of 'psf_instances' must be instances of pampelmuse.core.psf.profiles.Profile, not {0}.".format(
                    psf.__class__)
            if not psf.transform.shape == cube.instrument.transform.shape \
                    and (psf.transform == cube.instrument.transform).all():
                psf.transform = cube.instrument.transform
                logging.warning("PSF array of source #{0} did not match data, modified it.".format(psf.src_id))
        self.psf_instances = psf_instances

        self.nsources = len(self.psf_instances)
        logging.info("Prepared aperture spectrophotometry for {0} source[s].".format(self.nsources))

    def __call__(self, r_aper=3., layer_range=(0, -1), calc_moments=True, thresh=3., background="global", ncpu=1,
                 nbin=1):
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
        ncpu : integer, optional
            The number of CPUs used in the calculation.
        nbin : int, optional
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
            start = self.cube.ndisp + start
        if stop < 0:
            stop = self.cube.ndisp + (stop + 1)
        self.nbin = nbin
        layers_to_analyse = np.arange(start + self.nbin//2, stop, self.nbin, dtype=np.int16)

        self.thresh = thresh
        self.calc_moments = calc_moments
        self.ncpu = ncpu

        if background not in ["local", "global"]:
            logging.critical("Unknown value '{0}' for parameter 'background'. Must be 'local' or 'global'.".format(
                background))
        assert(background in ["local", "global"]),\
            "Parameter 'background' must be  either 'local' or 'global', not '{0}'.".format(background)

        logging.info("Starting calculation with the following setup:")
        logging.info("   aperture radius [spaxel]: {0:.1f}".format(r_aper))
        logging.info("   range of considered layers: {0} - {1}, ".format(start, stop))
        logging.info("   binning factor of input data: {0}".format(self.nbin))
        logging.info("   pixel threshold [sigma]: {0}".format(thresh))
        logging.info("   calculating moments?: {0}".format(calc_moments))
        logging.info("   background estimation: {0}".format(background))
        logging.info("   number of CPUs: {0}".format(ncpu))

        # prepare record array that will contains output.
        logging.info("   preparing array to hold results...")
        dtype = [("id", np.int32), ("flux", np.float32, (self.cube.ndisp,))]
        if calc_moments:
            for name in ["x", "y", "x2", "y2", "xy"]:
                dtype.append((name, np.float32, (self.cube.ndisp,)))
        self.results = np.zeros(self.nsources + (background == "global"), dtype=dtype)
        self.results[:] = np.nan

        # collect centroid coordinates for all sources and add their IDs to output array:
        centroids = np.zeros((self.nsources, 2), dtype=np.float32)
        i = 0
        for psf in self.psf_instances:
            centroids[i] = psf.yc, psf.xc
            self.results["id"][i] = psf.src_id
            i += 1

        # use ID=-1 for a global background if it is used
        if background == "global":
            self.results["id"][-1] = -1
            for name in ["x", "y", "x2", "y2", "xy"]:
                self.results[name][-1] = np.nan

        # loop over layers, assign layers to CPUs if parallelization should be used, and perform aperture photometry
        _results = []

        i = 0
        while i < layers_to_analyse.size:

            if ncpu > 1:
                pool = Pool(ncpu)
                threads = []

            for nthreads in xrange(ncpu):
                layer_index = layers_to_analyse[i]

                logging.info("   processing layer {0:{1}d}/{2}".format(i+1, len(str(layers_to_analyse.size)),
                                                                       layers_to_analyse.size))

                # load data
                if nbin == 1:
                    data = self.cube.data.layer(layer_index)
                    if self.cube.mask is not None:
                        mask = self.cube.mask.layer(layer_index)
                        data[mask] = np.nan
                else:
                    if layer_index != layers_to_analyse[-1]:
                        data = self.cube.integrate(layer_range=(layer_index-nbin//2, layer_index+nbin//2))
                    else:
                        data = self.cube.integrate(layer_range=(layer_index-nbin//2, stop))

                if ncpu > 1:
                    threads.append(pool.apply_async(run_ap, args=(
                        centroids, self.cube.instrument.transform, data, background, r_aper, calc_moments, thresh)))
                else:
                    # put result in list to make output comparable with multi-CPU case
                    _results.append(run_ap(
                        centroids, self.cube.instrument.transform, data, background, r_aper, calc_moments, thresh))

                i += 1
                if i == layers_to_analyse.size:
                    break

            if ncpu > 1:
                pool.close()
                pool.join()
                _results.extend([threads[n].get() for n in xrange(nthreads + 1)])

        if background == "global":
            results = np.vstack([r[0] for r in _results])
            self.results["flux"][-1] = np.hstack((r[1] for r in _results))
        else:
            results = np.vstack(_results)

        self.results["flux"][:results.shape[1], layers_to_analyse] = results[..., 0].T
        if calc_moments:
            i = 1
            for name in ["x", "y", "x2", "y2", "xy"]:
                self.results[name][:results.shape[1], layers_to_analyse] = results[..., i].T
                i += 1

        return self.results
