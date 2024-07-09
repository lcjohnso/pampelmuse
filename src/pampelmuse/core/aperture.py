"""
aperture.py
===========
Copyright 2013-2024 Sebastian Kamann

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
This module implements the class 'AperPhoto' that is designed to perform
aperture photometry on a 2dim. image or aperture spectrophotometry a 3dim.
data cube.

An instance of the class can be initialized from IFS data and a list of
centroids. To this aim, a valid instance of pampelmuse.core.cube.Cube
and a pandas DataFrame containing columnc 'x' and 'y' must be provided.
The aperture photometry is performed for each layer of the IFS data
independently. A local background around every source is estimated and
subtracted before its flux and moments are determined. If requested, the
routine also estimates the global background in each layer of the data.

Provides
--------
function pampelmuse.core.aperture.run_ap

Latest Git revision
-------------------
2024/07/09
"""
import logging
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.stats import sigmaclip
from ..instruments import Instrument, MusePixtable


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20240709


logger = logging.getLogger(__name__)


def init_worker(centroids, background, r_aperture, calc_moments, threshold, *args):
    """
    This function is used to set the parameters for the parallel execution of
    the aperture photometry that do not change from one layer to the next.
    Therefore, they need to be provided only once at the beginning of the
    analysis.

    Parameters
    ----------
    centroids : pandas.DataFrame
        The initial guesses for the centroids of the sources. At least the
        columns 'x' and 'y' must be present.
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
    global g_properties

    g_centroids = centroids
    g_background = background
    g_r_aperture = r_aperture
    g_threshold = threshold
    if calc_moments:
        g_properties = ['flux', 'x', 'y', 'x2', 'y2', 'xy']
    else:
        g_properties = ['flux']


def run_ap(data):
    """
    This function performs aperture photometry on one image with a random
    arrangement of pixels.

    The image must be a provided as a pandas DataFrame, containing athe flux,
    x-, and y-coordinate of every pixel.

    For the purpose of parallelizing the code, this function is defined
    outside of the AperPhoto class. The parallelization via multi-threading
    seems not to work if a method of the same class is called.
    
    Parameters
    ----------
    data : pandas DataFrame
        The data on which to perform aperture photometry. It should be
        provided as a pandas DataFrame containing at least three columns,
        containing x-, y-coordinate, and flux of every pixel.

    Returns
    -------
    results : pandas.DataFrame
        The calculated values are returned as a pandas DataFrame containing
        the measured properties for the individual centroids as columns of a
        pandas MultiIndex (centroids on first level, properties on second).
        If calc_moments is set to False, the only property will be the
        measured flux, otherwise, the flux will be returned together with the
        first and second order spatial moments.
    background : float
        The flux of the global background.
    """
    global g_centroids
    global g_background
    global g_r_aperture
    global g_threshold
    global g_properties

    # prepare output array
    results = pd.DataFrame(np.nan, index=[0], columns=pd.MultiIndex.from_product([g_centroids.index, g_properties],
                                                                                 names=['id', 'property']))

    # exclude masked pixels, make sure at least some pixels have signal
    valid = np.isfinite(data['value'])
    if not valid.any():
        logger.warning("Layer contains only masked pixels.")
        return results, np.nan

    # calculate distance of centroid of each source to each spaxel. Note that the shape of the returned array is
    # (N_spaxel, N_sources)
    _offsets = spatial.distance.cdist(data.loc[valid, ['x', 'y']].values, g_centroids[['x', 'y']].values)
    # create data frame containing spaxels as rows and centroids as columns
    offsets = pd.DataFrame(_offsets, index=data[valid].index, columns=g_centroids.index)

    # select global background pixels, defined as having distances >r_aper to all sources, and calculate global
    # background
    logger.debug("Determining global background ...")
    min_offsets = offsets.min(axis=1)
    back_pixels = min_offsets[min_offsets > g_r_aperture].index
    if len(back_pixels) < 20:
        logger.warning("Using < 20 pixels to estimate global background.")

    # perform kappa-sigma clipping to determine background level and noise
    back_data, _, _ = sigmaclip(data.loc[back_pixels, 'value'])
    back_mean = back_data.mean()
    back_std = back_data.std()
    global_background = back_mean
    logger.debug("... global background is {0:.2g}/pixel".format(global_background))

    # loop over all centroids
    for i, (index, row) in enumerate(g_centroids.iterrows(), start=1):

        logger.debug("Processing source #{0} [{1}/{2}] ...".format(index, i, g_centroids.shape[0]))

        # estimate local background in annulus between 2*r_aper and 4*r_aper (if requested, otherwise global
        # background is used)
        if g_background == "local":
            # determine local background pixels, defined as having distances >2*r_aper and <4*r_aper to one source
            back_pixels = (offsets[index] > (2 * g_r_aperture)) & (offsets[index] <= (4 * g_r_aperture))
            back_data, _, _ = sigmaclip(data.loc[offsets.index[back_pixels], 'value'])
            if np.isfinite(back_data).sum() > 1:
                back_mean = back_data.mean()
                back_std = back_data.std()
            logger.debug("... local background is {0:.2g}+/-{1:0.2g}".format(back_mean, back_std))

        # determine flux pixels, defined a having distances <=r_aper to one source.
        flux_pixels = (offsets[index] <= g_r_aperture)

        # Get array for calculations & subtract background estimate
        flux_data = data.loc[offsets.index[flux_pixels], 'value'] - back_mean
        results.loc[0, (index, 'flux')] = flux_data.sum()

        logger.debug("... flux inside aperture is {0:.3g}".format(results.loc[0, (index, 'flux')]))

        # Moment calculation, only if requested:
        if len(g_properties) > 1:

            # note that only pixels above the background noise should be used
            moment_pixels = flux_pixels & (data.loc[valid, 'value'] > (back_mean + g_threshold * back_std))
            moment_data = data.loc[offsets.index[moment_pixels], 'value']
            logger.debug("... {0} pixel(s) available for moment calculation".format(moment_pixels.sum()))
            if moment_pixels.sum() > 3:
                x_grid = data.loc[moment_pixels.index, 'x']
                y_grid = data.loc[moment_pixels.index, 'y']

                total = float(moment_data.sum())
                x = (x_grid * moment_data).sum() / total
                y = (y_grid * moment_data).sum() / total
                x2 = (x_grid * x_grid * moment_data).sum() / total - x ** 2
                y2 = (y_grid * y_grid * moment_data).sum() / total - y ** 2
                xy = (x_grid * y_grid * moment_data).sum() / total - x * y

                logger.debug("... moments: x={0:.1f}, y={1:.1f}, x2={2:.1f}, y2={3:.1f}, xy={4:.1f}".format(
                    x, y, x2, y2, xy))

                # check validity: physical moments measured and centroid inside measurement area
                if x_grid.min() < x < x_grid.max() and y_grid.min() < y < y_grid.max():
                    if x2 > 0 and y2 > 0 and xy ** 2 < (x2 * y2):
                        results.loc[0, (index, 'x')] = x
                        results.loc[0, (index, 'y')] = y
                        results.loc[0, (index, 'x2')] = x2
                        results.loc[0, (index, 'y2')] = y2
                        results.loc[0, (index, 'xy')] = xy
            else:
                logger.debug("...too few pixels - skipping moment calculation.")

    return results, global_background


class AperPhoto:
    """
    Perform aperture (spectro)photometry on 2 or 3dimensional data using a
    provided list of sources.

    The calculation of the first and second order moments for the provided
    sources is optional. Prior to any flux or moment measurement, a background
    (global or local if requested) is determined and subtracted.
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
        centroids : pandas.DataFrame
            Initial guesses for the centroids of the sources, provided as a
            pandas DataFrame containing at least the columns 'x' and 'y'. The
            various sources should constitute the rows of the DataFrame.
        psf_instance : an instances of pampelmuse.psf.profiles.Profile
            To estimate the aperture correction, i.e. the fraction of flux
            outside the aperture, provide a model of the PSF of the sources.
            WARNING: the aperture correction is not implemented yet.
        """
        logger.info("Initializing aperture photometry, validating input...")

        # Checking cube
        if not isinstance(cube, Instrument):
            logger.error(
                'Parameter "cube" must be an instance of a pampelmuse.instrument class, not {0}.'.format(type(cube)))
            return
        self.cube = cube

        # Checking centroids
        if not isinstance(centroids, pd.DataFrame):
            logger.error('Parameter "centroids" must be pandas DataFrame, not {0}.'.format(type(centroids)))
            return
        elif 'x' not in centroids.columns or 'y' not in centroids.columns:
            logger.error('Columns "x" and "y" must be in centroids.')
            return
        self.centroids = centroids

        self.n_sources = self.centroids.shape[0]
        logger.info('Centroids for {0} source(s) provided.'.format(self.n_sources))

        if psf_instance is not None:
            raise(NotImplementedError, 'Aperture corrections are not implemented yet.')
        self.psf_instance = None

        self.layers_to_analyse = None

    def __call__(self, r_aper=3., layer_range=(0, -1), calc_moments=True, thresh=3., background="global", n_cpu=1,
                 n_bin=1):
        """
        Method performs the actual aperture (spectro-) photometry and measures
        the moments if requested.

        If requested, an estimate of the global background is first determined
        using a sigma-clipping approach. Optionally, a local background around
        every source can be determined.

        Note that aperture (spectro-) photometry is always performed around
        the input coordinates, NOT around the first order moments (even if they
        are calculated).

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
        results : pandas.DataFrame
            The data frame will contain one row per layer in the IFS data. The
            sources define the first layer of the columns and the measurements
            the second level of the columns (using a a pandas MultiIndex).
            For every source, the flux/spectrum is provided in 'flux' and the
            moments (if calculated) are provided in fields 'x', 'y', 'x2',
            'y2', and 'xy'.
        background : pandas.Series
            The global  background determined in every layer (if the parameter
            'background' is set to 'global').
        """
        self.r_aper = r_aper

        start, stop = layer_range
        if start < 0:
            start = self.cube.wave.size + start
        if stop < 0:
            stop = self.cube.wave.size + (stop + 1)
        self.n_bin = n_bin
        self.layers_to_analyse = np.arange(start + self.n_bin // 2, stop, self.n_bin, dtype=np.int16)

        self.thresh = thresh
        self.calc_moments = calc_moments
        if n_cpu is None or n_cpu == -1:
            n_cpu = cpu_count()

        if background not in ["local", "global"]:
            logger.error("Unknown value '{0}' for parameter 'background'. Must be 'local' or 'global'.".format(
                background))
            return

        logger.info("Starting calculation with the following setup:")
        logger.info("   aperture radius [pixel]: {0:.1f}".format(r_aper))
        logger.info("   range of considered layers: {0} - {1}, ".format(start, stop))
        logger.info("   binning factor of input data: {0}".format(self.n_bin))
        logger.info("   pixel threshold [sigma]: {0}".format(thresh))
        logger.info("   calculating moments?: {0}".format(calc_moments))
        logger.info("   background estimation: {0}".format(background))
        logger.info("   number of CPUs: {0}".format(n_cpu))

        # prepare list to store intermediate results
        _results = []

        # initialize parallelisation
        init_arguments = (self.centroids, background, r_aper, calc_moments, thresh)
        if n_cpu > 1:
            pool = Pool(n_cpu, initializer=init_worker, initargs=init_arguments)
        else:
            pool = None
            init_worker(*init_arguments)

        # loop over layers, collect input data.
        # every time number of layers is equal to number of CPUs, trigger parallel analysis
        input_data = []  # container for input data of each layer
        layer_indices = []  # container for index of each layer
        for i, layer_index in enumerate(self.layers_to_analyse):
            logger.info("Processing layer {0}/{1}".format(i + 1, self.layers_to_analyse.size))
            layer_indices.append(layer_index)

            # load data - note that layer binning is currently not performed for pixtables.
            if n_bin == 1 or isinstance(self.cube, MusePixtable):
                input_data.append(self.cube.get_layer(i_min=layer_index))
            else:
                i_min = layer_index - n_bin // 2
                i_max = min(layer_index + n_bin // 2, stop)
                input_data.append(self.cube.get_layer(i_min=i_min, i_max=i_max))

            if not i % n_cpu or layer_index == self.layers_to_analyse[-1]:
                if n_cpu > 1:
                    mapper = pool.map_async(run_ap, input_data)
                    _results.extend(mapper.get())
                else:
                    _results.append(run_ap(input_data[0]))
                input_data = []

        results = pd.concat([r[0] for r in _results], axis=0, ignore_index=True)
        results.index = layer_indices
        background = pd.Series([r[1] for r in _results], index=layer_indices)

        return results, background
