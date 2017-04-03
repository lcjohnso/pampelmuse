"""
fit_cube.py
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
class pampelmuse.core.fit_cube.FitCube

Purpose
-------
The module implements the class 'FitCube' which performs the PSF-based
extraction for a data cube.

Latest SVN revision
-------------------
375, 2016/09/23
"""
import logging
import time
import numpy as np
from multiprocessing import cpu_count, Pool
from .fit_layer import FitLayer
from .background import BackgroundGrid


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 375


logger = logging.getLogger(__name__)


def init_worker(psf_type, magnitudes, components, last_component_is_unresolved, maxrad, osfactor, osradius,
                psf_parameters_to_fit, position_parameters_to_fit, sources, psf_sources, maxiter, *args):
    """
    This function defines the global parameters for the parallel part of the
    analysis. The global parameters are those that do not change from one
    layer to the next, therefore they do not have to be initialized anew for
    each layer.

    Parameters
    ----------
    psf_type : string
        The name of the PSF profile in use.
    magnitudes : array_like
        The relative magnitudes of the sources. A magnitude of 0 corresponds
        to an integrated flux of 1.
    components : array_like
        The component in the fit to which each source belongs.
    last_component_is_unresolved : boolean
        Flag indicating if the last component (i.e. with the highest value in
        the 'components' array) is unresolved.
    maxrad : int
        The radius in pixels out to which the PSF profiles are calculated
    osfactor : int
        The oversampling factor used to improve the accuracy for the central
        PSF pixels.
    osradius : float
        The radius out to which oversampling is being used.
    psf_parameters_to_fit : array_like
        The names of the PSF parameters that should be fitted, or None for a
        fixed PSF.
    position_parameters_to_fit : array_like
        The parameters of the coordinate transformation that should be fitted,
        or None for fixed coordinates.
    sources : array_like
        The indices of the sources used to fit source coordinates.
    psf_sources : array_like
        The indices used to fit the PSF parameters.  If None, use same sources
        as for coordinates.
    maxiter : int
        The maximum number of iterations.
    args

    Returns
    -------
    Nothing is returned, only the global variables are set. Their names are
    derived from the names of the parameters by adding a prefix 'g_'
    """
    global g_psf_type
    global g_magnitudes
    global g_components
    global g_last_component_is_unresolved
    global g_maxrad
    global g_osfactor
    global g_osradius
    global g_psf_parameters_to_fit
    global g_position_parameters_to_fit
    global g_sources
    global g_psf_sources
    global g_maxiter

    g_psf_type = psf_type
    g_magnitudes = magnitudes
    g_components = components
    g_last_component_is_unresolved = last_component_is_unresolved
    g_maxrad = maxrad
    g_osfactor = osfactor
    g_osradius = osradius
    g_psf_parameters_to_fit = psf_parameters_to_fit
    g_position_parameters_to_fit = position_parameters_to_fit
    g_sources = sources
    g_psf_sources = psf_sources
    g_maxiter = maxiter


def parallel_fit_layer(fit_parameters):
    """
    This function is designed to carry out the fit to several layers in
    parallel. To this aim it is called by multiprocess.Pool from within the
    FitCube class. It basically initializes and executes a new instance of
    the pampelmuse.core.fit_layer.FitLayer class every time it is called.

    Parameters
    ----------
    fit_parameters : dict
        A dictionary containing the parameters necessary to launch the fit:
        data - the data array that defines the layer.
        variances - the variances for the data array.
        transformation - The transformation from pixel to x- & y-coordinates
        xy - the reference or pixel coordinates of the sources.
        psf_parameters - the initial guesses/fixed values of the PSF
                         parameters.
        position_parameters - the initial guesses/fixed values of the coord.
                              transformation parameters.
        background - the matrix containing the footprints of the individual
                     background components.
        wave - the wavelengths of each pixel in the data array.

    Returns
    -------
    results : list
        The list with the results contains the following entries:
        fluxes - The fitted flux of each component
        uncertainties - The flux uncertainty of each component
        positions - The fitted values of the source positions or the coord.
                    transformation
        psf - The fitted values of the PSF parameters.
        wave - The average wavelength determined for each component.
    """

    # fit parameters is a dictionary that must contain the following keys:
    # data, variances, transformation, xy, psf_parameters, position_parameters
    data = fit_parameters['data']
    variances = fit_parameters['variances']
    transformation = fit_parameters['transformation']
    xy = fit_parameters['xy']
    psf_parameters = fit_parameters['psf_parameters']
    position_parameters = fit_parameters['position_parameters']
    background = fit_parameters['background']
    wave = fit_parameters['wave']

    global g_psf_type, g_magnitudes, g_components, g_last_component_is_unresolved, g_maxrad, g_osfactor, g_osradius
    global g_psf_parameters_to_fit, g_position_parameters_to_fit, g_sources, g_psf_sources, g_maxiter

    layerfit = FitLayer(data=data, variances=variances, transformation=transformation, psf_type=g_psf_type,
                        psf_parameters=psf_parameters, xy=xy, position_parameters=position_parameters,
                        magnitudes=g_magnitudes, components=g_components, background=background,
                        last_component_is_unresolved=g_last_component_is_unresolved, wave=wave,
                        maxrad=g_maxrad, osfactor=g_osfactor, osradius=g_osradius)

    results = layerfit(fluxes=None, psf_parameters_to_fit=g_psf_parameters_to_fit,
                       position_parameters_to_fit=g_position_parameters_to_fit,
                       sources=g_sources, psf_sources=g_psf_sources, maxiter=g_maxiter)

    del layerfit
    return results


class FitCube(object):
    """
    This class implements the core-routine of PampelMuse. It uses a catalog of
    input sources, created e.g. using the dedicated subroutine 'INITFIT', to
    analyse a complete data cube in a layer-by-layer manner. The fitting
    process itself is split into three parts:
    
        (i) Fluxes:
        A linear least-squares fit is performed for all the resolved sources
        simultaneously. This is done using a sparse-matrix representation of
        the individual sources.

        (ii) PSF:
        The PSF is modelled currently as an analytical profile, usually
        assuming a Moffat-shaped PSF. The individual parameters that govern
        the shape of the profile can be optimized during the fit.

        (iii) Source positions:
        Usually, the source positions are derived from a set of reference
        coordinates via a coordinate formation that is defined by several
        parameters. These parameters can be optimized during the fit. The
        routine also supports direct fitting of source coordinates. However,
        this is only recommended for individual stars instead of crowded
        stellar fields.

    Note that the routine first analyzes the central layer of the cube and
    than alternating proceeds to the red and blue. This is done because
    usually the highest signal is expected near the center of the observed
    wavelength range.
    """

    def __init__(self, ifs_data, sources, psf_attributes, maxrad=np.inf, osfactor=5, osradius=2, **kwargs):
        """
        Initialize an instance of the `StarFit'-class.
    
        Parameters
        ----------
        ifs_data : instance of a pampelmuse.instruments class
            The data to be analyzed, provided as an instance of one of the
            dedicated 'Instrument' classes of PampelMuse. Check that routines
            documentation for further information.
        sources : a pampelmuse.core.sources.Sources-instance
            The required information (positions, magnitudes,...) about the
            sources that should be included in the optimization.
        psf_attributes : a pampelmuse.core.psf.psf_attributes-instance
            The required information about the PSF to be used in the opti-
            mization, i.e., type of PSF, initial guesses of shape parameters
            (i.e. variables), ...
        maxrad : float, optional
            The radius out to which each PSF will be defined, measured in
            pixels and from the centre of the profile.
        osfactor : The improve the accuracy of the PSF modelling close to the
            centre of a profile, the central pixels can be supersampled by a
            factor 'osfactor' along x and y, i.e. each pixel is split into
            'osfactor'^2 subpixels. Default is no supersampling.
        osradius : All pixels with distances <'osradius' to the centre of a
            PSF profile are supersampled by a factor 'osfactor'. Default is
            no supersampling.

        Returns
        -------
            The newly created instance
        """
        logger.debug("Initializing FitCube instance (revision: {0})".format(__revision__))

        self.ifs_data = ifs_data

        # Check provided sources class for consistency, check how positions should be fitted:
        self.sources = sources

        # get assignment of sources to components
        self.components = self.sources.specrow[~self.sources.is_background]
        # index of unresolved component must be highest source component index
        if self.sources.includes_unresolved:
            assert(self.sources.specrow_unresolved == self.components.max())
        # indices of all background components must be > highest source component index
        if self.sources.includes_background:
            assert(self.sources.specrow_background.min() > self.components.max())
        # get relative weights of sources in each component
        self.magnitudes = self.sources.weights[~self.sources.is_background]

        # The following two dictionaries will contain the name of ALL parameters needed to define the source
        # positions and the PSF. The keys are flags indicating if the should be optimized during the fit
        self.position_parameters = {}
        self.psf_parameters = {}

        # For obtaining source coordinates, the following order is used:
        # 1) polynomial fit exists for the source -> use it
        # 2) the position is a free fit parameter for the source -> fit it
        # 3) use the coordinate transformation, fit its parameters if requested
        # IMPORTANT: the coordinates for all sources must be obtained from the same step, i.e. it is not possible
        #            to get half of the positions from a coordinate transformation and the other half from polynomial
        #            fits.
        if self.sources.ifs_coordinates_fitted[~self.sources.is_background].any():
            if not self.sources.ifs_coordinates_fitted[~self.sources.is_background].all():
                logger.error('The provided status of the source positions is not supported.')
                logger.error('Only {0} out of {1} source positions have polynomial fits.'.format(
                    self.sources.ifs_coordinates_fitted.sum(), self.sources.n_sources - self.sources.n_background))
            else:
                self.position_parameters['xy'] = False
        elif self.sources.ifs_coordinates_free[~self.sources.is_background].any():
            if not self.sources.ifs_coordinates_free[~self.sources.is_background].all():
                logger.error('The provided status of the source positions is not supported.')
                logger.error('Only {0} out of {1} source positions are free fit parameters.'.format(
                    self.sources.ifs_coordinates_free.sum(), self.sources.n_sources - self.sources.n_background))
            else:
                self.position_parameters['xy'] = True
        else:
            if not self.sources.transformation.valid:
                logger.error('No coordinate transformation found. Cannot determine source positions.')
            for i, parameter in enumerate(self.sources.transformation.names):
                self.position_parameters[parameter] = self.sources.transformation.free[i]

        # work on PSF, check if basic PSF model is valid
        self.psf_attributes = psf_attributes
        if self.psf_attributes.profile not in ["gauss", "moffat", "double_gauss"]:
            logger.error('Invalid or unsupported PSF type: "{0}"'.format(self.psf_attributes.profile))
        else:
            logger.info('Using PSF of type {0}.'.format(self.psf_attributes.profile.upper()))

        # collect PSF parameters and their status in the fit
        self.psf_attributes = psf_attributes
        for i, parameter in enumerate(self.psf_attributes.names):
            self.psf_parameters[parameter] = self.psf_attributes.free[i] & ~self.psf_attributes.fitted[i]

        # Information needed to set up PSF profiles later
        self.maxrad = maxrad
        self.osfactor = osfactor  # supersampling of central pixels
        self.osradius = osradius

        # initialize background
        if self.sources.is_background.sum() == 1:
            self.background = BackgroundGrid.constant()
        elif self.sources.is_background.sum() > 1:
            self.background = BackgroundGrid(
                np.vstack((self.sources.y[self.sources.is_background], self.sources.x[self.sources.is_background])).T)
        else:
            self.background = None

    def __call__(self, layer_range=(0, -1), n_track=10, n_cpu=1, maxiter=20, use_variances=True, n_bin=1):
        """
        Execute the fitting process. For each layer of the provided
        data cube, the following steps are performed:
         1) Load the data for the current layer
         2) Update the initial guesses or values of the parameters of the
            PSF and/or the coordinate transformation.
         3) Perform the optimization, this step will be executed in parallel
            for n_cpu layers.
         4) Store the fit results in their respective class instances.
        
        Parameters
        ----------
        layer_range : tuple
            The indices of the first and last layer of the spectral range to
            be analysed. By default, the whole cube is analysed.
        n_track :  integer, optional
            In case the PSF and/or the source positions are fitted, the 
            routine will update the initial guesses of the parameters that
            should be fitted prior to fitting a layer. To this aim, it uses
            the results from the fits to the 'n_track' neighboring layers. If
            you do not want this, set `n_track' to zero. Default is 10. Note
            that `n_track' will be divided by `n_bin', if layer binning is used.
        n_cpu : integer
            The number of CPUs that should be used.
        maxiter : int, optional
            The maximum number of iterations per image. The default is 20.
        use_variances : boolean, optional
            A flag indicating whether the uncertainties of the IFS data should
            be used in the analysis.
        n_bin : int, optional
            The number of layers of the IFS data combined into a single layer
            for the analysis.
        """
        # check which layers should be analysed
        start, stop = layer_range
        if start < 0:
            start = self.sources.ndisp + start
        if stop < 0:
            stop = self.sources.ndisp + (stop + 1)
        logger.info("Layer range to be analysed: {0}-{1}".format(start, stop))
        logger.info("Binning factor per layer: {0}".format(n_bin))

        # define array containing the layers (or centres of combined layers) that should be analysed. They must be
        # in the correct order, i.e. starting from the central wavelength and then moving to the red and blue ends
        # of the cube in an alternating manner.

        # first define layers that should be used in increasing order
        _layers_to_analyse = np.arange(start + n_bin // 2, stop, n_bin, dtype=np.int16)

        # then rearrange layers so that central layer comes first.
        layers_to_analyse = np.zeros(_layers_to_analyse.shape, dtype=_layers_to_analyse.dtype)
        layers_to_analyse[0::2] = _layers_to_analyse[_layers_to_analyse.size // 2:]
        layers_to_analyse[1::2] = _layers_to_analyse[:_layers_to_analyse.size // 2][::-1]

        # also get list of layers that are not included in this run
        # (if the binning factor is >1, they may still be considered)
        omitted_layers = ~np.in1d(np.arange(self.sources.ndisp, dtype=np.int16), layers_to_analyse)

        # Number of iterations per layer.
        if maxiter > 1:
            logger.info("Maximum iterations per layer: {0}".format(maxiter))
        else:
            logger.info("No iteration on individual layers performed.")

        # begin loading data required for fitting process into memory
        # whole array with spectra needs to be loaded, fluxes of omitted layers are set to NaN
        self.sources.load_spectra()
        self.sources.data[omitted_layers, ...] = np.nan
        if self.sources.data_wave is not None:
            if np.isnan(self.sources.wave).any() or (self.sources.wave <= 0).any():
                self.sources.data_wave[:, 0] = self.ifs_data.wave

        # if parameters (of the PSF or the coordinate transformation) are fitted, it must be ensured that the instances
        # have correct dispersion information
        if self.psf_attributes.ndisp != self.sources.ndisp:
            self.psf_attributes.resize(self.sources.ndisp)
        if self.sources.transformation.valid and self.sources.transformation.ndisp != self.sources.ndisp:
            self.sources.transformation.resize(self.sources.ndisp)

        # Interpolate across wavelength range to ensure the parameters have valid values for all fitted layers.
        # Afterwards set parameters of omitted layers to NaN. Note that this is not done for polynomial fits to the
        # parameters
        self.psf_attributes.interpolate()
        self.psf_attributes.data[omitted_layers, :, 0] = np.nan
        if self.sources.transformation.valid:
            self.sources.transformation.interpolate()
            self.sources.transformation.data[omitted_layers, :, 0] = np.nan

        # Load full coordinate array into memory because doing it per layer is quite slow. Switch on interpolation to
        # make sure valid coordinates are available for all layers.
        self.sources.load_ifs_coordinates(interpolate=True)
        self.sources.ifs_coordinates[omitted_layers, ..., 0] = np.nan

        # Check how many CPUs should be used
        if n_cpu is None:
            n_cpu = cpu_count()
        logger.info("Starting analysis using {0} CPU(s).".format(n_cpu))

        # The number of previously fitted layers used to obtain initial guesses
        # n_track = (n_track//n_bin) + 1 if n_bin > 1 else n_track

        # For all layers to be fitted in parallel, we first need to collect the input data in a common list.
        # The list will contain one dictionary per layer to be fitted in parallel
        fit_input = []

        # List containing the indices of the last fitted layers
        last_layers = []

        # Set global values for fits to individual layers:
        # check which parameters of the PSF should be fitted
        psf_parameters_to_fit = [key for key in self.psf_parameters.keys() if self.psf_parameters[key]]
        if len(psf_parameters_to_fit) == 0:
            psf_parameters_to_fit = None

        # check if/how the source positions should be fitted
        position_parameters_to_fit = [key for key in self.position_parameters.keys() if self.position_parameters[key]]
        if len(position_parameters_to_fit) == 0:
            position_parameters_to_fit = None

        # check which sources should be used in single PSF fits
        # if no PSF sources exist, use method for layers with few sources/spaxels
        if (self.sources.status == 3).sum() == 0:
            sources = None
            psf_sources = None
        # if source positions are fitted directly, all affected sources must be used, plus a subset to constrain the PSF
        elif self.sources.ifs_coordinates_free.any():
            sources = np.flatnonzero(self.sources.ifs_coordinates_free[~self.sources.is_background])
            psf_sources = np.flatnonzero(self.sources.status[~self.sources.is_background] == 3)
        # otherwise, just use the subset to constrain the PSF
        else:
            sources = np.flatnonzero(self.sources.status[~self.sources.is_background] == 3)
            psf_sources = None

        initargs = (self.psf_attributes.profile,
                    self.magnitudes,
                    self.components,
                    self.sources.includes_unresolved,
                    self.maxrad,
                    self.osfactor,
                    self.osradius,
                    psf_parameters_to_fit,
                    position_parameters_to_fit,
                    sources,
                    psf_sources,
                    maxiter
                    )
        if n_cpu > 1:
            pool = Pool(n_cpu, initializer=init_worker, initargs=initargs)
        else:
            pool = None
            init_worker(*initargs)

        # loop over all layers in cube that should be analysed
        for i, jj in enumerate(layers_to_analyse):

            # Add a common prefix for all logging messages related to the analysis of the current layer.
            n_digits = int(np.log10(layers_to_analyse.size)) + 1  # for logging prefix, get number of digits of N_layer
            fmt = logging.Formatter("%(asctime)s[%(levelname)-8s]: [{0:0{1}d}/{2}] %(message)s".format(
                i+1, n_digits, layers_to_analyse.size), datefmt='%m/%d/%Y %H:%M:%S')
            handler = logging.getLogger().handlers[0]
            handler.setFormatter(fmt)

            # call fitting routine
            logger.info("Collecting input data for fit to layer #{0}.".format(jj))

            # among stored layers of last fits, find those nearest to current layer
            if len(last_layers) > n_track:
                nearest_layers = last_layers[np.argsort(abs(last_layers - jj))[:n_track]]
            else:
                nearest_layers = last_layers
            logger.debug("Layers used to update initial guesses: {0}".format(nearest_layers))

            # get 1D arrays containing valid data pixels, also sigma-values if requested
            data, variances, transform, wave = self.get_layer(jj, n_bin=n_bin, use_variances=use_variances)
            fit_input.append({'data': data, 'variances': variances, 'transformation': transform, 'wave': wave})

            # update PSF parameters
            fit_input[-1]['psf_parameters'] = self.read_psf_parameters(jj, last=nearest_layers)

            # update parameters of coordinate transformation (if available and needed)
            if self.sources.transformation.valid:
                if not self.sources.ifs_coordinates_fitted[~self.sources.is_background].all():
                    fit_input[-1]['position_parameters'] = self.read_coordinate_transformation(jj, last=nearest_layers)

            # update direct coordinates if required
            if self.sources.ifs_coordinates_fitted.any() or self.sources.ifs_coordinates_free.any():
                fit_input[-1]['xy'] = self.read_ifs_coordinates(jj, last=nearest_layers)
                fit_input[-1]['position_parameters'] = None
            else:
                fit_input[-1]['xy'] = np.dstack((self.sources.x[~self.sources.is_background],
                                                 self.sources.y[~self.sources.is_background]))[0]

            if self.background is not None:
                self.background.transform = transform
                fit_input[-1]['background'] = self.background.get_sparse_matrix(
                    offset=self.sources.n_components - self.sources.n_background)
            else:
                fit_input[-1]['background'] = None

            # The first two layers are fitted sequentially in order to obtain useful initial guesses for the
            # following ones. Otherwise a parallel fit is triggered every n_cpu's layer
            if i in [0, 1] or np.mod(i - 1, n_cpu) == 0 or jj == layers_to_analyse[-1]:
                if n_cpu > 1:  # and i > 1:
                    # pool = Pool(n_cpu, initializer=init_worker, initargs=initargs)
                    _results = pool.map_async(parallel_fit_layer, fit_input)
                    # pool.close()
                    # pool.join()
                    results = _results.get()  # list(itertools.chain.from_iterable(_results.get()))
                else:
                    # init_worker(*initargs)
                    results = [parallel_fit_layer(fit_input[0])]

                # write results for the individual layers
                for k, result in enumerate(results, start=1):
                    # recover layer for which the results were obtained
                    current = layers_to_analyse[i - len(results) + k]
                    # for each layer, four arguments are returned: fluxes, uncertainties, positions, and PSF parameters
                    self.write_fluxes(current, result[0], result[1], result[-1])
                    if position_parameters_to_fit is not None and 'xy' in position_parameters_to_fit:
                        self.write_ifs_coordinates(current, result[2])
                    else:
                        self.write_coordinate_transformation(current, position_parameters_to_fit, result[2])
                    self.write_psf_parameters(current, psf_parameters_to_fit, result[3])

                # reset list containing input data for parallel fit
                fit_input = []

                #  after each fit, update layers used to get parameters and/or initial guesses for PSF/positions
                # - first check if results from previous fits can be used
                if n_track == 0:
                    # no updating requested
                    last_layers = []
                else:
                    assert (jj in layers_to_analyse)
                    i = np.where(layers_to_analyse == jj)[0][0]
                    if i < 2:
                        # nothing to update yet
                        last_layers = []
                    elif i <= 2 * n_track:
                        # use all images that have been fitted so far
                        last_layers = layers_to_analyse[:i]
                    else:
                        last_layers = layers_to_analyse[i - 2:max(0, i - 2 - 2 * n_track):-1]

    def get_layer(self, current, n_bin, use_variances=True, mask_nan=True):
        """
        Returns the current layer of the IFS data. If requested, the layer
        is binned with adjacent ones to increase the S/N. Note that binning
        currently does not work when using MUSE pixtables because the spatial
        grid changes from one layer to the next.

        Parameters
        ----------
        current : integer
            The index of the layer that should be obtained.
        n_bin : integer
            The binning factor, i.e. the number of adjacent layers included.
        use_variances : boolean
            Flag indicating if the variances of the data should be returned.

        Returns
        -------
        data : nd_array
            The data layer
        variances : nd_array
            The variance layer, will  be None if the 'use_variances' parameter
             is set to False.
        transformation : ndarray
            The transformation from pixels to spatial (x,y) coordinates that
             is valid for the returned data layer.
        wave : ndarray or None
            If a pixtable is used, this array will contain a wavelength for
            each pixel of 'data'.
        mask_nan : boolean, optional
            Flag indicating if NaN pixels should be removed before the data are
            returned.
        """
        # Get data layer, also  variance layer if requested; bin data if requested
        if self.ifs_data.name == 'MUSE_Pixtable':
            data, variances = self.ifs_data.get_layer(current, return_var=use_variances)
            wave = self.ifs_data.rawwave_for_current_layer
        elif n_bin == 1:
            data, variances = self.ifs_data.get_layer(current, return_var=use_variances)
            wave = None
        else:
            jjmin = current - n_bin // 2
            jjmax = min(current + n_bin // 2, self.ifs_data.nlayer)
            data, variances = self.ifs_data.integrate(jjmin, jjmax, apply_variances=use_variances)
            wave = None

        # NaN pixels will be masked if requested
        if mask_nan:
            mask = np.isnan(data)
            logger.debug("Layer contains {0} masked pixels".format(mask.sum()))

            # apply mask - note that any 2D data will be flattened by this
            data = data[~mask]
            if variances is not None:
                variances = variances[~mask]

            # Set spaxel to (x,y) transformation of the PSF-instances, using the transformation of the Instrument
            # instance and the mask
            transform = self.ifs_data.transform[~mask]
        else:
            transform = self.ifs_data.transform

        return data, variances, transform, wave

    def read_psf_parameters(self, current, last=None):
        """
        Method to determine updated PSF parameters for a given layer using the
        results from adjacent layers. If possible, the method obtains values
        by averaging over the results from the nearest layers, otherwise it
        uses the values from the psf_attributes instance for the current layer.
        If a polynomial fit exists for a parameter, it is always used.

        Parameters
        ----------
        current : integer
            The index of the layer for which the parameters should be obtained
        last : list of integers, optional
            A list containing the indices of the layers that are averaged to
            get the new value.

        Returns
        -------
        parameters : dictionary
            The updated value of each PSF parameter for the current layer.
        """
        logstring = "Initial PSF model:"

        if last is None:
            last = []

        # loop over PSF parameters
        parameters = {}
        for i, prm in enumerate(self.psf_attributes.names):
            # check if polynomial fit exists; if so, use it.
            if self.psf_attributes.fitted[i]:
                value = self.psf_attributes.data[current, i, 1]
            else:
                # find successful fits in provided 'last' layers
                success = np.intersect1d(last, np.nonzero(~self.psf_attributes.mask[:, i, 0])[0])
                if len(success) > 0:
                    value = np.mean(self.psf_attributes.data[success, i, 0])
                else:
                    value = self.psf_attributes.data[current, i, 0]
            parameters[prm] = value
            logstring += " {0}={1:.2f},".format(prm, float(value))
        logger.info(logstring[:-1])

        return parameters

    def read_coordinate_transformation(self, current, last=None):
        """
        Method to update the parameters of the coordinate transformation for
        a given layer of data. If possible, the method obtains values
        by averaging over the results from the nearest layers, otherwise it
        uses the values from the transformation instance for the current
        layer. If a polynomial fit exists for a parameter, it is always used.

        Parameters
        ----------
        current : integer
            The index of the layer for which the parameters should be obtained
        last : list of integers, optional
            A list containing the indices of the layers that are averaged to
            get the new value.

        Returns
        -------
        parameters : dictionary
            The updated value of each PSF parameter for the current layer.
        """
        logstring = "Initial coord. transformation:"

        if last is None:
            last = []

        # loop over parameters of coordinate transformation
        parameters = {}
        for i, prm in enumerate(self.sources.transformation.names):
            # check if polynomial fit exists; if so, use it.
            if self.sources.transformation.fitted[i]:
                value = self.sources.transformation.data[current, i, 1]
            else:
                # find successful fits in provided 'last' layers
                success = np.intersect1d(last, np.nonzero(~self.sources.transformation.mask[:, i, 0])[0])
                if len(success) > 0:
                    value = np.mean(self.sources.transformation.data[success, i, 0])
                else:
                    value = self.sources.transformation.data[current, i, 0]
            parameters[prm] = value
            logstring += " {0}={1:.2f},".format(prm, float(value))
        logger.info(logstring[:-1])

        return parameters

    def read_ifs_coordinates(self, current, last=None):
        """
        Method to update the IFS coordinates for all sources for a given
        layer. If possible, obtain updated values by averaging over the
        results from nearest the layers, otherwise just use the values in the
        sources instance. If a polynomial fit exists for a parameter, it is
        always used.

        Parameters
        ----------
        current : integer
            The index of the layer for which the parameters should be obtained
        last : list of integers, optional
            A list containing the indices of the layers that are averaged to
            get the new value.

        Returns
        -------
        centroids : ndarray
            The updated value of each PSF parameter for the current layer.
        """
        if last is None:
            last = []

        logger.info("Updating coordinates of PSF instances...")
        t0 = time.time()

        # loop over sources
        centroids = []
        for k, i in enumerate(np.flatnonzero(self.sources.status < 4)):
            # check if polynomial fit exists; if so, use it.
            if self.sources.ifs_coordinates_fitted[i]:
                centroids.append(self.sources.ifs_coordinates[current, k, :, 1])
            elif self.sources.ifs_coordinates_free[i]:
                # find successful fits in provided 'last' layers
                success = np.intersect1d(last, np.nonzero(~np.isnan(self.sources.ifs_coordinates[:, k, 0, 0]))[0])
                if len(success) > 0:
                    centroids.append([np.mean(self.sources.ifs_coordinates[success, k, 0, 0]),
                                      np.mean(self.sources.ifs_coordinates[success, k, 1, 0])])
                else:
                    centroids.append(self.sources.ifs_coordinates[current, k, :, 0])

        logger.debug("Time required to update source coordinates: {0:6.4f}s".format(time.time() - t0))

        return np.asarray(centroids, dtype=np.float32)

    def write_fluxes(self, current, fluxes, uncertainties, wave=None):
        """
        This method writes the fitted fluxes back to the instance of
        pampelmuse.core.Sources that was provided when initialising
        the instance of FitCube.

        Parameters
        ----------
        current : integer
            The index of the layer for which the fluxes were fitted.
        fluxes : array_like
            The fitted fluxes
        uncertainties : array_like
            The associated uncertainties
        wave : array_like, optional
            The average wavelength values for the components. Currently
            only used with MUSE pixtables.
        """
        assert(len(fluxes) == len(uncertainties)), '"fluxes" and "uncertainties" must have same length'
        assert(len(fluxes) == self.sources.n_components), 'Length of "fluxes" must match number of components in fit.'
        self.sources.data[current, :, 0] = fluxes
        self.sources.data[current, :, 1] = uncertainties
        if wave is not None:
            if self.sources.data_wave is None:
                logging.error('Cannot store wavelength values in Sources instance. Array not initialized.')
            else:
                assert(len(wave) == self.sources.n_components)
                self.sources.data_wave[current, :] = wave

    def write_psf_parameters(self, current, parameters, values):
        """
        This method writes the fitted PSF parameters back to the instance of
        pampelmuse.psf.Variables that was provided when initialising the
        instance of FitCube.

        Parameters
        ----------
        current : int
            The index of the layer for which the fluxes were fitted.
        parameters : array_like
            The names of the PSF parameters that were fitted.
        values : array_like
            The fitted values of the PSF parameters.
        """

        if parameters is None:  # nothing to do
            return
        else:
            assert(len(parameters) == len(values)), '"parameters" and "values" must have same length'
            for i, parameter in enumerate(parameters):
                j = self.psf_attributes.names.index(parameter)
                self.psf_attributes.data[current, j, 0] = values[i]

    def write_coordinate_transformation(self, current, parameters, values):
        """
        This method writes the fitted parameters of a coord. transformation
        back to the instance of pampelmuse.core.Sources that was provided
        when initialising the instance of FitCube.

        Parameters
        ----------
        current : int
            The index of the layer for which the fluxes were fitted.
        parameters : array_like
            The names of the transformation parameters that were fitted.
        values : array_like
            The fitted values of the transformation parameters.
        """
        if parameters is None:  # nothing to do
            return
        else:
            assert(len(parameters) == len(values)), '"parameters" and "values" must have same length'
            for i, parameter in enumerate(parameters):
                j = self.sources.transformation.names.index(parameter)
                self.sources.transformation.data[current, j, 0] = values[i]

    def write_ifs_coordinates(self, current, values):
        """
        This method writes the fitted centroid coordinates back to
        the instance of pampelmuse.core.Sources that was provided
        when initialising the instance of FitCube.

        Parameters
        ----------
        current : int
            The index of the layer for which the fluxes were fitted.
        values : array_like
            The fitted values of the source centroids.
        """

        if values is None:  # nothing to do
            return
        else:
            for i, xy in enumerate(values):
                # The assignment assumes that all sources with lower indices have their IFS coordinates loaded
                assert (np.all(self.sources.ifs_coordinates_loaded[:i]))

                self.sources.ifs_coordinates[current, i, 0, 0] = xy[0]
                self.sources.ifs_coordinates[current, i, 1, 0] = xy[1]
