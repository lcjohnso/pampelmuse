"""
fit_cube.py
===========
Copyright 2013-2019 Sebastian Kamann

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

Latest GIT revision
-------------------
2019/04/12
"""
import logging
import time
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from .fit_layer import FitLayer
from .background import BackgroundGrid
from ..instruments.muse import MusePixtable


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20190412


logger = logging.getLogger(__name__)


def init_worker(catalog, psf_type, psf_radius, psf_parameters_to_fit, position_parameters_to_fit,
                last_component_is_unresolved, osfactor, osradius, sources, psf_sources, max_iterations, analyse_psf,
                *args):
    """
    This function defines the global parameters for the parallel part of the
    analysis.
    
    The global parameters are those that do not change from one layer to the
    next, therefore they do not have to be initialized anew for each layer.

    Parameters
    ----------
    catalog : pandas.DataFrame
        The pixel coordinates, statuses, components, and relative magnitudes
        of all sources to be used in fit.
    psf_type : string
        The name of the PSF profile in use.
    psf_radius : int
        The radius in pixels out to which the PSF profiles are calculated
    psf_parameters_to_fit : array_like
        The names of the PSF parameters that should be fitted, or None for a
        fixed PSF.
    position_parameters_to_fit : array_like
        The parameters of the coordinate transformation that should be fitted,
        or None for fixed coordinates.
    last_component_is_unresolved : boolean
        Flag indicating if the last component (i.e. with the highest value in
        the 'components' array) is unresolved.
    osfactor : int
        The oversampling factor used to improve the accuracy for the central
        PSF pixels.
    osradius : float
        The radius out to which oversampling is being used.
    sources : array_like
        The indices of the sources used to fit source coordinates.
    psf_sources : array_like
        The indices used to fit the PSF parameters. If None, use same sources
        as for coordinates.
    max_iterations : int
        The maximum number of iterations.
    analyse_psf : bool
       Flag indicating if the radial profiles of the stars used to fit the PSF
       plus the parameters of the individual PSF fits should be returned.
    args

    Returns
    -------
    Nothing is returned, only the global variables are set. Their names are
    derived from the names of the parameters by adding a prefix 'g_'
    """
    global g_catalog
    global g_psf_type
    global g_psf_radius
    global g_psf_parameters_to_fit
    global g_position_parameters_to_fit
    global g_last_component_is_unresolved
    global g_osfactor
    global g_osradius
    global g_sources
    global g_psf_sources
    global g_max_iterations
    global g_analyse_psf

    g_catalog = catalog
    g_psf_type = psf_type
    g_psf_radius = psf_radius
    g_psf_parameters_to_fit = psf_parameters_to_fit
    g_position_parameters_to_fit = position_parameters_to_fit
    g_last_component_is_unresolved = last_component_is_unresolved
    g_osfactor = osfactor
    g_osradius = osradius
    g_sources = sources
    g_psf_sources = psf_sources
    g_max_iterations = max_iterations
    g_analyse_psf = analyse_psf


def parallel_fit_layer(fit_parameters):
    """
    This function is designed to carry out the fit to several layers in
    parallel.
    
    To this aim it is called by multiprocess.Pool from within the FitCube
    class. It basically initializes and executes a new instance of the
    pampelmuse.core.fit_layer.FitLayer class every time it is called.

    Parameters
    ----------
    `fit_parameters` : dictionary with the following keys defined.
        data - pandas.DataFrame
            The value, variance, x-, and y-positions, and wavelength
            (optional) of each pixel of the layer.
        psf_parameters - dictionary
            The initial guesses/fixed values of the PSF parameters.
        coordinates - pandas.DataFrame
            The initial guesses/fixed values of the pixel coordinates if no
             coordinate transformation is being used.
        position_parameters - dictionary
            The initial guesses/fixed values of the coord. transformation
            parameters (if used).
        background - sparse matrix
            The matrix containing the footprints of the individual background
            components.
        background_index - pandas.Index
            The index/indices used for the background components.

    Returns
    -------
    `FitResult` with the following fields defined.
    fluxes : pandas.Series
        The fitted fluxes of the components that actually contributed to
        the observed flux. Sources entirely outside the observed field of
        view are excluded from this series.
    uncertainties : pandas.Series
        The associated uncertainties of the returned fluxes.
    positions : pandas.Series or pandas.DataFrame
        The best-fit parameters of the coordinate transformation or the
        best-fit centroids of all components (in this case a DataFrame is
        returned containing columns 'x' and 'y' and one row per source),
        or None if fixed positions were used.
    psf : pandas.Series
        The best-fit parameters of the PSF, or None if a fixed PSF was
        used.
    wave : pandas.Series
        If dedicated wavelength information per pixel was provided with
        the data, this Series will contain the weighted wavelength value
        for each returned flux. Otherwise None is returned.
    profiles : pandas.DataFrame
        The radial profiles of the PSF stars if requested. Otherwise None.
    parameters : pandas.DataFrame
        The parameters of the individual PSF fits it requested. Otherwise
        None.
    lut: instance of InterpolatedUnivariateSpline
        The look-up table created from the average residuals of the PSF
        fits if requested, otherwise None
    """
    # fit parameters is a dictionary that must contain the following keys:
    # data, variances, transformation, xy, psf_parameters, position_parameters
    data = fit_parameters['data']
    psf_parameters = fit_parameters['psf_parameters']
    psf_lut = fit_parameters['psf_lut']
    coordinates = fit_parameters['coordinates']
    position_parameters = fit_parameters['position_parameters']
    background = fit_parameters['background']
    background_index = fit_parameters['background_index']

    global g_catalog, g_psf_type, g_psf_radius, g_psf_parameters_to_fit, g_position_parameters_to_fit
    global g_last_component_is_unresolved, g_osfactor, g_osradius, g_sources, g_psf_sources, g_max_iterations
    global g_analyse_psf

    if coordinates is not None:
        catalog = pd.merge(coordinates, g_catalog[['component', 'magnitude']], left_index=True, right_index=True)
    else:
        catalog = g_catalog

    layerfit = FitLayer(data=data, catalog=catalog, psf_type=g_psf_type, psf_parameters=psf_parameters,
                        psf_radius=g_psf_radius, psf_lut=psf_lut, position_parameters=position_parameters,
                        last_component_is_unresolved=g_last_component_is_unresolved, background=background,
                        osfactor=g_osfactor, osradius=g_osradius, background_index=background_index)

    results = layerfit(fluxes=None, psf_parameters_to_fit=g_psf_parameters_to_fit,
                       position_parameters_to_fit=g_position_parameters_to_fit,
                       sources=g_sources, psf_sources=g_psf_sources, max_iterations=g_max_iterations,
                       analyse_psf=g_analyse_psf)

    del layerfit
    return results


class FitCube(object):
    """
    This class implements the PSF fit on a complete cube.

    The class uses a catalog of input sources, created e.g. using the
    dedicated subroutine 'INITFIT', to analyse a complete data cube in a
    layer-by-layer manner. The fitting process itself is split into three
    parts:
    
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

    def __init__(self, ifs_data, sources, psf_attributes, psf_radius=np.inf, osfactor=5, osradius=2):
        """
        Initialize an instance of the `StarFit'-class.
    
        Parameters
        ----------
        ifs_data : instance of a pampelmuse.instruments class
            The data to be analyzed, provided as an instance of one of the
            dedicated 'Instrument' classes of PampelMuse. Check that routines
            documentation for further information.
        sources : a pampelmuse.core.sources.Sources-instance
            The instance must contain the required information (i.e. status,
            position, ...) of the sources that are included in the
            optimization.
        psf_attributes : a pampelmuse.core.psf.psf_attributes-instance
            The instance must contain the required information about the PSF
            to be used in the optimization, i.e., type of PSF, the initial
            guesses of its parameters, and whether they should be fitted.
        psf_radius : float, optional
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

        # initial assignments
        self.ifs_data = ifs_data
        self.sources = sources
        self.psf_attributes = psf_attributes
        self.psf_radius = psf_radius
        self.osfactor = osfactor
        self.osradius = osradius

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
                return
            else:
                self.position_parameters['xy'] = False
        elif self.sources.ifs_coordinates_free[~self.sources.is_background].any():
            if not self.sources.ifs_coordinates_free[~self.sources.is_background].all():
                logger.error('The provided status of the source positions is not supported.')
                logger.error('Only {0} out of {1} source positions are free fit parameters.'.format(
                    self.sources.ifs_coordinates_free.sum(), self.sources.n_sources - self.sources.n_background))
                return
            else:
                self.position_parameters['xy'] = True
        else:
            if not self.sources.transformation.valid:
                logger.error('No coordinate transformation found. Cannot determine source positions.')
                return
            for i, parameter in enumerate(self.sources.transformation.names):
                self.position_parameters[parameter] = self.sources.transformation.free[i]

        # collect PSF parameters and their status in the fit
        for i, parameter in enumerate(self.psf_attributes.names):
            self.psf_parameters[parameter] = self.psf_attributes.free[i] & ~self.psf_attributes.fitted[i]

        # initialize background
        if self.sources.is_background.sum() == 1:
            self.background = BackgroundGrid.constant()
        elif self.sources.is_background.sum() > 1:
            self.background = BackgroundGrid(
                np.vstack((self.sources.y[self.sources.status == 4], self.sources.x[self.sources.status == 4])).T)
        else:
            self.background = None

        # check if background-per-slice option used for MUSE pixel tables
        if (self.sources.status == 5).any():
            logger.warning('Assigning background components to slices still in testing phase.')
            n_ifu = self.sources.x[self.sources.status == 5].max()
            n_slice = self.sources.y[self.sources.status == 5].max()
            self.background.add_slices(n_ifu=n_ifu, n_slice=n_slice)

        # check if dispersion information are compatible everywhere
        if self.sources.n_dispersion != self.ifs_data.nlayer:
            logger.error('IFS data and Sources instance have incompatible dispersion information')

        # ensure that the instances of the PSF and the coordinate transformation (if any) use the same indices as the
        # Sources instance
        self.psf_attributes.resize(self.sources.wave)
        if self.sources.transformation.valid:
            self.sources.transformation.resize(self.sources.wave)

        # prepare containers for storing detailed PSF information if requested
        self.profiles = {}
        self.profile_radii = []
        self.parameters = {}
        self.residuals = {}

    def __call__(self, layer_range=(0, -1), n_track=10, n_cpu=1, max_iterations=20, use_variances=True, n_bin=1,
                 analyse_psf=False):
        """
        Run the fitting process for the requested layers.

        For each layer of the provided data cube, the following steps are
        performed:
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
        max_iterations : int, optional
            The maximum number of iterations per image. The default is 20.
        use_variances : boolean, optional
            A flag indicating whether the uncertainties of the IFS data should
            be used in the analysis.
        n_bin : int, optional
            The number of layers of the IFS data combined into a single layer
            for the analysis.
        analyse_psf : bool, optional
            Flag indicating if the radial profiles of the stars used to fit
            the PSF should be returned.
        
        Returns
        -------
        psf_data : astropy.io.fits.HDUList
            If requested, the HDU list will contain the radial profile and the
            fitted parameters for every PSF star as a function of wavelength.
            Otherwise None is returned
        """
        # check which layers should be analysed
        start, stop = layer_range
        if start < 0:
            start = self.sources.n_dispersion + start
        if stop < 0:
            stop = self.sources.n_dispersion + (stop + 1)
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

        # finally, get indices corresponding to requested layers
        indices_to_analyse = self.sources.wave[layers_to_analyse]

        # also get list of layers that are not included in this run
        # (if the binning factor is >1, they may still be considered)
        omitted_indices = self.sources.wave[~self.sources.wave.isin(indices_to_analyse)]

        # Number of iterations per layer.
        if max_iterations > 1:
            logger.info("Maximum iterations per layer: {0}".format(max_iterations))
        else:
            logger.info("No iteration on individual layers performed.")

        # begin loading data required for fitting process into memory
        # collect sources data in catalog
        # note that the source positions may be overwritten later if no coordinate transformation is being used
        stellar = self.sources.status < 4
        series = []
        series.extend([self.sources.x[stellar], self.sources.y[stellar]])
        series.append(self.sources.specrow[stellar])
        series.append(self.sources.weights[stellar])
        catalog = pd.concat(series, axis=1, keys=['x', 'y', 'component', 'magnitude'])

        # whole array with spectra needs to be loaded
        self.sources.load_spectra(init_wave=isinstance(self.ifs_data, MusePixtable))
        # fluxes of omitted layers are set to NaN
        self.sources.data.loc[omitted_indices] = np.nan

        # Source coordinates, parameters of the coordinate transformation, or PSF parameters are first interpolated
        # across the wavelength range to ensure valid values for all fitted layers. Afterwards - if any of the
        # positions/parameters are fitted - all parameters of omitted layers to NaN. his is not done for any polynomial
        # fits to the positions/parameters that may exist
        self.psf_attributes.interpolate()
        if len(self.psf_parameters) > 0:
            self.psf_attributes.data.loc[omitted_indices, (slice(None), 'value')] = np.nan

        if self.sources.transformation.valid:
            self.sources.transformation.interpolate()
            if len(self.position_parameters) > 0:
                self.sources.transformation.data.loc[omitted_indices, (slice(None), 'value')] = np.nan

        # Load full coordinate array into memory because doing it per layer is quite slow.
        self.sources.load_ifs_coordinates()
        # replace NaNs to make sure valid coordinates are available for all fitted layers.
        self.sources.ifs_coordinates.fillna(method='pad', inplace=True)
        # then set values for all omitted layers to NaN - only if any parameters are fitted
        if len(self.position_parameters) > 0:
            self.sources.ifs_coordinates.loc[omitted_indices] = np.nan

        # Check how many CPUs should be used
        if n_cpu is None or n_cpu == -1:
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
        if 3 not in self.sources.status.values:
            sources = None
            psf_sources = None
        # if source positions are fitted directly, all affected sources must be used, plus a subset to constrain the PSF
        elif self.sources.ifs_coordinates_free.any():
            sources = self.sources.ifs_coordinates_free[~self.sources.is_background].index
            psf_sources = self.sources.ids[self.sources.status == 3]
        # otherwise, just use the subset to constrain the PSF
        else:
            sources = self.sources.ids[self.sources.status == 3]
            psf_sources = None

        # initialize fitting process by providing arguments that do not change between layers and creating process pool
        initial_arguments = (catalog,
                             self.psf_attributes.profile,
                             self.psf_radius,
                             psf_parameters_to_fit,
                             position_parameters_to_fit,
                             self.sources.includes_unresolved,
                             self.osfactor,
                             self.osradius,
                             sources,
                             psf_sources,
                             max_iterations,
                             analyse_psf
                             )

        # initialize processes pool
        if n_cpu > 1:
            pool = Pool(n_cpu, initializer=init_worker, initargs=initial_arguments)
        else:
            pool = None
            init_worker(*initial_arguments)

        # loop over all layers in cube that should be analysed
        for i, jj in enumerate(indices_to_analyse):

            # Add a common prefix for all logging messages related to the analysis of the current layer.
            n_digits = int(np.log10(indices_to_analyse.size)) + 1  # for logging prefix, get number of digits of N_layer
            fmt = logging.Formatter("%(asctime)s[%(levelname)-8s]: [{0:0{1}d}/{2}] %(message)s".format(
                i+1, n_digits, indices_to_analyse.size), datefmt='%m/%d/%Y %H:%M:%S')
            handler = logging.getLogger().handlers[0]
            handler.setFormatter(fmt)

            # collecting data required for fit to layer
            # when logging, note that internal wavelength unit is meter
            logger.info("Collecting input data for fit to layer with wavelength {0:.1f} Angstrom.".format(1e10*jj))

            # among stored layers of last fits, find those nearest to current layer
            if len(last_layers) > n_track:
                if np.mod(i, 2):  # if i is odd, layer has low wavelength
                    nearest_layers = last_layers[:n_track]
                else:  # otherwise, layer has high wavelength
                    nearest_layers = last_layers[-n_track:]
            else:
                nearest_layers = last_layers
            logger.debug("Layers used to update initial guesses: {0}".format(nearest_layers))

            # get 1D arrays containing valid data pixels, also sigma-values if requested
            data = self.get_layer(layers_to_analyse[i], n_bin=n_bin, use_variances=use_variances,
                                  use_slices=(self.sources.status == 5).any())
            fit_input.append({'data': data})

            # update PSF parameters & look-up table if available.
            psf_parameters, lut = self.read_psf_parameters(jj, last=nearest_layers)
            fit_input[-1]['psf_parameters'] = psf_parameters
            fit_input[-1]['psf_lut'] = lut

            # update direct coordinates if required
            if self.sources.ifs_coordinates_fitted.any() or self.sources.ifs_coordinates_free.any():
                fit_input[-1]['coordinates'] = self.read_ifs_coordinates(jj, last=nearest_layers)
                fit_input[-1]['position_parameters'] = None
            # otherwise try to update parameters of coordinate transformation
            elif self.sources.transformation.valid:
                fit_input[-1]['coordinates'] = None
                fit_input[-1]['position_parameters'] = self.read_coordinate_transformation(jj, last=nearest_layers)
            else:
                logger.critical('Cannot determine IFS coordinates of sources.')

            # add background if requested
            if self.background is not None:
                self.background.transform = data[['x', 'y']].values
                # if oversampling of background should be used, split up wavelength values into requested number of
                # equi-distant bins
                if self.background.oversampling > 1:
                    bins = np.linspace(data['wave'].min(), data['wave'].max(), self.background.oversampling+1)[1:-1]
                    self.background.spectral_bins = np.digitize(data['wave'], bins)
                # if IFU/slice-dependent background components should be used, add values to background instance
                if 'ifu' in data.columns and 'slice' in data.columns:
                    self.background.ifus = data['ifu'].values
                    self.background.slices = data['slice'].values
                background_matrix, background_components = self.background.get_sparse_matrix(
                    offset=self.sources.n_components - self.sources.n_background, purge_empty=True)
            else:
                background_matrix = background_components = None
            fit_input[-1]['background'] = background_matrix
            fit_input[-1]['background_index'] = background_components

            # The first two layers are fitted sequentially in order to obtain useful initial guesses for the
            # following ones. Otherwise a parallel fit is triggered every n_cpu's layer
            if i in [0, 1] or np.mod(i - 1, n_cpu) == 0 or jj == indices_to_analyse[-1]:
                if n_cpu > 1:
                    _results = pool.map_async(parallel_fit_layer, fit_input)
                    results = _results.get()
                else:
                    results = [parallel_fit_layer(fit_input[0])]

                # write results for the individual layers
                for k, result in enumerate(results, start=1):
                    # recover layer for which the results were obtained
                    current = indices_to_analyse[i - len(results) + k]
                    # for each layer, the fit returns: fluxes, uncertainties, positions, and PSF parameters, wave
                    self.write_fluxes(current, result.fluxes, result.uncertainties, result.wave)
                    if position_parameters_to_fit is not None and 'xy' in position_parameters_to_fit:
                        self.write_ifs_coordinates(current, result.positions)
                    else:
                        self.write_coordinate_transformation(current, result.positions)
                    self.write_psf_parameters(current, result.psf)

                    # save PSF if requested
                    if analyse_psf and sources is not None:
                        self._write_psf_properties(result.profiles, result.parameters, current_wavelength=current)
                        self._write_minicubes(current, result.residuals, fit_input[k - 1]['data'])

                # reset list containing input data for parallel fit
                fit_input = []

                #  after each fit, update layers used to get parameters and/or initial guesses for PSF/positions
                # - first check if results from previous fits can be used
                if n_track == 0:
                    # no updating requested
                    last_layers = []
                else:
                    assert (jj in indices_to_analyse)
                    i = np.where(indices_to_analyse == jj)[0][0]
                    if i < 2:
                        # nothing to update yet
                        last_layers = []
                    elif i <= 2 * n_track:
                        # use all images that have been fitted so far
                        last_layers = indices_to_analyse[:i]
                    else:
                        last_layers = indices_to_analyse[i - 2:max(0, i - 2 - 2 * n_track):-1]

        # if available, return results of detailed PSF analysis
        return self._make_psf_hdus() if analyse_psf else None

    def get_layer(self, current, n_bin, use_variances=True, skip_nan=True, use_slices=False):
        """
        Returns a single layer of the IFS data as a pandas DataFrame.

        If requested, the layer is binned with adjacent ones to increase the
        S/N. Note that binning currently does not work when using MUSE pixel
        tables because the spatial grid changes from one layer to the next.

        Parameters
        ----------
        current : integer
            The index of the layer that should be obtained.
        n_bin : integer
            The binning factor, i.e. the number of adjacent layers included.
        use_variances : boolean, optional
            Flag indicating if the variances of the data should be returned.
        skip_nan : boolean, optional
           Flag indicating if NaN pixels should be included in the data that
           is returned.
        use_slices : boolean, optional
            Flag indicating if the IFU and slice numbers should be returned
            for a MUSE pixel table (when available). If so, they will be
            stored in columns 'ifu' and 'slice'.

        Returns
        -------
        data : pandas.DataFrame
            The data frame will contain at least the columns 'value', 'x', and
            'y', providing the data value and the spatial coordinates of each
            pixel. If requested, a column 'variance' will be included. If a
            pixel table is used, a column 'wave' will contain the wavelength
            of each pixel and optional columns 'ifu' and 'slice' the IFU and
            slice numbers of each pixel.
        """
        # Get data layer, also variance layer if requested; bin data if requested
        if self.ifs_data.name == 'MUSE_Pixtable' or n_bin == 1:
            values, variances = self.ifs_data.get_layer(current, return_var=use_variances)
        else:
            jjmin = current - n_bin // 2
            jjmax = min(current + n_bin // 2, self.ifs_data.nlayer)
            values, variances = self.ifs_data.integrate(jjmin, jjmax, apply_variances=use_variances)

        # get x- & y-coordinates of each pixel
        transform = self.ifs_data.transform

        # NaN pixels will be masked if requested
        if skip_nan:
            mask = np.isnan(values)
            logger.debug("Layer contains {0} masked pixels".format(mask.sum()))
        else:
            mask = np.zeros(values.size, dtype=np.bool)

        # collect arrays in dictionary for initialisation of pandas DataFrame
        data = {'value': values[~mask], 'x': transform[~mask, 1], 'y': transform[~mask, 0]}
        # add variances if requested
        if use_variances:
            data['variance'] = variances[~mask]
        # add wavelength information for pixel table
        if self.ifs_data.name == 'MUSE_Pixtable':
            data['wave'] = self.ifs_data.rawwave_for_current_layer[~mask]
            if use_slices:
                if self.ifs_data.ifus_for_current_layer is not None:
                    data['ifu'] = self.ifs_data.ifus_for_current_layer[~mask]
                else:
                    logger.error('Cannot get IFU numbers for layer because HDU not available.')
                if self.ifs_data.ifus_for_current_layer is not None:
                    data['slice'] = self.ifs_data.slices_for_current_layer[~mask]
                else:
                    logger.error('Cannot get slice numbers for layer because HDU not available.')

        # return, make sure byte-ordering is correct, see
        # https://pandas.pydata.org/pandas-docs/stable/gotchas.html#byte-ordering-issues
        return pd.DataFrame(data, index=np.flatnonzero(~mask), dtype='<f4')

    def read_psf_parameters(self, current, last=None):
        """
        Method to determine updated PSF parameters for a given layer using the
        results from adjacent layers.
        
        If possible, the method obtains values by averaging over the results
        from the nearest layers, otherwise it uses the values from the
        psf_attributes instance for the current layer. If a polynomial fit
        exists for a parameter, it is always used.

        Parameters
        ----------
        current : float
            The index of the layer for which the parameters should be obtained
        last : list of floats, optional
            A list containing the indices of the layers that are averaged to
            get the new value.

        Returns
        -------
        parameters : dictionary
            The updated value of each PSF parameter for the current layer.
        lut : pandas Series or None
            The look-up table for the current layer (if available). The radii
            are provided as index.
        """
        log_string = "Initial PSF model:"

        if last is None:
            last = []

        # loop over PSF parameters
        parameters = {}
        for parameter in self.psf_attributes.names:
            # check if polynomial fit exists; if so, use it.
            if self.psf_attributes.fitted[parameter]:
                value = self.psf_attributes.data.at[current, (parameter, 'fit')]
            else:
                # find successful fits in provided 'last' layers
                success = ~self.psf_attributes.mask.loc[last, (parameter, 'value')]
                if success.sum() > 0:
                    value = np.mean(self.psf_attributes.data.loc[last[success], (parameter, 'value')])
                else:
                    value = self.psf_attributes.data.at[current, (parameter, 'value')]
            parameters[parameter] = value
            log_string += " {0}={1:.2f},".format(parameter, float(value))
        logger.info(log_string[:-1])

        if self.psf_attributes.lut is not None:
            lut = self.psf_attributes.lut.loc[current]
        else:
            lut = None

        return parameters, lut

    def read_coordinate_transformation(self, current, last=None):
        """
        Method to update the parameters of the coordinate transformation for
        a given layer of data.
        
        If possible, the method obtains values by averaging over the results
        from the nearest layers, otherwise it uses the values from the
        transformation instance for the current layer. If a polynomial fit
        exists for a parameter, it is always used.

        Parameters
        ----------
        current : float
            The index of the layer for which the parameters should be obtained
        last : list of floats, optional
            A list containing the indices of the layers that are averaged to
            get the new value.

        Returns
        -------
        parameters : dictionary
            The updated value of each PSF parameter for the current layer.
        """
        log_string = "Initial coordinate transformation:"

        if last is None:
            last = []

        # loop over parameters of coordinate transformation
        parameters = {}
        for parameter in self.sources.transformation.names:
            # check if polynomial fit exists; if so, use it.
            if self.sources.transformation.fitted[parameter]:
                value = self.sources.transformation.data.at[current, (parameter, 'fit')]
            else:
                # find successful fits in provided 'last' layers
                success = ~self.sources.transformation.mask.loc[last, (parameter, 'value')]
                if success.sum() > 0:
                    value = np.mean(self.sources.transformation.data.loc[last[success], (parameter, 'value')])
                else:
                    value = self.sources.transformation.data.at[current, (parameter, 'value')]
            parameters[parameter] = value
            log_string += " {0}={1:.2f},".format(parameter, float(value))
        logger.info(log_string[:-1])

        return parameters

    def read_ifs_coordinates(self, current, last=None):
        """
        Method to update the IFS coordinates for all sources for a given
        layer.
        
        If possible, obtain updated values by averaging over the results from
        nearest the layers, otherwise just use the values in the sources
        instance. If a polynomial fit exists for a centroid, it is always
        used.

        Parameters
        ----------
        current : float
            The index of the layer for which the centroids should be obtained
        last : list of floats, optional
            A list containing the indices of the layers that are averaged to
            get the new value.

        Returns
        -------
        centroids : pandas.DataFrame
            The updated centroid coordinates of the sources in the requested
            layer. The Data Frame will contain the columns 'x' and 'y' and use
            the sources identifiers as indices.
        """
        if last is None:
            last = []

        logger.info("Updating coordinates of PSF instances...")
        t0 = time.time()

        # loop over sources
        centroids = pd.DataFrame(0., index=self.sources.ids[self.sources.status < 4], columns=['x', 'y'])
        for index in centroids.index:

            # check if polynomial fit exists; if so, use it.
            if self.sources.ifs_coordinates_fitted[index]:
                centroids.at[index, 'x'] = self.sources.ifs_coordinates.at[current, (index, 'x', 'fit')]
                centroids.at[index, 'y'] = self.sources.ifs_coordinates.at[current, (index, 'y', 'fit')]
            else:
                # find successful fits in provided 'last' layers
                success = np.isfinite(self.sources.ifs_coordinates.loc[last, (index, 'x', 'value')])
                if success.sum() > 0:
                    centroids.at[index, 'x'] = np.mean(
                        self.sources.ifs_coordinates.loc[last[success], (index, 'x', 'value')])
                    centroids.at[index, 'y'] = np.mean(
                        self.sources.ifs_coordinates.loc[last[success], (index, 'y', 'value')])
                else:
                    centroids.at[index, 'x'] = self.sources.ifs_coordinates.at[current, (index, 'x', 'value')]
                    centroids.at[index, 'y'] = self.sources.ifs_coordinates.at[current, (index, 'y', 'value')]
        logger.debug("Time required to update source coordinates: {0:6.4f}s".format(time.time() - t0))

        return centroids

    def write_fluxes(self, current, fluxes, uncertainties, wave=None):
        """
        This method writes the fitted fluxes back to the instance of
        pampelmuse.core.Sources that was provided when initialising the
        instance of FitCube.

        Parameters
        ----------
        current : float
            The index of the layer for which the fluxes were fitted.
        fluxes : pandas.Series
            The fitted fluxes of the components.
        uncertainties : pandas.Series
            The uncertainties associated with the fluxes. The Series must use
            the same indices as the fluxes.
        wave : pandas.Series, optional
            The average wavelength values for the components. Currently
            only used with MUSE pixtables. If provided, the Series must use
            the same indices as the fluxes.
        """
        # perform consistency checks
        if not np.array_equal(fluxes.index, self.sources.data.columns.levels[0]):
            logger.warning('Input series "fluxes" has incompatible index.')
            fluxes = fluxes.reindex(self.sources.data.columns.levels[0])
        if not np.array_equal(uncertainties.index, self.sources.data.columns.levels[0]):
            logger.warning('Input series "uncertainties" has incompatible index.')
            uncertainties = uncertainties.reindex(self.sources.data.columns.levels[0])
        if wave is not None and not np.array_equal(wave.index, self.sources.data.columns.levels[0]):
            logger.warning('Input series "wave" has incompatible index.')
            wave = wave.reindex(self.sources.data.columns.levels[0])

        # it is faster to convert input data into a single 1D numpy array that is inserted at the given index than
        # inserting the provided Series consecutively.
        data = np.stack([fluxes, uncertainties, wave] if wave is not None else [fluxes, uncertainties], axis=-1)
        self.sources.data.loc[current] = data.flatten()

    def write_psf_parameters(self, current, values):
        """
        This method writes the fitted PSF parameters back to the instance of
        pampelmuse.psf.Variables that was provided when initialising the
        instance of FitCube.

        Parameters
        ----------
        current : float
            The index of the layer for which the fluxes were fitted.
        values : pandas.Series
            The fitted values of the PSF parameters. The indices of the Series
            should be the parameter names.
        """
        if values is None:  # nothing to do
            return
        else:
            for name, value in values.items():
                self.psf_attributes.data.at[current, (name, 'value')] = value

    def write_coordinate_transformation(self, current, values):
        """
        This method writes the fitted parameters of a coord. transformation
        back to the instance of pampelmuse.core.Sources that was provided
        when initialising the instance of FitCube.

        Parameters
        ----------
        current : float
            The index of the layer for which the fluxes were fitted.
        values : pandas.Series
            The fitted values of the transformation parameters. The indices of
            the Series should be the parameter names.
        """
        if values is None:  # nothing to do
            return
        else:
            for name, value in values.items():
                self.sources.transformation.data.at[current, (name, 'value')] = value

    def write_ifs_coordinates(self, current, values):
        """
        This method writes the fitted centroid coordinates back to
        the instance of pampelmuse.core.Sources that was provided
        when initialising the instance of FitCube.

        Parameters
        ----------
        current : float
            The index of the layer for which the fluxes were fitted.
        values : pandas.DataFrame
            The fitted values of the source centroids. The data frame should
            contain the columns 'xc' and 'yc' and use the source IDs as
            indices.
        """
        if values is None:  # nothing to do
            return
        else:
            for i, row in values.iterrows():
                # The assignment assumes that all sources with lower indices have their IFS coordinates loaded
                assert self.sources.ifs_coordinates_loaded[i]

                self.sources.ifs_coordinates.at[current, (i, 'x')] = row['xc']
                self.sources.ifs_coordinates.at[current, (i, 'y')] = row['yc']

    def _write_psf_properties(self, profiles, parameters, current_wavelength):
        """
        This method appends the detailed PSF information returned from the fit
        to a single layer to the dictionaries containing the combined PSF data.
        
        Parameters
        ----------
        profiles : pandas.DataFrame
            The radial PSF profiles. The data frame should contain the radial
            data as individual columns and use the source IDs as column names.
        parameters : pd.DataFrame
            The fitted PSF parameters, one per column. The data frame should
            contain one row per source.
        current_wavelength : float
            The wavelength of the layer for which the results were obtained. 
        """
        if parameters is None:
            return

        for source_id, row in parameters.iterrows():
            row.name = current_wavelength
            if source_id in self.parameters:
                self.parameters[source_id].append(row)
            else:
                self.parameters[source_id] = [row, ]

        for source_id in profiles:
            column = profiles[source_id]
            column.name = current_wavelength
            if source_id in self.profiles:
                self.profiles[source_id].append(column.reset_index(drop=True))
            else:
                self.profiles[source_id] = [column.reset_index(drop=True), ]

        self.profile_radii.append(profiles.index.values)

    def _write_minicubes(self, layer, data, xy):
        """
        Add the 2D maps of the PSF star fluxes and the fit residuals of the
        current layers to the StarCube instances used to create minicubes of
        the fluxes and residuals of each PSF star.
        
        Parameters
        ----------
        layer : float
            The wavelength of the layer to be added.
        data : pandas.DataFrame
            The source and residual flux in each pixel of the layer, provided
            as columns 'data' and 'residuals' of the data frame.
        xy : pandas.DataFrame
            The x- and y-coordinates of the pixels of the current layer,
            provided as columns 'x' and 'y' of the data frame.         
        """
        if data is None:
            return

        # loop over sources
        for source_id, source_data in data.items():
            # if it is a new source, initialize and add new instance os StarCube class.
            if source_id not in self.residuals:
                # spatial shape of minicube will be 2x the PSF radius along x and y
                shape = (2 * self.psf_radius, 2 * self.psf_radius)
                self.residuals[source_id] = StarCube(shape, offset=(xy.loc[source_data.index, 'x'].min(),
                                                                    xy.loc[source_data.index, 'y'].min()))
            # add current layer to StarCube instance
            self.residuals[source_id].add_layer(layer, xy.loc[source_data.index, ['x', 'y']].values,
                                                source_data['data'].values, source_data['residuals'].values)

    def _make_psf_hdus(self):
        """
        Create an astropy.io.fits.HDUList containing the individual PSF
        parameters and radial profiles as a function of wavelength.
        
        Returns
        -------
        hdu_list : astropy.io.fits.HDUList
            For each PSF source, the HDUList will contain two separate
            ImageHDUs. One containing the radial profiles, the other one
            containing the fitted PSF parameters.
        """
        from astropy.io import fits
        from astropy.table import Table
        hdu_list = fits.HDUList()

        # add HDU containing averaged radii valid for all profiles/wavelengths
        hdu_list.append(fits.ImageHDU(np.nanmean(self.profile_radii, axis=0)))
        hdu_list[-1].header['EXTNAME'] = 'RADIUS'

        # loop over HDUs containing PSF parameters
        for source_id, _data in self.parameters.items():
            data = pd.concat(_data, axis=1).T
            data.sort_index(axis=0, inplace=True)

            # create Table containing one column per parameter and the dispersion information in the various rows.
            hdu_list.append(fits.table_to_hdu(Table.from_pandas(data)))
            hdu_list[-1].header['EXTNAME'] = 'PM_{0}'.format(source_id)

            # add (approximate) dispersion information (python counting, need to swap for FITS)
            hdu_list[-1].header['CRVAL1'] = data.index[0]
            hdu_list[-1].header['CDELT1'] = np.mean(np.diff(data.index))
            hdu_list[-1].header['CRPIX1'] = 1

        # loop over HDUs containing radial profiles
        for source_id, _data in self.profiles.items():
            data = pd.concat(_data, axis=1)
            data.sort_index(axis=1, inplace=True)

            # zeroth axis contains profiles, first contains wavelength (python counting, need to swap for FITS)
            hdu_list.append(fits.ImageHDU(data.values))
            hdu_list[-1].header['EXTNAME'] = 'PR_{0}'.format(source_id)

            # add (approximate) dispersion information
            try:
                hdu_list[-1].header['CRVAL1'] = data.columns[0]
                hdu_list[-1].header['CDELT1'] = np.mean(np.diff(data.columns))
                hdu_list[-1].header['CRPIX1'] = 1
            except ValueError:
                logger.error('Failed to save header for source #{0}'.format(source_id))
                pass

        # loop over HDUs containing 2D maps of the PSF star fluxes and the fit residuals
        for source_id, minicube in self.residuals.items():

            # data
            hdu_list.append(fits.ImageHDU(minicube.data_cube))
            hdu_list[-1].header['EXTNAME'] = 'DAT_{0}'.format(source_id)

            # residuals
            hdu_list.append(fits.ImageHDU(minicube.residuals_cube))
            hdu_list[-1].header['EXTNAME'] = 'RES_{0}'.format(source_id)

            # add (approximate) dispersion information
            for i in [-1, -2]:
                try:
                    hdu_list[i].header['CRVAL3'] = minicube.crval3
                    hdu_list[i].header['CDELT3'] = minicube.cdelt3
                    hdu_list[i].header['CRPIX3'] = 1
                except ValueError:
                    logger.error('Failed to save header for source #{0}'.format(source_id))
                    pass

        return hdu_list


class StarCube(object):
    """
    This class provides an interface to handle 2D data from the PSF fit to
    individual stars.
    
    The idea behind this class is that each layer of the IFS data may be
    differently sampled, so creating a cube with the source profile or the fit
    residuals as a function of wavelength is not trivial. StarCube solves this
    issue by defining a grid upon instance initialization. Each new layer that
    is added will be resampled to this grid.
    """

    def __init__(self, shape, offset=(0, 0)):
        """
        Initialize a new instance of the StarCube class.
        
        Parameters
        ----------
        shape : tuple
            The 2D spatial shape of the data cube that will be generated. The
            output sampling will be regular within the provided shape.  
        offset : tuple, optional
            If the starting values of the grid coordinates should be non-zero,
            they can be provided through this paramater.
            
        Returns
        -------
        The newly created instance.
        """
        # Initial definitions
        self.shape = shape
        self.offset = offset

        # define grid
        self.x = np.arange(shape[0]) + offset[0]
        self.y = np.arange(shape[1]) + offset[1]

        # Each instance will contain two dictionaries, storing the data and the fit residuals for each individual
        # layer. The keys of the dictionaries are supposed to be the wavelengths of the layers.
        self.data = {}
        self.residuals = {}

    def add_layer(self, index, xy, data, residuals):
        """
        Add a new layer of data to the class.
        
        IMPORTANT: Currently this method assumes that the layer data is
        defined on a subset of the grid pixels defined upon class
        initialisation, i.e. it does not support resampling of the layers.
        Therefore it can currently not be used with pixtables.
        
        Parameters
        ----------
        index : float
            The wavelength of the new layer. 
        xy : nd_array, shape (N, 2)
            The x- and y-coordinates of the N pixels on which the layer is
            sampled. 
        data : nd_array, shape (N,)
            The data values of the layer
        residuals : nd_array, shape (N,)
            The residuals of the PSF fit to the layer data.
        """
        # check on which pixels of the grid the layer is defined
        in_image = np.in1d(xy[:, 0], self.x) & np.in1d(xy[:, 1], self.y)

        # account for offsets
        xy[:, 0] -= self.offset[0]
        xy[:, 1] -= self.offset[1]

        # define empty layer for data and add input at provided pixels.
        new_layer = np.zeros(self.shape)
        new_layer[xy[in_image, 0].astype(np.int32), xy[in_image, 1].astype(np.int32)] = data[in_image]
        # add new layer to dictionary
        self.data[index] = new_layer

        # same as above, only for the residuals
        new_layer = np.zeros(self.shape)
        new_layer[xy[in_image, 0].astype(np.int32), xy[in_image, 1].astype(np.int32)] = residuals[in_image]
        self.residuals[index] = new_layer

        # only for testing purposes: plot final 2D image of layer
        # import matplotlib.pyplot as plt
        # plt.imshow(new_layer, cmap=plt.cm.Blues)
        # plt.show()

        # from scipy.interpolate import interp2d
        # f = interp2d(x, y, data)
        # self.layers[index] = f(x, y)

    @property
    def data_cube(self):
        """
        Returns
        -------
        data_cube : nd_array
            Creates a data cube by stacking the currently available layers of
            data in order of increasing wavelength.
        """
        return np.stack([self.data[key] for key in sorted(self.data.keys())])

    @property
    def residuals_cube(self):
        """
        Returns
        -------
        residuals_cube : nd_array
            Creates a residuals cube by stacking the currently available
            layers of residuals in order of increasing wavelength.
        """
        return np.stack([self.residuals[key] for key in sorted(self.residuals.keys())])

    @property
    def wave(self):
        """
        Returns
        -------
        wave : nd_array
            Returns the wavelengths of the currently available layers in
            increasing order.
        """
        return np.array([value for value in sorted(self.data.keys())])

    @property
    def crval3(self):
        """
        Returns
        -------
        crval3 : float
            The minimum wavelength of the available layers.
        """
        return self.wave[0]

    @property
    def cdelt3(self):
        """
        Returns
        -------
        cdelt3 : float
            The (approximate) dispersion in wavelength direction.
        """
        return np.mean(np.diff(self.wave))
