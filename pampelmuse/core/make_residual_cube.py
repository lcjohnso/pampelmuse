"""
make_residuals_cube.py
======================
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
The class MakeResidualsCube that is used by the SUBTRES routine of PampelMuse
to create a data cube containing the fit residuals. It is designed as a
sub-class of the pampelmuse.core.FitCube class as it uses much of its
functionality.

Added
-----
rev. 343, 2016/08/01

Latest SVN revision
349, 2016/08/05
"""
import logging
from multiprocessing import cpu_count, Pool
import numpy as np
from .fit_cube import FitCube
from .fit_layer import FitLayer


__author__ = 'Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)'
__revision__ = 349


logger = logging.getLogger(__name__)


def init_worker(psf_type, magnitudes, components, maxrad, osfactor, osradius, *args):
    """
    This function defines the global parameters for the parallel part of the
    processing. The global parameters are those that do not change from one
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
    maxrad : int
        The radius in pixels out to which the PSF profiles are calculated
    osfactor : int
        The oversampling factor used to improve the accuracy for the central
        PSF pixels.
    osradius : float
        The radius out to which oversampling is being used.
    args

    Returns
    -------
    Nothing is returned, only the global variables are set. Their names are
    derived from the names of the parameters by adding a prefix 'g_'
    """
    global g_psf_type
    global g_magnitudes
    global g_components
    global g_maxrad
    global g_osfactor
    global g_osradius

    g_psf_type = psf_type
    g_magnitudes = magnitudes
    g_components = components
    g_maxrad = maxrad
    g_osfactor = osfactor
    g_osradius = osradius


def parallel_process_layers(fit_parameters):
    """
    This function is designed to calculate the fit residuals for several
    layers in parallel. To this aim it is called by multiprocess.Pool from
    within the MakeResidualCube class. It basically initializes a new instance
    of the pampelmuse.core.fit_layer.FitLayer class and afterwards executes
    the residuals_from_fluxes routine every time it is called.

    Parameters
    ----------
    fit_parameters : dict
        A dictionary containing the parameters necessary to launch the fit:
        data - the data array that defines the layer.
        transformation - The transformation from pixel to x- & y-coordinates
        xy - the reference or pixel coordinates of the sources.
        psf_parameters - the initial guesses/fixed values of the PSF
                         parameters.
        position_parameters - the initial guesses/fixed values of the coord.
                              transformation parameters.
        background - the matrix containing the footprints of the individual
                     background components.
        fluxes - the fluxes of all components
        background_fluxes - the fluxes of all background components (if any)

    Returns
    -------
    residuals : ndarray
        An array containing the residuals for all pixels provided in the
        'transform' parameter.
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
    fluxes = fit_parameters['fluxes']
    background_fluxes = fit_parameters['background_fluxes']

    global g_psf_type, g_magnitudes, g_components, g_maxrad, g_osfactor, g_osradius

    layerfit = FitLayer(data=data, variances=variances, transformation=transformation, psf_type=g_psf_type,
                        psf_parameters=psf_parameters, xy=xy, position_parameters=position_parameters,
                        magnitudes=g_magnitudes, components=g_components, background=background,
                        wave=wave, maxrad=g_maxrad, osfactor=g_osfactor, osradius=g_osradius)

    results = layerfit.residuals_from_fluxes(fluxes=fluxes, background_fluxes=background_fluxes)

    return results


class MakeResidualsCube(FitCube):
    """
    This class is designed to calculate the fit residuals for a given data
    cube and model setup.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the MakeResidualsCube class.

        Parameters
        ----------
        kwargs
            Any keyword arguments are passed on to the initialisation method
            of the parent class.
        """
        super(MakeResidualsCube, self).__init__(**kwargs)

        # Need to create the full array that will contain the residuals.
        logging.info('Creating array to store residuals...')
        self.residuals = np.zeros(self.ifs_data.datahdu.shape, dtype=np.float32)

    def __call__(self, layer_range=(0, -1), n_cpu=1, subtract_background=False, **kwargs):
        """
        Creates the fit residuals for a full data cube. For each layer of the
        provided cube, the following steps are performed:
         1) Load the data for the current layer
         2) Update the data required for creating the residuals:
            - fluxes of all components
            - PSF parameter values
            - source coordinates
         3) Process n_cpu layers in parallel.
         4) Write the residuals to an array that will be returned once the
            processing is completed.

        Parameters
        ----------
        layer_range : tuple
            The indices of the first and last layer of the spectral range to
            be analysed. By default, the whole cube is analysed.
        n_cpu : integer
            The number of CPUs that should be used.
        subtract_background : boolean, optional
            A flag indicating if the background component should also be
            subtracted from the data.

        Returns
        -------
        residuals
        """
        # check which layers should be analysed
        start, stop = layer_range
        if start < 0:
            start = self.sources.ndisp + start
        if stop < 0:
            stop = self.sources.ndisp + (stop + 1)
        logger.info("Layer range used to create residuals: {0}-{1}".format(start, stop))

        # define layers that should be used in increasing order
        layers_to_analyse = np.arange(start, stop, dtype=np.int16)

        # begin loading data required for fitting process into memory
        # whole array with spectra needs to be loaded
        self.sources.load_spectra()
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
        if self.sources.transformation.valid:
            self.sources.transformation.interpolate()

        # Load full coordinate array into memory because doing it per layer is quite slow. Switch on interpolation to
        # make sure valid coordinates are available for all layers.
        self.sources.load_ifs_coordinates(interpolate=True)

        # Check how many CPUs should be used
        if n_cpu is None:
            n_cpu = cpu_count()
        logger.info("Starting processing using {0} CPU(s).".format(n_cpu))

        # For all layers to be fitted in parallel, we first need to collect the input data in a common list.
        # The list will contain one dictionary per layer to be fitted in parallel
        fit_input = []

        # Set global values for processing to individual layers:
        initargs = (self.psf_attributes.profile,
                    self.magnitudes,
                    self.components,
                    self.maxrad,
                    self.osfactor,
                    self.osradius
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
            logger.info("Collecting input data processing layer #{0}.".format(jj))

            # get 1D arrays containing valid data pixels, also sigma-values if requested
            data, variances, transform, wave = self.get_layer(jj, n_bin=1, use_variances=False, mask_nan=False)
            fit_input.append({'data': data, 'variances': variances, 'transformation': transform, 'wave': wave})

            # update fluxes
            fluxes = self.read_fluxes(jj)
            fit_input[-1]['fluxes'] = fluxes[:-self.sources.nbackground]
            if subtract_background:
                fit_input[-1]['background_fluxes'] = fluxes[-self.sources.nbackground:]
            else:
                fit_input[-1]['background_fluxes'] = None

            # update PSF parameters
            fit_input[-1]['psf_parameters'] = self.read_psf_parameters(jj)

            # update parameters of coordinate transformation (if available and needed)
            if self.sources.transformation.valid:
                if not self.sources.ifs_coordinates_fitted[~self.sources.is_background].all():
                    fit_input[-1]['position_parameters'] = self.read_coordinate_transformation(jj)

            # update direct coordinates if required
            if self.sources.ifs_coordinates_fitted.any() or self.sources.ifs_coordinates_free.any():
                fit_input[-1]['xy'] = self.read_ifs_coordinates(jj)
                fit_input[-1]['position_parameters'] = None
            else:
                fit_input[-1]['xy'] = np.dstack((self.sources.x[~self.sources.is_background],
                                                 self.sources.y[~self.sources.is_background]))[0]

            if self.background is not None and subtract_background:
                self.background.transform = self.ifs_data.transform
                fit_input[-1]['background'] = self.background.get_sparse_matrix(
                    offset=self.sources.ncmp - self.sources.nbackground)
            else:
                fit_input[-1]['background'] = None

            # A parallel processing is triggered every n_cpu's layer
            if np.mod(i + 1, n_cpu) == 0:
                if n_cpu > 1:
                    # pool = Pool(n_cpu, initializer=init_worker, initargs=initargs)
                    _results = pool.map_async(parallel_process_layers, fit_input)

                    # pool.close()
                    # pool.join()
                    results = _results.get()  # list(itertools.chain.from_iterable(_results.get()))
                else:
                    # init_worker(*initargs)
                    results = [parallel_process_layers(fit_input[0])]

                for k, result in enumerate(results, start=1):
                    self.residuals[self.ifs_data.slice_for_layer(jj - n_cpu + k)] = result

                fit_input = []

        return self.residuals

    def read_fluxes(self, current, last=None):
        """
        Method to fluxes of all components for a given layer. If possible and
        requested, the method obtains updated values by averaging over the
        results from the nearest layers, otherwise it uses the values in the
        sources instance.

        Parameters
        ----------
        current : integer
            The index of the layer for which the parameters should be obtained
        last : list of integers, optional
            A list containing the indices of the layers that are averaged to
            get the new value.

        Returns
        -------
        fluxes : ndarray
            The updated value of each PSF parameter for the current layer.
        """
        logger.info("Updating fluxes of individual components...")

        if last is None:
            return self.sources.data[current, :, 0]
        else:
            return np.nanmean(self.sources.data[last, :, 0], axis=0)
