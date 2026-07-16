"""
make_residuals_cube.py
======================
Copyright 2013-2026 Sebastian Kamann

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

Latest Git revision
2026/04/11
"""
import collections
import logging
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from .fit_cube import FitCube
from .fit_layer import FitLayer
from ..instruments import GenericIFS, available_instruments


__author__ = 'Sebastian Kamann (s.kamann@ljmu.ac.uk)'
__revision__ = 20260411


logger = logging.getLogger(__name__)


FitLayerInit = collections.namedtuple('FitLayerInit', ['psf_type', 'psf_radius', 'psf_radius_scaling', 'osfactor',
                                                       'osradius'])
FitLayerInit.__new__.__defaults__ = (None,) * len(FitLayerInit._fields)
# see https://stackoverflow.com/questions/11351032/namedtuple-and-default-values-for-optional-keyword-arguments


FitLayerCall = collections.namedtuple('FitLayerCall', ['data', 'fluxes', 'psf_parameters', 'psf_lut', 'catalog',
                                                       'position_parameters', 'background', 'background_index', 'wvl'])


init_arguments = FitLayerInit()


def init_worker(cube_parameters, *args):
    """
    This function defines the global parameters for the parallel part of the
    processing.
    
    The global parameters are those that do not change from one layer to the
    next. Therefore they do not have to be initialized anew for each layer.

    Parameters
    ----------
    cube_parameters : instance of FitLayerInit
        The global parameters.
    args
        This function does not support additional arguments.
    """
    global init_arguments

    assert not args
    assert isinstance(cube_parameters, FitLayerInit)

    init_arguments = cube_parameters


def parallel_process_layers(layer_parameters):
    """
    This function is designed to calculate the fit residuals for several
    layers in parallel.
    
    To this aim it is called by multiprocess.Pool from within the
    MakeResidualCube class. It basically initializes a new instance of the
    pampelmuse.core.fit_layer.FitLayer class and afterwards executes the
    residuals_from_fluxes routine every time it is called.

    Parameters
    ----------
    layer_parameters : instance of FitLayerCall
        The paramters required to create the residuals for ta given layer

    Returns
    -------
    residuals : ndarray
        An array containing the fit residuals for all pixels provided via the
        FitLayerCall.data input parameter.
    """
    global init_arguments

    layerfit = FitLayer(data=layer_parameters.data,
                        catalog=layer_parameters.catalog,
                        psf_type=init_arguments.psf_type,
                        psf_parameters=layer_parameters.psf_parameters,
                        psf_radius=init_arguments.psf_radius,
                        psf_radius_scaling=init_arguments.psf_radius_scaling,
                        psf_lut=layer_parameters.psf_lut,
                        position_parameters=layer_parameters.position_parameters,
                        osfactor=init_arguments.osfactor,
                        osradius=init_arguments.osradius,
                        background=layer_parameters.background,
                        background_index=layer_parameters.background_index,
                        wvl=layer_parameters.wvl)

    results = layerfit.residuals_from_fluxes(fluxes=layer_parameters.fluxes)

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

    def __call__(self, layer_range=(0, -1), n_cpu=1, subtract_background=False,
                 include=None, exclude=None, **kwargs):
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
        include : array_like, optional
            In case only a subset of the resolved sources should be
            subtracted, their IDs should be provided here.
        exclude : array_like, optional
            In case some resolved sources should not be subtracted, their IDs
            should be provided here.

        Returns
        -------
        residuals
        """
        if kwargs:
            logger.error('Invalid keyword argument(s) provided: {0}'.format(kwargs))
            raise IOError('Cannot handle keyword argument(s): {0}'.format(kwargs))

        # either include of exclude can be provided, not both
        if include is not None and exclude is not None:
            raise IOError('Either parameter "include" or "exclude" may be set, not both.')

        # check if only certain sources should be subtracted
        elif include is not None:
            logging.info('Only subtracting the following source(s): {0}'.format(', '.join(map(str, include))))
            selected_ids = pd.Index(include)

            # add sky components if they should be subtracted
            if subtract_background:
                selected_ids = selected_ids.append(self.sources.ids[self.sources.is_background])
        else:
            selected_ids = self.sources.ids.copy()

            # remove any stellar components that should not be subtracted
            if exclude is not None:
                logging.info('Not subtracting the following source(s): {0}'.format(', '.join(map(str, exclude))))
                selected_ids = selected_ids.drop(exclude)

            # remove sky components if they should not be subtracted
            if not subtract_background:
                selected_ids = selected_ids.drop(self.sources.ids[self.sources.is_background])

        # get corresponding rows in spectrum data
        selected_components = self.sources.specrow[selected_ids].unique()

        # check which layers should be analysed
        start, stop = layer_range
        if start < 0:
            start = self.sources.ndisp + start
        if stop < 0:
            stop = self.sources.ndisp + (stop + 1)
        logger.info("Layer range used to create residuals: {0} - {1}".format(start, stop))

        # define layers that should be used in increasing order
        layers_to_analyse = np.arange(start, stop, dtype=np.int16)

        # get indices corresponding to requested layers
        indices_to_analyse = self.sources.wave[layers_to_analyse]

        # begin loading data required for fitting process into memory
        # whole array with spectra needs to be loaded
        self.sources.load_spectra()

        # if parameters (of the PSF or the coordinate transformation) are fitted, it must be ensured that the instances
        # have correct dispersion information
        if self.psf_attributes.n_dispersion != self.sources.n_dispersion:
            self.psf_attributes.resize(self.sources.wave)
        else:
            self.psf_attributes.data.index = self.sources.wave
        # Interpolate across wavelength range to ensure the parameters have valid values for all fitted layers.
        self.psf_attributes.interpolate()

        # do same for coordinate transformation (only if available)
        if self.sources.transformation.valid:
            if self.sources.transformation.n_dispersion != self.sources.n_dispersion:
                self.sources.transformation.resize(self.sources.wave)
            else:
                self.sources.transformation.data.index = self.sources.wave
            self.sources.transformation.interpolate()

        # Load full coordinate array into memory because doing it per layer is quite slow.
        self.sources.load_ifs_coordinates()
        # replace NaNs to make sure valid coordinates are available for all fitted layers.
        self.sources.ifs_coordinates.fillna(method='pad', inplace=True)

        # Check how many CPUs should be used
        if n_cpu is None or n_cpu == -1:
            n_cpu = cpu_count()
        logger.info("Starting processing using {0} CPU(s).".format(n_cpu))

        # Set global values for processing of individual layers and initialize pool of processes
        initial_arguments = FitLayerInit(psf_type=self.psf_attributes.profile,
                                         psf_radius=self.psf_radius,
                                         psf_radius_scaling=self.psf_radius_scaling,
                                         osfactor=self.osfactor,
                                         osradius=self.osradius)

        if n_cpu > 1:
            pool = Pool(n_cpu, initializer=init_worker, initargs=[initial_arguments])
        else:
            pool = None
            init_worker(initial_arguments)

        # prepare source catalog of _stellar_ sources
        # note that the catalog content may change from layer to layer when no coordinate transformation is being used
        series = []
        if subtract_background:
            stellar_ids = selected_ids.drop(self.sources.ids[self.sources.is_background])
        else:
            stellar_ids = selected_ids
        series.extend([self.sources.x[stellar_ids], self.sources.y[stellar_ids]])
        series.append(self.sources.specrow[stellar_ids])
        series.append(self.sources.weights[stellar_ids])
        catalog = pd.concat(series, axis=1, keys=['x', 'y', 'component', 'magnitude'])

        # For all layers to be fitted in parallel, we first need to collect the input data in a common list.
        # The list will contain one dictionary per layer to be fitted in parallel
        fit_input = []

        # loop over all layers in cube that should be analysed
        for i, jj in enumerate(indices_to_analyse):

            # Add a common prefix for all logging messages related to the analysis of the current layer.
            n_digits = int(np.log10(layers_to_analyse.size)) + 1  # for logging prefix, get number of digits of N_layer
            fmt = logging.Formatter("%(asctime)s[%(levelname)-8s]: [{0:0{1}d}/{2}] %(message)s".format(
                i+1, n_digits, layers_to_analyse.size), datefmt='%m/%d/%Y %H:%M:%S')
            handler = logging.getLogger().handlers[0]
            handler.setFormatter(fmt)

            # collect input data for processing of current layer
            logger.info("Processing layer with wavelength {0:.1f} Angstrom.".format(1e10*jj))

            # get 1D arrays containing valid data pixels
            data = self.get_layer(self.sources.wave.get_loc(jj), use_variances=False, n_bin=1, skip_nan=False)

            # update fluxes
            fluxes = self.read_fluxes(jj, fillna=False)
            fluxes = fluxes[fluxes.index <= selected_components.max()]

            # update PSF parameters
            psf_parameters, lut = self.read_psf_parameters(jj)

            # update parameters of coordinate transformation (if available and needed)
            position_parameters = None
            if self.sources.transformation.valid:
                if not self.sources.ifs_coordinates_fitted[~self.sources.is_background].all():
                    position_parameters = self.read_coordinate_transformation(jj)

            # update direct coordinates if required
            if self.sources.ifs_coordinates_fitted.any() or self.sources.ifs_coordinates_free.any():
                catalog[['x', 'y']] = self.read_ifs_coordinates(jj)

            # add background if requested
            if subtract_background and self.background is not None:
                self.background.transform = data[['x', 'y']].values
                if 'ifu' in data.columns and 'slice' in data.columns:
                    self.background.ifus = data['ifu'].values
                    self.background.slices = data['slice'].values
                background_matrix, background_components = self.background.get_sparse_matrix(
                    offset=self.sources.n_components - self.sources.n_background)
            else:
                background_matrix = background_components = None

            fit_input.append(FitLayerCall(data=data, fluxes=fluxes, psf_parameters=psf_parameters, psf_lut=lut,
                                          position_parameters=position_parameters, catalog=catalog,
                                          background=background_matrix, background_index=background_components, wvl=jj))

            # A parallel processing is triggered every n_cpu's layer
            if np.mod(i + 1, n_cpu) == 0 or jj == indices_to_analyse[-1]:
                if n_cpu > 1:
                    _results = pool.map_async(parallel_process_layers, fit_input)
                    results = _results.get()
                else:
                    results = [parallel_process_layers(fit_input[0])]

                for k, result in enumerate(results, start=1):
                    # How results are written to residuals array depends on IFU data type in use.
                    layer_index = layers_to_analyse[i] - len(results) + k
                    if isinstance(self.ifs_data, GenericIFS):
                        img = self.ifs_data.make_image(layer=result)
                        self.residuals[layer_index] = img
                    elif isinstance(self.ifs_data, available_instruments().get('PIXTABLE', type(None))):
                        i_min, i_max = self.ifs_data.layer_indices[k]
                        img = np.nan*np.ones((i_max - i_min), dtype=np.float64)
                        img[result.index] = result.values
                        self.residuals[i_min:i_max] = img
                    else:
                        raise NotImplementedError
                fit_input = []

        return self.residuals
