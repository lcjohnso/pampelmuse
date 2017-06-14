"""
make_residuals_cube.py
======================
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
The class MakeResidualsCube that is used by the SUBTRES routine of PampelMuse
to create a data cube containing the fit residuals. It is designed as a
sub-class of the pampelmuse.core.FitCube class as it uses much of its
functionality.

Added
-----
rev. 343, 2016/08/01

Latest SVN revision
431, 2017/06/13
"""
import logging
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from .fit_cube import FitCube
from .fit_layer import FitLayer


__author__ = 'Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)'
__revision__ = 431


logger = logging.getLogger(__name__)


def init_worker(psf_type, psf_radius, osfactor, osradius, *args):
    """
    This function defines the global parameters for the parallel part of the
    processing.
    
    The global parameters are those that do not change from one layer to the
    next. Therefore they do not have to be initialized anew for each layer.

    Parameters
    ----------
    psf_type : string
        The name of the PSF profile in use.
    psf_radius : int
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
    global g_psf_radius
    global g_osfactor
    global g_osradius

    g_psf_type = psf_type
    g_psf_radius = psf_radius
    g_osfactor = osfactor
    g_osradius = osradius


def parallel_process_layers(fit_parameters):
    """
    This function is designed to calculate the fit residuals for several
    layers in parallel.
    
    To this aim it is called by multiprocess.Pool from within the
    MakeResidualCube class. It basically initializes a new instance of the
    pampelmuse.core.fit_layer.FitLayer class and afterwards executes the
    residuals_from_fluxes routine every time it is called.

    Parameters
    ----------
    fit_parameters : dict
        A dictionary containing the parameters necessary to launch the fit:
        data - the value, variance, x-, and y-positions, and wavelength
               (optional) of each pixel of the layer.
        catalog - A pandas,.DataFrame containing at least the pixel
                  coordinates of all sources that should be subtracted from
                  the data.
        psf_parameters - the values of the PSF parameters.
        position_parameters - the values of the coordinate transformation
                              parameters.
        background - the matrix containing the footprints of the individual
                     background components.
        fluxes - the fluxes of all components

    Returns
    -------
    residuals : ndarray
        An array containing the fit residuals for all pixels provided via the
        'data' parameter.
    """

    # fit parameters is a dictionary that must contain the following keys:
    # data, variances, transformation, xy, psf_parameters, position_parameters
    data = fit_parameters['data']
    catalog = fit_parameters['catalog']
    psf_parameters = fit_parameters['psf_parameters']
    position_parameters = fit_parameters['position_parameters']
    background = fit_parameters['background']
    fluxes = fit_parameters['fluxes']

    global g_psf_type, g_psf_radius, g_osfactor, g_osradius

    layerfit = FitLayer(data=data, catalog=catalog, psf_type=g_psf_type, psf_parameters=psf_parameters,
                        psf_radius=g_psf_radius, position_parameters=position_parameters, background=background,
                        osfactor=g_osfactor, osradius=g_osradius)

    results = layerfit.residuals_from_fluxes(fluxes=fluxes)

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

        # check if any sources should be included/excluded, define boolean array of included (resolved) sources
        elif include is not None:
            logging.info('Only subtracting the following source(s): {0}'.format(', '.join(map(str, include))))
            selected_ids = pd.Index(include)
        elif exclude is not None:
            logging.info('Not subtracting the following source(s): {0}'.format(', '.join(map(str, exclude))))
            selected_ids = self.sources.ids.drop(exclude)
        else:
            selected_ids = self.sources.ids

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
        if n_cpu is None:
            n_cpu = cpu_count()
        logger.info("Starting processing using {0} CPU(s).".format(n_cpu))

        # Set global values for processing of individual layers and initialize pool of processes
        initial_arguments = (self.psf_attributes.profile,
                             self.psf_radius,
                             self.osfactor,
                             self.osradius
                             )
        if n_cpu > 1:
            pool = Pool(n_cpu, initializer=init_worker, initargs=initial_arguments)
        else:
            pool = None
            init_worker(*initial_arguments)

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
            logger.info("Collecting input data processing layer #{0}.".format(jj))

            # get 1D arrays containing valid data pixels
            data = self.get_layer(self.sources.wave.get_loc(jj), use_variances=False, n_bin=1, skip_nan=False)
            fit_input.append({'data': data})

            # update fluxes
            fluxes = self.read_fluxes(jj)
            fit_input[-1]['fluxes'] = fluxes[selected_components]

            # update PSF parameters
            fit_input[-1]['psf_parameters'] = self.read_psf_parameters(jj)

            # update parameters of coordinate transformation (if available and needed)
            if self.sources.transformation.valid:
                if not self.sources.ifs_coordinates_fitted[~self.sources.is_background].all():
                    fit_input[-1]['position_parameters'] = self.read_coordinate_transformation(jj)

            # update direct coordinates if required
            if self.sources.ifs_coordinates_fitted.any() or self.sources.ifs_coordinates_free.any():
                catalog[['x', 'y']] = self.read_ifs_coordinates(jj)
                fit_input[-1]['position_parameters'] = None

            # add sources catalog to fit input
            fit_input[-1]['catalog'] = catalog

            # add background if requested
            if subtract_background and self.background is not None:
                self.background.transform = data[['x', 'y']].values
                fit_input[-1]['background'] = self.background.get_sparse_matrix(
                    offset=self.sources.n_components - self.sources.n_background)
            else:
                fit_input[-1]['background'] = None

            # A parallel processing is triggered every n_cpu's layer
            if np.mod(i + 1, n_cpu) == 0 or jj == indices_to_analyse[-1]:
                if n_cpu > 1:
                    _results = pool.map_async(parallel_process_layers, fit_input)
                    results = _results.get()
                else:
                    results = [parallel_process_layers(fit_input[0])]

                for k, result in enumerate(results, start=1):
                    self.residuals[self.ifs_data.slice_for_layer(layers_to_analyse[i] - n_cpu + k)] = result

                fit_input = []

        return self.residuals

    def read_fluxes(self, current, last=None):
        """
        Method to fluxes of all components for a given layer.
        
        If possible and requested, the method obtains updated values by
        averaging over the results from the nearest layers, otherwise it uses
        the values in the sources instance.

        Parameters
        ----------
        current : integer
            The index of the layer for which the parameters should be obtained
        last : list of integers, optional
            A list containing the indices of the layers that are averaged to
            get the new value.

        Returns
        -------
        fluxes : pd.Series
            The updated value of the flux of each component for the current
            layer.
        """
        logger.info("Updating fluxes of individual components ...")

        if last is None:
            return self.sources.data.loc[current, (slice(None), 'flux')]
        else:
            return np.nanmean(self.sources.loc[last, (slice(None), 'flux')], axis=0)
