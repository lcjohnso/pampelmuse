"""
fit_array.py
============
Copyright 2013-2020 Sebastian Kamann

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
This module implements the optimization for a single layer of IFS data.

The optimization consists of three steps that are performed iteratively:
(1) An optimization for the fluxes based on the current estimate of the PSF
    parameters and the source coordinates. This is a linear optimization,
    carried out via direct inversion of a sparse matrix that contains the PSF
    profiles of all fitted components
(2) An optimization for the positions of the individual sources that is
    performed using a Levenberg-Marquard algorithm. Note that there are two
    different ways of how the to fit the fluxes.
    (a) via a coordinate transformation that translates the positions from a
        reference frame into the image that is being analysed. In that case,
        the parameters of the transformation are optimized, not individual
        source positions.
    (b) via direct coordinate fits without a transformation. This approach is
        currently only recommended for sparsely populated fields.
(3) An optimization for the parameters of the PSF profile that is also
    performed using a Levenberg-Marquard algorithm.

Note that steps (2a) and (3) can be performed in two different ways, either by
simultaneously fitting all sources or by selecting individual isolated sources
and only fitting this subset. The former approach is recommended for small
images while the latter approach works better on large images. Step (2b) only
works with the latter approach.

Finally, a not about the fitting weights of the individual pixels in the
analysis. By default, inverse-variance weighting is used. If the variances are
not provided, the code tries to estimate them from the current residuals as a
function of the fitted pixel intensities.
In addition, the code uses an additional weighting scheme to reduce the
weights of deviant pixels. As this method requires a reasonably good model to
stars with, it is currently only used if the source positions and PSF
parameters are not fitted.


Latest Git revision
-------------------
2020/03/06
"""
import collections
import contextlib
import logging
import time
import traceback
import numpy as np
import pandas as pd
from scipy import interpolate, optimize, sparse
from scipy.sparse import linalg
from ..psf.profiles import *
from ..core.coordinates import Transformation


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20200306


logger = logging.getLogger(__name__)


# for a nicer printing of numpy arrays, following suggestion from
# http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


# define namedtuple for returning the results of the fit in a useful structure
FitResult = collections.namedtuple('FitResult', ['fluxes', 'uncertainties', 'positions', 'psf', 'wave', 'profiles',
                                                 'parameters', 'residuals', 'lut'])


class FitLayer(object):
    """
    This class performs a PSF fit on a single layer of a data cube.

    If requested, it optimizes for the parameters of the PSF model and
    the positions of the provided sources. In any case, the fluxes of the
    sources are optimized.
    """

    _unresolved_matrix = None

    def __init__(self, data, catalog, psf_type, psf_parameters, psf_radius=np.inf, psf_radius_scaling=None,
                 psf_lut=None, position_parameters=None, last_component_is_unresolved=False, background=None,
                 osfactor=1, osradius=0, background_index=None, wvl=None):
        """
        Initialize a new instance of the FitLayer class.

        Parameters
        ----------
        data : pandas.DataFrame
            The data on which the PSF fit should be performed. Note that this
            should be provided as a pandas DataFrame containing at least the
            columns 'x', 'y', and 'value', providing the spatial coordinates
            and the flux in each pixel. Additional columns that are used when
            present are 'variance' and 'wave', containing the variance and the
            exact wavelength of each pixel.
        catalog : pandas.DataFrame
            The catalog of sources that should be optimized on the data. It
            must also be provided as a pandas DataFrame, containing one source
            per row and at least the columns 'x' and 'y', containing (initial
            guesses of) the centroid coordinates. Additional columns that are
            used when present are 'component' and 'magnitude'. If available,
            they should contain information on which sources are combined to
            components and what their relative magnitudes per component are.
        psf_type : string
            The type of PSF to be used. Currently 'gauss', 'moffat', and
            'double_gauss' are supported.
        psf_parameters : dictionary
            The initial guesses (or fixed values) of the individual PSF
            parameters. The required keys depend on which PSF profile is being
            used:
             - gauss: fwhm, e, theta
             - moffat: beta, fwhm, e, theta
             - double_gauss: fwhm1, fwhm2, f12, e, theta
             ...
        psf_radius_scaling : float, optional
            The PSF radius can be adapted based on the fitted flux of each
            star. If provided, 'psf_radius_scaling' defines the fractional
            flux (relative to the brightest star in the fit) that corresponds
            to a reduction of psf_radius by a factor of 2.
        psf_lut : pandas.Series, optional
            A look-up table for correcting the final fluxes obtained from the
            analytical PSF profiles.
        psf_radius : float, optional
            The radius in pixels out to which the PSF profiles will be defined.
        position_parameters : dictionary
            The initial guesses (or fixed values) of the individual parameters
            of the coordinate transformation (if any). If not provided, it is
            assumed that 'xy' contains pixel coordinates.
        last_component_is_unresolved : boolean, optional
            This flag should be set to true if the fit component with the
            highest index (i.e. the highest entries in 'components') is an
            unresolved component. See Kamann et al. (2013) for the idea of the
            unresolved component.
        background : sparse matrix, optional
            The spatial distributions of the background component to be used
            in the fit, provided as a sparse matrix with one component per
            column. Note that the first columns of the matrix must be empty
            to accommodate the stellar sources.
        osfactor : int, optional
            The oversampling factor used to increase the accuracy of the
            calculation of the core shape of the PSF profiles.
        osradius : float, optional
            The radius in pixels out to which the oversampling is being used.
        background_index : pandas.Index, optional
            The index/indices of the background component(s). If None (default)
            the background components are indexed descendingly from -1.
        wvl : float, optional
            The wavelength of the data (in m). This is needed for the set up
            of certain PSF models.
        """
        self.psf_radius = psf_radius
        self.psf_radius_scaling = psf_radius_scaling

        # check consistency of provided data
        if not isinstance(data, pd.DataFrame):
            logger.error('Parameter "data" must be pandas DataFrame, not {0}'.format(type(data)))
            return
        for column in ['x', 'y', 'value']:
            if column not in data.columns:
                logger.error('Missing required column "{0}" in provided data.'.format(column))
                return
        self.data = data
        self.n_pixel = self.data.shape[0]

        if 'wave' in self.data.columns:
            self.wave = sparse.csc_matrix(self.data['wave'].values)
        else:
            self.wave = None

        # check consistency of source catalog
        if not isinstance(catalog, pd.DataFrame):
            logger.error('Parameter "catalog" must be pandas DataFrame, not {0}'.format(type(catalog)))
            return
        for column in ['x', 'y']:
            if column not in catalog.columns:
                logger.error('Missing required column "{0}" in provided source catalog.'.format(column))
                return
        self.catalog = catalog
        self.n_sources = self.catalog.shape[0]

        # check if information on components available, otherwise assume each source fitted individually
        if 'component' not in self.catalog.columns or 'magnitude' not in self.catalog.columns:
            self.catalog['component'] = pd.Series(np.arange(self.n_sources, dtype=np.int32), index=self.catalog.index)
            self.catalog['magnitude'] = pd.Series(0., index=self.catalog.index)

        self.n_components = self.catalog['component'].max() + 1
        self.last_component_is_unresolved = last_component_is_unresolved

        # check consistency and handle background if any was provided
        if background is not None and not isinstance(background, sparse.coo_matrix):
            logger.error('Parameter "background" must be a scipy.sparse.coo_matrix, not {0}'.format(type(background)))
            return
        elif background is not None and background.shape[1] > 0:
            self.background = background
            self.n_background = self.background.shape[1] - self.n_components
        else:
            self.background = None
            self.n_background = 0

        # check if a fit will be possible
        if self.n_components + self.n_background > self.n_pixel:
            logger.error('Number of fit components exceeds number of pixels.')
            logger.error('Any fit will be unconstrained.')

        # Initialize PSF
        if psf_type.lower() == 'gauss':
            psf_class = Gaussian
        elif psf_type.lower() == 'moffat':
            psf_class = Moffat
        elif psf_type.lower() == 'double_gauss':
            psf_class = DoubleGaussian
        elif psf_type.lower() == 'moffat_gauss':
            psf_class = MoffatGaussian
        elif psf_type.lower() == 'double_moffat':
            psf_class = DoubleMoffat
        elif psf_type.lower() == 'maoppy':
            psf_class = Maoppy
        else:
            logger.error('Unknown PSF type provided: {0}'.format(psf_type))
            psf_class = None

        # define PSF profile for each source, check if di   rect positions were provided for some sources
        self.psf_instances = {}

        # Important: Extract transformation out of following loop as otherwise a new copy will be
        # created for each PSF!
        transform = self.data[['y', 'x']].values

        # check if look-up table should be used
        if psf_lut is not None:
            self.lut = interpolate.InterpolatedUnivariateSpline(psf_lut.index, psf_lut.values)
        else:
            self.lut = None

        for index, row in self.catalog.iterrows():
            new_psf = psf_class(transform=transform, uc=row['x'], vc=row['y'], mag=row['magnitude'],
                                src_id=index, maxrad=self.psf_radius, osfactor=osfactor, osradius=osradius,
                                lut=self.lut)
            if isinstance(psf_class, Maoppy) and wvl is not None:
                new_psf.wvl = wvl

            # initialize the PSF parameters:
            for key, value in psf_parameters.items():
                setattr(new_psf, key, value)
            # if provided, add information about coordinate transformation
            if position_parameters is not None:
                for key, value in position_parameters.items():
                    setattr(new_psf, key, value)
            # otherwise use reference coordinates as direct IFS coordinates
            else:
                new_psf.xc = row['x']
                new_psf.yc = row['y']
            self.psf_instances[index] = new_psf

        # place-holder for variables defined below:
        self.source_matrix = None
        self.fit_weights = None

        # prepare array for fluxes of all components (including background components if any)
        self.fluxes = pd.Series(0., index=np.arange(self.n_components))
        if self.n_background > 0:
            # check if valid indices for background components were provided
            if background_index is None or len(background_index) != self.n_background:
                background_index = np.arange(-1, -(self.n_background + 1), -1)
            self.fluxes = self.fluxes.append(pd.Series(0., index=background_index))

        self.offsets = None

    def _check_background(self, background):
        """
        Internal method that checks if the provided background is valid and
        whether it contains any components that do not contribute to the
        observed flux.
        
        Parameters
        ----------
        background : scipy.sparse.coo_matrix or None
            The provided background. Unless no background is being used
            (None), it must be provided as an instance of a scipy.sparse
            coo-matrix.

        Returns
        -------
        background : scipy.sparse.coo_matrix or None
            The provided matrix (if any, otherwise None is returned), cleaned
            for any components that do not contribute to the observed flux.
        n_background : int
            The number of (nonzero) background components.
        missing_background : array_like
            The list of background components that were found to have no
            contribution to the observed flux.
        """

        if background is None:
            return None, 0, None

        # background must be provided as a sparse matrix
        assert(isinstance(background, sparse.coo_matrix)), '"background" must be scipy.sparse.coo_matrix'

        # check if all background components actually contribute
        missing = np.flatnonzero(background.getnnz(axis=0) == 0)

        # need to account for offset because of stellar sources
        missing = missing[missing >= self.n_components]
        if missing.any():
            logger.warning('Provided background contains irrelevant components: {0}'.format(
                ', '.join([str(m - self.n_components) for m in missing])))

            # remove empty background columns - need to convert to csc matrix for slicing to work
            to_keep = np.in1d(np.arange(background.shape[1]), missing, assume_unique=True, invert=True)
            background = sparse.coo_matrix(background.tocsc()[:, to_keep])

        return background, 1 + background.col.max() - background.col.min(), missing

    @property
    def variances(self):
        """
        This method is designed to provide an estimate of the variances if
        they are not provided externally.

        To this aim, the method compares the current model with the data and
        determines the deviation between the two as a function of the model
        flux using a running MAD.

        Returns
        -------
        variances : pandas-Series
            The array containing the estimated variances. Note that it will
            be filled with 1s if no model is yet available.
        """
        # if available, use provided variances
        if 'variance' in self.data:
            return self.data['variance']
        # otherwise, we only can estimate the variances if a model exists
        elif self.fluxes.max() > 0:
            model_fluxes = self.model.sum(axis=1).getA1()
            x_bin, y_bin = FitLayer.calc_running_mad(model_fluxes, self.current)
            return pd.Series((1.4826*np.interp(model_fluxes, x_bin, y_bin))**2, index=self.data.index)
            # 1.4826 is for MAD -> sigma conversion
        # if not, just return 1s
        else:
            return pd.Series(1., index=self.data.index)

    def get_weight_factors(self, alpha=2, beta=2):
        """
        This method implements a weighting scheme similar to the one used by
        P. Stetson in DAOphot.

        The idea is to reduce the weights of deviant pixels (i.e. pixels with
        huge residuals) using a modified variance. The weight of a pixel i in
        the fit is modified from a pure variance-weighting as follows:

        w_i = f_i(residuals_i) / variance_i, where

        f_i = 1 / {1 + [abs(residuals_i)/(alpha*sigma_i)]^beta}, with

        sigma_i = SQRT(variance_i)

        Parameters
        ----------
        alpha : int
            The parameter alpha in the above equation.
        beta : int
            The parameter beta in the above equation.

        Returns
        -------
        f : pandas.Series
             The factors by which the variances should be divided to get
             modified variances for the fitting process.
        """
        if self.fluxes.max() > 0:
            return pd.Series(1. / (1. + np.abs(self.current / (alpha*np.sqrt(self.variances)))**beta),
                             index=self.data.index)
        else:
            return pd.Series(1., index=self.data.index)

    def __call__(self, fluxes=None, psf_parameters_to_fit=None, position_parameters_to_fit=None,
                 sources=None, psf_sources=None, max_iterations=5, fit_psf_background=False, analyse_psf=False):
        """
        This method carries out a least-squares fir to the data.

        The process is iterative: The model is optimized for the fluxes, the
        PSF, and the positions of the sources sequentially. This loop is
        repeated until convergence or the maximum number of iterations has
        been reached.

        Parameters
        ----------
        fluxes : pandas.Series, optional
            Initial values for the fluxes of the components. If provided, its
            length must match the number of components provided upon instance
            initialisation (including background components). Note that the
            fit of the fluxes is a linear optimization, so no initial guesses
            are required.
        psf_parameters_to_fit : array_like, optional
            The names of the parameters of the PSF that should be optimized in
            the fit. Default is to fix all the PSF parameters to their initial
            values.
        position_parameters_to_fit : array_like or string, optional
            The names of the parameters of the coordinate transformation that
            should be optimized in the fit. Default is to fix all the
            parameters to their initial values.
            IMPORTANT: If no coordinate transformation is used and instead an
            optimization of the individual source positions are wanted, set
            this parameter to 'xy'.
        sources : pandas.Index, optional
            The indices of the sources used to optimize the PSF parameters and
            the source coordinates. If none are provided, all sources are being
            used.
        psf_sources : pandas.Index, optional
            This parameter may be used if only a subset of the sources provided
            as parameter 'sources' should be used to optimize the PSF. The idea
            behind this is that usually a higher S/N is required to get the
            shape of the PSF right than to get its centroid.
        max_iterations : int, optional
            The maximum number of iterations.
        fit_psf_background : bool, optional
            Flag indicating if a local background should be included in the
            individual PSF fits.
        analyse_psf : bool, optional
            Set this flag to return the radial profiles of the stars that were
            used to fit the PSF and position parameters. This parameter only
            has an effect if 'sources' is not None and at least one of the
            parameters is being fitted.

        Returns
        -------
        `FitResult` with the following fields defined:
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
        # sanity check if fit is possible
        if self.n_components + self.n_background > self.n_pixel:
            logger.error('Encountered ill-defined data. Returning NanNs ...')

            if position_parameters_to_fit is None:
                return_positions = pd.Series()
            elif 'xy' in position_parameters_to_fit:
                return_positions = pd.DataFrame(np.nan, columns=['xc', 'yc'], index=sources)
            else:
                return_positions = pd.Series(np.nan, index=position_parameters_to_fit)

            if psf_parameters_to_fit is not None:
                return_psf = pd.Series(np.nan, index=psf_parameters_to_fit)
            else:
                return_psf = pd.Series()

            return FitResult(pd.Series(np.nan, index=self.fluxes.index),
                             pd.Series(np.nan, index=self.fluxes.index),
                             return_positions, return_psf,
                             None if self.wave is None else pd.Series(np.nan, index=self.fluxes.index),
                             None, None, None, None)

        if fluxes is not None:
            if isinstance(fluxes, (list, np.ndarray)):
                fluxes = pd.Series(fluxes, index=self.fluxes.index)
            elif not isinstance(fluxes, pd.Series):
                logger.error('Parameter "fluxes" must be convertible to pandas Series.')
                return
            elif fluxes.size != (self.n_components + self.n_background):
                logger.error('No. of provided fluxes does not match number of components.')
                return
            self.fluxes = fluxes

        profiles = None
        single_psf_results = None
        residuals = None
        included_components = None

        # Determine initial weights for each pixel in the fit.
        self.fit_weights = 1./self.variances

        # start iterations
        for iteration in range(1, max_iterations + 1):

            if max_iterations > 1:
                logger.info("Iteration {0} (of max. {1})".format(iteration, max_iterations))

            # if required, calculate matrix with all sources
            if self.source_matrix is None or included_components is None:
                self.source_matrix, included_components = self.build_matrix()

                # fluxes of components outside FoV are set to NaN
                self.fluxes.loc[~self.fluxes.index.isin(included_components)] = np.nan

            # Fit fluxes by direct matrix inversion
            # first obtain fit matrix by multiplying source matrix with sqrt of fitting weights
            n, m = self.source_matrix.shape  # n=number of pixels, m=number of fit components
            fit_matrix = sparse.dia_matrix((np.sqrt(self.fit_weights), [0]), shape=(n, n)) * self.source_matrix
            # -> shape m, n

            # The documentation of scipy states that to efficiently solve the matrix equation, the columns should
            # be scaled to have the same euclidean norm. I am still not convinced that this is true, but scale the
            # columns nevertheless.
            total_flux = np.sqrt((fit_matrix.T*fit_matrix).diagonal())  # -> length m
            fit_matrix = fit_matrix * sparse.dia_matrix((1./total_flux, [0]), shape=(m, m))
            # -> shape m, n

            # The fit for the source fluxes is carried out on the residuals, not the data itself. The fitted residual
            # fluxes are added to the existing ones
            t1 = time.time()
            best_fit = linalg.lsmr(fit_matrix, self.current*np.sqrt(self.fit_weights), atol=1e-5, show=False)
            t2 = time.time()
            logger.debug("Time required to solve matrix equation: {0:6.3f}s".format(t2-t1))
            logger.info("Chi**2 of current fit: {0:.1f}".format(best_fit[3] ** 2))

            # update fluxes of all components
            self.fluxes[included_components] += (best_fit[0] / total_flux)

            # Convergence criterion: the average relative change in flux is < 0.5 percent
            delta_flux = (best_fit[0] / total_flux) / self.fluxes[included_components]
            with printoptions(precision=4, suppress=True):
                logger.debug("Relative changes in fluxes: {0}".format(delta_flux))
            if np.nanmedian(abs(delta_flux)) < 0.005:
                break

            # # for testing, show scatter plot with current residuals and model
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(8, 10))
            # ax = fig.add_axes([0.04, 0.52, 0.92, 0.46])
            # ax.scatter(self.data.loc[:, 'x'], self.data.loc[:, 'y'], marker='s', s=3, c=self.current,
            #            cmap=plt.cm.hot, edgecolors='face', vmin=-100, vmax=1000)
            # ax2 = fig.add_axes([0.04, 0.02, 0.92, 0.46])
            # ax2.scatter(self.data.loc[:, 'x'], self.data.loc[:, 'y'], marker='s', s=3,
            #             c=self.model.sum(axis=1).getA1(), cmap=plt.cm.hot, edgecolors='face', vmin=-100, vmax=1000)
            # plt.show()

            # carry out optimization of PSF and/or source positions
            if psf_parameters_to_fit is not None or position_parameters_to_fit is not None:

                # if any sources are provided for the PSF fit, the method for large layers is used
                if sources is not None:
                    single_psf_results, profiles, residuals = self.fit_sources_consecutively(
                        sources=sources, psf_parameters=psf_parameters_to_fit, psf_sources=psf_sources,
                        fit_positions=(position_parameters_to_fit is not None), local_background=fit_psf_background,
                        return_psf_profiles=analyse_psf | (self.lut is not None))

                    if self.lut is not None:
                        self.lut = self.update_look_up_table(residuals=residuals, centroids=single_psf_results,
                                                             weights=single_psf_results['snr'])
                        [setattr(psf, 'lut', self.lut) for psf in self.psf_instances.values()]

                    # if positions are fitted directly, update them
                    if position_parameters_to_fit is not None and 'xy' in position_parameters_to_fit:
                        for index, row in single_psf_results.iterrows():
                            if np.isfinite(row['xc']) and np.isfinite(row['yc']):
                                self.psf_instances[index].xc = row['xc']
                                self.psf_instances[index].yc = row['yc']

                    # get average values from individual fits as pandas Series
                    if position_parameters_to_fit is None or 'xy' in position_parameters_to_fit:
                        _position_parameters_to_fit = None
                    else:
                        _position_parameters_to_fit = position_parameters_to_fit
                    final = self.average_psf_fits(
                        single_psf_results, psf_parameters_to_fit, _position_parameters_to_fit)

                else:
                    final = pd.Series()
                    # first fit positions
                    if position_parameters_to_fit is not None:
                        final = final.append(self.fit_sources_simultaneously(position_parameters_to_fit))
                    # then PSF parameters
                    if psf_parameters_to_fit is not None:
                        final = final.append(self.fit_sources_simultaneously(psf_parameters_to_fit))

                # update values in PSF instances
                parameters = None if position_parameters_to_fit is None else position_parameters_to_fit[:]
                if parameters is None:
                    parameters = psf_parameters_to_fit[:]
                elif psf_parameters_to_fit is not None:
                    parameters += psf_parameters_to_fit
                for parameter in parameters:
                    if parameter == 'xy':
                        continue
                    elif parameter in final and np.isfinite(final[parameter]):
                        value = final[parameter]
                        try:
                            [setattr(psf, parameter, value) for psf in self.psf_instances.values()]
                        except AssertionError:
                            logger.error('PSF fit yielded invalid value {0:.1f} for parameter {1}.'.format(
                                parameter, value))
                        else:
                            logger.info('Updated parameters "{0}" to {1}.'.format(parameter, value))
                    else:
                        logger.error('Cannot update parameter "{0}" because fit yielded NaN.'.format(parameter))

                # PSF and/or positions have changed, so the matrix needs to be recalculated (unless end of iteration)
                if iteration < max_iterations:
                    self.source_matrix = None

            elif iteration < max_iterations:
                # re-calculate fitting weights
                self.fit_weights = self.get_weight_factors() / self.variances
            else:
                logger.warning('Maximum number of iterations reached.')

        # estimate uncertainties that are returned with the data
        # note that fit_weights ~ 1/variances
        inverse_variances = pd.Series(np.nan, index=self.fluxes.index)
        inverse_variances[included_components] = self.source_matrix.multiply(self.source_matrix).T.dot(self.fit_weights)
        # invert and take square root to get uncertainties
        uncertainties = 1./np.sqrt(inverse_variances)

        # collect data that is returned as fit result
        i = self.catalog.index[0]  # extract final parameters from first PSF index
        if psf_parameters_to_fit is not None:
            return_psf = pd.Series([getattr(self.psf_instances[i], p) for p in psf_parameters_to_fit],
                                   index=psf_parameters_to_fit)
        else:
            return_psf = pd.Series()
        if position_parameters_to_fit is None:
            return_positions = pd.Series(dtype=np.float64)
        elif 'xy' in position_parameters_to_fit:
            return_positions = single_psf_results[['xc', 'yc']]
        else:
            return_positions = pd.Series([getattr(self.psf_instances[i], p) for p in position_parameters_to_fit],
                                         index=position_parameters_to_fit)

        if self.wave is not None:
            # calculate variance weighted wavelength of each component
            # first combine PSF contribution and variance information to final weight for each pixel
            n = self.source_matrix.shape[0]
            weight_matrix = sparse.dia_matrix((self.fit_weights, [0]), shape=(n, n)) * self.source_matrix
            # multiply wavelength information with weight matrix and divide by total weight to get final wavelength
            return_wave = pd.Series(np.nan, index=self.fluxes.index)
            return_wave[included_components] = np.asarray(self.wave.dot(weight_matrix)/weight_matrix.sum(axis=0))[0]
        else:
            return_wave = None

        return FitResult(self.fluxes, uncertainties, return_positions, return_psf, return_wave, profiles,
                         single_psf_results, residuals, self.lut)

    @property
    def model(self):
        """

        Returns
        -------
        The current best-fit model.
        """
        fluxes = self.fluxes[np.isfinite(self.fluxes)]
        return self.source_matrix * sparse.dia_matrix((fluxes.values, [0]), shape=(self.source_matrix.shape[1],
                                                                                   self.source_matrix.shape[1]))

    @property
    def current(self):
        """

        Returns
        -------
        The residuals of the current best fit model.
        """
        return self.data['value'] - self.model.sum(axis=1).getA1()

    def residuals_from_fluxes(self, fluxes):
        """
        This method returns the fit residuals determined from provided input
        values for the fluxes.
        
        The purpose of this method is to have a routine for fast residuals
        calculation as is required by the SUBTRES routine in PampelMuse.

        Parameters
        ----------
        fluxes : pd.Series
            The fluxes of the components. The length of the Series must match
            the number of components of the instance (including background
            components if any).

        Returns
        -------
        residuals : nd_array
            The model residuals calculated using the input fluxes.
        """
        # consistency checks on provided fluxes
        if isinstance(fluxes, (list, np.ndarray)):
            fluxes = pd.Series(fluxes, index=self.fluxes.index)
        elif not isinstance(fluxes, pd.Series):
            logger.error('Parameter "fluxes" must be convertible to pandas Series.')
            return
        elif fluxes.size != (self.n_components + self.n_background):
            logger.error('No. of provided fluxes does not match number of components.')
            return
        self.fluxes = fluxes

        # calculate source matrix
        self.source_matrix, included_components = self.build_matrix()

        # return residuals
        return self.current

    def build_matrix(self, update_unresolved=True, sources=None):
        """
        (Re)calculate the source matrix.

        The source matrix is a sparse matrix that contains the spatial
        profiles of all requested components, one per column.

        The method initially creates three empty arrays to hold
        (a) the source (PSF) coefficients, divided by the pixel uncertainties
        (b) the (1dim.) coordinates of the coefficients on the data array
            (they are the row indices of the final PSF coefficient matrix)
        (c) the component to which each source belongs
            (they are the column indices of the final PSF coefficient matrix)

        It then loops over all sources in the 'Sources'-instance, obtains
        their PSF and adds it to the respective column of the array. After-
        wards, the arrays are combined into a single sparse matrix.

        Parameters
        ----------
        update_unresolved : boolean, optional
            If False, the matrix column containing the unresolved stellar
            component (if included) is not recomputed, but taken from a
            previous calculation.
        sources : pandas.Index, optional
            The indices of the sources that should be included in the matrix.
            The default is to include all sources (minus the unresolved if
            'update_unresolved' is set to False).

        Returns
        -------
        source_matrix : a scipy.sparse.csc_matrix-instance
            This is the PSF coefficient matrix that is used in the linear
            least-squares fit to determine the source fluxes.
        included_components : pandas.Index
            The indices of the components that actually contribute to the
            source matrix, i.e. are defined on any of the data pixels.
        """
        if not self.last_component_is_unresolved:
            update_unresolved = True

        t0 = time.time()

        # result will be a list with length equal to the number of sources, each entry is again a list containing two
        # entries: the spaxels on which the PSF is defined and the corresponding fluxes
        pixels = []
        components = []
        fluxes = []

        # create a pandas DataFrame to calculate the integrated fluxes of all components
        psf_fluxes = pd.DataFrame({'flux': pd.Series(0., index=self.catalog.index),
                                   'component': self.catalog['component']})

        # define which sources should be included
        if not update_unresolved:
            resolved_sources = self.catalog[self.catalog['component'] < self.catalog['component'].max()].index
            if sources is None:
                sources = resolved_sources
            else:
                sources = pd.Index(np.intersect1d(sources, resolved_sources))
        elif sources is None:
            sources = self.catalog.index

        # loop over sources & extract information from PSF instances
        for index in sources:
            psf = self.psf_instances[index]
            current_component = self.catalog.at[index, 'component']

            if self.offsets is not None and index in self.offsets.index:
                psf.uc = self.catalog.at[index, 'x'] - self.offsets.at[index, 'x']
                psf.vc = self.catalog.at[index, 'y'] - self.offsets.at[index, 'y']

            # scale PSF radius based on flux (if requested) - the minimum PSF radius is set to (0.5^3) x PSF radius
            if np.isfinite(self.fluxes[current_component]) and self.fluxes.max() > 0 and self.psf_radius_scaling:
                flux_ratio = max(self.psf_radius_scaling**3, self.fluxes[current_component]/self.fluxes.max())
                radius = self.psf_radius*(2**(np.log10(flux_ratio)/np.log10(1./self.psf_radius_scaling)))
            else:
                radius = self.psf_radius
            psf.maxrad = radius

            current_pixels, current_fluxes = psf.get_flux(apply_lut=True)
            if current_pixels.size == 0:
                logger.warning("Source #{0} entirely outside FoV.".format(index))

            pixels.extend(current_pixels)
            fluxes.extend(current_fluxes)
            components.extend(len(current_pixels)*[current_component])

            psf_fluxes.at[index, 'flux'] = current_fluxes.sum()

        matrix_data = pd.DataFrame({'i0': pixels, 'i1':  components, 'flux': fluxes})

        t1 = time.time()
        logger.debug("Time required to calculate PSF profiles: {0:.3f}s".format(t1 - t0))

        # Calculate total flux of each component
        total_fluxes = psf_fluxes.groupby(by='component').agg(np.sum)['flux']

        t2 = time.time()
        logger.debug("Time required to combine components: {0:.3f}s".format(t2 - t1))

        # check if any row does not contribute at all
        empty_components = total_fluxes[total_fluxes <= 0].index
        if len(empty_components) > 0:
            logger.warning("Encountered component(s) that contribute zero flux: {0}".format(empty_components))

            # if so, reduce index number of higher components (start with highest empty component to avoid gaps)
            for i1 in np.sort(empty_components.values)[::-1]:
                matrix_data.loc[matrix_data['i1'] > i1, 'i1'] -= 1

        # create matrix: first, a COO matrix is constructed. Note that it handles the case when the input data contains
        # multiple entries for the some indices (row, column). After construction, the matrix is converted to a
        # CSC matrix to speed up future usage of it.
        n_rows = self.data.shape[0]
        n_columns = self.n_components - len(empty_components) + self.n_background

        # handle the unresolved component
        if self.last_component_is_unresolved:

            # need to take into account that the row of the unresolved component in the matrix changes if some
            # components do not contribute
            unresolved_component = self.n_components - 1 - len(empty_components)

            if update_unresolved or self._unresolved_matrix is None:
                # store the unresolved sources as a list for later re-use
                slc = (matrix_data['i1'] != unresolved_component)
                self._unresolved_matrix = sparse.coo_matrix(
                    (matrix_data.loc[~slc, 'flux'], (matrix_data.loc[~slc, 'i0'], matrix_data.loc[~slc, 'i1'])),
                    shape=(n_rows, n_columns))

                # then add it to matrix of resolved sources
                _psf_matrix = sparse.coo_matrix(
                    (matrix_data.loc[slc, 'flux'], (matrix_data.loc[slc, 'i0'], matrix_data.loc[slc, 'i1'])),
                    shape=(n_rows, n_columns)) + self._unresolved_matrix
            else:
                # create matrix for resolved sources, add matrix for unresolved ones
                _psf_matrix = sparse.coo_matrix(
                    (matrix_data['flux'], (matrix_data['i0'], matrix_data['i1'])),
                    shape=(n_rows, n_columns)) + self._unresolved_matrix
        else:
            # create full matrix
            _psf_matrix = sparse.coo_matrix((matrix_data['flux'], (matrix_data['i0'], matrix_data['i1'])),
                                            shape=(n_rows, n_columns))

        # add background matrix if available
        if self.n_background > 0:
            if not len(empty_components):
                _psf_matrix = _psf_matrix + self.background
            else:
                # if PSF matrix contains unresolved components, need to adapt indices of background matrix
                used = ~np.in1d(np.arange(self.n_components + self.n_background), empty_components)
                _psf_matrix = _psf_matrix + self.background.tocsc()[:, used].tocoo()

        t3 = time.time()
        logger.debug("Time required to create matrix: {0:.3f}s".format(t3 - t2))
        logger.debug("Full source matrix creation took {0:.2f}s.".format(t3 - t0))

        # return CSC matrix and indices of components that contribute
        source_indices = total_fluxes[total_fluxes > 0].index
        if self.n_background > 0:
            source_indices = source_indices.append(self.fluxes.index[-self.n_background:])

        return _psf_matrix.tocsc(), source_indices

    def fit_sources_consecutively(self, sources, psf_parameters=None, fit_positions=True, psf_sources=None,
                                  local_background=False, return_psf_profiles=True):
        """
        Perform the PSF fit for one or more sources individually after
        subtracting all other sources using their current best fit.

        Depending on the respective fit-flags, either the parameters of the
        PSF and/or the source coordinates are fitted. The results of the
        individual fits are used afterwards to update the PSF instances.

        This approach is recommended to determine the PSF in a layer that
        contains many sources (approx. more than hundred).

        Parameters
        ----------
        sources : pandas.Index
            The indices of the sources that should be fitted.
        psf_parameters : array_like, optional
            The names of the PSF parameters that should be optimized in the
            fits. Default is to use a constant PSF.
        fit_positions : boolean, optional
            Flag indicating if the source positions should be optimized in
            the fits.
        psf_sources : pandas.Series, optional
            The indices of the sources used to constrain the PSF parameters.
            This is useful if some of the provided 'sources' are too faint for
            this purpose. In this case, only the fluxes and centroids of the
            provided 'sources' that are not included in 'psf_sources' will be
            optimized. If None, use all 'sources' to constrain the PSF
            parameters (if requested).
        local_background : bool, optional
            Flag indicating if the local background should be determined prior
            to each PSF fit.
        return_psf_profiles : bool, optional
            Flag indicating if the radial profiles of the stars used in the
            PSF fits should be returned.

        Returns
        -------
        results : pandas.DataFrame
            The data frame containing the fit results, including one row per
            provided source and the following columns, where square brackets
            indicate columns that are only included if the corresponding
            parameters have been fitted:
            * 'flux': the fitted flux of each source.
            * 'snr': an estimate of the S/N of each source.
            * ['fwhm', 'e', 'theta', ...: values of the free PSF parameters.]
            * ['xc', 'yc': the x- and y-centroids obtained in the fits]
        profiles : pandas.DataFrame
            The radial profiles of the stars provided as 'sources' in a
            combined data frame that uses the radii as Index and the source
            IDs as column names. If 'return_psf_profiles' is not set, None is
            returned instead.
        residuals : dict
            The residuals of the successful PSF fits. The dictionary contains
            one entry per successful PSF fit. The source IDs are used as keys
            and the values are data frames containing the observed flux and
            the fit residuals for the pixels on which the PSF is defined. Only
            returned if 'return_psf_profiles' was set. Otherwise None is
            returned.
        """
        # The keyword 'psf_sources' may be used to define a subset of 'sources' that is used to fit the PSF parameters.
        # All 'sources' are used for this purpose if psf_sources=None
        if psf_sources is None:
            psf_sources = sources

        # prepare array that will hold the results
        # first collect columns that need to be included in the array
        columns = ['snr', 'flux']
        if psf_parameters is not None:
            columns.extend(psf_parameters)
        if fit_positions:
            columns.extend(['xc', 'yc'])
        if local_background:
            columns.append('back')

        # prepare empty data frame & fill with NaNs
        results = pd.DataFrame(0., index=sources, columns=columns)
        results[:] = np.nan

        # if PSF profiles and residuals should be returned, prepare empty dictionaries for this purpose
        radial_data = {} if return_psf_profiles else None
        residuals = {} if return_psf_profiles else None

        # Loop over all sources with free parameters; add source to residuals, fit resulting array and subtract source
        # again.
        for i in sources:
            psf = self.psf_instances[i]

            # for fit, get residuals and variances for pixels where the current PSF is defined
            used = psf.used
            data = self.current[used]
            weights = self.fit_weights[used]

            # add best-fit PSF-profile to residuals to get data used in fit
            data += self.fluxes[self.catalog.at[i, 'component']] * psf.get_f(apply_lut=True)

            # calculate radial profile if requested
            if return_psf_profiles:
                # calculate weight of every sub-pixel (=inverse of number of sub-pixels in pixel)
                _, unique_inverse, unique_counts = np.unique(psf.osmap, return_inverse=True, return_counts=True)
                weights = 1./unique_counts[unique_inverse]
                # account for sub-pixels by duplicating data of every pixel containing sub-pixels.
                os_data = data[self.data.index[psf.osmap]]
                # calculate profile
                profile = self._get_radial_profile(data=os_data, radii=psf.r, weights=weights)
                radial_data[i] = profile  # contains radii and values

            # # for debugging, show plot with current data
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(5, 8))
            # ax1 = fig.add_axes([0.12, 0.54, 0.83, 0.44])
            # ax1.scatter(psf.x[used], psf.y[used], marker='o', s=50, c=data, cmap=plt.cm.hot, vmin=-1000,
            #             vmax=10000)
            # ax2 = fig.add_axes([0.12, 0.05, 0.83, 0.44])
            # ax2.scatter(psf.x[used], psf.y[used], marker='o', s=50,
            #             c=self.current[used], cmap=plt.cm.hot, vmin=-1000, vmax=10000)
            # plt.show()

            # prior to fit, check if enough pixels available and source is actually detected
            if data.size <= (len(psf_parameters) + 1 + local_background + 2*fit_positions):
                # insufficient number of pixels, return initial guess
                logger.error('Cannot perform PSF fit for source #{0}: N_pixels <= N_parameters.'.format(i))
                continue

            # also, check if there is any significant peak inside a circle with radius FWHM centred on the initial
            # guess
            pixel = np.unique(psf.osmap, return_index=True)[1]  # this makes sure every sub-pixel only counted once
            in_fwhm = (psf.r[pixel] < psf.fwhm)

            if np.sum(in_fwhm) == 0:
                peak = 0
                noise = 1
            elif np.sum(~in_fwhm) >= 5:
                peak = data[in_fwhm].max() - data[~in_fwhm].mean()
                noise = np.std(data[~in_fwhm])
            else:
                peak = data.max() - data.min()
                noise = np.median(data) - data.min()
            if peak/noise <= 3 and i in psf_sources:
                logger.error('Cannot perform PSF fit for source #{0}: No significant peak detected.'.format(i))
                continue

            # TODO: get initial guess for local background as median of residuals outside FWHM
            # if local_background:
            #     x0.append(np.median(data[~in_fwhm]))

            result = psf.fitter(psf=data, weights=weights, parameters=psf_parameters, fit_positions=fit_positions,
                                fit_background=local_background, flux_x0=self.fluxes[self.catalog.at[i, 'component']],
                                psf_padding=int(round(self.psf_radius-psf.maxrad)))

            # check if result of a successful fit makes any sense:
            if not result.success:
                logger.warning("PSF fit failed for source #{0}: {1}".format(i, result.message))
                continue
            # fitted flux must be positive
            elif result.flux <= 0:
                logger.warning('PSF fit for source #{0} returned negative flux.'.format(i))
                continue
            # offset between fitted centroid and initial guess must be <= FWHM
            elif fit_positions:
                xc, yc = result.centroid
                offset = np.sqrt((xc - psf.xc)**2 + (yc - psf.yc)**2)
                if offset > psf.fwhm:
                    logger.warning(
                        'PSF fit for source #{0} yielded offset ({1:.2f}) > FWHM ({2:.2f}) wrt. initial guess.'.format(
                            i, offset, psf.fwhm))
                    continue

            # add results to data frame containing results and create log message summarizing fit
            results.at[i, 'snr'] = result.flux * np.sqrt(sum((psf.get_f()**2)*self.fit_weights[psf.used]))
            results.at[i, 'flux'] = result.flux
            logger.info('Successfully fitted PSF of source #{0}: flux={1:.1f} (S/N={2:.1f})'.format(
                i, results.at[i, 'flux'], results.at[i, 'snr']))
            if i in psf_sources and psf_parameters is not None:
                results.loc[i, psf_parameters] = result.values
                _string = ", ".join(
                    ["{0}={1:.2g}".format(result.names[k], result.values[k]) for k in range(len(result.names))])
                logger.info(' Best-fit PSF parameters: {0}'.format(_string))
            if fit_positions:
                results.loc[i, ['xc', 'yc']] = result.centroid
                logger.info(' Centroid coordinates: xc={0:.2f}, yc={1:.2f}'.format(results.at[i, 'xc'],
                                                                                   results.at[i, 'yc']))
            if local_background:
                results.at[i, 'back'] = result.background
                logger.info(' Local background: {0:.1f}'.format(result.background))

            # if requested, provide residuals
            if return_psf_profiles:
                residuals[i] = pd.DataFrame({'data': data, 'residuals': result.residuals},
                                            index=self.data.index[psf.used])

        # return
        return results, self._combine_psf_profiles(radial_data) if return_psf_profiles else None, residuals

    # @staticmethod
    # def single_psf_optimizer(x0, names, psf, data, sigma):
    #     """
    #     This is the optimization routine that is passed to the Levenberg-
    #     Marquardt solver scipy.optimize.leastsq.
    #
    #     During each iteration, a PSF array is created, scaled and subtracted
    #     from the data. The residuals of the fit are returned.
    #
    #     Parameters
    #     ----------
    #     x0 : array_like
    #         The initial guesses of the parameters that are fitted. The first
    #         parameter must be the flux of the source.
    #     names :  array_like
    #         The names of the parameters that are fitted. Must have same length
    #         as 'x0' of course.
    #     psf : an instance of class pampelmuse.core.psf.profiles.Profile or
    #           one of its subclasses
    #         The PSF model for the source.
    #     data : nd_array
    #         The data array used to fit the parameters. Note that the pixel
    #         values should be normalized by their respective uncertainties.
    #     sigma : nd_array
    #         The uncertainties of the 'data' array pixels.
    #
    #     Returns
    #     -------
    #     nd_array
    #         The sigma-weighted residuals after subtracting the PSF from the
    #         data.
    #     """
    #     # get local background if provided
    #     fit_background = names[-1] == 'back'
    #     back = x0[-1] if fit_background else 0
    #
    #     # update PSF using current parameter estimates
    #     for i in range(1, len(x0) - int(fit_background)):
    #         try:
    #             # the attribute is only set in the 'psf'-instance of the class, not in all instances! This is different
    #             # from the behaviour of getattr(self.PSF, "set_<PAR>")(<VALUE>)
    #             setattr(psf, names[i], x0[i])
    #         except AssertionError as error:  # if parameter cannot be updated, raise LinAlgError
    #             logger.error("Encountered AssertionError: {0}".format(error))
    #             raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], names[i]))
    #
    #     return data / sigma - x0[0] * (psf.get_f() / sigma) - back/sigma
    #
    # @staticmethod
    # def single_psf_jacobi(x0, names, psf, data, sigma):
    #     """
    #     Calculate the Jacobi matrix for the optimization carried out in
    #     StarFit.optimize_single_psf.
    #
    #     The method is passed to the Levenberg-Marquard solver
    #     scipy.optimize.leastsq as parameter Dfun.
    #
    #     Parameters
    #     ----------
    #     x0
    #     names
    #     psf
    #     data
    #     sigma
    #         This method takes the same arguments as StarFit.optimize_single_psf.
    #
    #     Returns
    #     -------
    #     jacob : nd_array
    #         The Jacobi matrix with respect to the model parameters included
    #         in 'x0'.
    #     """
    #     fit_background = names[-1] == 'back'
    #
    #     jacob = np.zeros((len(x0), psf.used.sum()), dtype=np.float64)
    #     jacob[0] -= psf.get_f()
    #     for i in range(1, len(x0) - int(fit_background)):
    #         diff_i = getattr(psf, "get_diff_{0}".format(names[i]))()
    #         jacob[i] -= x0[0] * diff_i
    #     if fit_background:
    #         jacob[-1] = x0[-1]
    #
    #     # Note that to get the right shape for the optimization, the matrix needs to be transposed.
    #     return (jacob / sigma).T

    def fit_sources_simultaneously(self, parameters, sources=None):
        """
        Perform the PSF fit for several sources simultaneously using a
        Levenberg-Marquardt optimization.

        This approach is recommended to optimize the parameters of the PSF or
        the coordinate transformation in a layer that contains only a small
        number of sources (approx. less than hundred).

        Parameters
        ----------
        sources : pandas.Index, optional
            The indices of the sources that should be included in the fit. The
            default is to use all sources.
        parameters : array_like
            The names of the parameters that should be optimized during the
            fit. Any parameter that is defined for each instance of the
            current PSF model (including those of a coordinate transformation)
            may be provided.

        Returns
        -------
        results : pd.Series
            The best-fit values for the provided parameters. The parameter
            names are used as indices of the pandas Series that is returned.
        """
        # check which sources are included:
        if sources is None:
            sources = self.catalog.index
        elif not isinstance(sources, pd.Index):
            sources = pd.Index(sources)

        # Fit position parameters
        logger.info('Optimizing parameters "{0}" in a multi-PSF fit ...'.format(", ".join(parameters)))

        # collect initial guesses, using PSF instance of first provided source
        x0 = [getattr(self.psf_instances[sources[0]], name) for name in parameters]

        # as additional input argument for the fitting routine, source indices are needed
        args = (parameters, sources)

        # call Levenberg-Marquardt solver from scipy.optimize (& provide Jacobian)
        try:
            best_fit = optimize.leastsq(self.multi_psf_optimizer, x0=np.asarray(x0), Dfun=self.multi_psf_jacobi,
                                        col_deriv=True, epsfcn=0.0, args=args)[0]
        # error handler to avoid code that code crashes
        except np.linalg.linalg.LinAlgError as msg:  # optimize.minpack.error:
            # seems to depend on numpy version which error is actually thrown
            logger.error("PSF fit failed: {0}".format(msg))
            #  fill results array with NaNs
            best_fit = len(x0)*[np.nan, ]
        else:
            # log fit results
            log_string = "Fitted parameter values:"
            for i, name in enumerate(parameters):
                log_string += " {0}={1:.2f},".format(name, best_fit[i])
            logger.info(log_string[:-1])

        # return
        return pd.Series(best_fit, index=parameters)

    def multi_psf_optimizer(self, x0, names, sources):
        """
        The optimization routine that is passed to the 'leastsq'-routine of
        the 'scipy.optimize'-package.

        During each iteration, a PSF array is prepared for every resolved
        source, the PSFs are scaled and subtracted from the data. The
        residuals are returned.

        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted.
        names : array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        sources : pandas.Index
            The indices of the sources included in the fit.

        Returns
        -------
        residuals : nd_array
            The residuals of the fit.
        """
        # logger.debug("Shape of Jacobi matrix: {0}".format(self.jacobi_multi_psf(x0, names).shape))
        # logger.debug("Shape of data: {0}".format(self.data.shape))
        # update parameters in PSF-class
        for i, name in enumerate(names):
            try:
                [setattr(self.psf_instances[j], name, x0[i]) for j in sources]
            except AssertionError:  # if parameter cannot be updated, raise LinAlgError
                raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], name))

        # calculate new model
        psf_matrix, included_components = self.build_matrix(update_unresolved=False, sources=sources)

        # inverse variance weighting of each pixel
        n = psf_matrix.shape[0]
        psf_matrix = sparse.dia_matrix((np.sqrt(self.fit_weights), [0]), shape=(n, n)) * psf_matrix
        # psf_matrix.data *= np.take(np.sqrt(self.fit_weights), psf_matrix.indices)

        # remove fluxes of components that do not contribute
        # if len(included_components) < len(fluxes.index):
        fluxes = self.fluxes[included_components]

        # calculate model matrix
        n_comp = included_components.size
        model = psf_matrix * sparse.dia_matrix((fluxes, [0]), shape=(n_comp, n_comp))

        # return residuals
        return (self.data['value'] * np.sqrt(self.fit_weights) - model.sum(axis=1).getA1()).values

    def multi_psf_jacobi(self, x0, names, sources):
        """
        Routine that calculates the Jacobi matrix of the residuals. This method
        is parsed to the Levenberg-Marquard solver scipy.optimize.leastsq a
        parameter Dfun when the PSF of multiple sources is fitted simultaneously.

        Parameters
        ----------
        x0
        names
        sources
            The method takes the same arguments as StarFit.optimize_multi_psf

        Returns
        -------
        jacob : nd_array
            The Jacobi matrix with respect to the model parameters included
            in 'x0'.
        """
        t0 = time.time()
        jacob = np.zeros((len(x0), self.data.shape[0]), dtype=np.float64)

        # The influence of the unresolved sources (if any) is currently ignored when calculating the Jacobi matrix,
        # any sky component has no influence anyway.
        if self.last_component_is_unresolved:
            resolved_sources = self.catalog[self.catalog['Component'] < self.catalog['Component'].max()].index
            used_sources = pd.Index(np.intersect1d(sources, resolved_sources))
        else:
            used_sources = sources

        # loop over sources
        for i in used_sources:
            # get PSF instance
            psf = self.psf_instances[i]

            #  get component
            ii = self.catalog.at[i,  'component']

            # for each parameter, subtract partial derivative of PSF profile (x flux) from Jacobi matrix
            for j in range(len(x0)):

                diff_i = getattr(psf, "get_diff_{0}".format(names[j]))()
                jacob[j, psf.used] -= self.fluxes[ii] * (diff_i * np.sqrt(self.fit_weights[psf.used].values))

        t1 = time.time()
        logger.debug("Time needed to create Jacobi matrix: {0:.3f}s".format(t1 - t0))

        return jacob

    @staticmethod
    def _update_coordinate_transformation(xy_reference, xy_true, fixed=None):
        """
        This method uses a set of reference and best-fit coordinates to obtain
        a new coordinate transformation.

        Parameters
        ----------
        xy_reference : pandas.DataFrame
            The reference coordinates of the sources, provided as pandas
            DataFrame containing aat least the columns 'x' and 'y'.
        xy_true : pandas.DataFrame
            The best-fit (or true) coordinates of the same N sources, again
            provided as pandas DataFrame containing the columns 'xc' and 'yc'.
        fixed : dictionary, optional
            The names and values of the parameters that should be fixed to
            their given values when determining the new coordinate
            transformation.

        Returns
        -------
        best_fit : pandas.Series
            The results of the least squares fit.
        offsets : pandas.DataFrame
            The offsets relative to the reference coordinates when applying
            the inverted coordinate transformation to the fitetd coordinates.

        Raises
        ------
        IOError
            If provided data does not allow to fit the coordinate
            transformation.
        """
        # need to change structure of array with best-fit coordinates; source IDs and (x,y) coordinates must be two
        # levels of a pandas MultiIndex
        xy_out = pd.DataFrame(xy_true[['xc', 'yc']].values.reshape(1, -1),
                              columns=pd.MultiIndex.from_product([xy_true.index, ['x', 'y']]))

        # need to convert fixed parameters dictionary to pandas DataFrame
        if fixed is not None and len(fixed) > 0:
            fixed = pd.DataFrame(fixed, index=xy_out.index)
        else:
            fixed = None

        try:
            tf = Transformation.from_coordinates(xy_in=xy_reference[['x', 'y']], xy_out=xy_out, fixed=fixed)
        except IOError:
            logger.error('Fit of coordinate transformation failed.')
            return None

        # use fitted coordinate transformation to determine reference coordinates corresponding to best-fit
        # coordinates
        xy_inverted = tf.inverse(xy_out)
        xy_reference_recovered = pd.DataFrame(xy_inverted.values.reshape((-1, 2)),
                                              columns=xy_inverted.columns.levels[1],
                                              index=xy_inverted.columns.levels[0])

        # calculate offsets for sources
        indices = np.intersect1d(xy_reference.index, xy_reference_recovered.index, assume_unique=True)
        offsets = xy_reference.loc[indices, ['x', 'y']] - xy_reference_recovered.loc[indices, ['x', 'y']]

        # return Series containing only fitted parameters
        best_fit = tf.data.loc[tf.data.index[0], (tf.data.columns.levels[0][tf.free], 'value')]

        return best_fit.droplevel(level=1), offsets

    @staticmethod
    def calc_running_mad(x, y, n=500):
        """
        Given two 1D arrays x and y, the routine will determine the median
        absolute deviation (MAD) of y as a function of x.
        
        To this aim, the data are sorted by increasing x values and then the
        MAD is measured in bins of n datapoints with increasing x values.

        NOTE: It is assumed that the median value of y is zero.

        Parameters
        ----------
        x : array_like
            The x-values of the data, used for sorting
        y : array_like
            The y-values of the data, used to determine the MAD values.
        n : int, optional
            The number of bins in which the MAD is calculated

        Returns
        -------
        xbin : ndarray
            The value of x at the centre of each bin.
        ybin :  ndarray
            The MAD determined for each bin.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        indices = np.argsort(x)

        bincenters = np.arange(n//2, indices.size, n)

        xbin = np.zeros(bincenters.size, dtype=np.float32)
        ybin = np.zeros(bincenters.size, dtype=np.float32)

        for i, j in enumerate(bincenters):
            imin = j - n//2
            imax = min(j + n//2, len(x))
            xbin[i] = x[indices[j]]

            # use robust median absolute deviation (MAD) around an initial median value of zero
            cury = y[indices[imin:imax]]
            ybin[i] = np.nanmedian(abs(cury))
        return xbin, ybin

    def average_psf_fits(self, results, psf_parameters=None, position_parameters=None):
        """
        Average the results of several PSF fits.

        The purpose of this method is first to get mean values of the PSF
        parameters that were obtained in single-source PSF fits. If available,
        the averaging uses the S/N of each source as weight factor.

        In addition, if fitted centroid coordinates are provided for the
        sources, the code will also obtain a new coordinate transformation.

        Parameters
        ----------
        results : pandas.DataFrame
            The results from the individual fits, containing one row per
            fitted source and one column per fitted PSF parameter. If it also
            contains the columns 'x' and 'y' and 'position_parameters' is not
            None, the code will fit a coordinate transformation.
        psf_parameters : array_like, optional
            The list of coordinate transformation parameters that should be
            fitted.
        position_parameters : array_like, optional
            The list of PSF parameters that have been fitted.

        Returns
        -------
        final : pd.Series
            The averaged values of the PSF parameters and/or the parameters
            of the coordinate transformation obtained.
        """
        # create empty Series
        final = pd.Series(dtype=np.float64)

        # update PSF parameters, using S/N as weights:
        if psf_parameters is not None:
            _final = pd.Series(0., index=psf_parameters)
            for name in psf_parameters:
                # TODO: properly implement code from next three lines
                from astropy.stats import sigma_clip
                from scipy.stats import median_absolute_deviation as mad
                _, vmin, vmax = sigma_clip(results[name], return_bounds=True,
                                           stdfunc=lambda x, axis: mad(x, nan_policy='omit', axis=axis))

                valid = (results[name] > vmin) & (results[name] < vmax)
                # valid = np.isfinite(results[name])
                if valid.any():
                    _final[name] = np.average(results.loc[valid, name], weights=results.loc[valid, 'snr'])
                else:
                    logger.warning('Cannot calculate mean of "{0}" because no PSF fit succeeded.'.format(name))
                    _final[name] = np.nan
            final = final.append(_final)

        # fit coordinate transformation to provided centroid coordinates
        if position_parameters is not None and 'xy' not in position_parameters:

            # check if any parameters were not provided. If so, get fixed values from PSF of first source
            index = self.catalog.index[0]
            fixed = {}
            for parameter in Transformation.PARAMETERS:
                if parameter not in position_parameters:
                    fixed[parameter] = getattr(self.psf_instances[index], parameter)
            # select valid fits
            valid = np.isfinite(results[['xc', 'yc']]).all(axis=1)

            # perform fit
            try:
                _final, self.offsets = FitLayer._update_coordinate_transformation(
                    self.catalog, results.loc[valid, ['xc', 'yc']], fixed=fixed)
            except IOError:
                logging.error('Fit of coordinate transformation failed.')
            else:
                final = final.append(_final)

        return final

    def _get_radial_profile(self, data, radii, weights=None, sampling=None, median=False):
        """
        Internal method to determine the radial profile of the data used in a
        PSF fit.
        
        Parameters
        ----------
        data : nd_array
            The data used in the PSF fit.
        radii : nd_array
            The distance of each pixel to the centre of the profile. Must have
            same length as data array.
        weights : nd_array, optional
            An optional weighting applied to each data pixel. This is useful
            if the central pixels were oversampled, for example. If provided,
            must have same length as data array. Default is to use no
            weighting. 
        sampling : array-like
            The edges of the radial bins in which the profile is measured.
            The default is to use bins of 1 pixel width out to the PSF radius.
        median : bool, optional
            Set this flag to use median instead of weighted mean to average
            data.

        Returns
        -------
        profile : pandas.Series
            The radial profile determined from the data.
        """
        # specify bin edges if they are not provided.
        if sampling is None:
            sampling = np.arange(self.psf_radius)

        # use unity weights if none provided
        if weights is None:
            weights = np.ones(data.size, dtype=np.float32)

        # prepare arrays to hold results
        output_radii = np.zeros(len(sampling) - 1, dtype=np.float32)
        profile = np.zeros(len(sampling) - 1, dtype=np.float32)

        for i, r_min in enumerate(sampling[:-1]):
            r_max = sampling[i + 1]

            in_bin = (radii >= r_min) & (radii <= r_max) & np.isfinite(data)
            if in_bin.any():
                output_radii[i] = np.average(radii[in_bin], weights=weights[in_bin])
                if median:
                    profile[i] = np.median(data[in_bin])
                else:
                    profile[i] = np.average(data[in_bin], weights=weights[in_bin])
            else:
                output_radii[i] = np.nan
                profile[i] = np.nan

        return output_radii, profile

    @staticmethod
    def _combine_psf_profiles(profiles):
        """
        Combine the radial profiles derived for the various PSF stars to a
        single pandas DataFrame.
        
        Parameters
        ----------
        profiles : dict
            The individual PSF profiles, using the source IDs as keys and
            tuples containing radii and intensities as values.

        Returns
        -------
        combined : pandas.DataFrame
            The combined profiles, one column per source (using source IDs as
            labels) and using average radii (in pixels) as indices. 
        """
        source_ids = []
        radii = []
        data = []
        for key, value in profiles.items():
            source_ids.append(key)
            radii.append(value[0])
            data.append(value[1])

        combined = pd.DataFrame(np.stack(data, axis=1), columns=source_ids, index=np.nanmean(radii, axis=0))
        return combined

    def update_look_up_table(self, residuals, centroids, weights=None):
        """
        Create look-up table containing the relative residuals of the PSF fits
        averaged over all sources.

        Parameters
        ----------
        residuals : dict
            A dictionary containing for each source a pandas DataFrame with
            the columns 'data' and 'residuals' that uses the same Index values
            as the full data set.
        centroids : pandas.DataFrame
            A pandas DataFrame containing one row per source, using the source
            IDs as Index values and containing the centroid coordinates in
            columns named 'xc' and 'yc'.

        Returns
        -------
        lut : instance of scipy.interpolate.InterpolatedUnivariateSpline
            The calculated luck-up table. It can be called using a set of
            radii and will return the interpolated relative residuals at
            these radii.
        """
        # for each source, measure distances of provided pixels to centroid and collect them together with the
        # relative residuals in a single DataFrame.
        combined = []

        for source_id, data in residuals.items():
            x = self.data.loc[data.index, 'x']
            y = self.data.loc[data.index, 'y']
            weight = weights[source_id] if weights is not None else 1

            combined.append(pd.DataFrame({'dx': x - centroids.at[source_id, 'xc'],
                                          'dy': y - centroids.at[source_id, 'yc'],
                                          'rel_res': data['residuals']/(data['data'] - data['residuals']),
                                          'residuals': data['residuals'],
                                          'data': data['data']/np.sum(data['data']),
                                          'model': (data['data'] - data['residuals'])/np.sum(data['data'] - data['residuals']),
                                          'weights': weight*np.ones_like(x)}))

        combined_residuals = pd.concat(combined, ignore_index=True)

        radii, correction = self._get_radial_profile(
            data=combined_residuals['rel_res'],
            radii=np.sqrt(combined_residuals['dx']**2 + combined_residuals['dy']**2),
            weights=combined_residuals['weights'],
            median=True)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.12, 0.83, 0.83])
        ax.axhline(y=0, ls='-', lw=1.5, c='0.5')
        ax.scatter(np.sqrt(combined_residuals['dx']**2 + combined_residuals['dy']**2), combined_residuals['rel_res'],
                   marker='o', s=combined_residuals['weights']/100., c='g', alpha=0.1)
        # ax.hexbin(np.sqrt(combined_residuals['dx']**2 + combined_residuals['dy']**2), combined_residuals['rel_res'],
        #           bins='log')
        # ax.plot(np.sqrt(combined_residuals['dx'] ** 2 + combined_residuals['dy'] ** 2),
        #         np.log10(combined_residuals['model']), ls='None', marker='+', ms=5, mew=1.0, mec='b', alpha=0.5)
        ax.plot(radii, correction, ls='None', marker='+', mew=2.5, mec='r', ms=20)
        # ax.set_ylim(-10, 10)
        plt.show()

        return interpolate.InterpolatedUnivariateSpline(radii, 1. + correction)
