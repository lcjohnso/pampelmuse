"""
fit_array.py
============
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


Latest GIT revision
-------------------
2017/08/07
"""
import contextlib
import logging
import time
import numpy as np
from scipy import optimize, sparse
from scipy.sparse import linalg
from ..psf import Gaussian, Moffat, DoubleGaussian
from ..core.coordinates import Transformation


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 20170807


logger = logging.getLogger(__name__)


def sum_by_group(values, groups):
    """
    This is a fast algorithm for summing up the data in one
    array based on the groups provided in the second array.
    It is taken from http://stackoverflow.com/questions\
    /4373631/sum-array-by-number-in-numpy?lq=1

    Parameters
    ----------
    values : array_like, 1dim.
        The data that should be summed up.
    groups : array_like, 1dim.
        The groups according to which the data in 'values'
        should be summed. It must have the same length as
        'values'

    Returns
    -------
    values : nd_array
        The summed up data. Note that the entries are sorted
        according to the provided group numbers.
    groups : nd_array
        A sorted list of all the groups provided as input.
    """
    order = np.argsort(groups)
    groups = np.asarray(groups)[order]
    values = np.asarray(values)[order]
    values.cumsum(out=values, dtype=np.float64)
    # print "CUMSUM:", values
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups


# for a nicer printing of numpy arrays, following suggestion from
# http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


class FitLayer(object):
    """
    This class is designed to perform a PSF fit on a single layer of a data
    cube. If requested, it optimizes for the parameters of the PSF model and
    the positions of the provided sources. In any case, the fluxes of the
    sources are optimized.
    """

    _unresolved_matrix = None

    def __init__(self, data, variances, transformation, psf_type, psf_parameters, xy, position_parameters=None,
                 magnitudes=None, components=None, last_component_is_unresolved=False, background=None,
                 wave=None, maxrad=np.inf, osfactor=1, osradius=0):
        """
        Initialize a new instance of the FitLayer class.

        Parameters
        ----------
        data : array_like, 1D
            The data on which the PSF fit should be performed. Note that this
            should be a 1dim. array and that the 'transformation' parameter is
            used to translate from pixels to spatial (x, y)-positions.
        variances : array_like, 1D
            The variances of the data array. If provided, both arrays must have
            the same size. It is also possible to provide None, in which case
            the routine will try to estimate the variances from the deviation
            between data and model.
        transformation : array_like, 2D
            This array performs assigns spatial (x,y) coordinates to the
            pixels of the 'data' and 'variance' arrays. Therefore, it must be
            2dim, with the zeroth axis having the same length as 'data' and
            the first axis having a length of 2.
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
        xy : array_like, 2D
            The reference coordinates of the sources used in the fit. Zeroth
            axis defines the number of sources, first axis must have length 2.
        position_parameters : dictionary
            The initial guesses (or fixed values) of the individual parameters
            of the coordinate transformation (if any). If not provided, it is
            assumed that 'xy' contains pixel coordinates.
        magnitudes : array_like, optional
            Relative magnitudes of the sources to be fitted. If none are
            provided, it is assumed that all sources have the same magnitudes
            in the fit. If provided, its length must match the number of
            sources defined by 'xy'.
            IMPORTANT: The idea of this parameter is to define fixed flux
            ratios for sources that are fitted as a combined component. If no
            such components exist, this parameter can safely be ignored.
        components : array_like, optional
            This array defines how the sources are combined into components in
            the fit. To this aim, each entry should be an integer defining the
            corresponding fit component. The relative weight of the sources in
            each component is determined from the 'magnitudes' array. If
            provided, the length of 'components' must match the number of
            sources defined by 'xy'.
        last_component_is_unresolved : boolean, optional
            This flag should be set to true if the fit component with the
            highest index (i.e. the highest entries in 'components') is an
            unresolved component. See Kamann et al. (2013) for the idea of the
            unresolved component.
        wave : array_like, optional
            This parameter may be used to provide a separate wavelength array
            for the data array. If provided, the code will use it to determine
            an effective wavelength for each fitted component. This is only
            useful if not all pixels of the data array have the same
            wavelength.
        maxrad : float, optional
            The radius in pixels out to which the PSF profiles will be defined.
        osfactor : int, optional
            The oversampling factor used to increase the accuracy of the
            calculation of the core shape of the PSF profiles.
        osradius : float, optional
            The radius in pixels out to which the oversampling is being used.
        """
        self.maxrad = maxrad

        # Prepare data and variances, make sure they are 1dim. and have the same size
        self.data = np.asarray(data)
        if self.data.ndim != 1:
            logger.error('Data array must be 1dimensional.')
        if variances is not None:
            self._variances = np.asarray(variances)
            if self._variances.shape != self.data.shape:
                logger.error('Arrays containing data and variances must have same shape.')
        else:
            self._variances = None
        if wave is not None:
            self.wave = sparse.csc_matrix(wave)
            if self.wave.shape[-1] != self.data.shape[-1]:
                logging.error('If provided, wavelength array must have same shape as data array.')
        else:
            self.wave = None
        self.nspaxel = self.data.size

        # this is the transformation from spaxels to (x,y) coordinates
        self.transformation = np.asarray(transformation)
        if self.transformation.ndim != 2 or self.transformation.shape != (self.nspaxel, 2):
            logger.error('Spaxel-to-xy transformation has invalid shape.')

        # work on source catalogue
        self.xy = np.asarray(xy)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            logger.error('Array containing source coordinates has invalid shape.')
        self.n_sources = self.xy.shape[0]

        self.magnitudes = np.asarray(magnitudes) if magnitudes is not None else np.zeros(self.n_sources, dtype='<f4')
        if self.magnitudes.size != self.n_sources:
            logger.error('Number of provided magnitudes ({0}) does not match number of sources ({1}).'.format(
                self.magnitudes.size, self.n_sources))

        self.components = components if components is not None else np.arange(self.n_sources, dtype='<i4')
        if self.components.size != self.n_sources:
                logger.error('No. of entries in components data ({0}) does not match number of sources ({1}).'.format(
                    self.components.size, self.n_sources))
        self.n_components = self.components.max() + 1
        self.last_component_is_unresolved = last_component_is_unresolved

        # check and prepare background if any was provided
        self.background, self.n_background, self.missing_background = self._check_background(background)

        # Initialize PSF
        if psf_type.lower() == 'gauss':
            psf_class = Gaussian
        elif psf_type.lower() == 'moffat':
            psf_class = Moffat
        elif psf_type.lower == 'double_gauss':
            psf_class = DoubleGaussian
        else:
            logger.error('Unknown PSF type provided: {0}'.format(psf_type))
            psf_class = None

        # define PSF profile for each source, check if direct positions were provided for some sources
        self.psf_instances = []
        for i in range(self.n_sources):
            new_psf = psf_class(transform=self.transformation, uc=self.xy[i, 0], vc=self.xy[i, 1],
                                mag=self.magnitudes[i], src_id=i, maxrad=maxrad, osfactor=osfactor, osradius=osradius)
            # initialize the PSF parameters:
            for key, value in psf_parameters.items():
                setattr(new_psf, key, value)
            # if provided, add information about coordinate transformation
            if position_parameters is not None:
                for key, value in position_parameters.items():
                    setattr(new_psf, key, value)
            # otherwise use reference coordinates as direct IFS coordinates
            else:
                new_psf.xc = self.xy[i, 0]
                new_psf.yc = self.xy[i, 1]
            self.psf_instances.append(new_psf)

        self.source_matrix = None
        self.fit_weights = None
        self.fluxes = np.zeros((self.n_components,), dtype='<f4')
        self.background_fluxes = np.zeros((self.n_background,), dtype='<f4')

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
        This method is designed to provide an estimate of the variances
        if they are not provided externally. To this aim, the method compares
        the current model with the data and determines the deviation between
        the two as a function of the model flux using a running MAD.

        Returns
        -------
        variances : ndarray
            The array containing the estimated variances. Note that it will
            be filled with 1s if no model is yet available.
        """
        # if available, use provided variances
        if self._variances is not None:
            return self._variances
        # otherwise, we only can estimate the variances if a model exists
        elif (self.fluxes != 0).any():
            model_fluxes = self.model.sum(axis=1).getA1()
            xbin, ybin = FitLayer.calc_running_mad(model_fluxes, self.current)
            return 1.4826*np.interp(model_fluxes, xbin, ybin)  # 1.4826 is for MAD -> sigma conversion
        # if not, just return 1s
        else:
            return np.ones(self.data.shape, dtype=np.float32)

    def get_weight_factors(self, alpha=2, beta=2):
        """
        This method implements a weighting scheme similar to the one used by
        P. Stetson in DAOphot. The idea is to reduce the weights of deviant
        pixels (i.e. pixels with huge residuals) using a modified variance.
        The weight of a pixel i in the fit is modified from a pure variance-
        weighting as follows:

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
        f : ndarray
             The factors by which the variances should be divided to get
             modified variances for the fitting process.
        """
        if (self.fluxes != 0).any():
            return 1. / (1. + np.abs(self.current / (alpha*np.sqrt(self.variances)))**beta)
        else:
            return np.ones(self.data.shape, dtype=np.float32)

    def __call__(self, fluxes=None, background_fluxes=None, psf_parameters_to_fit=None, position_parameters_to_fit=None,
                 sources=None, psf_sources=None, maxiter=5):
        """
        This method carries out a least-squares fir to the data. The process
        is iterative: The model is optimized for the fluxes, the PSF, and the
        positions of the sources sequentially. This loop is repeated until
        convergence or the maximum number of iterations have been reached.

        Parameters
        ----------
        fluxes : array_like, optional
            Initial values for the fluxes of the components. Note that the fit
            of the fluxes is a linear optimization, hence no initial guesses
            are required.
        background_fluxes : array_like, optional
            Initial values for the fluxes of the background components. Again,
            as the optimization of the fluxes is a linear process, no initial
            guesses are required.
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
        sources : array_like, optional
            The indices of the sources used to optimize the PSF parameters and
            the source coordinates. If none are provided, all sources are being
            used.
        psf_sources : array_like, optional
            This parameter may be used if only a subset of the sources provided
            as parameter 'sources' should be used to optimize the PSF. The idea
            behind this is that usually a higher S/N is required to get the
            shape of the PSF right than to get its centroid.
        maxiter : int, optional
            The maximum number of iterations.

        Returns
        -------
        all_fluxes : nd_array
            The fluxes determined for all the components in the fit.
        uncertainties : nd_array
             The associated uncertainties of all the components in the fit.
        return_positions, nd_array
             The best-fit parameters of the coordinate transformation or the
             best-fit centroids of all components, or None if fixed positions
             were used.
        return_psf : nd_array
             The best-fit parameters of the PSF, or None if a fixed PSf was
             used.
        return_wave : nd_array
             If a 'wave' parameter was provided at the initialization of this
             instance, this array will contain the weighted wavelength value
             of each component. Otherwise None is returned.
        """
        if fluxes is not None:
            if fluxes.size != self.n_components:
                logger.error('No. of provided fluxes does not match number of components.')
            else:
                self.fluxes = fluxes
        if background_fluxes is not None:
            if 0 < self.n_background != background_fluxes.size:
                logger.error('No. of provided background fluxes does not match number of background components.')
            else:
                self.background_fluxes = background_fluxes

        # Determine initial weights for each pixel in the fit.
        self.fit_weights = 1./self.variances

        for iteration in range(1, maxiter + 1):
            if maxiter > 1:
                logger.info("Iteration {0} (of max. {1})".format(iteration, maxiter))

            if self.source_matrix is None:
                self.source_matrix, included_components = self.build_matrix()

                # fluxes of components outside FoV are set to NaN
                self.fluxes[~included_components] = np.nan

            # Fit fluxes by direct matrix inversion
            # first obtain fit matrix by multiplying source matrix with sqrt of fitting weights
            n, m = self.source_matrix.shape
            fit_matrix = sparse.dia_matrix((np.sqrt(self.fit_weights), [0]), shape=(n, n)) * self.source_matrix

            # The documentation of scipy states that to efficiently solve the matrix equation, the columns should
            # be scaled to have the same euclidean norm. I am still not convinced that this is true, but scale the
            # columns nevertheless.
            total_flux = np.sqrt((fit_matrix.T*fit_matrix).diagonal())
            fit_matrix = fit_matrix * sparse.dia_matrix((1./total_flux, [0]), shape=(m, m))

            # for testing, show scatter plot with current residuals
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(8, 10))
            # ax = fig.add_axes([0.04, 0.52, 0.92, 0.46])
            # ax.scatter(self.transformation[:, 1], self.transformation[:, 0], marker='s', s=300, c=self.current,
            #            cmap=plt.cm.hot, edgecolors='face', vmin=-100, vmax=1000)
            # ax2 = fig.add_axes([0.04, 0.02, 0.92, 0.46])
            # ax2.scatter(self.transformation[:, 1], self.transformation[:, 0], marker='s', s=300,
            #             c=self.model.sum(axis=1).getA1(), cmap=plt.cm.hot, edgecolors='face', vmin=-100, vmax=1000)
            # plt.show()

            # The fit for the source fluxes is carried out on the residuals, not the data itself. The fitted residual
            # fluxes are added to the existing ones
            t1 = time.time()
            bestfit = linalg.lsmr(fit_matrix, self.current*np.sqrt(self.fit_weights), atol=1e-5, show=False)
            t2 = time.time()
            logger.debug("Time required to solve matrix equation: {0:6.3f}s".format(t2-t1))
            logger.info("Chi**2 of current fit: {0:.1f}".format(bestfit[3] ** 2))

            # The background fluxes are currently stored in a separate array
            self.fluxes[included_components] += (bestfit[0] / total_flux)[:included_components.sum()]
            self.background_fluxes += (bestfit[0] / total_flux)[included_components.sum():]

            # Convergence criterion: the average relative change in flux is < 0.5 percent
            delta_flux = (bestfit[0] / total_flux)[:included_components.sum()] / self.fluxes[included_components]
            with printoptions(precision=4, suppress=True):
                logger.debug("Relative changes in fluxes: {0}".format(delta_flux))
            if np.median(abs(delta_flux)) < 0.005:
                break

            # carry out optimization of PSF and/or source positions
            if psf_parameters_to_fit is not None or position_parameters_to_fit is not None:
                # if any sources are provided for the PSF fit, the method for large layers is used
                if sources is not None:
                    results = self.fit_psf_single_source(sources=sources, psf_parameters=psf_parameters_to_fit,
                                                         psf_sources=psf_sources,
                                                         fit_positions=(position_parameters_to_fit is not None))
                    # update PSF parameters, using S/N as weights:
                    if psf_parameters_to_fit is not None:
                        for i, name in enumerate(psf_parameters_to_fit, start=2):  # first two entries are S/N and flux
                            valid = np.isfinite(results[:, i])
                            if valid.any() and results[valid, 0].sum() > 0:
                                new_value = np.average(results[valid, i], weights=results[valid, 0])
                                try:
                                    [setattr(psf, name, new_value) for psf in self.psf_instances]
                                    logger.info('Updated PSF parameter {0} to {1:.2f}'.format(name, new_value))
                                except AssertionError:
                                    logger.error('PSF fit yielded invalid value {0:.1f} for parameter {1}.'.format(
                                        name, new_value))
                            else:
                                logger.warning('Cannot update PSF parameter {0} because no PSF fit succeeded.'.format(
                                    name))

                    # update positions
                    if 'xy' in position_parameters_to_fit:  # positions are fitted directly, no coord. transf.
                        for i, source in enumerate(sources):
                            if np.isfinite(results[i, -2:]).all():
                                self.psf_instances[source].xc = results[i, -2]
                                self.psf_instances[source].yc = results[i, -1]
                    elif position_parameters_to_fit is not None:
                        fixed = {}
                        for parameter in Transformation.PARAMETERS:
                            if parameter not in position_parameters_to_fit:
                                fixed[parameter] = getattr(self.psf_instances[0], parameter)
                        valid = np.isfinite(results[:, -2:]).all(axis=1)
                        try:
                            bestfit = FitLayer._update_coordinate_transformation(
                                    self.xy[sources[valid]], results[valid, -2:], fixed=fixed)
                        except IOError:
                            logging.error('Fit of coordinate transformation failed.')
                        else:
                            for i, parameter in enumerate(Transformation.PARAMETERS):
                                if parameter in position_parameters_to_fit:
                                    [setattr(psf, parameter, bestfit[0][i]) for psf in self.psf_instances]
                                    logger.info('Updated position parameter {0} to {1:.2f}'.format(
                                            parameter, bestfit[0][i]))
                else:
                    position_results, psf_results = self.fit_psf_all_sources(
                        psf_parameters=psf_parameters_to_fit, position_parameters=position_parameters_to_fit)
                    if position_results is not None:
                        for i, parameter in enumerate(position_parameters_to_fit):
                            [setattr(psf, parameter, position_results[i]) for psf in self.psf_instances]
                    if psf_results is not None:
                        for i, parameter in enumerate(psf_parameters_to_fit):
                            try:
                                [setattr(psf, parameter, psf_results[i]) for psf in self.psf_instances]
                            except AssertionError:
                                logger.error('PSF fit yielded invalid value {0:.1f} for parameter {1}.'.format(
                                    parameter, psf_results[i]))

                # PSF and/or positions have changed, so the matrix needs to be recalculated (unless end of iteration)
                if iteration < maxiter:
                    self.source_matrix = None

            elif iteration < maxiter:
                # re-calculate fitting weights
                self.fit_weights = self.get_weight_factors() / self.variances
            else:
                logger.warning('Maximum number of iterations reached.')

        # combine fluxes, need to add missing background components again
        # IMPORTANT: when using np.insert, need to lower indices by number of previous missing indices
        indices = self.missing_background - self.n_components - np.arange(len(self.missing_background))
        all_fluxes = np.hstack((self.fluxes, np.insert(self.background_fluxes, indices, np.nan)))

        # estimate uncertainties that are returned with the data
        # note that fit_weights ~ 1/variances
        inverse_variances = self.source_matrix.multiply(self.source_matrix).T.dot(self.fit_weights)
        # fill missing components (sources and background) with NaN
        indices = np.flatnonzero(~included_components) - np.arange((~included_components).sum())
        inverse_variances = np.insert(inverse_variances, indices, np.nan)
        indices = self.missing_background - np.arange(len(self.missing_background))
        inverse_variances = np.insert(inverse_variances, indices, np.nan)
        # invert and take square root to get uncertainties
        uncertainties = 1./np.sqrt(inverse_variances)

        # collect data that is returned as fit result
        if psf_parameters_to_fit is not None:
            return_psf = [getattr(self.psf_instances[0], p) for p in psf_parameters_to_fit]
        else:
            return_psf = None
        if position_parameters_to_fit is None:
            return_positions = None
        elif 'xy' in position_parameters_to_fit:
            return_positions = [[psf.xc, psf.yc] for psf in self.psf_instances]
        else:
            return_positions = [getattr(self.psf_instances[0], p) for p in position_parameters_to_fit]

        if self.wave is not None:
            # calculate variance weighted wavelength of each component
            n = self.source_matrix.shape[0]
            wave_matrix = sparse.dia_matrix((self.fit_weights, [0]), shape=(n, n)) * self.source_matrix
            return_wave = np.asarray(self.wave.dot(wave_matrix)/wave_matrix.sum(axis=0))[0]
            if not included_components.all():
                indices = np.flatnonzero(~included_components) - np.arange((~included_components).sum())
                return_wave = np.insert(return_wave, indices, np.nan)
        else:
            return_wave = None

        return all_fluxes, uncertainties, return_positions, return_psf, return_wave

    @property
    def model(self):
        """

        Returns
        -------
        The current best-fit model.
        """
        fluxes = np.hstack((self.fluxes[np.isfinite(self.fluxes)], self.background_fluxes))
        return self.source_matrix * sparse.dia_matrix((fluxes, [0]), shape=(
            self.source_matrix.shape[1], self.source_matrix.shape[1]))

    @property
    def current(self):
        """

        Returns
        -------
        The residuals of the current best fit model.
        """
        return self.data - self.model.sum(axis=1).getA1()

    def residuals_from_fluxes(self, fluxes, background_fluxes=None):
        """
        This method returns the fit residuals provided input values for the
        fluxes and background fluxes (if any). The purpose of this method is
        to have a routine for fast residuals calculation as is required by
        the SUBTRES routine in PampelMuse.

        Parameters
        ----------
        fluxes : nd_array
            The fluxes of the components. The length of this array must match
            the number of components of the instance.
        background_fluxes :  nd_array, optional
            The fluxes of the background components (if any). If background
            components are used, the length of this array must match their
            number. Otherwise, None should be provided.

        Returns
        -------
        residuals : nd_array
            The model residuals calculated using the input fluxes.
        """
        if fluxes.size != self.n_components:
            logger.error('No. of provided fluxes does not match number of components.')
        if background_fluxes is not None and 0 < self.n_background != background_fluxes.shape:
            logging.error('No of. provided background fluxes does not match number of background components.')

        self.source_matrix, included_components = self.build_matrix()

        if background_fluxes is not None:
            self.fluxes = np.hstack((fluxes, background_fluxes))[included_components]
        else:
            self.fluxes = fluxes[included_components]

        return self.current

    def build_matrix(self, update_unresolved=True):
        """
        Define the source matrix. The source matrix is a sparse matrix that
        contains the spatial profiles of all resolved components, one per
        column.
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

        Returns
        -------
        source_matrix : a scipy.sparse.csc_matrix-instance
            This is the PSF coefficient matrix that is used in the linear
            least-squares fit to determine the source fluxes.
        included_components : nd_array
            A 1dim. boolena array with length equal to the number of
            components that is True for each component that contributes
            to the PSF source matrix.
        """
        if not self.last_component_is_unresolved:
            update_unresolved = True

        t0 = time.time()

        # result will be a list with length equal to the number of sources, each entry is again a list containing three
        # entries: the spaxels on which the PSF is defined, the corresponding fluxes, and the integrated flux
        results = []
        for i, psf in enumerate(self.psf_instances):
            if self.last_component_is_unresolved and not update_unresolved:
                if self.components[i] == self.n_components - 1:
                    continue

            spaxels, fluxes = psf.get_flux()
            if spaxels.size == 0:
                logger.warning("Source no. {0} entirely outside FoV.".format(psf.src_id))
            results.append([spaxels, fluxes, fluxes.sum()])

        t1 = time.time()
        logger.debug("Time required to calculate PSF profiles: {0:.3f}s".format(t1 - t0))

        # Calculate total flux of each component
        if not self.last_component_is_unresolved or update_unresolved:
            components = self.components
        else:
            components = self.components[self.components < self.components.max()]
        totflux, ordered_components = sum_by_group([r[-1] for r in results], components)
        # note that the total fluxes are already in the correct order, i.e. sorted by component

        t2 = time.time()
        logger.debug("Time required to combine components: {0:.3f}s".format(t2 - t1))

        # check if any row does not contribute at all
        empty_components = ordered_components[totflux <= 0]
        if len(empty_components) > 0:
            logger.warning("Encountered component(s) that contribute zero flux: {0}".format(empty_components))

        # Fill the matrix: the indices and intensities of all PSF instances are combined and serve as row indices (i0)
        # and values of the matrix. To obtain the column indices of the matrix (i1), we use the property 'specrow' of
        # the Sources class and correct for components with zero flux contribution (if they exist)
        i0 = np.hstack((r[0] for r in results))  # the row indices

        i1 = np.zeros(i0.shape, dtype=np.int32)
        jstart = 0
        for i in range(len(results)):
            if components[i] in empty_components:
                assert(len(results[i][0]) == 0), results[i]
                continue
            jend = jstart + len(results[i][0])
            i1[jstart:jend] = components[i]
            jstart = jend

        f = np.hstack((r[1] for r in results))  # the fluxes

        # If components are entirely outside the field of view, the indices in i1 will not be contiguous but there will
        # be gaps that must be removed.
        assert np.intersect1d(i1, empty_components).size == 0, "i1 contains indices of components with zero flux."
        for empty_component in empty_components[::-1]:
            i1[i1 > empty_component] -= 1

        # create matrix: first, a COO matrix is constructed. Note that it handles the case when the input data contains
        # multiple entries for the some indices (row, column). After construction, the matrix is converted to a
        # CSC matrix to speed up future usage of it.
        nrows = self.data.size
        ncols = self.n_components - len(empty_components) + self.n_background

        if self.last_component_is_unresolved:  # handle the unresolved component
            # need to take into account that the row of the unresolved component in the matrix changes if some
            # components do not contribute
            unresolved_component = self.n_components - 1 - len(empty_components)
            if update_unresolved or self._unresolved_matrix is None:
                # just store the unresolved sources as a list for later re-use
                r = (i1 != unresolved_component)
                self._unresolved_matrix = sparse.coo_matrix((f[~r], (i0[~r], i1[~r])), shape=(nrows, ncols))
                _Apsf = sparse.coo_matrix((f[r], (i0[r], i1[r])), shape=(nrows, ncols)) + self._unresolved_matrix
            else:
                _Apsf = sparse.coo_matrix((f, (i0, i1)), shape=(nrows, ncols)) + self._unresolved_matrix
        else:
            _Apsf = sparse.coo_matrix((f, (i0, i1)), shape=(nrows, ncols))

        if self.n_background > 0:
            if empty_components is None:
                _Apsf = _Apsf + self.background
            else:
                used = ~np.in1d(np.arange(self.n_components + self.n_background), empty_components)
                _Apsf = _Apsf + self.background.tocsc()[:, used].tocoo()

        t3 = time.time()
        logger.debug("Time required to create matrix: {0:.3f}s".format(t3 - t2))
        logger.debug("Full source matrix creation took {0:.2f}s.".format(t3-t0))

        # return CSC matrix and array containing the components that contribute flux
        zero_component = np.in1d(np.arange(self.n_components, dtype=np.int32), empty_components)
        return _Apsf.tocsc(), ~zero_component

    def fit_psf_single_source(self, sources, psf_parameters=None, fit_positions=True, psf_sources=None):
        """
        Perform the PSF fit for one or more individual sources. This is
        recommended if the layer contains many sources (approx. more than
        hundred). Depending on the respective fit-flags, either the
        parameters of the PSF and/or the source coordinates are fitted.
        The results of the individual fits are afterwards used to update
        the PSF instances.

        Parameters
        ----------
        psf_sources : array_like
            The indices of the sources used to constrain the parameters of
            the PSF.
        psf_parameters : array_like, optional
            The names of the PSF parameters that should be optimized in the
            fits. Default is to use a constaant PSF.
        fit_positions : boolean, optional
            Flag indicating if the source positions should be optimized in
            the fits.
        sources : array_like, optional
            The indices of the sources for which centroid coordinates are
            fitted. If None, use the sources provided as 'psf_sources'.

        Returns
        -------
        results : nd_array
            An array containing the results of all sources used in the fits,
            one per row. Each row contains the following entries:
            0: the flux of each source.
            1: the S/N of each source.
            [2 - ?: The values of the free PSF parameters.]
            [-2 - -1: the x- and y-centroids obtained in the fits]
            [] = only if among fitted parameters
        """
        # The keyword 'psf_sources' may be used to define a subset of 'sources' that is used to fit the PSF parameters.
        # All 'sources' are used for this purpose if psf_sources=None
        if psf_sources is None:
            psf_sources = sources

        if psf_parameters is None:
            psf_parameters = []

        # prepare array that will hold the results
        results = np.zeros((len(sources), 2 + len(psf_parameters) + 2*fit_positions), dtype='<f4')

        # Loop over all sources with free parameters; add source to residuals, fit resulting array and subtract source
        # again.
        for n, i in enumerate(sources):
            psf = self.psf_instances[i]

            # Collect parameters and initial guesses
            # (a) fluxes are always fitted
            names = ["flux"]  # the names of the free parameters for this PSF
            x0 = [self.fluxes[self.components[i]]]  # the initial guesses for the parameter fit

            # (b) check if PSF parameters should be fitted
            if len(psf_parameters) > 0 and i in psf_sources:
                names.extend(psf_parameters)
                for psf_parameter in psf_parameters:
                    x0.append(getattr(psf, psf_parameter))

            # (c) check if positions should be fitted
            if fit_positions:
                names.extend(["xc", "yc"])
                x0.extend([psf.xc, psf.yc])

            logger.debug("Initial guesses for fit of source no. {0}: {1}".format(psf.src_id, dict(zip(names, x0))))

            # get pixels of layer where the current PSF is defined
            variances = 1. / self.fit_weights[psf.used]
            data = self.current[psf.used] + x0[0] * psf.f

            psf.transform = self.transformation[psf.used]
            psf.maxrad = np.inf

            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_axes([0.12, 0.12, 0.83, 0.83])
            # # print(psf.uc, psf.vc, psf.xc, psf.yc)
            # ax.scatter(psf.x, psf.y, marker='o', s=50, c=data, cmap=plt.cm.hot)
            # plt.show()

            # perform fit, check first if fit makes sense
            if data.size <= len(x0):
                result = (x0, None, {}, "Number of free parameters exceeds number of datapoints.", 99)
            else:
                # before fitting, first check if there is any significant peak inside a circle with radius FWHM
                # centred on the initial guess
                spaxel = np.unique(psf.osmap, return_index=True)[1]  # make sure every subspaxel only counted once
                in_fwhm = (psf.r[spaxel] < psf.fwhm)
                if in_fwhm.sum() == 0:
                    peak = 0
                    noise = 1
                elif (~in_fwhm).sum() >= 5:
                    peak = data[in_fwhm].max() - data[~in_fwhm].mean()
                    noise = np.std(data[~in_fwhm])
                else:
                    peak = data.max() - data.min()
                    noise = np.median(data) - data.min()
                if peak/noise <= 3:
                    result = (x0, None, {}, "No significant peak detected.", 99)
                else:
                    try:
                        result = optimize.leastsq(self.single_psf_optimizer, x0=x0, Dfun=self.single_psf_jacobi,
                                                  col_deriv=1, args=(names, psf, data, np.sqrt(variances)),
                                                  full_output=1, epsfcn=0.0)
                    except:  # optimize.minpack.error as error:
                        # error handler to avoid code crashes completely
                        # type of exception thrown seems to depend on scipy version
                        result = (x0, None, {}, "Exception of type optimize.minpack.error encountered during fit.", 99)

                    # check if result of a successful fit makes any sense:
                    if 0 < result[-1] <= 4:
                        if result[0][0] <= 0:  # fitted flux must be positive
                            result = (x0, None, {}, "Fitted flux is not positive.", 99)
                        elif 'xc' in names:  # offset between fitted centroid and initial guess must be <= FWHM
                            ix = names.index("xc")
                            iy = names.index("yc")
                            offset = np.sqrt((result[0][ix]-x0[ix])**2 + (result[0][iy]-x0[iy])**2)
                            if offset > psf.fwhm:
                                result = (x0, None, {}, "Offset between fitted centroid and initial guess >FWHM.", 99)

            snratio = result[0][0] * np.sqrt((psf.f**2/variances).sum())
            results[n, 0] = snratio

            if 0 < result[-1] <= 4:  # fit successful
                results[n, 1] = results[0][0]
                if i in psf_sources:
                    results[n, 2:(2 + len(psf_parameters))] = result[0][1:(1 + len(psf_parameters))]
                else:
                    results[n, 2:(2 + len(psf_parameters))] = np.nan
                if fit_positions:
                    results[n, -2:] = result[0][-2:]
                # log outcome of fit
                prmstring = ", ".join(["{0}={1:.2f}".format(names[k], result[0][k]) for k in range(1, len(names))])
                logger.info("Successfully fitted PSF of source no. {0} (S/N={1:.1f}): f={2:.1f}, {3}".format(
                    i, snratio, result[0][0], prmstring))
            else:
                logger.warning("PSF fit failed for source no. {0}: {1}".format(i, result[-2]))
                results[n] = np.nan

            # reset PSF instance to state it had before the fit
            psf.transform = self.transformation
            psf.maxrad = self.maxrad

        return results

    @staticmethod
    def single_psf_optimizer(x0, names, psf, data, sigma):
        """
        This is the optimization routine that is passed to the Levenberg-
        Marquardt solver scipy.optimize.leastsq. During each iteration, a
        PSF array is created, scaled and subtracted from the data. The
        residuals of the fit are returned.

        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted. The first
            parameter must be the flux of the source.
        names :  array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        psf : an instance of class pampelmuse.core.psf.profiles.Profile or
              one of its subclasses
            The PSF model for the source.
        data : nd_array
            The data array used to fit the parameters. Note that the pixel
            values should be normalized by their respective uncertainties.
        sigma : nd_array
            The uncertainties of the 'data' array pixels.

        Returns
        -------
        bestfit : nd_array
            The values of the free parameters obtained in the fit.
        """
        # update PSF using current parameter estimates
        for i in range(1, len(x0)):
            try:
                # the attribute is only set in the 'psf'-instance of the class, not in all instances! This is different
                # from the behaviour of getattr(self.PSF, "set_<PAR>")(<VALUE>)
                setattr(psf, names[i], x0[i])
            except AssertionError as error:  # if parameter cannot be updated, raise LinAlgError
                logger.error("Encountered AssertionError: {0}".format(error))
                raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], names[i]))

        return data / sigma - x0[0] * (psf.f / sigma)

    @staticmethod
    def single_psf_jacobi(x0, names, psf, data, sigma):
        """
        Calculate the Jacobi matrix for the optimization carried out in
        StarFit.optimize_single_psf. The method is passed to the Levenberg-
        Marquard solver scipy.optimize.leastsq as parameter Dfun.

        Parameters
        ----------
        x0
        names
        psf
        data
        sigma
            This method takes the same arguments as StarFit.optimize_single_psf.

        Returns
        -------
        jacob : nd_array
            The Jacobi matrix with respect to the model parameters included
            in 'x0'.
        """
        jacob = np.zeros((len(x0), psf.used.sum()), dtype=np.float64)
        jacob[0] -= psf.f
        for i in range(1, len(x0)):
            diff_i = getattr(psf, "diff_{0}".format(names[i]))
            jacob[i] -= x0[0] * diff_i
        return jacob / sigma

    def fit_psf_all_sources(self, psf_parameters=None, position_parameters=None):
        """
        Perform the positions/PSF fit on the whole layer. A non-linear least-
        squares fit that uses Levenberg-Marquardt optimization is performed
        with all resolved sources. This fitting procedure is recommended only
        for layers with a small number of sources (approx. less than 100). The
        results of the fit are used to update the PSF and/or the parameters of
        the coordinate transformation.

        Parameters
        ----------
        position_parameters : array_like, optional
            The names of the parameters of the coordinate transformation that
            are free fit parameters. Default is a fixed coordinate
            transformation.
        psf_parameters : array_like, optional
            The names of the PSF parameters that  are free fit parameters.
            Default is to use a fixed PSF.

        Returns
        -------
        bestfit_positions : array_like
            The best-fit values of the coordinate transformation parameters.
            Or None if a fixed transformation is used.
        bestfit_psf : array_like
            The best-fit values of the PSF parameters. Or None if a constant
            PSF is used.
        """

        # Fit position parameters
        if position_parameters is not None:
            logger.info("Optimizing source coordinates...")

            # collect initial guesses
            x0 = [getattr(self.psf_instances[0], name) for name in position_parameters]

            # call Levenberg-Marquardt solver from scipy.optimize (& provide Jacobian)
            args = (position_parameters, np.hstack((self.fluxes, self.background_fluxes)))
            bestfit_position = optimize.leastsq(self.multi_psf_optimizer, x0=x0, Dfun=self.multi_psf_jacobi,
                                                col_deriv=1, epsfcn=0.0, args=args)[0]

            # update parameters using fit results & some logging
            logstring = "Fitted coord. transf. parameters:"
            for i, name in enumerate(position_parameters):
                [setattr(psf, name, bestfit_position[i]) for psf in self.psf_instances]
                logstring += " {0}={1:.2f},".format(name, bestfit_position[i])
            logger.info(logstring[:-1])
        else:
            bestfit_position = None

        # fit PSF parameters
        if psf_parameters is not None:
            logger.info("Optimizing PSF parameters...")

            # collect free parameters
            x0 = [getattr(self.psf_instances[0], name) for name in psf_parameters]

            # call Levenberg-Marquardt solver from scipy.optimize (& provide Jacobian)
            try:
                args = (psf_parameters, np.hstack((self.fluxes, self.background_fluxes)))
                bestfit_psf = optimize.leastsq(self.multi_psf_optimizer, x0=x0, Dfun=self.multi_psf_jacobi,
                                               col_deriv=1, args=args, epsfcn=0.0)[0]

                # update parameters using fit results & do some logging
                logstring = "Fitted PSF model:"
                for i, name in enumerate(psf_parameters):
                    [setattr(psf, name, bestfit_psf[i]) for psf in self.psf_instances]
                    logstring += " {0}={1:.2f},".format(name, bestfit_psf[i])
                logger.info(logstring)

            except Exception as msg:  # optimize.minpack.error: # error handler to avoid code that code crashes
                # completely, seems to depend on numpy version which error is actually thrown
                logger.error("PSF fit failed: {0}".format(msg))
                bestfit_psf = None
        else:
            bestfit_psf = None

        return bestfit_position, bestfit_psf

    def multi_psf_optimizer(self, x0, names, fluxes):
        """
        The optimization routine that is passed to the 'leastsq'-routine of
        the 'scipy.optimize'-package. During each iteration, a PSF array is
        prepared for every resolved source, the PSFs are scaled and subtracted
        from the data. The residuals are returned.

        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted.
        names : array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        fluxes : nd_array
            The fluxes of the components included in the fit

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
                [setattr(psf, name, x0[i]) for psf in self.psf_instances]
            except AssertionError:  # if parameter cannot be updated, raise LinAlgError
                raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], name))

        # calculate new model
        psf_matrix, included_components = self.build_matrix(update_unresolved=False)
        n_comp = included_components.sum() + self.n_background
        psf_matrix.data *= np.take(np.sqrt(self.fit_weights), psf_matrix.indices)

        if (~included_components).any():
            fluxes = fluxes[np.hstack((included_components, np.ones(self.n_background, dtype=np.bool)))]

        model = psf_matrix * sparse.dia_matrix((fluxes, [0]), shape=(n_comp, n_comp))

        # return residuals
        return (self.data * np.sqrt(self.fit_weights)) - model.sum(axis=1).getA1()

    def multi_psf_jacobi(self, x0, names, fluxes):
        """
        Routine that calculates the Jacobi matrix of the residuals. This method
        is parsed to the Levenberg-Marquard solver scipy.optimize.leastsq a
        parameter Dfun when the PSF of multiple sources is fitted simultaneously.

        Parameters
        ----------
        x0
        names
        fluxes
            The method takes the same arguments as StarFit.optimize_multi_psf

        Returns
        -------
        jacob : nd_array
            The Jacobi matrix with respect to the model parameters included
            in 'x0'.
        """
        t0 = time.time()
        jacob = np.zeros((len(x0), self.data.size), dtype=np.float64)

        # The influence of the unresolved sources (if any) is currently ignored when calculating the Jacobi matrix,
        # any sky component has no influence anyway.
        if not self.last_component_is_unresolved:
            used_sources = np.arange(self.n_sources, dtype='<i4')
        else:
            used_sources = np.flatnonzero(self.components < self.components.max())
        for i in used_sources:
            psf = self.psf_instances[i]
            ii = self.components[i]
            for j in range(len(x0)):
                diff_i = getattr(psf, "diff_{0}".format(names[j]))
                jacob[j, psf.used] -= fluxes[ii] * (diff_i * np.sqrt(self.fit_weights[psf.used]))

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
        xy_reference : array_like
            The reference coordinates of N sources, provided as an array of
            shape (N, 2)
        xy_true : array_like
            The best-fit (or true) coordinates of the same N sources, again
            provided as an array of shape (N, 2)
        fixed : dictionary, optional
            The names and values of the parameters that should be fixed to
            their given values when determining the new coordinate
            transformation.

        Returns
        -------
        bestfit : nd_array
            The results of the least squares fit.

        Raises
        ------
        IOError
            If provided data does not allow to fit the coordinate
            transformation.
        """
        xy_reference = np.asarray(xy_reference)
        xy_true = np.asarray(xy_true)
        if xy_reference.shape != xy_true.shape or xy_reference.shape[1] != 2:
            logger.error('Cannot fit coordinate transformation. Arrays have incompatible shapes.')
            raise IOError('Incompatible shapes og input arrays: {0} and {1}'.format(xy_reference.shape, xy_true.shape))
        if xy_reference.shape[0] < 4:
            logger.error('Cannot fit coordinate transformation. Less than 4 sources provided.')
            raise IOError('Len of first axis of input arrays must be >=4 but is {0}'.format(xy_true.shape[0]))

        bestfit, residuals = Transformation.fit_to_coordinates(xy_reference, xy_true, fixed=fixed)
        return bestfit

    @staticmethod
    def calc_running_mad(x, y, n=500):
        """
        Given two 1D arrays x and y, the routine will determine the
        median absolute deviation (MAD) of y as a function of x. To this
        aim, the data are sorted by increasing x values and then the MAD
        is measured in bins of n datapoints with increasing x values.
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
            ybin[i] = np.median(abs(cury))
        return xbin, ybin
