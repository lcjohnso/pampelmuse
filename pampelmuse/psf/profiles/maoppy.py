"""
maoppy.py
==========
Copyright 2013-2021 Sebastian Kamann

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
function pampelmuse.core.psf.profiles.maoppy.resample_to_2d
function pampelmuse.core.psf.profiles.maoppy.resample_to_1d
class pampelmuse.core.psf.profiles.Maoppy

Latest Git revision
-------------------
2022/11/29
"""
import importlib.util
import io
import logging
import warnings
import numpy as np
from contextlib import redirect_stdout
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline, interp2d
from .profile import Profile, FitResult

if importlib.util.find_spec('maoppy') is not None:
    from maoppy.psfmodel import Psfao, oversample
    from maoppy.psffit import psffit
    from maoppy.instrument import muse_nfm
    from maoppy.utils import circavgplt
else:
    logging.error('Package "maoppy" is not installed. Cannot use MAOPPY PSF model.')


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20221129


logger = logging.getLogger(__name__)


def resample_to_2d(x, y, z, shape=None):
    """
    Given a set of values z defined on arbitrary coordinates (x, y), the method
    will translate the data to a regularly sampled 2d array.

    The idea is that actual resampling is only performed if the input `x` and
    `y` coordinates do not correspond to grid points of the final 2d array.
    Otherwise, the available pixels values are just inserted into the output
    array.

    Parameters
    ----------
    x : sequence (1D) of floats
       The x-coordinates on which the input data are defined. Note that
       following numpy conventions, the x-axis will be the second axis of the
       output array.
    y : sequence (1D) of floats
       The y-coordinates on which the input data are defined. Note that
       following numpy conventions, the y-axis will be the first axis of the
       output array.
    z : sequence (1D) of floats
        The values of the input data. Parameters `x`, `y`, and `z` must have
        the same length.
    shape : tuple of length 2, optional
        The 2D shape of the requested output array. If not defined, the data
        will be resampled to the smallest 2d grid that encompasses all input
        data.

    Returns
    -------
    out : array (2D) of floats
        The array of resampled values.
    """
    if shape is None:
        shape = (int(y.max()) + 1, int(x.max()) + 1)

    is_int = np.all(x == x.astype(np.int32)) & np.all(y == y.astype(np.int32))

    if is_int:
        out = np.zeros(shape, dtype=z.dtype)
        slc = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
        out[y[slc].astype(np.int32), x[slc].astype(np.int32)] = z[slc]
    else:
        logger.warning('Input data irregularly sampled. Need to interpolate.')

        ip = interp2d(x=x, y=y, z=z, kind='linear')
        out = ip(np.arange(shape[1], dtype=np.int32), np.arange(shape[0], dtype=np.int32), assume_sorted=True)

    return out


def resample_to_1d(z, x, y,):
    """
    Given a regularly sampled 2D array of values z, the method will convert
    the data to the provided (randomly sampled) coordinates x and y.

    The idea is that actual resampling is only performed when the requested
    output coordinates are not a subset of the grid points on which the input
    data are defined. Otherwise, the output values are obtained by basic
    slicing.

    Parameters
    ----------
    z : array (2D) of floats
        The input grid of values.
    x : array (1D) of floats
        The x-coordinates of the output grid.
    y : array (1D) of floats
        The y-coordinates of the output grid. Parameters `x` and `y` must have
        the same length.

    Returns
    -------
    znew : array (1D) of floats
        The interpolated values at the coordinates provided by `x` and `y`.
    """
    is_int = np.all(x == x.astype(np.int32)) & np.all(y == y.astype(np.int32))

    if x.min() < 0 or y.min() < 0 or x.max() > (z.shape[1] - 1) or y.max() > (z.shape[0] - 1):
        raise IOError('Provided output coordinates extend beyond grid.')

    if is_int:
        return z[y.astype(np.int32), x.astype(np.int32)]
    else:
        logger.warning('Output data irregularly sampled. Need to interpolate.')

        ip = RectBivariateSpline(x=np.arange(z.shape[1], dtype=np.in32), y=np.arange(z.shape[0], dtype=np.in32),
                                 z=z, kx=1, ky=1)
        return ip(x, y, grid=False)


class Maoppy(Profile):
    """
    This class implements the MAOPPY PSF profile described by Fetick et al.
    (2019). It is optimized for the modelling of PSF profiles in adaptive
    optics (AO) observations, e.g. performed with the MUSE narrow field mode.

    References
    ----------
    1) Fetick et al. (2019, A&A, 628, A99)
       https://ui.adsabs.harvard.edu/abs/2019A&A...628A..99F/abstract
    """

    def __init__(self, uc, vc, mag, r0=0.2, bck=0.005, amp=1, alpha=0.05, beta=2.0, wvl=5e-7, **kwargs):
        """
        Initialize a new instance of the MAOPPY PSF.

        Notes
        -----
        There are some differences in the parameter set of the MAOPPY PSF that
        is used by PampelMuse compared to the paper by Fetick et al. (2019).:

        1) The parameters C (the AO-corrected phase PSD background) and A (the
           residual variance) have been renamed to `bck` and `amp`.

        2) Instead of defining two values for the effective radius of the
          Moffat, alpha_x and alpha_y, a single value `alpha` is used in
          combination with an ellipticity `e`.

        Parameters
        ----------
        uc : float
            The central coordinate of the PSF profile along the 1st reference
            axis.
        vc : float
            The central coordinate of the PSF profile along the 2nd reference
            axis.
        mag : float
            The integrated magnitude of the PSF profile, defined such that the
            integrated flux is unity for mag=0.
        r0 : float, optional
            The Fried parameter, given in units of [m].
        bck : float, optional
            The AO-corrected phase PSD background, given in units of
            [m^2 rad^2].
        amp : float, optional
            The residual variance, given in units of [rad^2].
        alpha : float, optional
            The effective radius of the Moffat profile, given in units of
            [m^-1].
        beta : float, optional
            The kurtosis parameter of the Moffat profile.
        wvl : float, optional
            The wavelength of the observation, given in units of [m].
        kwargs
            Any additional keyword arguments are passed on to the init method
            of the parent class.

        Returns
        -------
        The newly defined instance.
        """

        # oversampling of the PSF on the PampelMuse level is not supported, because MAOPPY performs its
        # own oversampling when computing a PSF profile.
        if 'osradius' in kwargs and kwargs['osradius'] > 0:
            logger.error('Oversampling not supported for MAOPPY PSF. Setting osradius=0 (was {})'.format(
                kwargs['osradius']))
            kwargs['osradius'] = 0
        if 'osfactor' in kwargs and kwargs['osfactor'] != 1:
            logger.error('Oversampling not supported for MAOPPY PSF. Setting osfactor=1 (was {})'.format(
                kwargs['osfactor']))
            kwargs['osfactor'] = 1

        # call initializer of parent class.
        super(Maoppy, self).__init__(uc=uc, vc=vc, mag=mag, **kwargs)

        # initial parameter definitions
        self._r0 = r0
        self._bck = bck
        self._amp = amp
        self._alpha = alpha
        self._beta = beta
        self._wvl = wvl

    @property
    def r0(self):
        """
        Returns the current Fried parameter of the PSF profile.
        """
        return self._r0

    @r0.setter
    def r0(self, value):
        """
        Set the Fried parameter of the PSF profile.

        Parameters
        ----------
        value : float
            The Fried parameter, given in units of [m].
        """
        self._r0 = value

    @property
    def bck(self):
        """
        Returns the current AO-corrected phase PSD background (labeled C in
        Fetick et al. 2019) of the PSF profile.
        """
        return self._bck

    @bck.setter
    def bck(self, value):
        """
        Set the AO-corrected phase PSD background (labeled C in Fetick et al.
        2019) of the PSF profile.

        Parameters
        ----------
        value : float
            The AO-corrected phase PSD background, given in units of
            [m^2 rad^2].
        """

        self._bck = value

    @property
    def amp(self):
        """
        Returns the current residual variance parameter (labeled A in Fetick
        et al. 2019) of the PSF profile.
        """
        return self._amp

    @amp.setter
    def amp(self, value):
        """
        Set the residual variance parameter (labeled A in Fetick et al. 2019)
        of the PSF profile.

        Parameters
        ----------
        value : float
            The residual variance parameter, given in units of [rad^2].
        """
        self._amp = value

    @property
    def alpha(self):
        """
        Returns the current Moffat effective radius of the PSF profile.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """
        Set the Moffat effective radius alpha of the PSF profile.

        Parameters
        ----------
        value : float
            The Moffat effective radius, given in units of [m^-1].
        """
        self._alpha = value

    @property
    def beta(self):
        """
        Returns the current Moffat beta of the PSF profile.
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        """
        Set the Moffat beta for the PSF profile.

        Parameters
        ----------
        value : float
            The Moffat beta.
        """
        self._beta = value

    @property
    def wvl(self):
        """
        Returns the current wavelength of the PSF profile.
        """
        return self._wvl

    @wvl.setter
    def wvl(self, value):
        """
        Set the wavelength for the PSF profile.

        Parameters
        ----------
        value : float
            The wavelength, given in units of [m].
        """
        self._wvl = value

    @property
    def q(self):
        """
        Returns the axis ratio corresponding to the current ellipticity.
        """
        return 1. - self.e

    @property
    def theta_maoppy(self):
        """
        Returns the position angle used internally by MAOPPY - which is offset
        from the one used in PampelMuse. While for PampelMuse, the +y-axis is
        used as reference, it is the +x-axis for MAOPPY.
        """
        return self.theta + np.pi/2

    @property
    def samp(self):
        """
        Returns the sampling of the PSF profile at the current wavelength.
        """
        return muse_nfm.samp(self.wvl)

    @property
    def f(self):
        """
        Returns the pixel values of the MAOPPY PSF for all pixels defined in
        the instance and for the current set of PSF parameters.
        """
        # check if flux needs to be recalculated
        if self._f is None:

            if not self.used.any():
                self._f = np.array([])
            else:
                # init MAOPPY code
                psf = Psfao(npix=self.maoppy_shape, system=muse_nfm, samp=self.samp)

                # get offsets of centre to nearest lower pixels.
                dx = self.xc - int(self.xc)
                dy = self.yc - int(self.yc)

                # run MAOPPY code
                __f = psf((self.r0, self.bck, self.amp, self.alpha, self.q, self.theta_maoppy, self.beta), dx=dy, dy=dx)

                # account for offset between PSF array and array used by Maoppy
                shift_x = int(self.xc) - self.maoppy_shape[0]//2
                shift_y = int(self.yc) - self.maoppy_shape[1]//2

                # MAOPPY returns 2D array. Need to revert to pixel sampling used by instance.
                ip = RectBivariateSpline(x=np.arange(self.maoppy_shape[1]) + shift_x,
                                         y=np.arange(self.maoppy_shape[0]) + shift_y,
                                         z=__f, kx=1, ky=1)

                # scale integrated flux to provided magnitude of source
                self._f = self.int_flux*ip(self.x[self.used], self.y[self.used], grid=False)

        return self._f

    @property
    def maoppy_shape(self):
        """
        Returns the minimum shape of a 2D array with an even number of pixels
        along both axes that fits the currently defined PSF profile.
        """
        return 2*int(round(self.maxrad)), 2*int(round(self.maxrad))

    def get_maoppy_centre(self, k, shape=None):
        """
        DEPRECATED

        Determine the centre of the PSF profile that is used by the MAOPPY
        code for a given oversampling factor `k` and array size `shape`.

        Parameters
        ----------
        k : int
            The oversampling used by MAOPPY.
        shape : tuple of 2 integers, optional
            The size of the array on which the PSF is computed. By default,
            the array size of the currently defined PSF is used.

        Returns
        -------
        centre : tuple of 2 floats
            The central coordinates of the PSF along the y- and x-axes.
        """
        warnings.warn('This method should not be used with Maoppy versions >=1.4.0', DeprecationWarning)

        if shape is None:
            shape = self.maoppy_shape
        return shape[0]/2 - (k - 1.)/(2.*k), shape[1]/2 - (k - 1.)/(2.*k)

    @classmethod
    def calculate_fwhm(cls, r0, bck, amp, alpha, e, beta, wvl=6e-7, **kwargs):
        """
        Estimate and return the FWHM of the MAOPPY PSF profile for a given
        set of parameters.

        To estimate the FWHM, the method creates the MAOPPY PSF on a 2dim.
        grid, then determines the symmetric radial profile, and tries to
        obtain the radii where the intensity drops to half its central value
        using univariate spline interpolation.

        Parameters
        ----------
        r0 : float
            The Fried parameter of the PSF profile, given in units of [m].
        bck : float
            The AO-corrected phase PSD (power spectral density) background,
            given in units of [rad^2 m^2]. Note that this parameter is called
            C in the publication by Fetick et al. (2019).
        amp : float
            The residual variance of the PSD, given in units of [rad^2]. Note
            that this parameter is called A in the publication of Fetick et
            al. (2019).
        alpha : float
            The effective radius of the Moffat profile, given in units of
            [m^-1].
        e : float
            The ellipticity of the Moffat profile. Must be >= 0 and < 1.
        beta : float
            The kurtosis paramter of the Moffat profile.
        wvl : float, optional
            The wavelength for which the PSF should be calculated, given in
            units of [m].
        kwargs

        Returns
        -------
        fwhm : float
            The FWHM of the MAOPPY PSF for the given input parameters.
        """
        # make sure no additional keywords were provided.
        if kwargs:
            raise IOError('Unknown argument(s) to call of Maoppy.fwhm: {0}'.format(kwargs))

        psf = Psfao(npix=(30, 30), system=muse_nfm, samp=muse_nfm.samp(wvl))
        theta = 0.
        f = psf((r0, bck, amp, alpha, 1 - e, theta, beta))

        r, fr = circavgplt(f)

        g = InterpolatedUnivariateSpline(r, fr - fr[abs(r).argmin()] / 2.)
        roots = g.roots()

        if len(roots) != 2:
            logger.error("Could not estimate FWHM of MAOPPY profile.")
            return np.nan
        else:
            return roots.max() - roots.min()

    @property
    def fwhm(self):
        """
        Returns
        -------
        fwhm : float
            The FWHM of the MAOPPY PSF profile with the given parameters.
        """
        return Maoppy.calculate_fwhm(r0=self.r0, bck=self.bck, amp=self.amp, alpha=self.alpha, e=self.e,
                                     beta=self.beta, wvl=self.wvl)

    @property
    def strehl(self):
        """
        Returns
        -------
        strehl : float
            The Strehl ratio for the current set of parameters.
        """
        psf = Psfao(npix=(30, 30), system=muse_nfm, samp=muse_nfm.samp(self.wvl))
        return psf.strehlOTF((self.r0, self.bck, self.amp, self.alpha, 1 - self.e, self.theta_maoppy, self.beta))

        # Analytical calculation based on eq. 14 from the paper by Fetick et al. (2019). This seems to underestimate
        # the Strehl significantly for NFM data.
        # f_ao = muse_nfm.Nact/(2.*muse_nfm.D)
        # return np.exp(-self.amp - self.bck*np.pi*(f_ao**2) - (0.023*6.*np.pi*(self.r0*f_ao)**(-5./3.))/5.)

    def fitter(self, psf, weights, parameters, fit_positions=True, fit_background=False, psf_padding=0, **kwargs):
        """
        This methods implements the fit of a MAOPPY PSF to a given data array.

        Notes
        -----
        The actual fit is performed using the method `psffit` available in the
        MAOPPY package. It requires a 2D array with an even number of pixels
        along each dimension. Therefore, the provided 1D data arrays need to
        be resampled prior to the fit.

        Parameters
        ----------
        psf : 1D-sequence of floats
            The data values for the pixels on which the PSF profile is
            defined.
        weights : 1D-sequence of floats
            The fitting weights for the pixels on which the PSF profile is
            defined.
        parameters : list of strings
            The names of the PSF parameters that should be optimized during
            the fit.
        fit_positions : bool, optional
            Flag indicating if the x- and y-coordinates of the PSF centre
            should be optimized
        fit_background : bool, optional
            Flag indicating if a constant background component should be
            included in the PSF fit.
        psf_padding : int, optional
            The number of pixels attached to each edge of the 2D array on
            which the PSF fit is performed.
        kwargs

        Returns
        -------
        fit_result : instance of pampelmuse.psf.profile.FitResult
            A named tuple holding the results of the PSF fit.
        """

        # if kwargs:
        #     raise IOError('Unknown parameter(s) {} for method Maoppy.fitter.'.format(kwargs))

        if not fit_positions:
            logger.warning('Maoppy.fitter always includes the centroid coordinates in the fit.')

        x0 = []
        names = ['r0', 'bck', 'amp', 'alpha', 'e', 'theta', 'beta']
        fixed = np.ones(len(names), dtype=bool)
        for i, parameter in enumerate(names):
            if parameter in parameters:
                fixed[i] = False
            # axis ratio is fitted instead of ellipticity
            if parameter == 'e':
                x0.append(self.q)
            elif parameter == 'theta':
                x0.append(self.theta_maoppy)
            else:
                x0.append(getattr(self, parameter))

        y_center, x_center = self.maoppy_shape[0]//2, self.maoppy_shape[1]//2
        shift_x = int(round(x_center - self.xc))
        shift_y = int(round(y_center - self.yc))

        # # subtract minimum of x- and y-coordinates, so that array starts at (0, 0)
        # x_min = int(self.x[self.used].min()) + max(0, self.maxrad - int(self.xc))
        # y_min = int(self.y[self.used].min()) + max(0, self.maxrad - int(self.yc))

        # code requires 2D array with even number of pixels along both axes
        psf2d = resample_to_2d(x=self.x[self.used] + shift_x, y=self.y[self.used] + shift_y, z=psf,
                               shape=self.maoppy_shape)
        weights2d = resample_to_2d(x=self.x[self.used] + shift_x, y=self.y[self.used] + shift_y, z=weights,
                                   shape=self.maoppy_shape)

        if np.isnan(psf2d).any() or np.isnan(weights2d).any():
            logger.warning('Input to Maoppy PSF fit contains invalid data.')
            psf2d[np.isnan(psf2d) | np.isnan(weights2d)] = 0
            weights2d[np.isnan(psf2d) | np.isnan(weights2d)] = 0

        # from astropy.io import fits
        # out = fits.HDUList([fits.PrimaryHDU(psf2d), fits.ImageHDU(weights2d)])
        # out.writeto('test_{0}.fits'.format(self.src_id), overwrite=1)

        # get initial guess for shifts along x- and y-axis
        dx = round(x_center - self.xc) - (x_center - self.xc)
        dy = round(y_center - self.yc) - (y_center - self.yc)

        npix = max(psf2d.shape)
        if psf_padding > 0:
            npixfit = npix + 2*psf_padding
        else:
            npixfit = None

        # perform fit
        # code prints out some "-" symbols to indicate progression, which are captured so that is does not interfere
        # with the logging
        f = io.StringIO()
        pmodel = Psfao(npix=(npix+2*psf_padding, npix+2*psf_padding), system=muse_nfm, samp=self.samp)
        with redirect_stdout(f):
            fitresult = psffit(psf2d, pmodel, x0, weights=weights2d, dxdy=(dx, dy), fixed=fixed,
                               flux_bck=(True, fit_background), positive_bck=False, npixfit=npixfit)
            logger.debug(f.getvalue())

        # return ellipticity e instead of axis ratio q
        x0 = [x0[names.index(p)] for p in parameters]
        if 'e' in parameters:
            i = parameters.index('e')
            x0[i] = 1. - x0[i]
        # return position angle as used by PampelMuse
        if 'theta' in parameters:
            i = parameters.index('theta')
            x0[i] = x0[i] - np.pi/2.

        if fitresult.success:
            centroid = (x_center + fitresult.dxdy[0] - shift_x, y_center + fitresult.dxdy[1] - shift_y)

            if psf_padding == 0:
                bestfit_model = fitresult.psf
            else:
                # note that max(psf2d.shape) - psf2d.shape[?] is always even, given that both axes not to have even
                # number of pixels
                pad_x = psf_padding + (max(psf2d.shape) - psf2d.shape[1])//2
                pad_y = psf_padding + (max(psf2d.shape) - psf2d.shape[0])//2
                bestfit_model = fitresult.psf[pad_y:-pad_y, pad_x:-pad_x]

            residuals_2d = psf2d - fitresult.flux_bck[1] - fitresult.flux_bck[0] * bestfit_model
            residuals_2d = np.pad(residuals_2d, 1, mode='constant', constant_values=0)
            residuals_1d = resample_to_1d(z=residuals_2d, x=self.x[self.used] + shift_x + 1,
                                          y=self.y[self.used] + shift_y + 1)

            # return ellipticity e instead of axis ratio q
            values = [fitresult.x[names.index(p)] for p in parameters]
            if 'e' in parameters:
                i = parameters.index('e')
                values[i] = 1. - values[i]
            # return position angle as used by PampelMuse
            if 'theta' in parameters:
                i = parameters.index('theta')
                values[i] = values[i] - np.pi / 2.

            return FitResult(names=parameters,
                             x0=x0,
                             values=values,
                             flux=fitresult.flux_bck[0],
                             background=fitresult.flux_bck[1] if fit_background else None,
                             centroid=centroid if fit_positions else None,
                             success=fitresult.success,
                             residuals=residuals_1d,
                             message=fitresult.message)

        else:
            return FitResult(names=parameters,
                             x0=x0,
                             values=np.nan*np.ones_like(x0),
                             flux=np.nan,
                             background=np.nan if fit_background else None,
                             centroid=(np.nan, np.nan) if fit_positions else None,
                             success=fitresult.success,
                             residuals=None,
                             message=fitresult.message)
