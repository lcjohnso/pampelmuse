import logging
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp2d
from .profile import Profile, FitResult

from maoppy.psfmodel import Psfao, psffit, oversample
from maoppy.instrument import MUSE_NFM


logger = logging.getLogger(__name__)


def resample_to_2d(x, y, z, shape=None):
    """
    Given a set of values z defined on arbitrary coordinates (x, y), the method
    will translate the data to a regularly sampled 2d array.

    Parameters
    ----------
    x
    y
    z
    shape

    Returns
    -------

    """
    if shape is None:
        shape = (int(y.max()) + 1 - int(y.min()), int(x.max()) + 1 - int(x.min()))

    is_int = np.all(x == x.astype(np.int32)) & np.all(y == y.astype(np.int32))

    if is_int:
        out = np.zeros(shape, dtype=z.dtype)
        out[y.astype(np.int32) - int(y.min()), x.astype(np.int32) - int(x.min())] = z
    else:
        logger.warning('Input data irregularly sampled. Need to interpolate.')

        ip = interp2d(x=x, y=y, z=z, kind='linear')
        out = ip(np.arange(shape[1], dtype=np.int32), np.arange(shape[0], dtype=np.int32), assume_sorted=True)

    return out


def resample_to_1d(z, x, y,):
    """
    Given a regularly sampled 2D array of values z, the method will convert
    the data to the provided (randomly sampled) coordinates x and y.

    Parameters
    ----------
    z
    x
    y

    Returns
    -------

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

    def __init__(self, uc, vc, mag, r0=0.2, bck=0.005, amp=1, alpha=0.05, beta=2.0, wvl=5e-7, **kwargs):

        if 'osradius' in kwargs and kwargs['osradius'] > 0:
            logger.error('Oversampling not supported for MAOPPY PSF. Setting osradius=0 (was {})'.format(
                kwargs['osradius']))
            kwargs['osradius'] = 0
        if 'osfactor' in kwargs and kwargs['osfactor'] != 1:
            logger.error('Oversampling not supported for MAOPPY PSF. Setting osfactor=1 (was {})'.format(
                kwargs['osfactor']))
            kwargs['osfactor'] = 1

        super(Maoppy, self).__init__(uc=uc, vc=vc, mag=mag, **kwargs)

        self._r0 = r0
        self._bck = bck
        self._amp = amp
        self._alpha = alpha
        self._beta = beta
        self._wvl = wvl

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, value):
        self._r0 = value

    @property
    def bck(self):
        return self._bck

    @bck.setter
    def bck(self, value):
        self._bck = value

    @property
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, value):
        self._amp = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def wvl(self):
        return self._wvl

    @wvl.setter
    def wvl(self, value):
        self._wvl = value

    @property
    def q(self):
        return 1. - self.e

    @property
    def samp(self):
        return MUSE_NFM.samp(self.wvl)

    @property
    def f(self):

        if self._f is None:

            if not self.used.any():
                self._f = np.array([])
            else:
                psf = Psfao(Npix=self.maoppy_shape, system=MUSE_NFM, samp=self.samp)

                dx = self.xc - int(self.xc)
                dy = self.yc - int(self.yc)

                __f = psf((self.r0, self.bck, self.amp, self.alpha, self.q, self.theta, self.beta), dx=dy, dy=dx)

                x_center, y_center = self.get_maoppy_centre(k=psf._k)
                shift_x = int(self.xc) - x_center
                shift_y = int(self.yc) - y_center

                ip = RectBivariateSpline(x=np.arange(self.maoppy_shape[1]) + shift_x,
                                         y=np.arange(self.maoppy_shape[0]) + shift_y,
                                         z=__f, kx=1, ky=1)

                self._f = self.int_flux*ip(self.x[self.used], self.y[self.used], grid=False)

        return self._f

    @property
    def maoppy_shape(self):
        return 2*int(self.maxrad), 2*int(self.maxrad)

    def get_maoppy_centre(self, k, shape=None):
        if shape is None:
            shape = self.maoppy_shape
        return shape[0]/2 - (k - 1.)/(2.*k), shape[1]/2 - (k - 1.)/(2.*k)

    @classmethod
    def calculate_fwhm(cls, r0, bck, amp, alpha, e, beta, wvl=6e-7, **kwargs):
        # make sure no additional keywords were provided.
        if kwargs:
            raise IOError('Unknown argument(s) to call of Maoppy.fwhm: {0}'.format(kwargs))

        psf = Psfao(Npix=(30, 30), system=MUSE_NFM, samp=MUSE_NFM.samp(wvl))

        theta = 0.
        f = psf((r0, bck, amp, alpha, 1-e, theta, beta))[:15]

        return f[f > (0.5*f.max())].sum()

    def fitter(self, psf, weights, parameters, fit_positions=True, fit_background=False, psf_padding=0, **kwargs):

        if kwargs:
            raise IOError('Unknown parameter(s) {} for method Maoppy.fitter.'.format(kwargs))

        if not fit_positions:
            logger.warning('Maoppy.fitter always includes the centroid coordinates in the fit.')

        x0 = []
        names = ['r0', 'bck', 'amp', 'alpha', 'e', 'theta', 'beta']
        fixed = []
        for parameter in names:
            fixed.append(False if parameter in parameters else True)
            if parameter == 'e':
                x0.append(self.q)
            else:
                x0.append(getattr(self, parameter))

        # subtract minimum of x- and y-coordinates, so that array starts at (0, 0)
        x_min = int(self.x[self.used].min())
        y_min = int(self.y[self.used].min())

        # code requires 2D array with even number of pixels along both axes
        psf2d = resample_to_2d(x=self.x[self.used] - x_min, y=self.y[self.used] - y_min, z=psf)
        weights2d = resample_to_2d(x=self.x[self.used] - x_min, y=self.y[self.used] - y_min, z=weights)

        pad = ((0, 1) if psf2d.shape[0] % 2> 0 else (0, 0), (0, 1) if psf2d.shape[1] % 2> 0 else (0, 0))
        psf2d = np.pad(psf2d, pad)
        weights2d = np.pad(weights2d, pad)

        # get initial guess for shifts along x- and y-axis
        y_center, x_center = self.get_maoppy_centre(k=oversample(self.samp)[1], shape=psf2d.shape)
        dx = self.xc - x_min - x_center
        dy = self.yc - y_min - y_center

        # weights = 1. / (psf2d + MUSE_NFM.ron ** 2.0)  # np.ones_like(psf_data)
        # weights[psf2d < 0.01] = 0

        if psf_padding > 0:
            npix = max(psf2d.shape)
            npixfit = npix + 2*psf_padding
        else:
            npixfit = None

        # print("PSFAO fitting (please wait)")
        fitresult = psffit(psf2d, Psfao, x0, weights=weights2d, system=MUSE_NFM, samp=self.samp, dxdy=(dx, dy),
                           fixed=fixed, flux_bck=(True, fit_background), positive_bck=False, npixfit=npixfit)

        if fitresult.success:
            centroid = (x_min + x_center + fitresult.dxdy[0], y_min + y_center + fitresult.dxdy[1])

            if psf_padding is None:
                bestfit_model = fitresult.psf
            else:
                # note that max(psf2d.shape) - psf2d.shape[?] is always even, given that both axes not to have even
                # number of pixels
                pad_x = psf_padding + (max(psf2d.shape) - psf2d.shape[1])//2
                pad_y = psf_padding + (max(psf2d.shape) - psf2d.shape[0])//2
                bestfit_model = fitresult.psf[pad_y:-pad_y, pad_x:-pad_x]

            residuals_2d = psf2d - fitresult.flux_bck[1] - fitresult.flux_bck[0] * bestfit_model
            residuals_1d = resample_to_1d(z=residuals_2d, x=self.x[self.used] - x_min, y=self.y[self.used] - y_min)

            return FitResult(names=parameters,
                             x0=[x0[names.index(p)] for p in parameters],
                             values=[fitresult.x[names.index(p)] for p in parameters],
                             flux=fitresult.flux_bck[0],
                             background=fitresult.flux_bck[1] if fit_background else None,
                             centroid=centroid if fit_positions else None,
                             success=fitresult.success,
                             residuals=residuals_1d)

        else:
            return FitResult(names=parameters,
                             x0=[x0[names.index(p)] for p in parameters],
                             values=np.nan*np.ones_like(x0),
                             flux=np.nan,
                             background=np.nan if fit_background else None,
                             centroid=(np.nan, np.nan) if fit_positions else None,
                             success=fitresult.success,
                             residuals=None)
