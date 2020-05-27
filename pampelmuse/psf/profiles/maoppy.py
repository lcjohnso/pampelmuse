import logging
import numpy as np
from scipy.interpolate import RectBivariateSpline
from .profile import Profile

from maoppy.psfmodel import Psfao
from maoppy.instrument import MUSE_NFM


logger = logging.getLogger(__name__)


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

    def get_maoppy_centre(self, k):
        return self.maoppy_shape[0]/2 - (k - 1.)/(2.*k), self.maoppy_shape[1]/2 - (k - 1.)/(2.*k)

    @classmethod
    def calculate_fwhm(cls, r0, bck, amp, alpha, e, beta, wvl=6e-7, **kwargs):
        # make sure no additional keywords were provided.
        if kwargs:
            raise IOError('Unknown argument(s) to call of Maoppy.fwhm: {0}'.format(kwargs))

        psf = Psfao(Npix=(30, 30), system=MUSE_NFM, samp=MUSE_NFM.samp(wvl))

        theta = 0.
        f = psf((r0, bck, amp, alpha, 1-e, theta, beta))[:15]

        return f[f > (0.5*f.max())].sum()

    # def fitter(self, fluxes, x0, fit_positions=True, fit_background=False):
    #