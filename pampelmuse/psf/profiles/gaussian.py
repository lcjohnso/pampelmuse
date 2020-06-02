"""
gaussian.py
===========
Copyright 2013-2018 Sebastian Kamann

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
class pampelmuse.core.psf.profiles.Gaussian


Purpose
-------
This module implements an analytical PSF profile in the form of a Gaussian.


Latest Git revision
-------------------
2019/06/05
"""
from numbers import Real
import numpy as np
from .profile import Profile
from ..functions import gaussian


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20180605


class Gaussian(Profile):
    """
    This class implements a PSF that follow an analytical Gaussian function.
    """
    function = "gaussian"

    def __init__(self, uc, vc, mag, fwhm=3.0, **kwargs):
        """
        Initialize an instance of the Gaussian-class.

        Parameters
        ----------
        uc : float
            The x-centroid of the profile in the reference system.
        vc : float
            The y-centroid of the profile in the reference system.
        mag : float
            The integrated magnitude of the profile. The integrated flux is
            unity for a magnitude of zero.
        fwhm : float, optional
            The FWHM of the profile, given in units of pixels.
        src_id : int, optional
            The ID of the source in any catalogue.
        **kwargs
            Any other keyword provided to this method are used to initialize
            the super-class.
        """
        self._fwhm = fwhm

        # call super-class
        super(Gaussian, self).__init__(uc=uc, vc=vc, mag=mag, **kwargs)

        self._diff_fwhm = None  # place-holder for partial differential wrt. the FWHM
        self.profile_properties.append("_diff_fwhm")

    @property
    def fwhm(self):
        """
        Returns the FWHM of the current profile.
        """
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        """
        Set the FWHM for this instance.

        Parameters
        ----------
        value : float
            The FWHM of the profile.
        """
        assert isinstance(value, Real), "'fwhm' must be a float, not {0:s}".format(type(value).__name__)
        assert value > 0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))
        if value != self._fwhm:
            self._fwhm = value

            # all properties that use the FWHM need to be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    # The property 'f' that returns the actual spatial intensity profile
    @property
    def f(self):
        """
        Returns the intensities of the current Gauss profile for all defined
        (sub)pixels.
        """
        if self._f is None:
            self._f = self.int_flux*gaussian.flux(self.r, self.fwhm, self.e)
        return self._f

    # In the following, the properties are provided to calculate the partial derivatives of the intensity distribution
    # provided by the Gaussian profile wrt. the individual parameters.
    @property
    def diff_fwhm(self):
        """
        Returns the partial derivative of the profile with respect to fwhm,
        i.e. the full-width at half maximum of the Gaussian.
        """
        if self._diff_fwhm is None:
            self._diff_fwhm = gaussian.pderiv_fwhm(self.r, self.fwhm, self.e)
        return self._diff_fwhm

    def get_diff_fwhm(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to fwhm, i.e. the full-width at half maximum of the Gaussian.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_fwhm : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if integrate:
            return self.sum_os(self.diff_fwhm)
        else:
            return self.diff_fwhm

    # Overwrite partial derivatives with respect to parameters already defined in parent class.
    @property
    def diff_e(self):
        """
        Returns the partial derivative of the profile with respect to e, i.e.
        the ellipticity.
        """
        if self._diff_e is None:
            r_pderiv_e = super().diff_e
            self._diff_e = gaussian.pderiv_e(self.r, self.fwhm, self.e, r_pderiv_e)
        return self._diff_e

    @property
    def diff_theta(self):
        """
        Returns the partial derivative of the profile with respect to theta,
        i.e. the position angle.
        """
        if self._diff_theta is None:
            r_pderiv_theta = super().diff_theta
            self._diff_theta = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_theta
        return self._diff_theta

    @property
    def diff_xc(self):
        """
        Returns the partial derivative of the profile with respect to xc, i.e.
        the centroid x-coordinate.
        """
        if self._diff_xc is None:
            r_pderiv_xc = super().diff_xc
            self._diff_xc = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_xc
        return self._diff_xc

    @property
    def diff_yc(self):
        """
        Returns the partial derivative of the profile with respect to yc, i.e.
        the centroid y-coordinate.
        """
        if self._diff_yc is None:
            r_pderiv_yc = super().diff_yc
            self._diff_yc = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_yc
        return self._diff_yc

    @property
    def diff_A(self):
        """
        Returns the partial derivative of the profile with respect to A, i.e.
        the [0,0] entry of the coordinate transformation matrix.
        """
        if self._diff_A is None:
            r_pderiv_A = super().diff_A
            self._diff_A = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_A
        return self._diff_A

    @property
    def diff_B(self):
        """
        Returns the partial derivative of the profile with respect to B, i.e.
        the [0,1] entry of the coordinate transformation matrix.
        """
        if self._diff_B is None:
            r_pderiv_B = super().diff_B
            self._diff_B = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_B
        return self._diff_B

    @property
    def diff_C(self):
        """
        Returns the partial derivative of the profile with respect to C, i.e.
        the [1,0] entry of the coordinate transformation matrix.
        """
        if self._diff_C is None:
            r_pderiv_C = super().diff_C
            self._diff_C = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_C
        return self._diff_C

    @property
    def diff_D(self):
        """
        Returns the partial derivative of the profile with respect to D, i.e.
        the [1,1] entry of the coordinate transformation matrix.
        """
        if self._diff_D is None:
            r_pderiv_D = super().diff_D
            self._diff_D = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_D
        return self._diff_D

    @property
    def diff_x0(self):
        """
        Returns the partial derivative of the profile with respect to x0, i.e.
        the x-coordinate shift of the coordinate transformation.
        """
        if self._diff_x0 is None:
            r_pderiv_x0 = super().diff_x0
            self._diff_x0 = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_x0
        return self._diff_x0

    @property
    def diff_y0(self):
        """
        Returns the partial derivative of the profile with respect to y0, i.e.
        the y-coordinate shift of the coordinate transformation.
        """
        if self._diff_y0 is None:
            r_pderiv_y0 = super().diff_y0
            self._diff_y0 = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_y0
        return self._diff_y0

    @property
    def bounds(self):
        bounds = super(Gaussian, self).bounds
        bounds['fwhm'] = (0, np.inf)
        return bounds
