"""
double_gaussian.py
==================
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
class pampelmuse.core.psf.profiles.DoubleGaussian


Purpose
-------
This module implements an analytical PSF profile in the form of a combination
of two Gaussian profiles.


Latest Git revision
-------------------
2018/04/24
"""
from numbers import Real
from .profile import Profile
from ..functions import gaussian


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20180424


class DoubleGaussian(Profile):
    """
    This class describes a two dimensional double Gaussian profile. The
    functional form of the double Gaussian with unity flux is given as

    dG(r) = f*G1(r) + (1-f)*G2(r),

    where G1 and G2 are both Gaussian profiles, i.e. have functional form

    G(r) = 1/(2*pi*std^2*(1-e))*exp(-r^2/(2*std**2)).

    In this implementation, each Gaussian has up to three free parameters, the
    FWHM, the ellipticity e, and the position angle theta. The ratio between
    the flux in profile 1 and 2 is labeled f12.
    """
    function = "double gaussian"

    def __init__(self, uc, vc, mag, f12=0.5, fwhm1=3, fwhm2=10, **kwargs):
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
        f12 : float, optional
            The flux ratio between the two Gaussians, provided as
            f_1/(f_1 + f_2). Must be between 0 and 1.
        fwhm1 : float, optional
            The FWHm of the first Gaussian in units oof pixels. Muse be >0.
        fwhm2 : float, optional
            The FWHm of the second Gaussian in units oof pixels. Muse be >0.
        kwargs
            Any other keyword provided to this method will be parsed to the
            initialization method of the current super-class.
        """
        self._f12 = f12
        self._fwhm1 = fwhm1
        self._fwhm2 = fwhm2

        # call super-class
        super(DoubleGaussian, self).__init__(uc=uc, vc=vc, mag=mag, **kwargs)

        self._diff_f12 = None  # place-holder for partial differential wrt. the flux ratio between Gaussians 1 and 2
        self._diff_fwhm1 = None  # place-holder for partial differential wrt. the FWHM of Gaussian 1
        self._diff_fwhm2 = None  # place-holder for partial differential wrt. the FWHM of Gaussian 2
        self.profile_properties.extend(["_diff_f12", "_diff_fwhm1", "_diff_fwhm2"])

    # Define the 'f12' property to set and return the flux ratio of the two Gaussians ########
    @property
    def f12(self):
        """
        Returns the fractional flux in the first Gaussian. Note that the value
        is defined as f_1/(f_1 + f_2).
        """
        return self._f12

    @f12.setter
    def f12(self, value):
        """
        Set the flux ratio of the Gaussians for this instance.

        Parameters
        ----------
        value : float
            The flux ratio of the Gaussians of the profile.
        """
        assert isinstance(value, Real), "'f12' must be a float, not {0:s}".format(type(value).__name__)
        assert 0. < value < 1, "'f12' must be in range [0, 1] but is {0:.1f}".format(float(value))
        if value != self._f12:
            self._f12 = value

            # all properties that use the FWHM need to  be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    # Define the 'fwhm1' property to set and return the FWHM of Gaussian 1 ########
    @property
    def fwhm1(self):
        """
        Return the FWHM of Gaussian 1.

        Returns
        -------
        fwhm1 : float
            The FWHM of the profile.
        """
        return self._fwhm1

    @fwhm1.setter
    def fwhm1(self, value):
        """
        Set the FWHM of Gaussian 1 for this instance.

        Parameters
        ----------
        value : float
            The FWHM of the profile.
        """
        assert isinstance(value, Real), "'fwhm1' must be a float, not {0:s}".format(type(value).__name__)
        assert value > 0, "'fwhm1' must be >0 but is {0:.1f}".format(float(value))
        if value != self._fwhm1:
            self._fwhm1 = value

            # all properties that use the FWHM need to  be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    # Define the 'fwhm2' property to set and return the FWHM of Gaussian 2 ########
    @property
    def fwhm2(self):
        """
        Return the FWHM of Gaussian 2.

        Returns
        -------
        fwhm2 : float
            The FWHM of Gaussian 2.
        """
        return self._fwhm2

    @fwhm2.setter
    def fwhm2(self, value):
        """
        Set the FWHM of Gaussian 2 for this instance.

        Parameters
        ----------
        value : float
            The FWHM of the profile.
        """
        assert isinstance(value, Real), "'fwhm2' must be a float, not {0:s}".format(type(value).__name__)
        assert value > 0, "'fwhm2' must be >0 but is {0:.1f}".format(float(value))
        if value != self._fwhm2:
            self._fwhm2 = value

            # all properties that use the FWHM need to  be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    @property
    def fwhm(self):
        """
        Estimate the FWHM of the PSF as the weighted ratio of the two
        individual Gaussians.
        """
        return self.f12*self.fwhm1 + (1. - self.f12)*self.fwhm2

    # The property 'f' that returns the actual spatial intensity profile
    def get_f(self, integrate=True):
        """
        Method that calculates the intensity profile of the double Gaussian.

        Parameters:
        -----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns:
        --------
        f : nd_array
            The intensity distribution for the pixels used in the calculation.
            If parameter 'integrate' is set, the intensities are returned only
            for the parents pixels after averaging over the sub-pixels.
        """
        if self._f is None:
            self._f = 10 ** (-0.4 * self.mag) * (self.f12 * gaussian.flux(self.r, self.fwhm1, self.e) + (
                1. - self.f12) * gaussian.flux(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._f)
        else:
            return self._f

    def get_diff_f12(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile with
        respect to the fractional flux of Gaussian 1.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_f12 : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_f12 is None:
            self._diff_f12 = self.int_flux * (gaussian.flux(self.r, self.fwhm1, self.e) - gaussian.flux(
                self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_f12)
        else:
            return self._diff_f12

    @property
    def diff_f12(self):
        return self.get_diff_f12()

    def get_diff_fwhm1(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile with
        respect to the FWHM of Gaussian 1.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_fwhm1 : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_fwhm1 is None:
            self._diff_fwhm1 = self.int_flux * gaussian.pderiv_fwhm(self.r, self.fwhm1, self.e)
        if integrate:
            return self.sum_os(self._diff_fwhm1)
        else:
            return self._diff_fwhm1

    @property
    def diff_fwhm1(self):
        return self.get_diff_fwhm1()

    def get_diff_fwhm2(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile with
        respect to the FWHM of Gaussian 2.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_fwhm2 : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_fwhm2 is None:
            self._diff_fwhm2 = self.int_flux * gaussian.pderiv_fwhm(self.r, self.fwhm2, self.e)
        if integrate:
            return self.sum_os(self._diff_fwhm2)
        else:
            return self._diff_fwhm2

    @property
    def diff_fwhm2(self):
        return self.get_diff_fwhm2()

    # The ellipticity 'e'
    def get_diff_e(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile with
        respect to the ellipticity 'e'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_e : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_e is None:
            r_pderiv_e = super(DoubleGaussian, self).get_diff_e(integrate=False)
            self._diff_e = self.int_flux*(
                self.f12 * gaussian.pderiv_e(self.r, self.fwhm1, self.e, r_pderiv_e) + (
                    (1. - self.f12) * gaussian.pderiv_e(self.r, self.fwhm2, self.e, r_pderiv_e)))
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    # The position angle 'theta'
    def get_diff_theta(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the position angle 'theta'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_theta : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_theta is None:
            r_pderiv_theta = super(DoubleGaussian, self).get_diff_theta(integrate=False)
            self._diff_theta = self.int_flux * r_pderiv_theta * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    # The centroid coordinates
    def get_diff_xc(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        xc, the centroid of the profile in x-direction.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_xc : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_xc is None:
            r_pderiv_xc = super(DoubleGaussian, self).get_diff_xc(integrate=False)
            self._diff_xc = self.int_flux * r_pderiv_xc * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_xc)
        else:
            return self._diff_xc

    def get_diff_yc(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        yc, the centroid of the profile in y-direction.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_yc : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_yc is None:
            r_pderiv_yc = super(DoubleGaussian, self).get_diff_yc(integrate=False)
            self._diff_yc = self.int_flux * r_pderiv_yc * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_yc)
        else:
            return self._diff_yc

    # The parameters of the coordinate transformation
    def get_diff_A(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the parameter 'A' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_A : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_A is None:
            r_pderiv_A = super(DoubleGaussian, self).get_diff_A(integrate=False)
            self._diff_A = self.int_flux * r_pderiv_A * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A

    def get_diff_B(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the parameter 'B' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_B : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_B is None:
            r_pderiv_B = super(DoubleGaussian, self).get_diff_B(integrate=False)
            self._diff_B = self.int_flux * r_pderiv_B * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    def get_diff_C(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the parameter 'C' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_C : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_C is None:
            r_pderiv_C = super(DoubleGaussian, self).get_diff_C(integrate=False)
            self._diff_C = self.int_flux * r_pderiv_C * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    def get_diff_D(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the parameter 'D' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_B : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_D is None:
            r_pderiv_D = super(DoubleGaussian, self).get_diff_D(integrate=False)
            self._diff_D = self.int_flux * r_pderiv_D * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    def get_diff_x0(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the parameter 'x0' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_x0 : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_x0 is None:
            r_pderiv_x0 = super(DoubleGaussian, self).get_diff_x0(integrate=False)
            self._diff_x0 = self.int_flux * r_pderiv_x0 * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_x0)
        else:
            return self._diff_x0

    def get_diff_y0(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the parameter 'y0' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_y0 : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_y0 is None:
            r_pderiv_y0 = super(DoubleGaussian, self).get_diff_y0(integrate=False)
            self._diff_y0 = self.int_flux * r_pderiv_y0 * (
                self.f12 * gaussian.pderiv_r(self.r, self.fwhm1, self.e) + (
                    1. - self.f12) * gaussian.pderiv_r(self.r, self.fwhm2, self.e))
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0
