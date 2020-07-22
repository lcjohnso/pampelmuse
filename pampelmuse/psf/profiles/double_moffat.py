"""
double_moffat.py
================
Copyright 2013-2019 Sebastian Kamann

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
class pampelmuse.core.psf.profiles.DoubleMoffat


Purpose
-------
This module implements an analytical PSF profile in the form of a combination
of two Moffat profiles.

Added
-----
2019/02/03

Latest Git revision
-------------------
2019/06/30
"""
import logging
import numpy as np
from numbers import Real
from scipy import optimize
from .profile import Profile
from ..functions import moffat


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20190630


class DoubleMoffat(Profile):
    """
    This class describes a two dimensional profile of a Moffat and a central
    Gaussian component. The functional form of the profile with unity flux is
    given as

    GM(r) = f*M_1(r) + (1-f)*M_2(r),

    where both M_1 and M_2 are Moffat profiles, i.e. have functional form
    
    M(r) = (beta-1)/(pi*r_d^2*(1-e))*[1+(r/r_d)^2]^-beta,
    
    In this implementation, the hybrid profile has up to 7 free parameters,
    the FWHM and beta of each Moffat profile, the (global) ellipticity e,
    the position angle theta, and the fractional flux in the central Moffat
    component, labelled f12.
    """
    function = "double moffat"

    @classmethod
    def calculate_fwhm(cls, f12, beta1, fwhm1, beta2, fwhm2, e=0, **kwargs):
        """
        Method to determine the FWHM of the full profile for a given set of
        input parameters.

        Parameters
        ----------
        f12 : float
            The fractional flux in the first Moffat component.
        beta1 : float
            The beta-parameter of the first Moffat component.
        fwhm1 : float
            The full-width at half maximum of the first Moffat component.
        beta2 : float
            The beta-parameter of the second Moffat component.
        fwhm2 : float
            The full-width at half maximum of the second Moffat component.
        e : float, optional
            The ellipticity of the full profile.
        kwargs

        Returns
        -------
        fwhm : float
            The full-width at half maximum of the combined profile.
        """
        # make sure no additional keywords were provided.
        if kwargs:
            raise IOError('Unknown argument(s) to call of DoubleMoffat.fwhm: {0}'.format(kwargs))

        # get central flux
        f0 = f12*moffat.flux(0, beta=beta1, fwhm=fwhm1, e=e) + (1. - f12)*moffat.flux(0, beta=beta2, fwhm=fwhm2, e=e)

        # find FWHM by locating root of profile minus 0.5 times the central flux.
        f = lambda r: 0.5 * f0 - f12 * moffat.flux(r, beta=beta1, fwhm=fwhm1, e=e) - (1. - f12) * moffat.flux(
            r, beta=beta2, fwhm=fwhm2, e=e)

        root = optimize.root_scalar(f, method='brentq', bracket=[0, 1.1 * max(fwhm1, fwhm2)])
        return 2. * root.root

    def __init__(self, uc, vc, mag, f12=0.5, beta1=2.5, fwhm1=3.0, beta2=2.5, fwhm2=6.0, **kwargs):
        """
        Initialize an instance of the DOubleMoffat-class.

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
            The fractional flux of the first Moffat component, provided as
            f_M_1/(f_M_1 + f_M_2). Must be between 0 and 1.
        beta1 : float, optional
            The beta-parameter of the first Moffat component.
        fwhm1 : float, optional
            The FWHM of the first Moffat component in units of pixels. Must
            be >0.
        beta1 : float, optional
            The beta-parameter of the first Moffat component.
        fwhm2 : float, optional
            The FWHM of the second Moffat component in units of pixels. Must
            be >0.
        kwargs
            Any other keyword provided to this method will be parsed to the
            initialization method of the current super-class.
        """
        self._f12 = f12
        self._beta1 = beta1
        self._fwhm1 = fwhm1
        self._beta2 = beta2
        self._fwhm2 = fwhm2

        # call super-class
        super(DoubleMoffat, self).__init__(uc=uc, vc=vc, mag=mag, **kwargs)

        self._diff_f12 = None  # place-holder for partial differential wrt. the flux ratio between Gaussians 1 and 2
        self._diff_beta1 = None  # place-holder for partial derivative wrt. the beta of Moffat 1.
        self._diff_fwhm1 = None  # place-holder for partial differential wrt. the FWHM of Moffat 1.
        self._diff_beta2 = None  # place-holder for partial derivative wrt. the beta of Moffat 2.
        self._diff_fwhm2 = None  # place-holder for partial differential wrt. the FWHM of Moffat 2.
        self.profile_properties.extend(["_diff_f12", "_diff_beta1", "_diff_fwhm1", "_diff_beta2", "_diff_fwhm2"])

    # Define the 'f12' property to set and return the fractional flux in the first Moffat ########
    @property
    def f12(self):
        """
        Returns the fractional flux in the first Moffat component. Note that the
        flux ratio is defined as f_Moffat(1)/(f_Moffat(1) + f_Moffat(2)).
        """
        return self._f12

    @f12.setter
    def f12(self, value):
        """
        Set the fractional flux in the first Moffat component for this instance.

        Parameters
        ----------
        value : float
            The fractional flux in the first Moffat component of the profile.
        """
        assert isinstance(value, Real), "'f12' must be a float, not {0:s}".format(type(value).__name__)
        assert 0. < value < 1, "'f12' must be in range [0, 1] but is {0:.1f}".format(float(value))
        if value != self._f12:
            self._f12 = value

            # all properties that use f12 need to  be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    @property
    def beta1(self):
        """
        Return the beta-parameter of the first Moffat component.

        Returns
        -------
        beta : float
            The beta-parameter of the profile.
        """
        return self._beta1

    @beta1.setter
    def beta1(self, value):
        """
        Set the beta-parameter of the first Moffat component for this instance.

        Parameters
        ----------
        value : float
            The new beta-parameter of the first Moffat component.
        """
        assert isinstance(value, Real), "'beta1' must be a float, not {0:s}".format(type(value).__name__)

        if not 1. < value < 10:
            logging.error("'beta1' must be >1 and <=10 but is {0:.1f}.".format(float(value)))
        assert 1. < value <= 10, "'beta1' must be >1 and <=10 but is {0:.1f}.".format(float(value))

        if value != self._beta1:
            self._beta1 = value

            # all properties that use the beta-parameter need to  be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    @property
    def fwhm1(self):
        """
        Return the FWHM of the first Moffat component.

        Returns
        -------
        fwhm1 : float
            The FWHM of the first Moffat compontent.
        """
        return self._fwhm1

    @fwhm1.setter
    def fwhm1(self, value):
        """
        Set the FWHM of the first Moffat component for this instance.

        Parameters
        ----------
        value : float
            The new FWHM of the first Moffat component.
        """
        assert isinstance(value, Real), "'fwhm1' must be a float, not {0:s}".format(type(value).__name__)
        assert value > 0, "'fwhm1' must be >0 but is {0:.1f}".format(float(value))
        if value != self._fwhm1:
            self._fwhm1 = value

            # all properties that use the FWHM need to  be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    @property
    def beta2(self):
        """
        Return the beta-parameter of the second Moffat component.

        Returns
        -------
        beta : float
            The beta-parameter of the second Moffat component.
        """
        return self._beta2

    @beta2.setter
    def beta2(self, value):
        """
        Set the beta-parameter of the second Moffat component for this instance.

        Parameters
        ----------
        value : float
            The new beta-parameter of the second Moffat component.
        """
        assert isinstance(value, Real), "'beta2' must be a float, not {0:s}".format(type(value).__name__)

        if not 1. < value < 10:
            logging.error("'beta2' must be >1 and <=10 but is {0:.1f}.".format(float(value)))
        assert 1. < value <= 10, "'beta2' must be >1 and <=10 but is {0:.1f}.".format(float(value))

        if value != self._beta2:
            self._beta2 = value

            # all properties that use the beta-parameter need to  be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    # Define the 'fwhm2' property to set and return the FWHM of the second Moffat component ########
    @property
    def fwhm2(self):
        """
        Return the FWHM of the second Moffat component.

        Returns
        -------
        fwhm2 : float
            The FWHM of the second Moffat component.
        """
        return self._fwhm2

    @fwhm2.setter
    def fwhm2(self, value):
        """
        Set the FWHM of the second Moffat component for this instance.

        Parameters
        ----------
        value : float
            The new FWHM of the second Moffat component.
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
        individual components.
        """
        return self.f12*self.fwhm1 + (1. - self.f12)*self.fwhm2

    @property
    def f(self):
        """
        Returns the intensities of the current double-Moffat profile for all
        defined (sub)pixels.
        """
        if self._f is None:
            self._f = self.int_flux * (self.f12 * moffat.flux(self.r, self.beta1, self.fwhm1, self.e) + (
                1. - self.f12) * moffat.flux(self.r, self.beta2, self.fwhm2, self.e))
        return self._f

    # In the following, the properties are provided to calculate the partial derivatives of the Moffat profile with
    # respect to its individual parameters.
    @property
    def diff_f12(self):
        """
        Returns the partial derivative of the profile with respect to f12,
        i.e. the flux contribution of the first Moffat component.
        """
        if self._diff_f12 is None:
            self._diff_f12 = self.int_flux * (moffat.flux(self.r, self.beta1, self.fwhm1, self.e) - moffat.flux(
                self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_f12

    def get_diff_f12(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to f12, i.e. the flux contribution of the first Moffat
        component.

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
        if integrate:
            return self.sum_os(self.diff_f12)
        else:
            return self.diff_f12

    @property
    def diff_beta1(self):
        """
        Returns the partial derivative of the profile with respect to beta1,
        i.e. the parameter defining the kurtosis of Moffat #1.
        """
        if self._diff_beta1 is None:
            self._diff_beta1 = self.int_flux * self.f12 * moffat.pderiv_beta(self.r, self.beta1, self.fwhm1, self.e)
        return self._diff_beta1

    def get_diff_beta1(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to beta1, i.e. the parameter defining the kurtosis of
        Moffat #1.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_beta : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if integrate:
            return self.sum_os(self.diff_beta1)
        else:
            return self.diff_beta1

    @property
    def diff_fwhm1(self):
        """
        Returns the partial derivative of the profile with respect to fwhm1,
        i.e. the full-width at half maximum of Moffat #1.
        """
        if self._diff_fwhm1 is None:
            self._diff_fwhm1 = self.int_flux * self.f12 * moffat.pderiv_fwhm(self.r, self.beta1, self.fwhm1, self.e)
        return self._diff_fwhm1

    def get_diff_fwhm1(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to fwhm1, i.e. the full-width at half maximum of Moffat #1.

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
        if integrate:
            return self.sum_os(self.diff_fwhm1)
        else:
            return self.diff_fwhm1

    @property
    def diff_beta2(self):
        """
        Returns the partial derivative of the profile with respect to beta2,
        i.e. the parameter defining the kurtosis of Moffat #2.
        """
        if self._diff_beta2 is None:
            self._diff_beta2 = self.int_flux * (1. - self.f12) * moffat.pderiv_beta(self.r, self.beta2, self.fwhm2,
                                                                                    self.e)
        return self._diff_beta2

    def get_diff_beta2(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to beta1, i.e. the parameter defining the kurtosis of
        Moffat #1.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_beta : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if integrate:
            return self.sum_os(self.diff_beta2)
        else:
            return self.diff_beta2

    @property
    def diff_fwhm2(self):
        """
        Returns the partial derivative of the profile with respect to fwhm2,
        i.e. the full-width at half maximum of Moffat #2.
        """
        if self._diff_fwhm2 is None:
            self._diff_fwhm2 = self.int_flux * (1. - self.f12) * moffat.pderiv_fwhm(self.r, self.beta2, self.fwhm2,
                                                                                    self.e)
        return self._diff_fwhm2

    def get_diff_fwhm2(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to fwhm2, i.e. the full-width at half maximum of Moffat #2.

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
        if integrate:
            return self.sum_os(self.diff_fwhm2)
        else:
            return self.diff_fwhm2

    # Overwrite partial derivatives with respect to parameters already defined in parent class.
    @property
    def diff_e(self):
        """
        Returns the partial derivative of the profile with respect to e, i.e.
        the ellipticity.
        """
        if self._diff_e is None:
            r_pderiv_e = super().diff_e
            self._diff_e = self.int_flux * (
                self.f12 * moffat.pderiv_e(self.r, self.beta1, self.fwhm1, self.e, r_pderiv_e) + (
                    1. - self.f12) * moffat.pderiv_e(self.r, self.beta2, self.fwhm2, self.e, r_pderiv_e))
        return self._diff_e

    @property
    def diff_theta(self):
        """
        Returns the partial derivative of the profile with respect to theta,
        i.e. the position angle.
        """
        if self._diff_theta is None:
            r_pderiv_theta = super().diff_theta
            self._diff_theta = self.int_flux * r_pderiv_theta * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_theta

    @property
    def diff_xc(self):
        """
        Returns the partial derivative of the profile with respect to xc, i.e.
        the centroid x-coordinate.
        """
        if self._diff_xc is None:
            r_pderiv_xc = super().diff_xc
            self._diff_xc = self.int_flux * r_pderiv_xc * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_xc

    @property
    def diff_yc(self):
        """
        Returns the partial derivative of the profile with respect to yc, i.e.
        the centroid y-coordinate.
        """
        if self._diff_yc is None:
            r_pderiv_yc = super().diff_yc
            self._diff_yc = self.int_flux * r_pderiv_yc * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_yc

    @property
    def diff_A(self):
        """
        Returns the partial derivative of the profile with respect to A, i.e.
        the [0,0] entry of the coordinate transformation matrix.
        """
        if self._diff_A is None:
            r_pderiv_A = super().diff_A
            self._diff_A = self.int_flux * r_pderiv_A * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_A

    @property
    def diff_B(self):
        """
        Returns the partial derivative of the profile with respect to B, i.e.
        the [0,1] entry of the coordinate transformation matrix.
        """
        if self._diff_B is None:
            r_pderiv_B = super().diff_B
            self._diff_B = self.int_flux * r_pderiv_B * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_B

    @property
    def diff_C(self):
        """
        Returns the partial derivative of the profile with respect to C, i.e.
        the [1,0] entry of the coordinate transformation matrix.
        """
        if self._diff_C is None:
            r_pderiv_C = super().diff_C
            self._diff_C = self.int_flux * r_pderiv_C * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_C

    @property
    def diff_D(self):
        """
        Returns the partial derivative of the profile with respect to D, i.e.
        the [1,1] entry of the coordinate transformation matrix.
        """
        if self._diff_D is None:
            r_pderiv_D = super().diff_D
            self._diff_D = self.int_flux * r_pderiv_D * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_D

    @property
    def diff_x0(self):
        """
        Returns the partial derivative of the profile with respect to x0, i.e.
        the x-coordinate shift of the coordinate transformation.
        """
        if self._diff_x0 is None:
            r_pderiv_x0 = super().diff_x0
            self._diff_x0 = self.int_flux * r_pderiv_x0 * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_x0

    @property
    def diff_y0(self):
        """
        Returns the partial derivative of the profile with respect to y0, i.e.
        the y-coordinate shift of the coordinate transformation.
        """
        if self._diff_y0 is None:
            r_pderiv_y0 = super().diff_y0
            self._diff_y0 = self.int_flux * r_pderiv_y0 * (
                self.f12 * moffat.pderiv_r(self.r, self.beta1, self.fwhm1, self.e) + (
                    1. - self.f12) * moffat.pderiv_r(self.r, self.beta2, self.fwhm2, self.e))
        return self._diff_y0

    @property
    def bounds(self):
        bounds = super(DoubleMoffat, self).bounds
        bounds['fwhm1'] = (self.fwhm2, np.inf)
        bounds['fwhm2'] = (0, self.fwhm1)
        bounds['beta1'] = (0, np.inf)
        bounds['beta2'] = (0, np.inf)
        return bounds
