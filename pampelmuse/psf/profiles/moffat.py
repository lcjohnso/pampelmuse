"""
moffat.py
=========
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
class pampelmuse.core.psf.profiles.Moffat


Purpose
-------
This module implements an analytical PSF profile in the form of a Moffat.


Latest Git revision
-------------------
2018/05/08
"""
import logging
from numbers import Real
from .profile import Profile
from ..functions import moffat


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20180508


class Moffat(Profile):
    """
    This class describes a two dimensional Moffat profile. The functional
    form of the Moffat with unity flux is given as

    M(r) = (beta-1)/(pi*r_d^2*(1-e))*[1+(r/r_d)^2]^-beta

    The width of the Gaussian is defined through the effective radius 'r_d'
    while 'beta' defines the kurtosis of the profile. Smaller values of
    'beta' corresponds to more pronounced wings.
    The FWHM of the Moffat profile can be calculated via

    FWHM = 2*r_d*sqrt(2^(1/beta)-1)

    In this implementation of the Moffat profile, the FWHM and 'beta' are
    used as the free parameters and the effective radius is just computed
    internally.
    """
    function = "moffat"

    def __init__(self, uc, vc, mag, beta=2.0, fwhm=3.0, **kwargs):
        """
        Initialize an instance of the Moffat-class.

        Parameters
        ----------
        vc : float
            The x-centroid of the profile in the reference system.
        vc : float
            The y-centroid of the profile in the reference system.
        mag : float
            The integrated magnitude of the profile. The integrated flux is
            unity for a magnitude of zero.
        beta : float, optional
            The beta-value of the Moffat profile. Must be >1.
        fwhm : float, optional
            The FWHM of the profile. Must be >0.
        kwargs
            Any other keyword provided to this method will be parsed to the
            initialization method of the super-class.
        """
        super(Moffat, self).__init__(uc=uc, vc=vc, mag=mag, **kwargs)

        self._beta = beta
        self._fwhm = fwhm

        self._diff_beta = None  # place-holder for partial derivatives wrt. beta
        self._diff_fwhm = None  # and the FWHM

        # add new partial derivatives to list of parameter-dependent properties
        self.profile_properties.extend(["_diff_beta", "_diff_fwhm"])

    @property
    def beta(self):
        """
        Return the 'beta' of the profile.

        Returns
        -------
        beta : float
            The 'beta' parameter of the profile.
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        """
        Set the 'beta' parameter for this instance.

        Parameters
        ----------
        value : float
            The 'beta' parameter of the profile.
        """
        assert isinstance(value, Real), "'beta' must be a float, not {0:s}".format(type(value).__name__)

        if not 1. < value < 10:
            logging.error("'beta' must be >1 and <=10 but is {0:.1f}.".format(float(value)))
        assert 1. < value <= 10, "'beta' must be >1 and <=10 but is {0:.1f}.".format(float(value))

        if value != self._beta:
            self._beta = value

            # all arrays that depend on 'beta' must be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    @property
    def fwhm(self):
        """
        Return the FWHM of the profile.

        Returns
        -------
        fwhm : float
            The FWHM of the profile.
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

            # all arrays that depend on 'beta' must be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    # The property that returns the actual intensity profile of the Moffat
    def get_f(self, integrate=True):
        """
        Method that calculates the actual intensity profile of the Moffat.

        Parameters:
        -----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.            
        **kwargs
            Any other keyword provided to this method will be parsed to the
            'get_f' method of the current super-class.

        Returns:
        --------
        f : nd_array
            The intensity distribution for the pixels used in the calculation.
            If parameter 'integrate' is set, the intensities are returned only
            for the parents pixels after averaging over the sub-pixels.
        """
        if self._f is None:
            self._f = self.int_flux * moffat.flux(self.r, self.beta, self.fwhm, self.e)
        if integrate:
            return self.sum_os(self._f if not self.lut else self._f*self.lut(self.r))
        else:
            return self._f if not self.lut else self._f*self.lut(self.r)

    # In the following, the properties are provided to calculate the partial derivatives of the intensity distribution
    # provided by the Moffat profile wrt. the individual parameters.

    def get_diff_beta(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        kurtosis parameter 'beta'.

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
        if self._diff_beta is None:
            self._diff_beta = moffat.pderiv_beta(self.r, self.beta, self.fwhm, self.e)
        if integrate:
            return self.sum_os(self._diff_beta if not self.lut else self._diff_beta*self.lut(self.r))
        else:
            return self._diff_beta if not self.lut else self._diff_beta*self.lut(self.r)

    @property
    def diff_beta(self):
        return self.get_diff_beta()

    # The FWHM
    def get_diff_fwhm(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the FWHM.

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
        if self._diff_fwhm is None:
            self._diff_fwhm = moffat.pderiv_fwhm(self.r, self.beta, self.fwhm, self.e)
        if integrate:
            return self.sum_os(self._diff_fwhm if not self.lut else self._diff_fwhm*self.lut(self.r))
        else:
            return self._diff_fwhm if not self.lut else self._diff_fwhm*self.lut(self.r)

    @property
    def diff_fwhm(self):
        return self.get_diff_fwhm()

    # The ellipticity 'e'
    def get_diff_e(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        ellipticity 'e'.

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
            r_pderiv_e = super(Moffat, self).get_diff_e(integrate=False)
            self._diff_e = moffat.pderiv_e(self.r, self.beta, self.fwhm, self.e, r_pderiv_e)
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    # The position angle 'theta'
    def get_diff_theta(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        position angle 'theta'.

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
            r_pderiv_theta = super(Moffat, self).get_diff_theta(integrate=False)
            self._diff_theta = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_theta
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    # The centroid coordinates
    def get_diff_xc(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. xc, the
        centroid of the profile in x-direction.

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
            r_pderiv_xc = super(Moffat, self).get_diff_xc(integrate=False)
            self._diff_xc = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_xc
        if integrate:
            return self.sum_os(self._diff_xc if not self.lut else self._diff_xc*self.lut(self.r))
        else:
            return self._diff_xc if not self.lut else self._diff_xc*self.lut(self.r)

    def get_diff_yc(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. yc, the
        centroid of the profile in y-direction.

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
            r_pderiv_yc = super(Moffat, self).get_diff_yc(integrate=False)
            self._diff_yc = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_yc
        if integrate:
            return self.sum_os(self._diff_yc if not self.lut else self._diff_yc*self.lut(self.r))
        else:
            return self._diff_yc if not self.lut else self._diff_yc*self.lut(self.r)

    # The parameters of the coordinate transformation
    def get_diff_A(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'A' of the coordinate transformation.

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
            r_pderiv_A = super(Moffat, self).get_diff_A(integrate=False)
            self._diff_A = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_A
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A

    def get_diff_B(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'B' of the coordinate transformation.

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
            r_pderiv_B = super(Moffat, self).get_diff_B(integrate=False)
            self._diff_B = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_B
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    def get_diff_C(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'C' of the coordinate transformation.

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
            r_pderiv_C = super(Moffat, self).get_diff_C(integrate=False)
            self._diff_C = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_C
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    def get_diff_D(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'B' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.

        Returns
        -------
        diff_D : nd_array
            The partial derivative, either for all (sub-)pixels or after
            averaging over the sub-pixels.
        """
        if self._diff_D is None:
            r_pderiv_D = super(Moffat, self).get_diff_D(integrate=False)
            self._diff_D = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_D
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    def get_diff_x0(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'x0' of the coordinate transformation.

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
            r_pderiv_x0 = super(Moffat, self).get_diff_x0(integrate=False)
            self._diff_x0 = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_x0
        if integrate:
            return self.sum_os(self._diff_x0)
        else:
            return self._diff_x0

    def get_diff_y0(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'y0' of the coordinate transformation.

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
            r_pderiv_y0 = super(Moffat, self).get_diff_y0(integrate=False)
            self._diff_y0 = moffat.pderiv_r(self.r, self.beta, self.fwhm, self.e) * r_pderiv_y0
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0
