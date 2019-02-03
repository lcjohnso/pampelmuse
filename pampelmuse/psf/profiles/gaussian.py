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
2018/04/22
"""
from numbers import Real
from .profile import Profile
from ..functions import gaussian


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20180422


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
    def get_f(self, integrate=True):
        """
        Method that calculates the actual intensity profile of the Gaussian.

        Parameters:
        -----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.            
        kwargs
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
            self._f = self.int_flux*gaussian.flux(self.r, self.fwhm, self.e)
        if integrate:
            return self.sum_os(self._f)
        else:
            return self._f

    # In the following, the properties are provided to calculate the partial derivatives of the intensity distribution
    # provided by the Gaussian profile wrt. the individual parameters.
    def get_diff_fwhm(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        FWHM.

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
            self._diff_fwhm = gaussian.pderiv_fwhm(self.r, self.fwhm, self.e)
        if integrate:
            return self.sum_os(self._diff_fwhm)
        else:
            return self._diff_fwhm

    @property
    def diff_fwhm(self):
        return self.get_diff_fwhm()

    # The ellipticity 'e'
    def get_diff_e(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
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
            r_pderiv_e = super(Gaussian, self).get_diff_e(integrate=False)
            self._diff_e = gaussian.pderiv_e(self.r, self.fwhm, self.e, r_pderiv_e)
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    # The position angle 'theta'
    def get_diff_theta(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
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
            r_pderiv_theta = super(Gaussian, self).get_diff_theta(integrate=False)
            self._diff_theta = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_theta
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    # The centroid coordinates
    def get_diff_xc(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. xc, the
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
            r_pderiv_xc = super(Gaussian, self).get_diff_xc(integrate=False)
            self._diff_xc = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_xc
        if integrate:
            return self.sum_os(self._diff_xc)
        else:
            return self._diff_xc

    def get_diff_yc(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. yc, the
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
            r_pderiv_yc = super(Gaussian, self).get_diff_yc(integrate=False)
            self._diff_yc = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_yc
        if integrate:
            return self.sum_os(self._diff_yc)
        else:
            return self._diff_yc

    # The parameters of the coordinate transformation
    def get_diff_A(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
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
            r_pderiv_A = super(Gaussian, self).get_diff_A(integrate=False)
            self._diff_A = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_A
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A

    def get_diff_B(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
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
            r_pderiv_B = super(Gaussian, self).get_diff_B(integrate=False)
            self._diff_B = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_B
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    def get_diff_C(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
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
            r_pderiv_C = super(Gaussian, self).get_diff_C(integrate=False)
            self._diff_C = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_C
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    def get_diff_D(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        parameter 'D' of the coordinate transformation.

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
            r_pderiv_D = super(Gaussian, self).get_diff_D(integrate=False)
            self._diff_D = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_D
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    def get_diff_x0(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
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
            r_pderiv_x0 = super(Gaussian, self).get_diff_x0(integrate=False)
            self._diff_x0 = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_x0
        if integrate:
            return self.sum_os(self._diff_x0)
        else:
            return self._diff_x0

    def get_diff_y0(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
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
            r_pderiv_y0 = super(Gaussian, self).get_diff_y0(integrate=False)
            self._diff_y0 = gaussian.pderiv_r(self.r, self.fwhm, self.e)*r_pderiv_y0
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0
