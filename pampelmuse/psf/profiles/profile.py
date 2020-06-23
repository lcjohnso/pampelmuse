"""
profiles.py
===========
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
class pampelmuse.core.psf.profiles.classproperty
class pampelmuse.core.psf.profiles.Profile

Purpose
-------
This module implements the Profile-class which serves as a parent-class for
all PSF profiles. Its main purpose is to carry out the transformation from
pixel coordinates to polar coordinates centred on a PSF profile.

The calculation of the PSF profiles is the most time-consuming step in the
analysis and its duration basically defines how fast PampelMuse can analyse an
IFS data set. For this reason, the PSF classes are designed to maximize the
efficiency of the intensity calculations. Especially, all pixel-based
properties, such as the (angle dependent) radius of a pixel to the profile
centre, the intensity distribution across the pixels, or the partial
derivatives of a profile with respect to its variables (that are needed to
fill the Jacobi matrix in a Levenberg-Marquard solver) are defined as
properties that are only recalculated if any of the underlying variables
(such as the profile centre or a PSF variable) changes its value.

Latest Git revision
-------------------
2019/06/11
"""
import collections
import logging
import numbers
import numpy as np
from scipy import optimize
from ...psf import Canvas


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20190611


logger = logging.getLogger(__name__)


# define namedtuple for returning the results of the fit in a useful structure
FitResult = collections.namedtuple(
    'FitResult', ['names', 'x0', 'values', 'flux', 'background', 'centroid', 'success', 'residuals', 'message'])


class Profile(Canvas):
    """
    This class implements a function of the form

    F(r) = r.

    The radius r is given as a function of u, v, and e as follows.

    r = SQRT(u^2 + v^2/(1 - e)^2)

    The coordinates (u, v) correspond to the semi-major and semi-minor axes in
    an elliptical coordinate system that is centred on the source. They can be
    related to the coordinates (x, y) in an image by the position angle theta
    as follows.

    u = (x - x_c) * SIN(theta) - (y - y_c) * COS(theta),
    v = (y - y_c) * SIN(theta) + (x - x_c) * COS(theta),

    where (x_c, y_c) are the centroid coordinates of the source. Note that in
    this definition, theta is measured from the y-axis in counterclockwise
    direction.

    In a common situation for PampelMuse, the centroid coordinates are
    obtained by applying a coordinate transformation to reference coordinates
    (x_ref, y_ref) as follows.

    x_c = A * x_ref + C * y_ref + x0
    y_c = D * y_ref - B * x_ref + y0

    This class provides the partial derivatives of F(r) with respect to all
    relevant parameters used in the equations above, i.e. e, theta, x_c, y_c,
    A, B, C, D, x0, y0.

    F(r) is not supposed to be an actual PSF profile (obviously0. Instead, it
    serves as a parent class for all classes that define a PSF profile. It
    provides the basic functionality to calculate the centroid distance of
    each pixel or the partial derivatives with respect to the various position
    parameters that define the centroid distance of a pixel as described
    above.
    """

    def __init__(self, uc, vc, mag, e=0, theta=0, A=1, B=0, C=0, D=1, x0=0, y0=0, src_id=0,
                 maxrad=np.inf, osradius=0, osfactor=1, lut=None, **kwargs):
        """
        Initialize an instance of the Profiles-class.

        Parameters
        ----------
        uc : float
            The x-centroid of the profile in the reference system.
        vc : float
            The y-centroid of the profile in the reference system.
        mag : float
            The integrated magnitude of the profile. The integrated flux is
            unity for a magnitude of zero.
        e : float, optional
            The ellipticity value of the profile.
        theta : float, optional
            The position angle of the profile, measured in radian and from the
             y-axis in counter-clockwise direction.
        A : float, optional
            The value of parameter A in the coordinate transformation.
        B : float, optional
            The value of parameter B in the coordinate transformation.
        C : float, optional
            The value of parameter C in the coordinate transformation.
        D : float, optional
            The value of parameter D in the coordinate transformation.
        x0 : float, optional
            The value of parameter x0 in the coordinate transformation.
        y0 : float, optional
            The value of parameter y0 in the coordinate transformation.
        src_id : int
            The ID of the source in any catalogue.
        maxrad : float
            The maximum distance from the profile centroid where the
            contribution of the profile will be calculated. Default is to
            calculate it for the full field of view.
        osradius : float
            The radius from the profile centroid within which the pixels
            will be supersampled to improve the accuracy of the calculation.
            Default is no supersampling.
        osfactor : int
            The supersampling factor. Each pixel within 'osradius' will be
            supersampled into 'osfactor' times 'osfactor' subpixels.
        lut : function, optional
            A look-up table used to modify the profile fluxes as a function of
            radius. The function must take a set of radii as input and return
            the correction factor for every radius provided.
        """
        # initialize parent class
        super(Profile, self).__init__(**kwargs)

        # initialize basic properties
        self._uc = uc  # Coordinates of source in reference system
        self._vc = vc
        self._xc = None
        self._yc = None
        self._mag = mag
        self.src_id = src_id
        self._e = e
        self._theta = theta
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self._x0 = x0
        self._y0 = y0
        self._maxrad = maxrad  # The radius (in pixels) out to which the PSF will be calculated

        # Initialize the arrays that will hold the per-pixel-information of the profile
        self._used = None  # will be a boolean array with all pixels within distance 'maxrad' to centroid set to True.
        self._oversampled = None  # will be a boolean array with all supersampled pixels set to True.
        self._osmap = None  # The arrays osmap and ossummer will be needed to properly include the oversampling
        self._ossummer = None
        self._xos = None  # The x- & y-coordinates of the pixels including the sub-pixels in supersampled ones.
        self._yos = None
        self._u = None  # The u- & v-coordinates of the pixels.
        self._v = None
        self._r = None  # The distances of the pixels wrt. the profile centroid
        self._f = None  # The flux is only a place-holder in this class.

        # Initialize the arrays that will hold the partial derivatives wrt. the individual profile parameters.
        self._diff_e = None
        self._diff_theta = None
        self._diff_xc = None
        self._diff_yc = None
        self._diff_A = None
        self._diff_B = None
        self._diff_C = None
        self._diff_D = None
        self._diff_x0 = None
        self._diff_y0 = None

        # All the arrays must eventually be recalculated if certain parameters are changed.
        self.profile_properties = ["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r",
                                   "_f",
                                   "_diff_e", "_diff_theta", "_diff_xc", "_diff_yc", "_diff_A", "_diff_B", "_diff_C",
                                   "_diff_D", "_diff_x0", "_diff_y0"]

        # Properly implement possibility to oversample central pixels
        self.osr = osradius
        if not self.osr > 0:
            self.osf = 1
            self.osxgrid = None
            self.osygrid = None
        else:
            self.osf = max(1, osfactor)

            # osxgrid and osygrid specify the offset (in pixels) of the individual sub-pixels wrt. to the oversampled
            # pixel.
            osgrid = np.linspace(-0.5 + 1. / (2 * self.osf), 0.5 - 1. / (2 * self.osf), self.osf)
            self.osxgrid = np.resize(osgrid, (self.osf * self.osf,))
            self.osygrid = np.sort(self.osxgrid)

        # verify look-up table
        if lut is not None:
            if not callable(lut):
                logger.error('Parameter <lut> must be callable.')
            self.lut = lut
        else:
            self.lut = None

    @property
    def e(self):
        """
        Returns the ellipticity e of the profile.
        """
        return self._e

    @e.setter
    def e(self, value):
        """
        Set the ellipticity 'e' of all instances of the instance.

        Parameters
        ----------
        value : float
            The ellipticity of the profiles must be in the interval [0,1).
            value=0 corresponds to a round profile.
        """
        assert isinstance(value, numbers.Real), "'e' must be a float, not {0:s}".format(type(value).__name__)
        if not value >= 0 and value < 1:
            logging.error("'e' must be in interval [0,1) but is {0:.1f}".format(float(value)))
        assert -1e-3 <= value < 1, "'e' must be in interval [0,1) but is {0:.1f}".format(float(value))
        if value != self._e:
            self._e = value

            # all properties building on the radius values of the individual pixels must be recalculated.
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v"])

    @property
    def theta(self):
        """
        Returns the position angle of the profile, measured in radian and from
        the y-axis in counter-clockwise direction.
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        """
        Set the position angle 'theta' for this instance.

        Parameters
        ----------
        value : float
            The position angle, given in radian. Note that 'theta' is measured
            from the y-axis in counter-clockwise direction.
        """
        assert isinstance(value, numbers.Real), "'theta' must be a float, not {0:s}".format(type(value).__name__)
        if value != self._theta:
            self._theta = value

            # all properties building up on the u- & v-coordinates must be recalculated.
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos"])

    @property
    def A(self):
        """
        Returns the current value of the parameter A of the coordinate
        transformation.
        """
        return self._A

    @A.setter
    def A(self, value):
        """
        Sets the parameter A of the coordinate transformation.

        Parameters
        ----------
        value : float
            The parameter A of the coordinate transformation.
        """
        assert isinstance(value, numbers.Real), "'A' must be a float or int, not {0:s}".format(type(value).__name__)
        if value != self._A:
            self._A = value

            self._xc = None  # any direct centroid coordinate should be deleted
            self._yc = None

            # All arrays must be recalculated
            self.reset()

    @property
    def B(self):
        """
        Returns the current value of the parameter B of the coordinate
        transformation.
        """
        return self._B

    @B.setter
    def B(self, value):
        """
        Set the parameter B of the coordinate transformation.

        Parameters
        ----------
        value : float
            The parameter B of the coordinate transformation.
        """
        assert isinstance(value, numbers.Real), "'B' must be a float or int, not {0:s}".format(type(value).__name__)
        if value != self._B:
            self._B = value

            self._xc = None  # any direct centroid coordinate should be deleted
            self._yc = None

            # All arrays must be recalculated
            self.reset()

    @property
    def C(self):
        """
        Returns the current value of the parameter C of the coordinate
        transformation.
        """
        return self._C

    @C.setter
    def C(self, value):
        """
        Set the parameter C of the coordinate transformation.

        Parameters
        ----------
        value : float
            The parameter C of the coordinate transformation.
        """
        assert isinstance(value, numbers.Real), "'C' must be a float or int, not {0:s}".format(type(value).__name__)
        if value != self._C:
            self._C = value

            self._xc = None  # any direct centroid coordinate should be deleted
            self._yc = None

            # All arrays must be recalculated
            self.reset()

    @property
    def D(self):
        """
        Returns the current value of the parameter D of the coordinate
        transformation.
        """
        return self._D

    @D.setter
    def D(self, value):
        """
        Set the parameter D of the coordinate transformation.

        Parameters
        ----------
        value : float
            The parameter D of the coordinate transformation.
        """
        assert isinstance(value, numbers.Real), "'D' must be a float or int, not {0:s}".format(type(value).__name__)
        if value != self._D:
            self._D = value

            self._xc = None  # any direct centroid coordinate should be deleted
            self._yc = None

            # All arrays must be recalculated
            self.reset()

    @property
    def x0(self):
        """
        Returns the current value of the parameter x0 of the coordinate
        transformation.
        """
        return self._x0

    @x0.setter
    def x0(self, value):
        """
        Set the parameter x0 of the coordinate transformation.

        Parameters
        ----------
        value : float
            The parameter x0 of the coordinate transformation.
        """
        assert isinstance(value, numbers.Real), "'x0' must be a float or int, not {0:s}".format(type(value).__name__)
        if value != self._x0:
            self._x0 = value

            self._xc = None  # any direct centroid coordinate should be deleted
            self._yc = None

            # All arrays must be recalculated
            self.reset()

    @property
    def y0(self):
        """
        Returns the current value of the parameter y0 of the coordinate
        transformation.
        """
        return self._y0

    @y0.setter
    def y0(self, value):
        """
        Set the parameter y0 of the coordinate transformation.

        Parameters
        ----------
        value : float
            The parameter y0 of the coordinate transformation.
        """
        assert isinstance(value, numbers.Real), "'y0' must be a float or int, not {0:s}".format(type(value).__name__)
        if value != self._y0:
            self._y0 = value

            self._xc = None  # any direct centroid coordinate should be deleted
            self._yc = None

            # All properties must be recalculated
            self.reset()

    @property
    def xc(self):
        """
        Returns the x-centroid of the profile.
        """
        if self._xc is None:
            return self.A * self.uc + self.C * self.vc + self.x0
        else:
            return self._xc

    @xc.setter
    def xc(self, value):
        """
        Sets the direct x-coordinate of the centroid of the profile.

        Parameters
        ----------
        value : float
            The new x-centroid of the profile.
        """
        assert isinstance(value, numbers.Real), "'xc' must be a float, not {0:s}".format(type(value).__name__)
        if self._xc != value:
            self._xc = value
            self.reset()

    @xc.deleter
    def xc(self):
        """
        Deletes the direct x-centroid for this profile.
        """
        if self._xc is not None or self._yc is not None:
            self._xc = None
            self._yc = None
            self.reset()

    @property
    def yc(self):
        """
        Returns the y-centroid of the profile.
        """
        if self._yc is None:
            return self.D * self.vc - self.B * self.uc + self.y0
        else:
            return self._yc

    @yc.setter
    def yc(self, value):
        """
        Sets the direct y-coordinate of the centroid of the profile.

        Parameters
        ----------
        value : float
            The new y-centroid of the profile.
        """
        assert isinstance(value, numbers.Real), "'yc' must be a float, not {0:s}".format(type(value).__name__)
        if self._yc != value:
            self._yc = value
            self.reset()

    @yc.deleter
    def yc(self):
        """
        Deletes the direct y-centroid for this profile.
        """
        if self._xc is not None or self._yc is not None:
            self._xc = None
            self._yc = None
            self.reset()

    def reset(self, omit=None):
        """
        When parameters of the profile are updated, all arrays that are affected
        by the parameter must eventually be recalculated. To assure this, they
        can be set to None using this method.

        Parameters
        ----------
        omit : list, optional
            A list containing the arrays that should not be reset because they are
            still valid with the updated profile parameters.
        """
        for prop in self.profile_properties:
            if omit is None or prop not in omit:
                setattr(self, prop, None)

    @property
    def maxrad(self):
        """
        Return the calculation radius for this instance.
        """
        return self._maxrad

    @maxrad.setter
    def maxrad(self, value):
        """
        Set the calculation radius of the profile.

        Parameters
        ----------
        value : float
            The radius (in pixels) out to which the profile will be
            calculated.
        """
        assert isinstance(value, numbers.Real), "'maxrad' must be a float, not {0:s}".format(type(value).__name__)
        if value != self._maxrad:
            self._maxrad = value

            # Reset all arrays calculated for this instance
            self.reset()

    @property
    def uc(self):
        return self._uc

    @uc.setter
    def uc(self, value):
        assert isinstance(value, numbers.Real), "'uc' must be a float, not {0:s}".format(type(value).__name__)
        if value != self._uc:
            self._uc = value

            # Reset all arrays
            self.reset()

    @property
    def vc(self):
        return self._vc

    @vc.setter
    def vc(self, value):
        assert isinstance(value, numbers.Real), "'vc' must be a float, not {0:s}".format(type(value).__name__)
        if value != self._vc:
            self._vc = value

            # Reset all arrays
            self.reset()

    @property
    def mag(self):
        """
        Return the current integrated magnitude of the profile.
        """
        return self._mag

    @mag.setter
    def mag(self, value):
        """
        Set the integrated magnitude of the profile.

        Parameters
        ----------
        value : float
            The integrated magnitude. A magnitude of zero yields a flux of
            one, i.e. f=10^(-0.4*m).
        """
        assert isinstance(value, numbers.Real), "'mag' must be a float, not {0:s}".format(type(value).__name__)
        if value != self._mag:
            self._mag = value

            # Reset all arrays that inherit from mag
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "yos", "_u", "_v", "_r"])

    @property
    def int_flux(self):
        """
        Returns the integrated flux of the profile.
        """
        return 10**(-0.4*self.mag)

    @property
    def used(self):
        """
        'used' is a boolean array that is set to True for each pixel that is
        included in the calculation, i.e. is within a distance 'maxrad' to the
        profile centroid.

        Returns
        -------
        used : nd_array
            A 1dim. boolean array with length = number of pixels in class.
        """
        if self._used is None:
            if self.maxrad == np.inf:  # use all pixels if requested
                self._used = np.ones((self.transform.shape[0],), dtype=np.bool)
            else:
                self._used = np.sqrt((self.x - self.xc) ** 2 + (self.y - self.yc) ** 2) <= self.maxrad
        return self._used

    @property
    def n_used(self):
        """
        Returns
        -------
        n_used : int
            The number of pixels included in the calculation.
        """
        return self.used.sum()

    @property
    def oversampled(self):
        """
        'oversampled' is a boolean array that is set to True for each pixel
        that is supersampled in the calculation, i.e. is within a distance
        'osradius' to the profile centroid.

        Returns
        -------
        oversampled : nd_array
            A 1dim. boolean array with length = number of pixels in class.
        """
        if self._oversampled is None:
            self._oversampled = np.sqrt((self.x - self.xc) ** 2 + (self.y - self.yc) ** 2) <= self.osr
        return self._oversampled

    @property
    def n_os(self):
        """
        Returns
        -------
        Nos : int
            The number of supersampled pixels included in the calculation.
        """
        return self.oversampled.sum()

    @property
    def norm(self):
        """
        Returns
        -------
        norm : nd_array
            A 1dim. array with length = number of pixels in class. It provides
            the normalization for each pixel, i.e. the value by which the
            calculated fluxes must be divided to return the true ones. It is
            basically 1 for each pixel used in the calculation except for the
            supersampled ones, where it is equal to the number of sub-pixels.
        """
        norm = np.asarray(self.used, dtype=np.float64)
        norm[self.oversampled] *= self.osf ** 2
        return norm

    # The supersampling of pixels is included in the following way:
    # All calculations are performed on 1dim. arrays, and every sub-pixel is included in those arrays. The property
    # 'osmap' keeps track of the parent pixel of each (sub)pixel used in the calculation. When a calculated property
    # is returned, by default the original pixels are restored by averaging over the sub-pixels. This is done using
    # the property 'ossummer' and the method 'sum_os'. The averaging can be turned off for internal calculations
    # (see below).
    @property
    def osmap(self):
        """
        Returns
        -------
        osmap : nd_array
            An array of integers providing the sorted indices of all pixels
            included in the calculation. The indices of the pixels that are
            supersampled are included 'osfactor'**2 times.
        """
        if self._osmap is None:
            # np.resize uses repeated copies of data to fill new values
            self._osmap = np.hstack((self.pixel[self.used],
                                     np.resize(self.pixel[self.oversampled], ((self.osf * self.osf - 1) * self.n_os,))))
            self._osmap.sort()
        return self._osmap

    @property
    def ossummer(self):
        """
        Returns
        -------
        ossummer : nd_array
            An 2dim. array of integers with shape (2, n_pixel). It contains
            the first and last index of every parent pixel in the 'osmap'
            array and is used by the method 'sum_os' to restore the values in
            the parent pixels.
        """
        if self._ossummer is None:
            # first calculate difference of adjacent entries in 'osmap'. Then find indices where this difference is >0
            # and include those in ossummer
            next_pixel = np.flatnonzero(np.diff(np.r_[self.osmap[0] - 1, self.osmap, self.osmap[-1] + 1]))
            self._ossummer = np.vstack((next_pixel[:-1], next_pixel[1:])).T
        return self._ossummer

    def sum_os(self, data):
        """
        This method calculates the value of 'data' in every parent pixel by
        averaging over its sub-pixels. This is done following suggestion at:
        http://stackoverflow.com/questions/7471474/\
            numpy-sum-of-values-in-subarrays-between-pairs-of-indices

        Parameters
        ----------
        data : nd_array
            A 1dim. array with the same size as property 'osmap'.

        Returns
        -------
        avg_data : nd_array
            The result of averaging the values in 'data' over all sub-pixels.
            Its length is equal to the number of used pixels.
        """
        if self.osf <= 1 or self.n_os == 0:
            return data
        else:
            tmp_data = np.r_[0, data.cumsum()][self.ossummer]
            return (tmp_data[:, 1] - tmp_data[:, 0]) / self.norm[self.used]

    @property
    def xos(self):
        """
        Returns
        -------
        xos : nd_array
            A one dimensional array containing the x-coordinate of each pixel
            used in the calculation (including sub-pixels!) in the field of
            view. It therefore has the same length as the 'osmap'-property.
        """
        if self._xos is None:
            self._xos = self.x[self.osmap]
            if self.osf > 1:
                # add offsets of sub-pixel wrt. parent pixel along x-axis
                is_oversampled = np.in1d(self.osmap, np.flatnonzero(self.oversampled))
                assert (is_oversampled.sum() == self.n_os * self.osxgrid.size)
                self._xos[is_oversampled] += np.resize(self.osxgrid, (self.n_os * self.osxgrid.size,))
        return self._xos

    @property
    def yos(self):
        """
        Returns
        -------
        yos : nd_array
            A one dimensional array containing the y-coordinate of each pixel
            used in the calculation (including sub-pixels!) in the field of
            view. It therefore has the same length as the 'osmap'-property.
        """
        if self._yos is None:
            self._yos = self.y[self.osmap]
            if self.osf > 1:
                # add offsets of sub-pixel wrt. parent pixel along y-axis
                is_oversampled = np.in1d(self.osmap, np.flatnonzero(self.oversampled))
                assert (is_oversampled.sum() == self.n_os * self.osygrid.size)
                self._yos[is_oversampled] += np.resize(self.osygrid, (self.n_os * self.osygrid.size,))
        return self._yos

    @property
    def u(self):
        """
        Returns
        -------
        u : nd_array
            A one dimensional array containing the coordinate of each pixel
            used in the calculation (including sub-pixels!) in u-direction,
            i.e. along the semi-major axis of the profile.
        """
        if self._u is None:
            self._u = (self.xos - self.xc) * np.sin(self.theta) - (self.yos - self.yc) * np.cos(self.theta)
        return self._u

    @property
    def v(self):
        """
        Returns
        -------
        v : nd_array
            A one dimensional array containing the coordinate of each pixel
            used in the calculation (including sub-pixels!) in v-direction,
            i.e. along the semi-minor axis of the profile.
        """
        if self._v is None:
            self._v = (self.yos - self.yc) * np.sin(self.theta) + (self.xos - self.xc) * np.cos(self.theta)
        return self._v

    @property
    def r(self):
        """
        Returns
        -------
        u : nd_array
            A one dimensional array containing the distance of each pixel
            used in the calculation (including sub-pixels!) to the profile
            centroid.
        """
        if self._r is None:
            self._r = np.sqrt(self.u ** 2 + self.v ** 2 / (1 - self.e) ** 2)
            self._r[self._r == 0] = 1e-5
        return self._r

    # The property 'f' returns the actual spatial intensity profile. In this class, it is basically only defined to
    # handle super() calls from instances of classes that inherit from Profile
    @property
    def f(self):
        """
        A place-holder for the intensity profile. Every child class of Profile
        should redefine this property to an actual profile.
        """
        return np.zeros_like(self.r)

    def get_f(self, integrate=True, apply_lut=False):
        """
        Method that calculates the actual intensity profile.

        Parameters:
        -----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-pixels before they are returned.
        apply_lut : bool, optional
            Whether the returned fluxes should be corrected using the look-up
            table.
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
        f = np.copy(self.f)
        if apply_lut and self.lut:
            correction = self.lut(self.r)
            correction *= f.sum()/(correction*f).sum()
            f *= correction
        if integrate:
            return self.sum_os(f)
        else:
            return f

    @property
    def diff_e(self):
        """
        Returns the partial derivative of the profile with respect to e, i.e.
        the ellipticity.
        """
        if self._diff_e is None:  # only re-calculate if necessary
            self._diff_e = self.v ** 2 / (self.r * (1. - self.e) ** 3)
        return self._diff_e

    def get_diff_e(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to the e, i.e. the ellipticity.

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
        if integrate:
            return self.sum_os(self.diff_e)
        else:
            return self.diff_e

    @property
    def diff_theta(self):
        """
        Returns the partial derivative of the profile with respect to theta,
        i.e. the position angle.
        """
        if self._diff_theta is None:  # only re-calculate if necessary
            self._diff_theta = self.u * self.v * (1. - 1. / (1. - self.e) ** 2) / self.r
        return self._diff_theta

    def get_diff_theta(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to theta, i.e. the position angle.

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
        if integrate:
            return self.sum_os(self.diff_theta)
        else:
            return self.diff_theta

    @property
    def diff_xc(self):
        """
        Returns the partial derivative of the profile with respect to xc, i.e.
        the centroid x-coordinate.
        """
        if self._diff_xc is None:  # only re-calculate if necessary
            self._diff_xc = -(self.u * np.sin(self.theta) + (self.v / (1 - self.e) ** 2) * np.cos(self.theta)) / self.r
        return self._diff_xc

    def get_diff_xc(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to xc, i.e. the centroid x-coordinate.

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
        if integrate:
            return self.sum_os(self.diff_xc)
        else:
            return self.diff_xc

    @property
    def diff_yc(self):
        """
        Returns the partial derivative of the profile with respect to yc, i.e.
        the centroid y-coordinate.
        """
        if self._diff_yc is None:  # only re-calculate if necessary
            self._diff_yc = (self.u * np.cos(self.theta) - (self.v / (1 - self.e) ** 2) * np.sin(self.theta)) / self.r
        return self._diff_yc

    def get_diff_yc(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to yc, the centroid y-coordinate.

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
        if integrate:
            return self.sum_os(self.diff_yc)
        else:
            return self.diff_yc

    @property
    def diff_A(self):
        """
        Returns the partial derivative of the profile with respect to A, i.e.
        the [0,0] entry of the coordinate transformation matrix.
        """
        if self._diff_A is None:  # only re-calculate if necessary
            self._diff_A = -self.uc * (self.u * np.sin(self.theta) + (
                    self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
        return self._diff_A

    def get_diff_A(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to A, i.e. the [0,0] entry of the coordinate transformation
        matrix.

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
        if integrate:
            return self.sum_os(self.diff_A)
        else:
            return self.diff_A

    @property
    def diff_B(self):
        """
        Returns the partial derivative of the profile with respect to B, i.e.
        the [0,1] entry of the coordinate transformation matrix.
        """
        if self._diff_B is None:  # only re-calculate if necessary
            self._diff_B = -self.uc * (self.u * np.cos(self.theta) - (
                    self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
        return self._diff_B

    def get_diff_B(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to B, i.e. the [0,1] entry of the coordinate transformation
        matrix.

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
        if integrate:
            return self.sum_os(self.diff_B)
        else:
            return self.diff_B

    @property
    def diff_C(self):
        """
        Returns the partial derivative of the profile with respect to C, i.e.
        the [1,0] entry of the coordinate transformation matrix.
        """
        if self._diff_C is None:  # only re-calculate if necessary
            self._diff_C = -self.vc * (self.u * np.sin(self.theta) + (
                self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
        return self._diff_C

    def get_diff_C(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to C, i.e. the [1,0] entry of the coordinate transformation
        matrix.

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
        if integrate:
            return self.sum_os(self.diff_C)
        else:
            return self.diff_C

    @property
    def diff_D(self):
        """
        Returns the partial derivative of the profile with respect to D, i.e.
        the [1,1] entry of the coordinate transformation matrix.
        """
        if self._diff_D is None:  # only re-calculate if necessary
            self._diff_D = self.vc * (self.u * np.cos(self.theta) - (
                self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
        return self._diff_D

    def get_diff_D(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to D, i.e. the [1,1] entry of the coordinate transformation
        matrix.

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
        if integrate:
            return self.sum_os(self.diff_D)
        else:
            return self.diff_D

    @property
    def diff_x0(self):
        """
        Returns the partial derivative of the profile with respect to x0, i.e.
        the x-coordinate shift of the coordinate transformation.
        """
        if self._diff_x0 is None:  # only re-calculate if necessary
            self._diff_x0 = -(self.u * np.sin(self.theta) + (self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
        return self._diff_x0

    def get_diff_x0(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to x0, i.e. the x-coordinate shift of the coordinate
        transformation.

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
        if integrate:
            return self.sum_os(self.diff_x0)
        else:
            return self.diff_x0

    @property
    def diff_y0(self):
        """
        Returns the partial derivative of the profile with respect to y0, i.e.
        the y-coordinate shift of the coordinate transformation.
        """
        if self._diff_y0 is None:  # only re-calculate if necessary
            self._diff_y0 = (self.u * np.cos(self.theta) - (self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
        return self._diff_y0

    def get_diff_y0(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile with
        respect to y0, i.e. the y-coordinate shift of the coordinate
        transformation.

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
        if integrate:
            return self.sum_os(self.diff_y0)
        else:
            return self.diff_y0

    def get_flux(self, **kwargs):
        """
        Get the calculated fluxes in each pixel used in the calculation

        Returns
        -------
        values : tuple
            The tuple contains the indices of the pixels used in the
            calculation as the first entry, the second entry contains
            the actual fluxes per pixel.
        """
        return self.pixel[self.used], self.get_f(**kwargs)

    @property
    def bounds(self):
        return {'e': (0, 1), 'theta': (0, np.pi)}

    def fitter(self, psf, weights, parameters, fit_positions=False, fit_background=False, flux_x0=None,
               background_x0=None, **kwargs):

        # TODO: find way to provide only parameters used by the different fitter() methods
        # if kwargs:
        #     raise IOError('Unknown parameter(s) {} for method Profile.fitter.'.format(kwargs))

        # make sure psf and variances are defined on same pixels as the profile
        if not len(psf) == len(weights):
            raise IOError('Parameters "psf" and "weights" must have same number of pixels.')
        elif len(psf) == self.n_used:
            psf2fit = np.asarray(psf, dtype=np.float64)
            weights2fit = np.asarray(weights, dtype=np.float64)
        elif len(psf) == self.n_pixel:
            psf2fit = np.asarray(psf, dtype=np.float64)[self.used]
            weights2fit = np.asarray(weights, dtype=np.float64)[self.used]
        else:
            raise IOError('Input data to Profile.fitter has inconsistent shape.')

        # collect initial guesses, set boundary conditions
        x0 = []
        names = []
        bounds = ([], [])

        # always fit flux of PSF
        x0.append(flux_x0 if flux_x0 is not None else np.nansum(psf))
        names.append('flux')
        bounds[0].append(-np.inf)
        bounds[1].append(np.inf)

        # PSF parameters
        for parameter in parameters:
            x0.append(getattr(self, parameter))
            names.append(parameter)
            if parameter in self.bounds.keys():
                bounds[0].append(self.bounds[parameter][0])
                bounds[1].append(self.bounds[parameter][1])
            else:
                bounds[0].append(-np.inf)
                bounds[1].append(np.inf)

        # positions if requested
        if fit_positions:
            x0.extend([self.xc, self.yc])
            names.extend(['xc', 'yc'])
            bounds[0].extend([-np.inf, -np.inf])
            bounds[1].extend([np.inf, np.inf])

        # background if requested
        if fit_background:
            x0.append(background_x0 if background_x0 is not None else psf.nanmin())
            names.append('back')
            bounds[0].append(-np.inf)
            bounds[1].append(np.inf)

        # need to avoid that grid of used pixels changes during optimization
        self.profile_properties.remove('_used')

        # fit
        try:
            result = optimize.least_squares(
                self.single_psf_optimizer, x0=np.asarray(x0), bounds=bounds, jac=self.single_psf_jacobi,
                args=(names, psf2fit, weights2fit))
        except Exception as msg:
            # error handler to avoid code crashes completely
            # note that type of exception thrown seems to depend on scipy version
            logger.error('PSF fit failed for source #{0}: {1}.'.format(self.src_id, msg),
                         exc_info=True if logger.getEffectiveLevel() <= 10 else False)
            values = np.nan*np.ones_like(x0)
            residuals = None
            success = False
            message = msg
        else:
            values = result['x']
            residuals = result['fun'] / np.sqrt(weights2fit)
            success = True
            message = result['message']
        finally:
            # add grid of used pixels to list of derived properties again
            self.profile_properties.insert(0, '_used')

            # reset fitted parameters back to initial guesses (i.e. recover state of Profile instance prior to fit)
            for i in range(1, len(names) - fit_background):
                setattr(self, names[i], x0[i])

        return FitResult(names=parameters,
                         x0=[x0[names.index(p)] for p in parameters],
                         values=[values[names.index(p)] for p in parameters],
                         flux=values[names.index('flux')],
                         background=values[names.index('back')] if fit_background else None,
                         centroid=(values[names.index('xc')], values[names.index('yc')]) if fit_positions else None,
                         success=success,
                         residuals=residuals,
                         message=message)

    def single_psf_optimizer(self, x0, names, data, weights):
        """
        This is the optimization routine that is passed to the Levenberg-
        Marquardt solver scipy.optimize.leastsq.

        During each iteration, a PSF array is created, scaled and subtracted
        from the data. The residuals of the fit are returned.

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
        weights : nd_array
            The pixel weights used in the optimization. Inverse-variance
            weighting is assumed.

        Returns
        -------
        nd_array
            The sigma-weighted residuals after subtracting the PSF from the
            data.
        """
        # get local background if provided
        fit_background = names[-1] == 'back'
        back = x0[-1] if fit_background else 0

        # update PSF using current parameter estimates
        for i in range(1, len(x0) - int(fit_background)):
            try:
                # the attribute is only set in the 'psf'-instance of the class, not in all instances! This is different
                # from the behaviour of getattr(self.PSF, "set_<PAR>")(<VALUE>)
                setattr(self, names[i], x0[i])
            except AssertionError as error:  # if parameter cannot be updated, raise LinAlgError
                logger.error("Encountered AssertionError: {0}".format(error))
                raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], names[i]))

        return np.sqrt(weights)*(data - x0[0] * self.get_f() - back)

    def single_psf_jacobi(self, x0, names, psf, weights):
        """
        Calculate the Jacobi matrix for the optimization carried out in
        StarFit.optimize_single_psf.

        The method is passed to the Levenberg-Marquard solver
        scipy.optimize.leastsq as parameter Dfun.

        Parameters
        ----------
        x0
        names
        psf
        weights
            This method takes the same arguments as StarFit.optimize_single_psf.

        Returns
        -------
        jacob : nd_array
            The Jacobi matrix with respect to the model parameters included
            in 'x0'.
        """
        fit_background = names[-1] == 'back'

        _jacob = np.zeros((len(x0), self.n_used), dtype=np.float64)
        _jacob[0] -= self.get_f()
        for i in range(1, len(x0) - int(fit_background)):
            diff_i = getattr(self, "get_diff_{0}".format(names[i]))()
            _jacob[i] -= x0[0] * diff_i
        if fit_background:
            _jacob[-1] = x0[-1]

        # Note that to get the right shape for the optimization, the matrix needs to be transposed.
        return (np.sqrt(weights)*_jacob).T

    @property
    def curve_of_growth(self):

        radii = np.arange(1, min(self.maxrad, 100), 1, dtype=np.float64)
        cog_flux = np.zeros_like(radii)

        for i, r_max in enumerate(radii):
            r_min = radii[i-1] if i > 0 else 0

            # get fraction of each pixel between r_min and r_max;
            # assume fraction increases/decreases linearly from 0 to 1 between r_min/max-0.5 and r_min/max+0.5
            weights = np.where(self.r < (r_max - 0.5), 1., r_max + 0.5 - self.r)
            weights[weights < 0] = 0.

            if i > 0:
                _weights = np.where(self.r < (r_min - 0.5), 1., r_min + 0.5 - self.r)
                _weights[_weights < 0] = 0
                weights -= _weights

            # get fraction of area between r_min and r_max covered by defined PSF pixels
            frac = weights.sum()/(np.pi*(r_max**2 - r_min**2))

            cog_flux[i] = (self.f*weights).sum()/frac

        return radii, np.cumsum(cog_flux)
