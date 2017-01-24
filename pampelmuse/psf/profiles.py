"""
profiles.py
===========
Copyright 2013-2016 Sebastian Kamann

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
class pampelmuse.core.psf.profiles.IterRegistry
class pampelmuse.core.psf.profiles.classproperty
class pampelmuse.core.psf.profiles.Canvas
class pampelmuse.core.psf.profiles.Profile
class pampelmuse.core.psf.profiles.Gaussian
class pampelmuse.core.psf.profiles.Moffat

Purpose
-------
This module implements the analytical PSF profiles that are used by Pampel-
Muse. The main classes defined here (class 'Canvas' and its subclasses)
serve the purpose of calculating the intensities of a given PSF for each
data point. This is the most time-consuming step in the analysis and its
duration basically defines how fast PampelMuse can analyse an IFS data set.
For this reason, the classes were designed to maximize the efficiency of the
intensity calculations. Especially, all pixel-based properties, such as the
(angle dependent) radius of a pixel to the profile centre, the intensity
distribution across the pixels, or the partial derivatives of a profile with
respect to its variables (that are needed to fill the Jacobi matrix in a
Levenberg-Marquard solver) are defined as properties that are only recalcu-
lated if any of the underlying variables (such as the profile centre or a
PSF variable) changes its value.

Latest SVN revision
-------------------
389, 2017/01/24
"""
import logging
import numpy as np


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 389


class Canvas(object):
    """
    This class does not define a profile, but only provides the frame for
    working on pixel arrays. The arrays may be irregularly sampled and must
    be initialized via directly providing the y- and x-position of each pixel.
    """

    def __init__(self, transform=None, **kwargs):
        """
        Initialize a new instance of the 'Canvas'-class.
        
        Parameters
        ----------
        transform : ndarray, optional
            This parameter may be used to provide the transformation from
            pixels to spatial positions as an array of shape (n, 2) for n
            pixels.

        Returns
        -------
        canvas : The newly created instance.
        """
        # initialize basic properties
        self._transform = transform

        # super(Canvas, self).__init__(**kwargs)

    @classmethod
    def from_shape(cls, shape, **kwargs):
        """
        This method can be used to initialize a new instance of the Canvas
        class from a size tuple, containing the grid dimensions along x and
        y. In this case, a homogeneous spatial sampling is assumed with a
        pixel size of 1 along x and y.

        Parameters
        ----------
        shape : tuple
            The shape of the canvas to create.
        kwargs
            Any additional keyword arguments are passed on to the call of
            Canvas.__init__.

        Returns
        -------
        canvas : The newly created instance.
        """
        assert len(shape) == 2, 'Parameter "shape" must be a tuple of length 2.'

        n_pixel = np.prod(shape)
        transform = np.vstack((np.arange(n_pixel, dtype=np.float32) // shape[1],
                               np.mod(np.arange(n_pixel, dtype=np.float32), shape[1]))).T
        return cls(transform=transform, **kwargs)

    @property
    def transform(self):
        """
        Returns the current transformation from pixels to x- & y-coordinates.
        """
        return self._transform

    @transform.setter
    def transform(self, value):
        """
        Specify the transformation from pixels to x-& y-coordinates.

        Parameters
        ----------
        value : nd_array
            An array specifying the x- and y-coordinate of each pixel, having
            size (n, 2) for n pixels. Note that following the numpy counting,
            the first number gives the size in y-direction.
        """
        if value.ndim != 2:
            logging.critical("'transform' must be 2dim., is {0}dim.".format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."

        if value.shape[1] != 2:
            logging.critical("'transform' must contain 2 spatial coordinates, is {0}.".format(value.shape[1]))
        assert value.shape[1] == 2, "'transform' must contain 2 spatial coordinates."

        # set basic class properties
        self._transform = value

        # reset all properties calculates so far in this instance that inherit from the transformation
        self.reset()

    def reset(self, **kwargs):
        """
        The is nothing to reset in this class. However, classes that inherit
        from 'Canvas' calculate arrays that must be recalculated if the size
        or the pixel positions of the array change.
        """
        return

    @property
    def pixel(self):
        """
        Return the indices of the individual pixels
        """
        return np.arange(self.transform.shape[0], dtype=np.int32)

    @property
    def n_pixel(self):
        """
        Returns the number of pixels on the canvas.
        """
        return self.transform.shape[0]

    @property
    def x(self):
        """
        Returns the x-coordinate of each pixel.
        """
        return self.transform[:, 1]

    @property
    def y(self):
        """
        Returns the y-coordinate of each pixel.
        """
        return self.transform[:, 0]


class Profile(Canvas):
    """
    This class does not describe an actual PSF profile but serves as a parent
    class for all classes that define a PSF profile. It provides the basic
    properties that calculate the centroid distance of each pixel or its
    partial derivatives with respect to the various parameters and that are
    inherited by the child classes of 'Profile'.

    The transformation from pixels (having coordinates u & v) to radii r
    is as follows (assuming a profile with origin [xc, yc]):
         (i)   x =  A*(x-xc) + C*(y-yc)
         (ii)  y = -B*(x-xc) + D*(y-yc)
         (iii) r = sqrt(x**2 +y**2/(1-e)**2)
    Where 'e' is the ellipticity of the profile and 'theta' the position
    angle, measured in radian and from the y-axis counter-clockwise. Note that
    'u' is the direction along the semi-major axis of the PSF, 'v' the one
    along the semi-minor axis.
    """

    def __init__(self, uc, vc, mag, e=0, theta=0, A=1, B=0, C=0, D=1, x0=0, y0=0, src_id=0,
                 maxrad=np.inf, osradius=0, osfactor=1, **kwargs):
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
        """
        # initialize parent class
        super(Profile, self).__init__(**kwargs)

        # initialize basic properties
        self.uc = uc  # Coordinates of source in reference system
        self.vc = vc
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
        self._maxrad = maxrad  # The radius (ins pixels) out to which the PSF will be calculated

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
        assert isinstance(value, (float, np.floating)), "'e' must be a float, not {0:s}".format(type(value).__name__)
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
        assert isinstance(value, (float, np.floating)), "'theta' must be a float, not {0:s}".format(type(value).__name__)
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
        assert isinstance(value, (float, int, np.floating)), "'A' must be a float or int, not {0:s}".format(
            type(value).__name__)
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
        assert isinstance(value, (float, int, np.floating)), "'B' must be a float or int, not {0:s}".format(
            type(value).__name__)
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
        assert isinstance(value, (float, int, np.floating)), "'C' must be a float or int, not {0:s}".format(
            type(value).__name__)
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
        assert isinstance(value, (float, int, np.floating)), "'D' must be a float or int, not {0:s}".format(
            type(value).__name__)
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
        assert isinstance(value, (float, int, np.floating)), "'x0' must be a float or int, not {0:s}".format(
            type(value).__name__)
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
        assert isinstance(value, (float, int, np.floating)), "'y0' must be a float or int, not {0:s}".format(
            type(value).__name__)
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
        assert isinstance(value, (float, np.floating)), "'xc' must be a float, not {0:s}".format(type(value).__name__)
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
        assert isinstance(value, (float, np.floating)), "'yc' must be a float, not {0:s}".format(type(value).__name__)
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
        assert isinstance(value, (float, int, np.floating)), "'maxrad' must be a float, not {0:s}".format(
            type(value).__name__)
        if value != self._maxrad:
            self._maxrad = value

            # Reset all arrays calculated for this instance
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
        assert isinstance(value, (float, int, np.floating)), "'mag' must be a float, not {0:s}".format(
            type(value).__name__)
        if value != self._mag:
            self._mag = value

            # Reset all arrays that inherit from mag
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "yos", "_u", "_v", "_r"])

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
                assert(is_oversampled.sum() == self.n_os * self.osxgrid.size)
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
                assert(is_oversampled.sum() == self.n_os * self.osygrid.size)
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
    def get_f(self, **kwargs):
        """
        Method that calculates the actual intensity profile. Note that because
        this class is not designed to compute an actual intensity profile, the
        method is mainly a place-holder for the `get_f`-methods of the
        individual sub-classes.

        Parameters:
        -----------

        Returns:
        --------
        result : zero value integer
            The return value is only used so that the method is fully
            compatible with the `make_array`-methods of the individual
            sub-classes.
        """
        assert not hasattr(super(Profile, self), 'get_f')
        # make sure no `get_f`-method is "hidden" behind this one (following recommendation from:
        # http://rhettinger.wordpress.com/2011/05/26/super-considered-super/)
        return 0

    @property
    def f(self):
        return self.get_f()

    def get_diff_e(self, integrate=True):
        """
        Calculate and return the partial derivative of the profile wrt. the
        ellipticity e.

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
        if self._diff_e is None:  # only re-calculate if necessary
            self._diff_e = self.v ** 2 / (self.r * (1. - self.e) ** 3)
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    @property
    def diff_e(self):
        return self.get_diff_e()

    def get_diff_theta(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the position
        angle 'theta'.

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
        if self._diff_theta is None:  # only re-calculate if necessary
            self._diff_theta = self.u * self.v * (1. - 1. / (1. - self.e) ** 2) / self.r
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    @property
    def diff_theta(self):
        return self.get_diff_theta()

    def get_diff_xc(self, integrate=True):
        """
        Calculate the partial derivative of the  profile wrt. xc, the centroid
        of the profile in x direction.
        
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
        if self._diff_xc is None:  # only re-calculate if necessary
            self._diff_xc = -(self.u * np.sin(self.theta) + (self.v / (1 - self.e) ** 2) * np.cos(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_xc)
        else:
            return self._diff_xc

    @property
    def diff_xc(self):
        return self.get_diff_xc()

    def get_diff_yc(self, integrate=True):
        """
        Calculate the partial derivative of the  profile wrt. xc, the centroid
        of the profile in y direction.
        
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
        if self._diff_yc is None:  # only re-calculate if necessary
            self._diff_yc = (self.u * np.cos(self.theta) - (self.v / (1 - self.e) ** 2) * np.sin(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_yc)
        else:
            return self._diff_yc

    @property
    def diff_yc(self):
        return self.get_diff_yc()

    def get_diff_A(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'A'
        of the coordinate transformation.

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
        if self._diff_A is None:  # only re-calculate if necessary
            self._diff_A = -self.uc * (self.u * np.sin(self.theta) + (
                self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A

    @property
    def diff_A(self):
        return self.get_diff_A()

    def get_diff_B(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'B'
        of the coordinate transformation.

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
        if self._diff_B is None:  # only re-calculate if necessary
            self._diff_B = -self.uc * (self.u * np.cos(self.theta) - (
                self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    @property
    def diff_B(self):
        return self.get_diff_B()

    def get_diff_C(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'C'
        of the coordinate transformation.

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
        if self._diff_C is None:  # only re-calculate if necessary
            self._diff_C = -self.vc * (self.u * np.sin(self.theta) + (
                self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    @property
    def diff_C(self):
        return self.get_diff_C()

    def get_diff_D(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'D'
        of the coordinate transformation.

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
        if self._diff_D is None:  # only re-calculate if necessary
            self._diff_D = self.vc * (self.u * np.cos(self.theta) - (
                self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    @property
    def diff_D(self):
        return self.get_diff_D()

    def get_diff_x0(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'y0'
        of the coordinate transformation.

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
        if self._diff_x0 is None:  # only re-calculate if necessary
            self._diff_x0 = -(self.u * np.sin(self.theta) + (self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_x0)
        else:
            return self._diff_x0

    @property
    def diff_x0(self):
        return self.get_diff_x0()

    # The property 'diff_y0' that returns the partial derivative of the intensity profile wrt. the parameter 'y0' of the
    # coordinate transformation.
    def get_diff_y0(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'y0'
        of the coordinate transformation.

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
        if self._diff_y0 is None:  # only re-calculate if necessary
            self._diff_y0 = (self.u * np.cos(self.theta) - (self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0

    @property
    def diff_y0(self):
        return self.get_diff_y0()

    def get_flux(self):
        """
        Get the calculated fluxes in each pixel used in the calculation

        Returns
        -------
        values : tuple
            The tuple contains the indices of the pixels used in the 
            calculation as the first entry, the second entry contains
            the actual fluxes per pixel.
        """
        return self.pixel[self.used], self.f


class Gaussian(Profile):
    """
    This class describes a two dimensional Gaussian profile. The functional
    form of the Gaussian with unity flux is given as

    G(r) = 1/(2*pi*std^2*(1-e))*exp(-r^2/(2*std**2))

    The width of the Gaussian is defined through the standard deviation 'std'
    that is related to the FWHM of the profile via

    FWHM = std*sqrt(8*ln(2))

    In this implementation of the Gaussian profile, the FWHM is used as the
    free parameter and the standard deviation is just computed internally.
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
        assert isinstance(value, (float, np.floating)), "'fwhm' must be a float, not {0:s}".format(type(value).__name__)
        assert value > 0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))
        if value != self._fwhm:
            self._fwhm = value

            # all properties that use the FWHM need to be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    @property
    def std(self):
        """
        Returns
        -------
        std : float
            The standard deviation of the profile, measured in pixels.
        """
        return self.fwhm / np.sqrt(8. * np.log(2.))

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
            self._f = 10 ** (-0.4 * self.mag) / (2. * np.pi * self.std ** 2 * (1. - self.e)) * np.exp(
                -self.r ** 2 / (2. * self.std ** 2)) + super(Gaussian, self).get_f(integrate=False)
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
            self._diff_fwhm = self.get_f(integrate=False) * (self.r ** 2 / self.std ** 3 - 2. / self.std) / np.sqrt(
                8. * np.log(2.))
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
            self._diff_e = self.get_f(integrate=False) * (
                1. / (1. - self.e) - (self.r / (self.std ** 2)) * super(Gaussian, self).get_diff_e(integrate=False))
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
            self._diff_theta = -self.get_f(integrate=False) * self.r / (self.std ** 2) \
                * super(Gaussian, self).get_diff_theta(integrate=False)
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
            self._diff_xc = -self.get_f(integrate=False) * self.r / (self.std ** 2) \
                * super(Gaussian, self).get_diff_xc(integrate=False)
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
            self._diff_yc = -self.get_f(integrate=False) * self.r / (self.std ** 2) * super(Gaussian, self).get_diff_yc(
                integrate=False)
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
            self._diff_A = -self.get_f(integrate=False) * self.r / (self.std ** 2) \
                * super(Gaussian, self).get_diff_A(integrate=False)
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
            self._diff_B = -self.get_f(integrate=False) * self.r / self.std ** 2 * super(Gaussian, self).get_diff_B(
                integrate=False)
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
            self._diff_C = -self.get_f(integrate=False) * self.r / (self.std ** 2) \
                * super(Gaussian, self).get_diff_C(integrate=False)
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
            self._diff_D = -self.get_f(integrate=False) * self.r / self.std ** 2 * super(Gaussian, self).get_diff_D(
                integrate=False)
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
            self._diff_x0 = -self.get_f(integrate=False) * self.r / self.std ** 2 * super(Gaussian, self).get_diff_x0(
                integrate=False)
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
            self._diff_y0 = -self.get_f(integrate=False) * self.r / self.std ** 2 * super(Gaussian, self).get_diff_y0(
                integrate=False)
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0


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
        assert isinstance(value, (float, np.floating)), "'beta' must be a float, not {0:s}".format(type(value).__name__)

        if not 1 <= value < 10:
            logging.error("'beta' must be >1 and <=10 but is {0:.1f}.".format(float(value)))
        assert 1 < value <= 10, "'beta' must be >1 and <=10 but is {0:.1f}.".format(float(value))

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
        assert isinstance(value, (float, np.floating)), "'fwhm' must be a float, not {0:s}".format(type(value).__name__)
        assert value > 0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))
        if value != self._fwhm:
            self._fwhm = value

            # all arrays that depend on 'beta' must be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    @property
    def r_d(self):
        """
        Returns
        -------
        r_d : float
            The effective radius of the profile.
        """
        return self.fwhm / (2. * np.sqrt(2 ** (1. / self.beta) - 1.))

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
            self._f = 10 ** (-0.4 * self.mag) * (self.beta - 1.) / (np.pi * (self.r_d ** 2) * (1. - self.e)) * (1. + (
                self.r / self.r_d) ** 2) ** (-self.beta) + super(Moffat, self).get_f(integrate=False)
        if integrate:
            return self.sum_os(self._f)
        else:
            return self._f

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
            self._diff_beta = self.get_f(integrate=False) * (
                1. / (self.beta - 1.) - np.log(1. + (self.r / self.r_d) ** 2) + np.log(2.) / (
                    self.beta * (1. - 2. ** (-1. / self.beta)))
                * ((self.r / self.r_d) ** 2 / (1. + (self.r / self.r_d) ** 2) - 1. / self.beta))
        if integrate:
            return self.sum_os(self._diff_beta)
        else:
            return self._diff_beta

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
            self._diff_fwhm = self.get_f(integrate=False) / (self.r_d * np.sqrt(2. ** (1. / self.beta) - 1.)) * (
                self.beta * (self.r / self.r_d) ** 2 / (1. + (self.r / self.r_d) ** 2) - 1.)
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
            self._diff_e = self.get_f(integrate=False) * (
                1. / (1. - self.e) - 2. * self.beta * self.r / ((self.r_d ** 2) * (1. + (self.r / self.r_d) ** 2))
                * super(Moffat, self).get_diff_e(integrate=False))
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
            self._diff_theta = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_theta(integrate=False)
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
            self._diff_xc = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_xc(integrate=False)
        if integrate:
            return self.sum_os(self._diff_xc)
        else:
            return self._diff_xc

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
            self._diff_yc = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_yc(integrate=False)
        if integrate:
            return self.sum_os(self._diff_yc)
        else:
            return self._diff_yc

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
            self._diff_A = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_A(integrate=False)
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
            self._diff_B = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_B(integrate=False)
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
            self._diff_C = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_C(integrate=False)
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
            self._diff_D = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_D(integrate=False)
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
            self._diff_x0 = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_x0(integrate=False)
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
            self._diff_y0 = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_y0(integrate=False)
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0


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

        self._exp1 = None
        self._exp2 = None
        self._fderiv = None
        self._diff_f12 = None  # place-holder for partial differential wrt. the flux ratio between Gaussians 1 and 2
        self._diff_fwhm1 = None  # place-holder for partial differential wrt. the FWHM of Gaussian 1
        self._diff_fwhm2 = None  # place-holder for partial differential wrt. the FWHM of Gaussian 2
        self.profile_properties.extend(["_exp1", "_exp2", "_fderiv", "_diff_f12", "_diff_fwhm1", "_diff_fwhm2"])

    # Define the 'f12' property to set and return the flux ratio of the two Gaussians ########
    @property
    def f12(self):
        """
        Returns the flux ratio between the two Gaussians. Note that the flux
        ratio is defined as f_1/(f_1 + f_2).
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
        assert isinstance(value, (float, np.floating)), "'f12' must be a float, not {0:s}".format(type(value).__name__)
        assert 0 < value < 1, "'f12' must be in range [0, 1] but is {0:.1f}".format(float(value))
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
        assert isinstance(value, (float, np.floating)), "'fwhm1' must be a float, not {0:s}".format(
            type(value).__name__)
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
        assert isinstance(value, (float, np.floating)), "'fwhm2' must be a float, not {0:s}".format(
            type(value).__name__)
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

    @property
    def std1(self):
        """
        Returns
        -------
        std1 : float
            The standard deviation of Gaussian 1, measured in pixels.
        """
        return self.fwhm1 / np.sqrt(8. * np.log(2.))

    @property
    def std2(self):
        """
        Returns
        -------
        std2 : float
            The standard deviation of Gaussian 2, measured in pixels.
        """
        return self.fwhm2 / np.sqrt(8. * np.log(2.))

    @property
    def exp1(self):
        """
        Helper function to get exponential term of Gaussian 1.

        Returns
        -------
        exp1 : nd_array
            The value of the exponential of Gaussian 1 in each pixel.
        """
        if self._exp1 is None:
            self._exp1 = np.exp(-self.r**2/(2.*self.std1**2))
        return self._exp1

    @property
    def exp2(self):
        """
        Helper function to get exponential term of Gaussian 2.

        Returns
        -------
        exp2 : nd_array
            The value of the exponential of Gaussian 2 in each pixel.
        """
        if self._exp2 is None:
            self._exp2 = np.exp(-self.r**2/(2.*self.std2**2))
        return self._exp2

    @property
    def fderiv(self):
        """
        Helper function to get partial derivatives wrt. r

        Returns
        -------
        fderiv : nd_array
            The partial derivative of the profile in each pixel with respect
            to the distance r to the centroid.
        """
        if self._fderiv is None:
            self._fderiv = -(self.f12 / (2.*np.pi*self.std1**4 * (1. - self.e)) * self.exp1 + (1. - self.f12) / (
                2.*np.pi*self.std2**4 * (1. - self.e)) * self.exp2)
        return self._fderiv

    # The property 'f' that returns the actual spatial intensity profile
    def get_f(self, integrate=True):
        """
        Method that calculates the actual intensity profile of the double
        Gaussian.

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
            self._f = 10 ** (-0.4 * self.mag) * (self.f12 / (2.*np.pi*self.std1**2 * (1. - self.e)) * self.exp1 + (
                1. - self.f12) / (2.*np.pi*self.std2**2 * (1. - self.e)) * self.exp2) + super(
                DoubleGaussian, self).get_f(integrate=False)
        if integrate:
            return self.sum_os(self._f)
        else:
            return self._f

    def get_diff_f12(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the flux ratio of Gaussians 1 and 2.

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
            self._diff_f12 = 10 ** (-0.4 * self.mag) * (
                1. / (2.*np.pi*self.std1**2 * (1. - self.e)) * self.exp1 - 1. / (
                    2.*np.pi*self.std2**2 * (1. - self.e)) * self.exp2)
        if integrate:
            return self.sum_os(self._diff_f12)
        else:
            return self._diff_f12

    @property
    def diff_f12(self):
        return self.get_diff_f12()

    def get_diff_fwhm1(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the FWHM of Gaussian 1.

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
            self._diff_fwhm1 = 10 ** (-0.4 * self.mag) * self.f12 / (2.*np.pi*self.std1**3 * (1. - self.e)) * (
                self.r**2/self.std1**2 - 2) / np.sqrt(8. * np.log(2.)) * self.exp1
        if integrate:
            return self.sum_os(self._diff_fwhm1)
        else:
            return self._diff_fwhm1

    @property
    def diff_fwhm1(self):
        return self.get_diff_fwhm1()

    def get_diff_fwhm2(self, integrate=True):
        """
        Calculate the partial derivative of the double Gaussian profile wrt.
        the FWHM of Gaussian 2.

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
            self._diff_fwhm2 = 10**(-0.4 * self.mag) * (1. - self.f12) / (2.*np.pi*self.std2**3 * (1. - self.e)) * (
                self.r**2/self.std2**2 - 2) / np.sqrt(8. * np.log(2.)) * self.exp2
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
        Calculate the partial derivative of the double Gaussian profile wrt.
        the ellipticity 'e'.

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
            self._diff_e = self.get_f(integrate=False) * (1. / (1. - self.e)) + 10**(-0.4 * self.mag) * self.r * super(
                DoubleGaussian, self).get_diff_e(integrate=False) * self.fderiv
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
            self._diff_theta = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_theta(integrate=False)
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
            self._diff_xc = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_xc(integrate=False)
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
            self._diff_yc = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_yc(integrate=False)
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
            self._diff_A = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_A(integrate=False)
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
            self._diff_B = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_B(integrate=False)
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
            self._diff_C = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_C(integrate=False)
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
            self._diff_D = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_D(integrate=False)
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
            self._diff_x0 = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_x0(integrate=False)
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
            self._diff_y0 = 10**(-0.4 * self.mag) * self.fderiv * self.r * super(
                DoubleGaussian, self).get_diff_y0(integrate=False)
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0
