"""
profiles.py
===========

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

There are two auxiliary classes defined in this module. 'IterRegistry' provides
an iterator over the available instances of the 'Profile' class (or one of its
subclasses), i.e., it is possible to write a loop like
>>> for psf in Profile:
>>>     ## do something
The second auxiliary class, 'classproperty', implements a property that is not
connected to an individual instance of a class, but to all instances of it. It
allows to change the value of a PSF variable (such as the FWHM) for all instan-
ces simultaneously without having to loop over them.

Latest SVN revision
-------------------
266, 2015/05/21
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 266

import logging
from scipy import spatial

import numpy as np


# class that defines an iterator over all its instances following suggestions at: 
# http://stackoverflow.com/questions/739882/iterating-over-object-instances-of-a-given-class-in-python
class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)


# define a property that is shared by all instances of a class following suggestions at:
# http://stackoverflow.com/questions/5189699/how-can-i-make-a-class-property-in-python?
class classproperty(object):
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


class Canvas(object):
    """
    This class does not define a profile, but only provides the frame for
    working on pixel arrays. The arrays may be irregularly sampled and can
    be initialized either via providing the (y,x)-size (in which case a
    regular sampling is assumed) or via directly providing the y- and x-
    positions of each pixel.

    Note that before any instance is created, the array to work on should be
    provided that is then inherited by all new instances. In addition, it is
    also possible to assign own arrays to instances after they have been
    created.
    """
    # allow for iteration over instances
    __metaclass__ = IterRegistry
    _registry = []

    # Initialization of basic properties
    _size = None  # The spatial size of the array
    _spaxel = None  # The index of each spaxel
    _transform = None  # The transformation from spaxel to x/y-coordinates

    @classmethod
    def set_size(cls, value):
        """
        Specify the size of the array on which to compute the profiles. Note
        that the size will be the same for all instances.

        Parameters
        ----------
        value : len=2 tuple
           The size must be provided as a tuple of length 2, providing the
           number of spaxels in x- & y-direction, respectively. Note that
           following the numpy counting, the first number gives the size in
           y-direction.
        """
        assert isinstance(value, (tuple, list)), "'size' should be a tuple, not {0}".format(type(value).__name__)
        assert len(value) == 2, "Array must be 2D, not {0:d}D".format(len(value))
        if value != cls._size:
            cls._spaxel = np.arange(value[0] * value[1], dtype=np.int32)

            # Assumption is that spaxels cover a rectangular area with filling factor 1.
            # Calculate x- & y-coordinates of each spaxel:
            _x = np.resize(np.arange(value[1], dtype=np.float64), (value[0] * value[1],))
            _y = np.arange(value[0] * value[1], dtype=np.float64) // value[1]
            cls._transform = np.vstack((_y, _x)).T
            cls._size = value

            # Delete all properties in all instances that were previously calculated.
            for inst in cls._registry:
                inst._own_size = None  # reset specific size set for the instance (if any)
                inst.reset()

    @classmethod
    def set_transform(cls, value):
        """
        Specify the transformation from spaxels to x-& y-coordinates. This is
        useful if the spaxels do not cover a rectangular area or are not
        quadratic and therefore the method 'set_size' is deprecated.

        Parameters
        ----------
        value : nd_array
            An array specifying the x- and y-coordinate of each spaxel, having
            size (<N_spaxels>, 2). Note that following the numpy counting, the
            first number gives the size in y-direction.
        """
        if value.ndim != 2:
            logging.critical("'transform' must be 2dim., is {0}dim.".format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."

        if value.shape[1] != 2:
            logging.critical("'transform' must contain 2 spatial coordinates, is {0}.".format(value.shape[1]))
        assert value.shape[1] == 2, "'transform' must contain 2 spatial coordinates."

        # set basic class properties
        cls._transform = np.asarray(value, dtype=np.float64)
        cls._size = (value.shape[0],)
        cls._spaxel = np.arange(value.shape[0], dtype=np.int32)

        # reset all properties calculates so far in any instance
        for inst in cls._registry:
            inst._own_transform = None  # reset specific transformation set for the instance (if any)
            inst.reset()

    def __init__(self, need_spaxels=True, **kwargs):
        """
        Initialize a new instance of the 'Canvas'-class.
        
        Parameters
        ----------
        need_spaxels : boolean, optional
            A flag indicating if the size or pixel positions of the array must
            be known before creating a new instance.
        """
        # add instance to registry of class (needed to iterate over instances)
        self._registry.append(self)

        # initialize basic properties
        self._own_transform = None
        self._own_size = None

        # Field of view must be specified beforehand
        if need_spaxels:
            if self.transform is None:
                logging.error("No (x,y)-coordinate system for spaxels defined.")
            assert self.transform is not None, "Specify size first or provide x- & y-coordinates of each spaxel."

    def reset(self, **kwargs):
        """
        The is nothing to reset in this class. However, classes that inherit
        from 'Canvas' calculate arrays that must be recalculated if the size
        or the pixel positions of the array change.
        """
        return

    def set_own_size(self, value):
        """
        Specify the size of the array on which to compute the profiles for this
        instance only.

        Parameters
        ----------
        value : len=2 tuple
           The size must be provided as a tuple of length 2, providing the
           number of spaxels in x- & y-direction, respectively. Note that
           following the numpy counting, the first number gives the size in
           y-direction.
        """
        assert isinstance(value, (tuple, list)), "'size' should be a tuple, not {0}".format(type(value).__name__)
        assert len(value) == 2, "Array must be 2D, not {0:d}D".format(len(value))
        if value != self._own_size:
            # Assumption is that spaxels cover a rectangular area with filling factor 1. Calculate x- & y-coordinates
            # of each spaxel:
            _x = np.resize(np.arange(value[1], dtype=np.float64), (value[0] * value[1],))
            _y = np.arange(value[0] * value[1], dtype=np.float64) // value[1]
            self._own_transform = np.vstack((_y, _x)).T
            self._own_size = value

            # Reset all arrays in this instance that were previously calculated.
            self.reset()

    def get_size(self):
        """
        The size of the spatial array on which the profile is calculated. 
        
        Returns
        -------
        size : tuple
            The size of the spatial array. Note that a size=2 tuple that can be
            related  to the y- & x-size of the array is only provided if the size
            has been explicitely defined. Otherwise, a size=1 tuple with the
            number of spaxels is returned.
        """
        if self._own_size is None:
            return self._size
        else:
            return self._own_size

    def del_own_size(self):
        """
        Delete a specific size set for this instance.
        """
        if self._own_size is None: return
        self._own_size = None

        # all properties for this instance must be recalculated.
        self.reset()

    size = property(get_size, set_own_size, del_own_size, "The size of the spatial array.")

    def set_own_transform(self, value):
        """
        Set an instance-specific transformation from spaxels to x-& y-coordinates.

        Parameters
        ----------
        value : nd_array
            An array specifying the x- and y-coordinate of each spaxel, having
            size (<N_spaxels>, 2). Note that following the numpy counting, the
            first number gives the size in y-direction.
        """
        if value.ndim != 2:
            logging.critical("'transform' must be 2dim., is {0}dim.".format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."
        if value.shape[1] != 2:
            logging.critical("'transform' must contain 2 spatial coordinates, is {0}.".format(value.shape[1]))
        assert value.shape[1] == 2, "'transform' must contain 2 spatial coordinates."

        # set basic class properties
        self._own_transform = np.asarray(value, dtype=np.float64)
        self._own_size = (value.shape[0],)

        # reset all arrays calculated so far in this instance
        self.reset()

    def get_transform(self):
        """
        The transformation from spaxels to x- and y-coordinates. The
        routine will use the specific transformation for this instance if it
        has been defined, otherwise it will use the global one defined for th
        class. Note that following the numpy counting, the first axis is the
        y-direction.
        
        Returns
        -------
        transform : nd_array
            The transformation from spaxels to x- and y-coordinates, provided as
            an array of shape (2, <number of spaxels>)
        """
        if self._own_transform is None:
            return self._transform
        else:
            return self._own_transform

    def del_own_transform(self):
        """
        Delete a specific transformation for this instance.
        """
        if self._own_transform is None: return
        self._own_transform = None

        # all properties for this instance must be recalculated.
        self.reset()

    transform = property(get_transform, set_own_transform, del_own_transform,
                         "The transformation from spaxels to x- & y-coordinates.")

    @property
    def spaxel(self):
        """
        Return the indices of the individual spaxels
        """
        return np.arange(self.transform.shape[0], dtype=np.int32)

    @property
    def x(self):
        """
        Returns the x-coordinate of each spaxel.
        """
        return self.transform[:, 1]

    @property
    def y(self):
        """
        Returns the y-coordinate of each spaxel.
        """
        return self.transform[:, 0]


class Profile(Canvas):
    """
    This class does not describe an actual PSF profile but serves as a parent
    class for all classes that define a PSF profile. It provides the basic
    properties that calculate the centroid distance of each spaxel or its
    partial derivatives with respect to the various parameters and that are
    inherited by the child classes of 'Profile'.

    The transformation from spaxels (having coordinates u & v) to radii r
    is as follows (assuming a profile with origin [xc, yc]):
         (i)   x =  A*(x-xc) + C*(y-yc)
         (ii)  y = -B*(x-xc) + D*(y-yc)
         (iii) r = sqrt(x**2 +y**2/(1-e)**2)
    Where 'e' is the ellipticity of the profile and 'theta' the position angle,
    measured in radian and from the y-axis counter-clockwise. Note that 'u' is
    the direction along the semi-major axis of the PSF, 'v' the one along the
    semi-minor axis.
    """

    # Initialization of basic properties
    _e = 0  # The ellipticity of the profile
    _theta = 0  # The position angle of the profile
    _A = 1  # The various parameters that specify the coordinate 
    _B = 0  # transformation; Note that 'Profile' supports two ways of
    _C = 0  # defining the source coordinates: either directly or via
    _D = 1  # a transformation from reference coordinates (see below).
    _x0 = 0
    _y0 = 0

    @classmethod
    def set_theta(cls, value):
        """
        Set the position angle 'theta' for all instances of the class. Calling
        this method will overwrite any instance-specific values of the position
        angle that might be present.

        Parameters
        ----------
        value : float
            The position angle, given in radian. Note that 'theta' is measured
            from the y-axis in counter-clockwise direction.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'theta' must be a float, not {0:s}".format(
            type(value).__name__)
        if value != cls._theta:
            cls._theta = value

            # all properties building up on the u- & v-coordinates must be recalculated.
            for inst in cls._registry:
                inst._own_theta = None  # remove possible position angle value of the specific instance
                inst.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos"])

    @classmethod
    def set_e(cls, value):
        """
        Set the ellipticity 'e' of all instances of the class. Calling this
        method will overwrite any instance-specific values of the ellipticity
        that might be present.

        Parameters
        ----------
        value : float
            The ellipticity of the profiles must be in the interval [0,1).
            value=0 corresponds to a round profile.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'e' must be a float, not {0:s}".format(
            type(value).__name__)
        if not value >= 0 and value < 1:
            logging.error("'e' must be in interval [0,1) but is {0:.1f}".format(float(value)))
        assert -1e-3 <= value < 1, "'e' must be in interval [0,1) but is {0:.1f}".format(float(value))
        if value != cls._e:
            cls._e = value

            # all properties building on the radius values of the individual spaxels must be recalculated.
            for inst in cls._registry:
                # remove possible ellipticity of the specific instance.
                inst._own_e = None
                inst.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v"])

    @classmethod
    def set_A(cls, value):
        """
        Set the parameter 'A' of the coordinate transformation for all
        instances of the class. Calling this method will overwrite any
        instance-specific values of the profile centroids that might be
        present.

        Parameters
        ----------
        value : float
            The parameter 'A' of the coordinate transformation.
        """
        assert isinstance(value, (float, int, np.float32, np.float64)), "'A' must be a float or int, not {0:s}".format(
            type(value).__name__)
        if value != cls._A:
            cls._A = value

            # All arrays must be recalculated
            for inst in cls._registry:
                inst._xc = None  # Remove a possible centroid defined for the specific instance.
                inst._yc = None
                inst.reset()

    @classproperty
    def A(cls):
        """
        Return the parameter 'A' of the coordinate transformation.

        Returns
        -------
        A : float
        """
        return cls._A

    @classmethod
    def set_B(cls, value):
        """
        Set the parameter 'B' of the coordinate transformation for all
        instances of the class. Calling this method will overwrite any
        instance-specific values of the profile centroids that might be
        present.

        Parameters
        ----------
        value : float
            The parameter 'B' of the coordinate transformation.
        """
        assert isinstance(value, (float, int, np.float32, np.float64)), "'B' must be a float or int, not {0:s}".format(
            type(value).__name__)
        if value != cls._B:
            cls._B = value

            # All arrays must be recalculated
            for inst in cls._registry:
                inst._xc = None  # Remove a possible centroid defined for the specific instance.
                inst._yc = None
                inst.reset()

    @classproperty
    def B(cls):
        """
        Return the parameter 'B' of the coordinate transformation.

        Returns
        -------
        B : float
        """
        return cls._B

    @classmethod
    def set_C(cls, value):
        """
        Set the parameter 'C' of the coordinate transformation for all
        instances of the class. Calling this method will overwrite any
        instance-specific values of the profile centroids that might be
        present.

        Parameters
        ----------
        value : float
            The parameter 'C' of the coordinate transformation.
        """
        assert isinstance(value, (float, int, np.float32, np.float64)), "'C' must be a float or int, not {0:s}".format(
            type(value).__name__)
        if value != cls._C:
            cls._C = value

            # All arrays must be recalculated
            for inst in cls._registry:
                inst._xc = None  # Remove a possible centroid defined for the specific instance.
                inst._yc = None
                inst.reset()

    @classproperty
    def C(cls):
        """
        Return the parameter 'A' of the coordinate transformation.

        Returns
        -------
        C : float
        """
        return cls._C

    @classmethod
    def set_D(cls, value):
        """
        Set the parameter 'D' of the coordinate transformation for all
        instances of the class. Calling this method will overwrite any
        instance-specific values of the profile centroids that might be
        present.

        Parameters
        ----------
        value : float
            The parameter 'D' of the coordinate transformation.
        """
        assert isinstance(value, (float, int, np.float32, np.float64)), "'D' must be a float or int, not {0:s}".format(
            type(value).__name__)
        if value != cls._D:
            cls._D = value

            # All arrays must be recalculated
            for inst in cls._registry:
                inst._xc = None  # Remove a possible centroid defined for the specific instance.
                inst._yc = None
                inst.reset()

    @classproperty
    def D(cls):
        """
        Return the parameter 'D' of the coordinate transformation.

        Returns
        -------
        D : float
        """
        return cls._D

    @classmethod
    def set_x0(cls, value):
        """
        Set the parameter 'x0' of the coordinate transformation for all
        instances of the class. Calling this method will overwrite any
        instance-specific values of the profile centroids that might be
        present.

        Parameters
        ----------
        value : float
            The parameter 'x0' of the coordinate transformation.
        """
        assert isinstance(value, (float, int, np.float32, np.float64)), "'x0' must be a float or int, not {0:s}".format(
            type(value).__name__)
        if value != cls._x0:
            cls._x0 = value

            # All arrays must be recalculated
            for inst in cls._registry:
                inst._xc = None  # Remove a possible centroid defined for the specific instance.
                inst._yc = None
                inst.reset()

    @classproperty
    def x0(cls):
        """
        Return the parameter 'x0' of the coordinate transformation.

        Returns
        -------
        x0 : float
        """
        return cls._x0

    @classmethod
    def set_y0(cls, value):
        """
        Set the parameter 'D' of the coordinate transformation for all
        instances of the class. Calling this method will overwrite any
        instance-specific values of the profile centroids that might be
        present.

        Parameters
        ----------
        value : float
            The parameter 'y0' of the coordinate transformation.
        """
        assert isinstance(value, (float, int, np.float32, np.float64)), "'y0' must be a float or int, not {0:s}".format(
            type(value).__name__)
        if value != cls._y0:
            cls._y0 = value

            # All properties must be recalculated
            for inst in cls._registry:
                inst._xc = None  # Remove a possible centroid defined for the specific instance.
                inst._yc = None
                inst.reset()

    @classproperty
    def y0(cls):
        """
        Return the parameter 'y0' of the coordinate transformation.

        Returns
        -------
        y0 : float
        """
        return cls._y0

    def __init__(self, uc, vc, mag, src_id=0, maxrad=np.inf, osradius=0, osfactor=1, **kwargs):
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
        src_id : int
            The ID of the source in any catalogue.
        maxrad : float
            The maximum distance from the profile centroid where the
            contribution of the profile will be calculated. Default is to
            calculate it for the full field of view.
        osradius : float
            The radius from the profile centroid within which the spaxels
            will be supersampled to improve the accuracy of the calculation.
            Default is no supersampling.
        osfactor : int
           The supersampling factor. Each spaxel within 'osradius' will be
           supersampled into 'osfactor' times 'osfactor' sub-spaxels.
        """
        # initialize parent class
        super(Profile, self).__init__(**kwargs)

        # intialize basic properties
        self.uc = uc  # Coordinates of source in reference system
        self.vc = vc
        self._xc = None  # can contain direct coordinates for the centroid of the profile
        self._yc = None
        self._mag = mag  # The source magnitude
        self.src_id = src_id  # The source ID
        self._own_e = None  # instance-specific values for the ellipticity and position angle may be provided.
        self._own_theta = None
        self._maxrad = maxrad  # The radius (ins spaxels) out to which the PSF will be calculated

        # Initialize the arrays that will hold the per-spaxel-information of the profile
        self._used = None  # will be a boolean array with all spaxels within distance 'maxrad' to centroid set to True.
        self._oversampled = None  # will be a boolean array with all supersampled spaxels set to True.
        self._osmap = None  # The arrays osmap and ossummer will be needed to properly include the oversampling
        self._ossummer = None
        self._xos = None  # The x- & y-coordinates of the spaxels including the sub-spaxels in supersampled ones.
        self._yos = None
        self._u = None  # The u- & v-coordinates of the spaxels.
        self._v = None
        self._r = None  # The distances of the spaxels wrt. the profile centroid
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

        # Properly implement possibility to oversample central spaxels
        self.osr = osradius
        if not self.osr > 0:
            self.osf = 1
            self.osxgrid = None
            self.osygrid = None
        else:
            self.osf = max(1, osfactor)

            # osxgrid and osygrid specify the offset (in spaxels) of the individual sub-spaxels wrt. to oversampled
            # spaxel.
            osgrid = np.linspace(-0.5 + 1. / (2 * self.osf), 0.5 - 1. / (2 * self.osf), self.osf)
            self.osxgrid = np.resize(osgrid, (self.osf * self.osf,))
            self.osygrid = np.sort(self.osxgrid)

    def reset(self, omit=[]):
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
            if prop not in omit:
                setattr(self, prop, None)

    def set_maxrad(self, value):
        """
        Set the calculation radius of the profile.
        
        Parameters
        ----------
        value : float
            The radius (in pixels) out to which the profile will be
            calculated.
        """
        assert isinstance(value, (float, int)), "'maxrad' must be a float, not {0:s}".format(type(value).__name__)
        if value != self._maxrad:
            self._maxrad = value

            # Reset all arrays calculated for this instance
            self.reset()

    def get_maxrad(self):
        """
        Return the calculation radius for this instance.
        """
        return self._maxrad

    maxrad = property(get_maxrad, set_maxrad, None, "The radius out to which the profile is calculated.")

    def set_mag(self, value):
        """
        Set the integrated magnitude of the profile.

        Parameters
        ----------
        value : float
            The integrated magnitude. A magnitude of zero yields a flux of
            one, i.e. f=10^(-0.4*m).
        """
        assert isinstance(value, (float, int)), "'mag' must be a float, not {0:s}".format(type(value).__name__)
        if value != self._mag:
            self._mag = value

            # Reset all arrays that inherit from mag
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "yos", "_u", "_v", "_r"])

    def get_mag(self):
        """
        Return the current integrated magnitude of the profile.
        """
        return self._mag

    mag = property(get_mag, set_mag, None, "The broadband magnitude of the profile star.")

    def get_xc(self):
        """
        Return the centroid of the profile along the x-axis. The routine will
        return the direct centroid if provided, and otherwise use the
        reference centroid in combination with the coordinate transformation
        of the class.

        Returns
        -------
        xc : float
            The centroid of the profile along the x-axis in units of spaxels.
        """
        if self._xc is None:  # use coordinate transformation if no direct centroid available
            return self.A * self.uc + self.C * self.vc + self.x0
        else:
            return self._xc

    def set_xc(self, value):
        """
        Set the direct centroid of the profile along the x-axis.

        Parameters
        ----------
        value : float
            The centroid should be provided in units of spaxels.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'xc' must be a float, not {0:s}".format(
            type(value).__name__)
        # Turned off following check as it turned out to be quite time-consuming
        #        assert (spatial.distance.cdist([[self.yc, value]], self.transform)>=-self.maxrad).any(), \
        #            "Provided xc={0:.2f} places source #{1} at d > maxrad to FoV.".format(float(value), self.src_id)
        if value != self._xc:
            self._xc = value

            # reset all arrays
            self.reset()

    def del_xc(self):
        """
        Delete the direct centroid of the profile along the x-axis (if
        available)
        """
        if self._xc is None:
            return
        self._xc = None

        # flag all properties for recalculation
        self.reset()

    xc = property(get_xc, set_xc, del_xc, "The centroid of the profile along the x-axis.")

    def get_yc(self):
        """
        Return the centroid of the profile along the y-axis. The routine will
        return the direct centroid if provided, and otherwise use the
        reference centroid in combination with the coordinate transformation
        of the class.

        Returns
        -------
        yc : float
            The centroid of the profile along the y-axis in units of spaxels.
        """
        if self._yc is None:  # use coordinate transformation if no direct centroid available
            return self.D * self.vc - self.B * self.uc + self.y0
        else:
            return self._yc

    def set_yc(self, value):
        """
        Set the direct centroid of the profile along the y-axis.

        Parameters
        ----------
        value : float
            The centroid should be provided in units of spaxels.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'yc' must be a float, not {0:s}".format(
            type(value).__name__)
        # Turned off following check as it turned out to be quite time-consuming
        #        assert (spatial.distance.cdist([[value, self.xc]], self.transform)>=-self.maxrad).any(), \
        #            "Provided yc={0:.2f} places source #{1} at d > maxrad to FoV.".format(value, self.src_id)
        if value != self._yc:
            self._yc = value

            # reset all arrays calculated for this  instance 
            self.reset()

    def del_yc(self):
        """
        Delete the direct centroid of the profile along the y-axis (if
        available)
        """
        if self._yc is None:
            return
        self._yc = None

        # reset all arrays calculated for this  instance 
        self.reset()

    yc = property(get_yc, set_yc, del_yc, "The centroid of the profile along the y-axis.")

    def get_theta(self):
        """
        Return the position angle of the profile. The routine will use the
        specific pos. angle of the instance if available and otherwise fall
        back to the one of the class.

        Returns
        -------
        theta : float
            The position angle in radian.
        """
        if self._own_theta is None:  # if no instance-specific angle has been defined, use the one of the class.
            return self._theta
        else:
            return self._own_theta

    def set_own_theta(self, value):
        """
        Set the position angle for this instance.

        Parameters
        ----------
        value : float
            The position angle should be provided in radian.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'theta' must be a float, not {0:s}".format(
            type(value).__name__)
        if value != self._own_theta:
            self._own_theta = value

            # all arrays that build up on u- & v-coordinates must be recalculated.
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos"])

    def del_own_theta(self):
        """
        Delete the value of the position angle for this instance.
        """
        if self._own_theta is None:
            return
        self._own_theta = None

        # all arrays that build up on u- & v-coordinates must be recalculated.
        self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos"])

    theta = property(get_theta, set_own_theta, del_own_theta, "The position angle theta of the PSF.")

    def get_e(self):
        """
        Return the ellipticity of the profile. The routine will use the
        specific value defined for the instance if available and otherwise
        fall back to the one of the class.

        Returns
        -------
        e : float
            The ellipticity of the profile.
        """
        if self._own_e is None:  # if no specific value available, use the one defined for the class.
            return self._e
        else:
            return self._own_e

    def set_own_e(self, value):
        """
        Set the ellipticity for this instance.

        Parameters
        ----------
        value : float
            The ellipticity of the profile.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'e' must be a float, not {0:s}".format(
            type(value).__name__)
        assert 0 <= value < 1, "'e' must be in interval [0,1), but is {0:.1f}".format(float(value))

        if value != self._own_e:
            self._own_e = value

            # all properties that build up on the distance r to the centroid of the profile must be recalculated.
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v"])

    def del_own_e(self):
        """
        Delete the value of the ellipticity for this instance.
        """
        if self._own_e is None:
            return
        self._own_e = None

        # all properties that build up on the distance r to the centroid of the profile must be recalculated.
        self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v"])

    e = property(get_e, set_own_e, del_own_e, "The ellipticity 'e' of the PSF.")

    @property
    def used(self):
        """
        'used' is a boolean array that is set to True for each spaxel that is
        included in the calculation, i.e. is within a distance 'maxrad' to the
        profile centroid.

        Returns
        -------
        used : nd_array
            A 1dim. boolean array with length = number of spaxels in class.
        """
        if self._used is None:
            if self.maxrad == np.inf:  # use all spaxels if requested
                self._used = np.ones((self.transform.shape[0],), dtype=np.bool)
            else:
                self._used = np.sqrt((self.x - self.xc) ** 2 + (self.y - self.yc) ** 2) <= self.maxrad
        return self._used

    @property
    def Nused(self):
        """
        Returns
        -------
        Nused : int
            The number of spaxels included in the calculation. 
        """
        return self.used.sum()

    @property
    def oversampled(self):
        """
        'oversampled' is a boolean array that is set to True for each spaxel
        that is supersampled in the calculation, i.e. is within a distance
        'osradius' to the profile centroid.

        Returns
        -------
        oversampled : nd_array
            A 1dim. boolean array with length = number of spaxels in class.
        """
        if self._oversampled is None:
            self._oversampled = np.sqrt((self.x - self.xc) ** 2 + (self.y - self.yc) ** 2) <= self.osr
        return self._oversampled

    @property
    def Nos(self):
        """
        Returns
        -------
        Nos : int
            The number of supersampled spaxels included in the calculation. 
        """
        return self.oversampled.sum()

    @property
    def norm(self):
        """
        Returns
        -------
        norm : nd_array
            A 1dim. array with length = number of spaxels in class. It provides
            the normalization for each spaxel, i.e. the value by which the
            calculated fluxes must be divided to return the true ones. It is
            basically 1 for each spaxel used in the calculation except for the
            supersampled ones, where it is equal to the number of sub-spaxels.
        """
        norm = np.asarray(self.used, dtype=np.float64)
        norm[self.oversampled] *= self.osf ** 2
        return norm

    # The supersampling of pixels is included in the following way:
    # All calculations are performed on 1dim. arrays, and every sub-spaxel is included in those arrays. The property
    # 'osmap' keeps track of the parent spaxel of each (sub)spaxel used in the calculation. When a calculated property
    # is returned, by default the original spaxels are restored by averaging over the sub-spaxels. This is done using
    # the property 'ossummer' and the method 'sum_os'. The averaging can be turned off for internal calculations
    # (see below).
    @property
    def osmap(self):
        """
        Returns
        -------
        osmap : nd_array
            An array of integers providing the sorted indices of all spaxels
            included in the calculation. The indices of the spaxels that are
            supersampled are included 'osfactor'**2 times.
        """
        if self._osmap is None:
            # np.resize uses repeated copies of data to fill new values
            self._osmap = np.hstack((self.spaxel[self.used],
                                     np.resize(self.spaxel[self.oversampled], ((self.osf * self.osf - 1) * self.Nos,))))
            self._osmap.sort()
        return self._osmap

    @property
    def ossummer(self):
        """
        Returns
        -------
        ossummer : nd_array
            An 2dim. array of integers with shape (2, Nspaxel). It contains
            the first and last index of every parent spaxel in the 'osmap'
            array and is used by the method 'sum_os' to restore the values in
            the parent spaxels.
        """
        if self._ossummer is None:
            # first calculate difference of adjacent entries in 'osmap'. Then find indices where this difference is >0
            # and include those in ossummer
            next_spaxel = np.flatnonzero(np.diff(np.r_[self.osmap[0] - 1, self.osmap, self.osmap[-1] + 1]))
            self._ossummer = np.vstack((next_spaxel[:-1], next_spaxel[1:])).T
        return self._ossummer

    def sum_os(self, data):
        """
        This method calculates the value of 'data' in every parent spaxel by
        averaging over its sub-spaxels. This is done following suggestion at:
        http://stackoverflow.com/questions/7471474/\
            numpy-sum-of-values-in-subarrays-between-pairs-of-indices
        
        Parameters
        ----------
        data : nd_array
            A 1dim. array with the same size as property 'osmap'.

        Returns
        -------
        avg_data : nd_array
            The result of averaging the values in 'data' over all subspaxels.
            Its length is equal to the number of used spaxels
            (property 'Nspaxel').
        """
        if self.osf <= 1 or self.Nos == 0:
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
            A one dimensional array containing the x-coordinate of each spaxel
            used in the calculation (including sub-spaxels!) in the field of
            view. It therefore has the same length as the 'osmap'-property.
        """
        if self._xos is None:
            self._xos = self.x[self.osmap]
            if self.osf > 1:
                # add offsets of sub-spaxel wrt. parent spaxel along x-axis
                is_oversampled = np.in1d(self.osmap, np.flatnonzero(self.oversampled))
                assert(is_oversampled.size == self.Nos * self.osxgrid.size)
                self._xos[is_oversampled] += np.resize(self.osxgrid, (self.Nos * self.osxgrid.size,))
        return self._xos

    @property
    def yos(self):
        """
        Returns
        -------
        yos : nd_array
            A one dimensional array containing the y-coordinate of each spaxel
            used in the calculation (including sub-spaxels!) in the field of
            view. It therefore has the same length as the 'osmap'-property.
        """
        if self._yos is None:
            self._yos = self.y[self.osmap]
            if self.osf > 1:
                # add offsets of sub-spaxel wrt. parent spaxel along y-axis
                is_oversampled = np.in1d(self.osmap, np.flatnonzero(self.oversampled))
                assert(is_oversampled.size == self.Nos * self.osygrid.size)
                self._yos[is_oversampled] += np.resize(self.osygrid, (self.Nos * self.osygrid.size,))
        return self._yos

    @property
    def u(self):
        """
        Returns
        -------
        u : nd_array
            A one dimensional array containing the coordinate of each spaxel
            used in the calculation (including sub-spaxels!) in u-direction,
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
            A one dimensional array containing the coordinate of each spaxel
            used in the calculation (including sub-spaxels!) in v-direction,
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
            A one dimensional array containing the distance of each spaxel
            used in the calculation (including sub-spaxels!) to the profile
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

    f = property(get_f, None, None, "A place-holder for the intensity distribution of the PSF.")

    # The property 'diff_e' returns the partial derivative of the intensity profile wrt. the ellipticity 'e'.
    def get_diff_e(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the ellipticity
        'e'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_e : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_e is None:  # only re-calculate if necessary
            self._diff_e = self.v ** 2 / (self.r * (1. - self.e) ** 3)
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    diff_e = property(get_diff_e, None, None, "The partial derivative of the profile wrt. the ellipticity 'e'.")

    # The property 'diff_theta' that returns the partial derivative of the intensity profile wrt. the position angle
    # 'theta'.
    def get_diff_theta(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the position
        angle 'theta'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_theta : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_theta is None:  # only re-calculate if necessary
            self._diff_theta = self.u * self.v * (1. - 1. / (1. - self.e) ** 2) / self.r
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    diff_theta = property(get_diff_theta, None, None,
                          "The partial derivative of the profile wrt. the pos. angle 'theta'.")

    def get_diff_xc(self, integrate=True):
        """
        Calculate the partial derivative of the  profile wrt. xc, the centroid
        of the profile in x direction.
        
        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.
            
        Returns
        -------
        diff_xc : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_xc is None:  # only re-calculate if necessary
            self._diff_xc = -(self.u * np.sin(self.theta) + (self.v / (1 - self.e) ** 2) * np.cos(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_xc)
        else:
            return self._diff_xc

    diff_xc = property(get_diff_xc, None, None,
                       "The partial derivative of the profile wrt. xc, the x-centroid of the profile.")

    def get_diff_yc(self, integrate=True):
        """
        Calculate the partial derivative of the  profile wrt. xc, the centroid
        of the profile in y direction.
        
        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.
            
        Returns
        -------
        diff_yc : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_yc is None:  # only re-calculate if necessary
            self._diff_yc = (self.u * np.cos(self.theta) - (self.v / (1 - self.e) ** 2) * np.sin(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_yc)
        else:
            return self._diff_yc

    diff_yc = property(get_diff_yc, None, None,
                       "The partial derivative of the profile wrt. yc, the y-centroid of the profile.")

    # The property 'diff_A' that returns the partial derivative of the intensity profile wrt. the parameter 'A' of the
    # coordinate transformation.
    def get_diff_A(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'A'
        of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_A : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_A is None:  # only re-calculate if necessary
            self._diff_A = -self.uc * (self.u * np.sin(self.theta) + (
                self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
#             self._diff_A = -(self.u * (self.uc * np.sin(self.theta) - self.vc * np.cos(self.theta)) + (
#                 self.v / (1. - self.e) ** 2) * (self.vc * np.sin(self.theta) + self.uc * np.cos(self.theta))) / self.r
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A

    diff_A = property(get_diff_A, None, None, "The partial derivative of the profile wrt. 'A'.")

    # The property 'diff_B' that returns the partial derivative of the intensity profile wrt. the parameter 'B' of the
    # coordinate. transformation.
    def get_diff_B(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'B'
        of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_B : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_B is None:  # only re-calculate if necessary
            self._diff_B = -self.uc * (self.u * np.cos(self.theta) - (
                self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
#            self._diff_B = -(self.u * (self.vc * np.sin(self.theta) + self.uc * np.cos(self.theta)) + (
#                self.v / (1. - self.e) ** 2) * (- self.uc * np.sin(self.theta) + self.vc * np.cos(self.theta))) / self.r
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    diff_B = property(get_diff_B, None, None, "The partial derivative of the profile wrt. 'B'.")

    # The property 'diff_C' that returns the partial derivative of the intensity profile wrt. the parameter 'C' of the
    # coordinate. transformation.
    def get_diff_C(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'C'
        of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_B : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_C is None:  # only re-calculate if necessary
            self._diff_C = -self.vc * (self.u * np.sin(self.theta) + (
                self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    diff_C = property(get_diff_C, None, None, "The partial derivative of the profile wrt. 'C'.")

    # The property 'diff_D' that returns the partial derivative of the intensity profile wrt. the parameter 'D' of the
    # coordinate. transformation.
    def get_diff_D(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'D'
        of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_B : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_D is None:  # only re-calculate if necessary
            self._diff_D = self.vc * (self.u * np.cos(self.theta) - (
                self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    diff_D = property(get_diff_D, None, None, "The partial derivative of the profile wrt. 'D'.")

    # The property 'diff_x0' that returns the partial derivative of the intensity profile wrt. the parameter 'x0' of the
    # coordinate transformation.
    def get_diff_x0(self, integrate=True):
        """
        Calculate the partial derivative of the profile wrt. the parameter 'y0'
        of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_x0 : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_x0 is None:  # only re-calculate if necessary
            self._diff_x0 = -(self.u * np.sin(self.theta) + (self.v / (1. - self.e) ** 2) * np.cos(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_x0)
        else:
            return self._diff_x0

    diff_x0 = property(get_diff_x0, None, None, "The partial derivative of the profile wrt. 'x0'.")

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
            sub-spaxels before they are returned.

        Returns
        -------
        diff_y0 : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_y0 is None:  # only re-calculate if necessary
            self._diff_y0 = (self.u * np.cos(self.theta) - (self.v / (1. - self.e) ** 2) * np.sin(self.theta)) / self.r
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0

    diff_y0 = property(get_diff_y0, None, None, "The partial derivative of the profile wrt. 'D'.")

    def get_flux(self):
        """
        Get the calculated fluxes in each spaxel used in the calculation

        Returns
        -------
        values : tuple
            The tuple contains the indices of the spaxels used in the 
            calculation as the first entry, the second entry contains
            the actual fluxes per spaxel.
        """
        return self.spaxel[self.used], self.f


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

    _fwhm = 3.
    function = "gaussian"

    @classmethod
    def set_fwhm(cls, value):
        """
        Set the FWHM for all instances of the class. Calling this method will
        overwrite any instance-specific values of the FWHM that might be
        present.

        Parameters
        ----------
        value : float
            The FWHM, given in spaxels
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'fwhm' must be a float, not {0:s}".format(
            type(value).__name__)
        assert value > 0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))
        if value != cls._fwhm:
            cls._fwhm = value

            for inst in cls._registry:
                inst._own_fwhm = None  # reset possible instance-specific value

                # all properties that include the FWHM need to be recalculated
                inst.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    def __init__(self, uc, vc, mag, src_id=0, maxrad=np.inf, osradius=0, osfactor=1, **kwargs):
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
        src_id : int
            The ID of the source in any catalogue.
        maxrad : float
            The maximum distance from the profile centroid where the
            contribution of the profile will be calculated. Default is to
            calculate it for the full field of view.
        osradius : float
            The radius from the profile centroid within which the spaxels
            will be supersampled to improve the accuracy of the calculation.
            Default is no supersampling.
        osfactor : int
           The supersampling factor. Each spaxel within 'osradius' will be
           supersampled into 'osfactor' times 'osfactor' sub-spaxels.
        **kwargs
            Any other keyword provided to this method will be parsed to the
            'get_f' method of the current super-class.
        """
        self._own_fwhm = None  # can be set to an instance-specific FWHM

        # call super-class
        super(Gaussian, self).__init__(uc=uc, vc=vc, mag=mag, src_id=src_id, maxrad=maxrad, osradius=osradius,
                                       osfactor=osfactor, **kwargs)

        self._diff_fwhm = None  # place-holder for partial differential wrt. the FWHM
        self.profile_properties.append("_diff_fwhm")

    # Define the 'fwhm' property to set and return the FWHM of the profile ########
    def get_fwhm(self):
        """
        Return the FWHM of the profile. The routine will use the
        specific value defined for the instance if available and otherwise
        fall back to the one of the class.

        Returns
        -------
        fwhm : float
            The FWHM of the profile.
        """
        if self._own_fwhm is None:
            return self._fwhm
        else:
            return self._own_fwhm

    def set_own_fwhm(self, value):
        """
        Set the FWHM for this instance.

        Parameters
        ----------
        value : float
            The FWHM of the profile.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'fwhm' must be a float, not {0:s}".format(
            type(value).__name__)
        assert value > 0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))
        if value != self._own_fwhm:
            self._own_fwhm = value

            # all properties that use the FWHM need to  be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    def del_own_fwhm(self):
        """
        Delete the value of the FWHM for this instance.
        """
        if self._own_fwhm is None:
            return
        self._own_fwhm = None

        # all properties that use the FWHM need to  be recalculated
        self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    fwhm = property(get_fwhm, set_own_fwhm, del_own_fwhm, "The FWHM of the profile.")

    @property
    def std(self):
        """
        Returns
        -------
        std : float
            The standard deviation of the profile, measured in spaxels.
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
            sub-spaxels before they are returned.            
        **kwargs
            Any other keyword provided to this method will be parsed to the
            'get_f' method of the current super-class.

        Returns:
        --------
        f : nd_array
            The intensity distribution for the spaxels used in the calculation.
            If parameter 'integrate' is set, the intensities are returned only
            for the parents spaxels after averaging over the sub-spaxels.
        """
        if self._f is None:
            self._f = 10 ** (-0.4 * self.mag) / (2. * np.pi * self.std ** 2 * (1. - self.e)) * np.exp(
                -self.r ** 2 / (2. * self.std ** 2)) + super(Gaussian, self).get_f(integrate=False)
        if integrate:
            return self.sum_os(self._f)
        else:
            return self._f

    f = property(get_f, None, None, "The intensity distribution corresponding to the current PSF.")

    # In the following, the properties are provided to calculate the partial derivatives of the intensity distribution
    # provided by the Gaussian profile wrt. the individual parameters.

    # The FWHM
    def get_diff_fwhm(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        FWHM.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_fwhm : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_fwhm is None:
            self._diff_fwhm = self.get_f(integrate=False) * (self.r ** 2 / self.std ** 3 - 2. / self.std) / np.sqrt(
                8. * np.log(2.))
        if integrate:
            return self.sum_os(self._diff_fwhm)
        else:
            return self._diff_fwhm

    diff_fwhm = property(get_diff_fwhm, None, None, "The partial derivative of the PSF wrt. the FWHM.")

    # The ellipticity 'e'
    def get_diff_e(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        ellipticity 'e'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_e : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_e is None:
            self._diff_e = self.get_f(integrate=False) * (
                1. / (1. - self.e) - (self.r / (self.std ** 2)) * super(Gaussian, self).get_diff_e(integrate=False))
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    diff_e = property(get_diff_e, None, None, "The partial derivative of the PSF wrt. the ellipticity.")

    # The position angle 'theta'
    def get_diff_theta(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        position angle 'theta'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_theta : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_theta is None:
            self._diff_theta = -self.get_f(integrate=False) * self.r / (self.std ** 2) \
                * super(Gaussian, self).get_diff_theta(integrate=False)
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    diff_theta = property(get_diff_theta, None, None, "The partial derivative of the PSF wrt. the pos. angle 'theta'.")

    # The centroid coordinates
    def get_diff_xc(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. xc, the
        centroid of the profile in x-direction.
        
        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.
            
        Returns
        -------
        diff_xc : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_xc is None:
            self._diff_xc = -self.get_f(integrate=False) * self.r / (self.std ** 2) \
                * super(Gaussian, self).get_diff_xc(integrate=False)
        if integrate:
            return self.sum_os(self._diff_xc)
        else:
            return self._diff_xc

    diff_xc = property(get_diff_xc, None, None,
                       "The partial derivative of the Gaussian  PSF wrt. xc, the x-centroid of the profile.")

    def get_diff_yc(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. yc, the
        centroid of the profile in y-direction.
        
        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.
            
        Returns
        -------
        diff_yc : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_yc is None:
            self._diff_yc = -self.get_f(integrate=False) * self.r / (self.std ** 2) * super(Gaussian, self).get_diff_yc(
                integrate=False)
        if integrate:
            return self.sum_os(self._diff_yc)
        else:
            return self._diff_yc

    diff_yc = property(get_diff_yc, None, None,
                       "The partial derivative of the Gaussian  PSF wrt. yc, the y-centroid of the profile.")

    # The parameters of the coordinate transformation
    def get_diff_A(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        parameter 'A' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_A : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_A is None:
            self._diff_A = -self.get_f(integrate=False) * self.r / (self.std ** 2) \
                * super(Gaussian, self).get_diff_A(integrate=False)
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A

    diff_A = property(get_diff_A, None, None, "The partial derivative of the PSF wrt. 'A'.")

    def get_diff_B(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        parameter 'B' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_B : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_B is None:
            self._diff_B = -self.get_f(integrate=False) * self.r / self.std ** 2 * super(Gaussian, self).get_diff_B(
                integrate=False)
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    diff_B = property(get_diff_B, None, None, "The partial derivative of the PSF wrt. 'B'.")

    def get_diff_C(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        parameter 'C' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_C : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_C is None:
            self._diff_C = -self.get_f(integrate=False) * self.r / (self.std ** 2) \
                * super(Gaussian, self).get_diff_C(integrate=False)
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    diff_C = property(get_diff_C, None, None, "The partial derivative of the PSF wrt. 'C'.")

    def get_diff_D(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        parameter 'D' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_B : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_D is None:
            self._diff_D = -self.get_f(integrate=False) * self.r / self.std ** 2 * super(Gaussian, self).get_diff_D(
                integrate=False)
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    diff_D = property(get_diff_D, None, None, "The partial derivative of the PSF wrt. 'D'.")

    def get_diff_x0(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        parameter 'x0' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_x0 : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """

        if self._diff_x0 is None:
            self._diff_x0 = -self.get_f(integrate=False) * self.r / self.std ** 2 * super(Gaussian, self).get_diff_x0(
                integrate=False)
        if integrate:
            return self.sum_os(self._diff_x0)
        else:
            return self._diff_x0

    diff_x0 = property(get_diff_x0, None, None, "The partial derivative of the PSF wrt. 'x0'.")

    def get_diff_y0(self, integrate=True):
        """
        Calculate the partial derivative of the Gaussian profile wrt. the
        parameter 'y0' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_y0 : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_y0 is None:
            self._diff_y0 = -self.get_f(integrate=False) * self.r / self.std ** 2 * super(Gaussian, self).get_diff_y0(
                integrate=False)
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0

    diff_y0 = property(get_diff_y0, None, None, "The partial derivative of the PSF wrt. 'y0'.")


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

    _beta = 2.
    _fwhm = 3.
    function = "moffat"

    @classmethod
    def set_beta(cls, value):
        """
        Set the parameter 'beta' for all instances of the class. Calling this
        method will overwrite any instance-specific values of 'beta' that
        might be present.

        Parameters
        ----------
        value : float
            The 'beta' parameter
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'beta' must be a float, not {0:s}".format(
            type(value).__name__)

        if value <= 1 or value > 10: logging.error("'beta' must be >1 and <=10 but is {0:.1f}.".format(float(value)))
        assert 1 < value <= 10, "'beta' must be >1 and <=10 but is {0:.1f}".format(float(value))

        if value != cls._beta:
            cls._beta = value

            for inst in cls._registry:
                inst._own_beta = None  # reset possible instance-specific value

                # all properties that depend on 'beta' need to be recalculated.
                inst.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    @classmethod
    def set_fwhm(cls, value):
        """
        Set the FWHM for all instances of the class. Calling this method will
        overwrite any instance-specific values of the FWHM that might be
        present.

        Parameters
        ----------
        value : float
            The FWHM, given in spaxels
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'fwhm' must be a float, not {0:s}".format(
            type(value).__name__)
        assert value > 0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))
        if value != cls._fwhm:
            cls._fwhm = value

            for inst in cls._registry:
                inst._own_fwhm = None  # reset possible instance-specific value

                # all properties that depend on 'beta' need to be recalculated.
                inst.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    def __init__(self, uc, vc, mag, src_id=0, maxrad=np.inf, osradius=0, osfactor=1, **kwargs):
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
        src_id : int
            The ID of the source in any catalogue.
        maxrad : float
            The maximum distance from the profile centroid where the
            contribution of the profile will be calculated. Default is to
            calculate it for the full field of view.
        osradius : float
            The radius from the profile centroid within which the spaxels
            will be supersampled to improve the accuracy of the calculation.
            Default is no supersampling.
        osfactor : int
           The supersampling factor. Each spaxel within 'osradius' will be
           supersampled into 'osfactor' times 'osfactor' sub-spaxels.
        **kwargs
            Any other keyword provided to this method will be parsed to the
            'get_f' method of the current super-class.
        """
        super(Moffat, self).__init__(uc=uc, vc=vc, mag=mag, src_id=src_id, maxrad=maxrad, osradius=osradius,
                                     osfactor=osfactor, **kwargs)

        self._own_beta = None  # can be set to an instance-specififc 'beta'
        self._own_fwhm = None  # can be set to an instance-specififc FWHM

        self._diff_beta = None  # place-holder for partial derivatives wrt. beta
        self._diff_fwhm = None  # and the FWHM

        # add new partial derivatives to list of parameter-dependent properties
        self.profile_properties.extend(["_diff_beta", "_diff_fwhm"])

    # Define the 'beta' property to set and return the 'beta' parameter of the profile
    def get_beta(self):
        """
        Return the 'beta' of the profile. The routine will use the
        specific value defined for the instance if available and otherwise
        fall back to the one of the class.

        Returns
        -------
        beta : float
            The 'beta' parameter of the profile.
        """
        if self._own_beta is None:
            return self._beta
        else:
            return self._own_beta

    def set_own_beta(self, value):
        """
        Set the 'beta' parameter for this instance.

        Parameters
        ----------
        value : float
            The 'beta' parameter of the profile.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'beta' must be a float, not {0:s}".format(
            type(value).__name__)

        if value <= 1 or value > 10:
            logging.error("'beta' must be >1 and <=10 but is {0:.1f}.".format(float(value)))
        assert 1 < value <= 10, "'beta' must be >1 and <=10 but is {0:.1f}.".format(float(value))

        if value != self._own_beta:
            self._own_beta = value

            # all arrays that depend on 'beta' must be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    def del_own_beta(self):
        """
        Delete the 'beta' parameter set for this instance.
        """
        if self._own_beta is None: return
        self._own_beta = None

        # all arrays that depend on 'beta' must be recalculated
        self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    beta = property(get_beta, set_own_beta, del_own_beta, "The 'beta' parameter of the profile.")

    # Define the 'fwhm' property to set and return the FWHM of the profile
    def get_fwhm(self):
        """
        Return the FWHM of the profile. The routine will use the
        specific value defined for the instance if available and otherwise
        fall back to the one of the class.

        Returns
        -------
        fwhm : float
            The FWHM of the profile.
        """
        if self._own_fwhm is None:
            return self._fwhm
        else:
            return self._own_fwhm

    def set_own_fwhm(self, value):
        """
        Set the FWHM for this instance.

        Parameters
        ----------
        value : float
            The FWHM of the profile.
        """
        assert isinstance(value, (float, np.float32, np.float64)), "'fwhm' must be a float, not {0:s}".format(
            type(value).__name__)
        assert value > 0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))
        if value != self._own_fwhm:
            self._own_fwhm = value

            # all arrays that depend on 'beta' must be recalculated
            self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    def del_own_fwhm(self):
        """
        Delete the FWHM set for this instance.
        """
        if self._own_fwhm is None:
            return
        self._own_fwhm = None

        # all arrays that depend on 'beta' must be recalculated
        self.reset(omit=["_used", "_oversampled", "_osmap", "_ossummer", "_xos", "_yos", "_u", "_v", "_r"])

    fwhm = property(get_fwhm, set_own_fwhm, del_own_fwhm, "The FWHM of the profile.")

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
            sub-spaxels before they are returned.            
        **kwargs
            Any other keyword provided to this method will be parsed to the
            'get_f' method of the current super-class.

        Returns:
        --------
        f : nd_array
            The intensity distribution for the spaxels used in the calculation.
            If parameter 'integrate' is set, the intensities are returned only
            for the parents spaxels after averaging over the sub-spaxels.
        """
        if self._f is None:
            self._f = 10 ** (-0.4 * self.mag) * (self.beta - 1.) / (np.pi * (self.r_d ** 2) * (1. - self.e)) * (1. + (
                self.r / self.r_d) ** 2) ** (-self.beta) + super(Moffat, self).get_f(integrate=False)
        if integrate:
            return self.sum_os(self._f)
        else:
            return self._f

    f = property(get_f, None, None, "The intensity distribution of the current PSF profile.")

    # In the following, the properties are provided to calculate the partial derivatives of the intensity distribution
    # provided by the Moffat profile wrt. the individual parameters.

    # The 'beta' parameter
    def get_diff_beta(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        kurtosis parameter 'beta'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_beta : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
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

    diff_beta = property(get_diff_beta, None, None, "The partial derivative of the PSF wrt. 'beta'.")

    # The FWHM
    def get_diff_fwhm(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the FWHM.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_fwhm : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_fwhm is None:
            self._diff_fwhm = self.get_f(integrate=False) / (self.r_d * np.sqrt(2. ** (1. / self.beta) - 1.)) * (
                self.beta * (self.r / self.r_d) ** 2 / (1. + (self.r / self.r_d) ** 2) - 1.)
        if integrate:
            return self.sum_os(self._diff_fwhm)
        else:
            return self._diff_fwhm

    diff_fwhm = property(get_diff_fwhm, None, None, "The partial derivative of the PSF wrt. the FWHM.")

    # The ellipticity 'e'
    def get_diff_e(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        ellipticity 'e'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_e : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_e is None:
            self._diff_e = self.get_f(integrate=False) * (
                1. / (1. - self.e) - 2. * self.beta * self.r / ((self.r_d ** 2) * (1. + (self.r / self.r_d) ** 2))
                * super(Moffat, self).get_diff_e(integrate=False))
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    diff_e = property(get_diff_e, None, None, "The partial derivative of the PSF wrt. the ellipticity 'e'.")

    # The position angle 'theta'
    def get_diff_theta(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        position angle 'theta'.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_theta : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_theta is None:
            self._diff_theta = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_theta(integrate=False)
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    diff_theta = property(get_diff_theta, None, None, "The partial derivative of the PSF wrt. the pos. angle 'theta'.")

    # The centroid coordinates
    def get_diff_xc(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. xc, the
        centroid of the profile in x-direction.
        
        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.
            
        Returns
        -------
        diff_xc : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_xc is None:
            self._diff_xc = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_xc(integrate=False)
        if integrate:
            return self.sum_os(self._diff_xc)
        else:
            return self._diff_xc

    diff_xc = property(get_diff_xc, None, None, "The partial derivative of the Moffat PSF wrt. xc, the x-centroid.")

    def get_diff_yc(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. yc, the
        centroid of the profile in y-direction.
        
        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.
            
        Returns
        -------
        diff_yc : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_yc is None:
            self._diff_yc = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_yc(integrate=False)
        if integrate:
            return self.sum_os(self._diff_yc)
        else:
            return self._diff_yc

    diff_yc = property(get_diff_yc, None, None, "The partial derivative of the Moffat PSF wrt. yc, the y-centroid.")

    # The parameters of the coordinate transformation
    def get_diff_A(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'A' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_A : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_A is None:
            self._diff_A = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_A(integrate=False)
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A

    diff_A = property(get_diff_A, None, None, "The partial derivative of the PSF wrt. 'A'.")

    def get_diff_B(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'B' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_B : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_B is None:
            self._diff_B = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_B(integrate=False)
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    diff_B = property(get_diff_B, None, None, "The partial derivative of the PSF wrt. 'B'.")

    def get_diff_C(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'C' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_C : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_C is None:
            self._diff_C = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_C(integrate=False)
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    diff_C = property(get_diff_C, None, None, "The partial derivative of the PSF wrt. 'C'.")

    def get_diff_D(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'B' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_D : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_D is None:
            self._diff_D = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_D(integrate=False)
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    diff_B = property(get_diff_D, None, None, "The partial derivative of the PSF wrt. 'D'.")

    def get_diff_x0(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'x0' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_x0 : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_x0 is None:
            self._diff_x0 = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_x0(integrate=False)
        if integrate:
            return self.sum_os(self._diff_x0)
        else:
            return self._diff_x0

    diff_x0 = property(get_diff_x0, None, None, "The partial derivative of the PSF wrt. 'x0'.")

    def get_diff_y0(self, integrate=True):
        """
        Calculate the partial derivative of the Moffat profile wrt. the
        parameter 'y0' of the coordinate transformation.

        Parameters
        ----------
        integrate : bool, optional
            Whether or not the values should be averaged over the individual
            sub-spaxels before they are returned.

        Returns
        -------
        diff_y0 : nd_array
            The partial derivative, either for all (sub-)spaxels or after
            averaging over the sub-spaxels.
        """
        if self._diff_y0 is None:
            self._diff_y0 = self.get_f(integrate=False) * -2. * self.beta * self.r / (
                self.r_d ** 2 * (1. + (self.r / self.r_d) ** 2)) * super(Moffat, self).get_diff_y0(integrate=False)
        if integrate:
            return self.sum_os(self._diff_y0)
        else:
            return self._diff_y0

    diff_y0 = property(get_diff_y0, None, None, "The partial derivative of the PSF wrt. 'y0'.")
