"""
PROFILES
========
The classes defined in this script provide the functionality to create a PSF
profile, i.e. the intensity distribution as a function of position, for a
given set of shape parameters and source coordinates. Besides the actual
distribution of intensities, the partial derivatives with respect to the
individual parameters are also incldued. They can be used to prepare
the Jacobi matrix that can be provided to the Levenberg-Marquard scheme of
scipy.optimize.
All parameters that influence the shape and the position of a profile can
be updated on the fly. The classes are designed in such a way that only the
properties that depend on the updates paramaters must be recalculated
afterwards. This allows for an efficient usage of the classes during the
analysis.
After initialization, the instances of each class are iterable, i.e. one can
loop over all instances of a class via
>>> for psf in Profile:
>>>  print psf.xc, psf.yc  ## returns central coordinates of each profile

CHANGELOG:
----------
2013/03/21, v0.1.0.0, initial version completed. Some documentation still
                      missing.
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.1.0.0"
import logging
import numpy as np


## class that defines an iterator over all its instances
## following suggestions at: http://stackoverflow.com/questions/739882/iterating-over-object-instances-of-a-given-class-in-python
class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)



## define a property that is shared by all instances of a class
## following suggestions at: http://stackoverflow.com/questions/5189699/how-can-i-make-a-class-property-in-python?
class classproperty(object):
    def __init__(self, getter):
        self.getter= getter
    def __get__(self, instance, owner):
        return self.getter(owner)



class Profile(object):
    """
    This class does not describe an actual profile but serves as a parent
    class for all following classes. It provides the basic properties that
    calculate the centroid distance of each spaxel or its partial derivatives
    with respect to the various parameters and that are inherited by the child
    classes of 'Profile'.
    """
    
    ## allow for iteration over instances
    __metaclass__ = IterRegistry
    _registry = []

    ## Initialization of basic properties
    _size = None  ## The spatial size of the array

    _spaxel = None  ## The index of each spaxel
    _transform = None  ## The transformation from spaxel to x/y-coordinates

    _e     = 0  ## The ellipticity of the profile
    _theta = 0  ## The position angle of the profile

    _A = 1  ## The various parameters that specify the coordinate 
    _B = 0  ## transformation; Note that 'Profile' supports two ways of
    _C = 0  ## defining the source coordinates: either directly or via
    _D = 0  ## a transformation from reference coordinates (see below).
 

    @classmethod
    def set_size(cls,value):

        assert isinstance(value, (tuple, list)), \
            "'size' should be a tuple, not {0}".format(type(value).__name__)

        assert len(value) == 2, "Array must be 2D, not {0:d}D".format(\
            len(value))
        
        if value != cls._size:
            cls._spaxel = np.arange(value[0]*value[1], dtype=np.int16)

            _x = np.resize(np.arange(value[0], dtype=np.float64),
                               (value[0]*value[1],))
            _y = np.arange(value[0]*value[1], dtype=np.float64)//value[0]
            
            cls._transform = np.vstack((_y, _x)).T

            cls._size = value

            for inst in cls._registry:
                for prop in inst.profile_properties:
                    inst.__dict__[prop] = None


    @classproperty
    def size(cls):
        return cls._size



    @classmethod
    def set_transform(cls, value):
        
        if value.ndim != 2:
            logging.critical("'transform' must be 2dim., is {0}dim."\
                                 .format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."

        if value.shape[1] != 2:
            logging.critical(\
                "'transform' must contain 2 spatial coordinates, is {0}."\
                    .format(value.shape[1]))
        assert value.shape[1]==2, \
            "'transform' must contain 2 spatial coordinates."
        
        cls._transform = np.asarray(value, dtype=np.float64)
        cls._size = (value.shape[0],)
        cls._spaxel = np.arange(value.shape[0], dtype=np.int16)

        for inst in cls._registry:
            for prop in inst.profile_properties:
                inst.__dict__[prop] = None


    @classproperty
    def transform(cls):
        return cls._transform


    @classproperty
    def x(cls):
        return cls._transform[:,1]
        

    @classproperty
    def y(cls):
        return cls._transform[:,0]
        

    @classmethod
    def set_theta(cls, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
            "'theta' must be a float, not {0:s}".format(type(value).__name__)

        if value != cls._theta:
            cls._theta = value
        
            for inst in cls._registry:
                inst._own_theta = None
                for prop in inst.profile_properties:
                    if not prop in ["_used", "_oversampled", "_osmap", 
                                    "_ossummer", "_xos", "_yos"]:
                        inst.__dict__[prop] = None


    @classmethod
    def set_e(cls, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
            "'e' must be a float, not {0:s}".format(type(value).__name__)
        assert value>=-1e-3 and value<1, \
            "'e' must be in interval [0,1) but is {0:.1f}".format(float(value))

        if value != cls._e:
            cls._e = value

            for inst in cls._registry:
                inst._own_e = None
                for prop in inst.profile_properties:
                    if not prop in ["_used", "_oversampled", "_osmap", 
                                    "_ossummer", "_xos", "_yos", "_u", "_v"]:
                        inst.__dict__[prop] = None


    @classmethod
    def set_A(cls,value):

        assert isinstance(value, (float, int, np.float32, np.float64)), \
            "'A' must be a float or int, not {0:s}".format(type(value).__name__)

        if value !=  cls._A:
            cls._A = value
            
            for inst in cls._registry:
                inst._xc = None
                inst._yc = None
                for prop in inst.profile_properties:
                    inst.__dict__[prop] = None

    
    @classproperty
    def A(cls):
        return cls._A


    @classmethod
    def set_B(cls, value):
        assert isinstance(value, (float, int, np.float32, np.float64)), \
            "'B' must be a float or int, not {0:s}".format(type(value).__name__)

        if value !=  cls._B:
            cls._B = value

            for inst in cls._registry:
                inst._xc = None
                inst._yc = None
                for prop in inst.profile_properties:
                    inst.__dict__[prop] = None

    
    @classproperty
    def B(cls):
        return cls._B


    @classmethod
    def set_C(cls, value):
        assert isinstance(value, (float, int, np.float32, np.float64)), \
            "'C' must be a float or int, not {0:s}".format(type(value).__name__)

        if value !=  cls._C:
            cls._C = value

            for inst in cls._registry:
                inst._xc = None
                inst._yc = None
                for prop in inst.profile_properties:
                    inst.__dict__[prop] = None

    
    @classproperty
    def C(cls):
        return cls._C


    @classmethod
    def set_D(cls, value):
        assert isinstance(value, (float, int, np.float32, np.float64)), \
            "'D' must be a float or int, not {0:s}".format(type(value).__name__)

        if value !=  cls._D:
            cls._D = value

            for inst in cls._registry:
                inst._xc = None
                inst._yc = None
                for prop in inst.profile_properties:
                    inst.__dict__[prop] = None

    
    @classproperty
    def D(cls):
        return cls._D


    def __init__(self, Xc, Yc, mag, src_id=0, maxrad=np.inf, osradius=0,
                 osfactor=1, **kwargs):

        if self.transform == None:
            logging.error("No (x,y)-coordinate system for spaxels defined.")

        assert self.transform is not None, \
            "Specify size first or provide x- & y-coordinates of each spaxel."

        self._registry.append(self)

        self.Xc = Xc
        self.Yc = Yc
        self._mag = mag
        
        self.src_id = src_id

        self.maxrad = maxrad
        self._used  = None

        self._xc = None
        self._yc = None
        
        self._own_e = None
        self._own_theta = None

        self._osmap = None
        self._ossummer = None

        self._xos = None
        self._yos = None

        self._u = None
        self._v = None

        self._r = None
        self._f = None
        
        self._diff_e = None
        self._diff_theta = None
        self._diff_A = None
        self._diff_B = None
        self._diff_C = None
        self._diff_D = None

        self.profile_properties = ["_used", "_oversampled", "_osmap",
                                   "_ossummer", "_xos", "_yos", "_u", "_v",
                                   "_r", "_f", "_diff_e", "_diff_theta",
                                   "_diff_A", "_diff_B", "_diff_C", "_diff_D"]

        ## Properly implement possibility to oversample central spaxels
        self.osr   = osradius
        if not self.osr > 0:
            self.osf = 1
            self.osxgrid = None
            self.osygrid = None
        else:
            self.osf = max(1, osfactor)
            osgrid = np.linspace(\
                -0.5+1./(2*self.osf), 0.5-1./(2*self.osf), self.osf)
            self.osxgrid = np.resize(osgrid,(self.osf*self.osf,))
            self.osygrid = np.sort(self.osxgrid)
        self._oversampled = None

##         self._seeing = seeing



    def set_mag(self, value):
        assert isinstance(value, (float, np.float32,  np.float64)), \
            "'mag' must be a float, not {0:s}".format(type(value).__name__)

        if value != self._mag:
            self._mag = value

            for prop in self.profile_properties:
                    if not prop in ["_used", "_oversampled", "_osmap",
                                    "_ossummer", "_xos", "yos",
                                    "_u", "_v", "_r"]:
                        self.__dict__[prop] = None


    def get_mag(self):
        return self._mag


    mag = property(get_mag, set_mag, None,
                   "The broadband magnitude of the profile star.")

        
    def get_spaxel(self):
        
        return self.__class__._spaxel

    spaxel = property(get_spaxel, None, None, "The spatial pixel array.")


    def get_xc(self):
        if self._xc == None:
            return self.A*self.Xc + self.B*self.Yc + self.C
        else:
            return self._xc


    def set_xc(self, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
           "'xc' must be a float, not {0:s}".format(type(value).__name__)
        self._xc = value

        for prop in self.profile_properties:
            self.__dict__[prop] = None


    def del_xc(self):
        self._xc = None

        for prop in self.profile_properties:
            self.__dict__[prop] = None



    xc = property(get_xc, set_xc, del_xc,
                  "The centroid of the profile along the x-axis.")


    def get_yc(self):
        if self._yc == None:
            return self.A*self.Yc - self.B*self.Xc + self.D
        else:
            return self._yc


    def set_yc(self, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
           "'yc' must be a float, not {0:s}".format(type(value).__name__)
        self._yc = value

        for prop in self.profile_properties:
            self.__dict__[prop] = None



    def del_yc(self):
        self._yc = None

        for prop in self.profile_properties:
            self.__dict__[prop] = None


    yc = property(get_yc, set_yc, del_yc,
                  "The centroid of the profile along the y-axis.")


    def get_theta(self):
        if self._own_theta == None:
            return self._theta
        else:
            return self._own_theta


    def set_own_theta(self, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
           "'theta' must be a float, not {0:s}".format(type(value).__name__)

        self._own_theta = value

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos"]:
                self.__dict__[prop] = None


    def del_own_theta(self):
        self._own_theta = None

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos"]:
                self.__dict__[prop] = None


        
    theta = property(get_theta, set_own_theta, del_own_theta,
                     "The position angle theta of the PSF.")


    def get_e(self):
        if self._own_e == None:
            return self._e
        else:
            return self._own_e


    def set_own_e(self, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
            "'e' must be a float, not {0:s}".format(type(value).__name__)
        assert value>=0 and value<1, \
           "'e' must be in interval [0,1), but is {0:.1f}".format(float(value))

        self._own_e = value

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos", "_u", "_v"]:
                self.__dict__[prop] = None

        
    def del_own_e(self):
        self._own_e = None

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos", "_u", "_v"]:
                self.__dict__[prop] = None


        
    e = property(get_e, set_own_e, del_own_e,
                 "The ellipticity 'e' of the PSF.")


    @property
    def used(self):
        if self._used == None:
            if self.maxrad == np.inf:
                self._used = np.ones((self.transform.shape[0],), dtype=np.bool)
            else:
                self._used = np.sqrt(\
                    (self.x-self.xc)**2+(self.y-self.yc)**2)<=self.maxrad
        return self._used


    @property
    def Nused(self):
        return self.used.sum()

    
    @property
    def oversampled(self):
        if self._oversampled == None:
            self._oversampled = np.sqrt(\
                (self.x-self.xc)**2 + (self.y-self.yc)**2)<=self.osr
        return self._oversampled

    
    @property
    def norm(self):
        norm = np.asarray(self.used, dtype=np.float64)
        norm[self.oversampled] *= self.osf**2
        return norm
    
    @property
    def Nos(self):
        return self.oversampled.sum()


    @property
    def osmap(self):
        if self._osmap == None:
            self._osmap = np.hstack((\
                    self.spaxel[self.used],
                    np.resize(self.spaxel[self.oversampled],
                              ((self.osf*self.osf-1)*self.Nos,))))
            self._osmap.sort()
        return self._osmap


    @property
    def ossummer(self):
        if self._ossummer == None:
            next_spaxel = np.flatnonzero(\
                np.diff(np.r_[self.osmap[0]-1, self.osmap, self.osmap[-1]+1]))
            self._ossummer = np.vstack((next_spaxel[:-1],next_spaxel[1:])).T
        return self._ossummer


    @property
    def xos(self):
        if self._xos == None:
            self._xos = self.x[self.osmap]
            if self.osf>1:
                self._xos[np.in1d(self.osmap,np.flatnonzero(self.oversampled))]\
                    += np.resize(self.osxgrid, (self.Nos*self.osxgrid.size,))
        return self._xos
    

    @property
    def yos(self):
        if self._yos == None:
            self._yos = self.y[self.osmap]
            if self.osf>1:
                self._yos[np.in1d(self.osmap,np.flatnonzero(self.oversampled))]\
                    += np.resize(self.osygrid, (self.Nos*self.osygrid.size,))
        return self._yos


    @property
    def u(self):
        if self._u == None:
            self._u = (self.xos - self.xc)*np.sin(self.theta) \
                -(self.yos - self.yc)*np.cos(self.theta)
        return self._u


    @property
    def v(self):
        if self._v == None:
            self._v = (self.yos - self.yc)*np.sin(self.theta) \
                +(self.xos - self.xc)*np.cos(self.theta)
        return self._v


    @property
    def r(self):
        if self._r == None:
            self._r = np.sqrt(self.u**2 + self.v**2/(1-self.e)**2)
            self._r[self._r==0] = 1e-5
        return self._r


    def get_f(self, **kwargs):
        """
        Method that calculates the actual intensity profile. Note that this
        method is mainly a place-holder for the `make_array`-methods of the
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

        assert not hasattr(super(Profile, self), 'make_array')
        ## make sure no `make_array`-method is "hidden" behind this one.
        ## (following recommondation from:
        ## http://rhettinger.wordpress.com/2011/05/26/super-considered-super/)

        return 0

    f = property(get_f, None, None, 
                 "A place-holder for the intensity distribution of the PSF.")

    
    def get_diff_e(self, integrate=True):
        if self._diff_e == None:
            self._diff_e = self.v**2/(self.r*(1.-self.e)**3)
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    diff_e = property(\
        get_diff_e, None, None,
        "The partial derivative of the profile wrt. the ellipticity 'e'.")



    def get_diff_theta(self, integrate=True):
        if self._diff_theta == None:
            self._diff_theta = self.u*self.v*(1.-1./(1.-self.e)**2)/self.r
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    diff_theta = property(\
        get_diff_theta, None, None,
        "The partial derivative of the profile wrt. the pos. angle 'theta'.")
    


    def get_diff_A(self, integrate=True):
        if self._diff_A == None:
            self._diff_A \
                = -(self.u*(self.Xc*np.sin(self.theta)\
                                -self.Yc*np.cos(self.theta)) \
                        + self.v/(1.-self.e)**2\
                        *(self.Yc*np.sin(self.theta) \
                              + self.Xc*np.cos(self.theta)))/self.r
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A
        
    diff_A = property(get_diff_A, None, None,
                      "The partial derivative of the profile wrt. 'A'.")

    
    def get_diff_B(self, integrate=True):
        if self._diff_B == None:
            self._diff_B \
                = -(self.u*(self.Yc*np.sin(self.theta) \
                                + self.Xc*np.cos(self.theta)) \
                        + self.v/(1.-self.e)**2\
                        *(-self.Xc*np.sin(self.theta)/ \
                               + self.Yc*np.cos(self.theta)))/self.r
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    diff_B = property(get_diff_B, None, None,
                      "The partial derivative of the profile wrt. 'B'.")



    def get_diff_C(self, integrate=True):
        if self._diff_C == None:            
            self._diff_C \
                = -(self.u*np.sin(self.theta) \
                        + self.v/(1.-self.e)**2*np.cos(self.theta))/self.r
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    diff_C = property(get_diff_C, None, None,
                      "The partial derivative of the profile wrt. 'C'.")


    def get_diff_D(self, integrate=True):
        if self._diff_D == None:
            self._diff_D \
                = (self.u*np.cos(self.theta) - \
                       self.v/(1.-self.e)**2*np.sin(self.theta))/self.r
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    diff_D = property(get_diff_D, None, None,
                      "The partial derivative of the profile wrt. 'D'.")


    def sum_os(self, data):
        """

        Following suggestion at: http://stackoverflow.com/questions/7471474/\
            numpy-sum-of-values-in-subarrays-between-pairs-of-indices
        """
        if self.osf<=1:
            return data
        else:
            tmp_data = np.r_[0, data.cumsum()][self.ossummer]
            return (tmp_data[:,1]-tmp_data[:,0])/self.norm[self.used]
##             return (self.ossummer * data).sum(axis=1)/self.ossummer.sum(axis=1)

    
    def get_flux(self):

        return (self.spaxel[self.used], self.f)



class Gaussian(Profile):


    _fwhm = 3.


    @classmethod
    def set_fwhm(cls, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
            "'fwhm' must be a float, not {0:s}".format(type(value).__name__)
        assert value>0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))

        if value != cls._fwhm:
            cls._fwhm = value
            
            for inst in cls._registry:
                inst._own_fwhm = None
                for prop in inst.profile_properties:
                    if not prop in ["_used", "_oversampled", "_osmap",
                                    "_ossummer", "_xos", "_yos", "_u", "_v",
                                    "_r"]:
                        inst.__dict__[prop] = None



    def __init__(self, **kwargs):

        self._own_fwhm = None

        super(Gaussian, self).__init__(**kwargs)

        self._diff_fwhm = None
        self.profile_properties.append("_diff_fwhm")



    def get_fwhm(self):
        if self._own_fwhm == None:
            return self._fwhm
        else:
            return self._own_fwhm


    def set_own_fwhm(self, value):

        assert isinstance(value, (float, np.float32, np.float64)), \
            "'fwhm' must be a float, not {0:s}".format(type(value).__name__)
        assert value>0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))

        self._own_fwhm = value

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos", "_u", "_v", "_r"]:
                self.__dict__[prop] = None

        
    def del_own_fwhm(self):
        
        self._own_fwhm = None

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos", "_u", "_v", "_r"]:
                self.__dict__[prop] = None


    fwhm = property(get_fwhm, set_own_fwhm, del_own_fwhm,
                    "The FWHM of the profile.")

        
    ## The standard deviation
    @property
    def std(self):
        return self.fwhm / np.sqrt(8.*np.log(2.))



    def get_f(self, integrate=True):
        if self._f == None:
            self._f \
                = 10**(-0.4*self.mag)/(2.*np.pi*self.std**2*(1.-self.e))\
                *np.exp(-self.r**2/(2.*self.std**2))\
                +super(Gaussian, self).get_f(integrate=False)
        if integrate:
            return self.sum_os(self._f)
        else:
            return self._f
    
    f = property(get_f, None, None,
                 "The intensity distribution corresponding to the current PSF.")



    def get_diff_fwhm(self, integrate=True):
        if self._diff_fwhm == None:
            self._diff_fwhm \
                = self.get_f(integrate=False)*(self.r**2/self.std**3 \
                                                   -2./self.std)
        if integrate:
            return self.sum_os(self._diff_fwhm)
        else:
            return self._diff_fwhm
                 
    diff_fwhm = property(get_diff_fwhm, None, None,
                         "The partial derivative of the PSF wrt. the FWHM.")


    def get_diff_e(self, integrate=True):
        if self._diff_e == None:
            self._diff_e \
                = self.get_f(integrate=False)\
                *(1./(1.-self.e) - self.r/self.std**2\
                      *super(Gaussian, self).get_diff_e(integrate=False))
        if integrate:
            return self.sum_os(self._diff_fwhm)
        else:
            return self._diff_e
    
    diff_e = property(get_diff_e, None, None,
                      "The partial derivative of the PSF wrt. the ellipticity.")

    
    def get_diff_theta(self, integrate=True):
        if self._diff_theta == None:
            self._diff_theta = \
                -self.get_f(integrate=False)*self.r/self.std**2\
                *super(Gaussian, self).get_diff_theta(integrate=False)
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    diff_theta = property(\
        get_diff_theta, None, None,
        "The partial derivative of the PSF wrt. the pos. angle 'theta'.")


    def get_diff_A(self, integrate=True):
        if self._diff_A == None:
            self._diff_A = \
                -self.get_f(integrate=False)*self.r/self.std**2\
                *super(Gaussian, self).get_diff_A(integrate=False)
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A
    
    diff_A = property(get_diff_A, None, None,
                      "The partial derivative of the PSF wrt. 'A'.")


    def get_diff_B(self, integrate=True):
        if self._diff_B == None:
            self._diff_B = \
                -self.get_f(integrate=False)*self.r/self.std**2\
                *super(Gaussian, self).get_diff_B(integrate=False)
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    diff_B = property(get_diff_B, None, None,
                      "The partial derivative of the PSF wrt. 'B'.")


    def get_diff_C(self, integrate=True):

        if self._diff_C == None:
            self._diff_C = \
                -self.get_f(integrate=False)*self.r/self.std**2\
                *super(Gaussian, self).get_diff_C(integrate=False)
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    diff_C = property(get_diff_C, None, None,
                      "The partial derivative of the PSF wrt. 'C'.")

    
    def get_diff_D(self, integrate=True):
        if self._diff_D == None:
            self._diff_D = \
                -self.get_f(integrate=False)*self.r/self.std**2\
                *super(Gaussian, self).get_diff_D(integrate=False)
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    diff_D = property(get_diff_D, None, None,
                      "The partial derivative of the PSF wrt. 'D'.")



class Moffat(Profile):

    _beta = 2.
    _fwhm = 3.


    @classmethod
    def set_beta(cls, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
            "'beta' must be a float, not {0:s}".format(type(value).__name__)
        assert value>=1, "'beta' must be >=1 but is {0:.1f}".format(\
            float(value))

        if value != cls._beta:
            cls._beta = value
            
            for inst in cls._registry:
                inst._own_beta = None
                for prop in inst.profile_properties:
                    if not prop in ["_used", "_oversampled", "_osmap",
                                    "_ossummer", "_xos", "_yos",
                                    "_u", "_v", "_r"]:
                        inst.__dict__[prop] = None



    @classmethod
    def set_fwhm(cls, value):
        assert isinstance(value, (float, np.float32, np.float64)), \
            "'fwhm' must be a float, not {0:s}".format(type(value).__name__)
        assert value>0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))

        if value != cls._fwhm:
            cls._fwhm = value
            
            for inst in cls._registry:
                inst._own_fwhm = None
                for prop in inst.profile_properties:
                    if not prop in ["_used", "_oversampled", "_osmap",
                                    "_ossummer", "_xos", "_yos",
                                    "_u", "_v", "_r"]:
                        inst.__dict__[prop] = None


    def __init__(self, **kwargs):

        super(Moffat, self).__init__(**kwargs)

        self._own_beta = None
        self._own_fwhm = None

        self._diff_beta = None
        self._diff_fwhm = None
    
        self.profile_properties.extend(["_diff_beta", "_diff_fwhm"])


    def get_beta(self):
        if self._own_beta == None:
            return self._beta
        else:
            return self._own_beta


    def set_own_beta(self, value):

        assert isinstance(value, (float, np.float32, np.float64)), \
            "'beta' must be a float, not {0:s}".format(type(value).__name__)
        assert value>1, "'beta' must be >1 but is {0:.1f}".format(float(value))

        self._own_beta = value

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos", "_u", "_v", "_r"]:
                self.__dict__[prop] = None

        
    def del_own_beta(self):
        
        self._own_beta = None

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos", "_u", "_v", "_r"]:
                self.__dict__[prop] = None


    beta = property(get_beta, set_own_beta, del_own_beta,
                    "The beta-parameter of the profile.")



    def get_fwhm(self):
        if self._own_fwhm == None:
            return self._fwhm
        else:
            return self._own_fwhm


    def set_own_fwhm(self, value):

        assert isinstance(value, (float, np.float32, np.float64)), \
            "'fwhm' must be a float, not {0:s}".format(type(value).__name__)
        assert value>0, "'fwhm' must be >0 but is {0:.1f}".format(float(value))

        self._own_fwhm = value

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos", "_u", "_v", "_r"]:
                self.__dict__[prop] = None

        
    def del_own_fwhm(self):
        
        self._own_fwhm = None

        for prop in self.profile_properties:
            if not prop in ["_used", "_oversampled", "_osmap", "_ossummer",
                            "_xos", "_yos", "_u", "_v", "_r"]:
                self.__dict__[prop] = None


    fwhm = property(get_fwhm, set_own_fwhm, del_own_fwhm,
                    "The FWHM of the profile.")


    ## The effective radius
    @property
    def r_d(self):
        return self.fwhm /(2. *np.sqrt(2**(1. /self.beta) -1.))



    def get_f(self, integrate=True):
        if self._f == None:
            self._f \
                = 10**(-0.4*self.mag)\
                *(self.beta-1.)/(np.pi*(self.r_d**2)*(1.-self.e))\
                *(1. + (self.r/self.r_d)**2)**(-self.beta)\
                +super(Moffat, self).get_f(integrate=False)
        if integrate:
            return self.sum_os(self._f)
        else:
            return self._f

    f = property(get_f, None, None,
                 "The intensity distribution of the current PSF profile.")


    def get_diff_beta(self, integrate=True):
        if self._diff_beta == None:
            self._diff_beta \
                = self.get_f(integrate=False)\
                *(1./(self.beta-1.) \
                      - np.log(1.+(self.r/self.r_d)**2)\
                      + np.log(2.)/(self.beta*(1.-2.**(-1./self.beta)))\
                      *((self.r/self.r_d)**2/(1.+(self.r/self.r_d)**2)\
                          - 1./self.beta))
        if integrate:
            return self.sum_os(self._diff_beta)
        else:
            return self._diff_beta

    diff_beta = property(get_diff_beta, None, None,
                         "The partial derivative of the PSF wrt. 'beta'.")


    def get_diff_fwhm(self, integrate=True):
        if self._diff_fwhm == None:
            self._diff_fwhm \
                = self.get_f(integrate=False)\
                /(self.r_d*np.sqrt(2.**(1./self.beta)-1.))\
                *(self.beta*(self.r/self.r_d)**2/(1.+(self.r/self.r_d)**2) - 1.)
        if integrate:
            return self.sum_os(self._diff_fwhm)
        else:
            return self._diff_fwhm

    diff_fwhm = property(get_diff_fwhm, None, None,
                         "The partial derivative of the PSF wrt. the FWHM.")


    def get_diff_e(self, integrate=True):
        if self._diff_e == None:
            self._diff_e \
                = self.get_f(integrate=False)\
                *(1./(1.-self.e) - 2.*self.beta*self.r\
                      /(self.r_d**2*(1.+(self.r/self.r_d)**2))\
                      *super(Moffat, self).get_diff_e(integrate=False))
        if integrate:
            return self.sum_os(self._diff_e)
        else:
            return self._diff_e

    diff_e = property(\
        get_diff_e, None, None,
        "The partial derivative of the PSF wrt. the ellipticity 'e'.")


    def get_diff_theta(self, integrate=True):
        if self._diff_theta  == None:
            self._diff_theta \
                = self.get_f(integrate=False)\
                *-2.*self.beta*self.r/(self.r_d**2*(1.+(self.r/self.r_d)**2))\
                *super(Moffat, self).get_diff_theta(integrate=False)
        if integrate:
            return self.sum_os(self._diff_theta)
        else:
            return self._diff_theta

    diff_theta = property(\
        get_diff_theta, None, None,
        "The partial derivative of the PSF wrt. the pos. angle 'theta'.")


    def get_diff_A(self, integrate=True):
        if self._diff_A == None:
            self._diff_A \
                = self.get_f(integrate=False)\
                *-2.*self.beta*self.r/(self.r_d**2*(1.+(self.r/self.r_d)**2))\
                *super(Moffat, self).get_diff_A(integrate=False)
        if integrate:
            return self.sum_os(self._diff_A)
        else:
            return self._diff_A

    diff_A = property(get_diff_A, None, None,
                      "The partial derivative of the PSF wrt. 'A'.")
    

    def get_diff_B(self, integrate=True):
        if self._diff_B == None:
            self._diff_B \
                = self.get_f(integrate=False)\
                *-2.*self.beta*self.r/(self.r_d**2*(1.+(self.r/self.r_d)**2))\
                *super(Moffat, self).get_diff_B(integrate=False)
        if integrate:
            return self.sum_os(self._diff_B)
        else:
            return self._diff_B

    diff_B = property(get_diff_B, None, None,
                      "The partial derivative of the PSF wrt. 'B'.")


    def get_diff_C(self, integrate=True):
        if self._diff_C == None:
            self._diff_C \
                = self.get_f(integrate=False)\
                *-2.*self.beta*self.r/(self.r_d**2*(1.+(self.r/self.r_d)**2))\
                *super(Moffat, self).get_diff_C(integrate=False)
        if integrate:
            return self.sum_os(self._diff_C)
        else:
            return self._diff_C

    diff_C = property(get_diff_C, None, None,
                      "The partial derivative of the PSF wrt. 'C'.")


    def get_diff_D(self, integrate=True):
        if self._diff_D == None:
            self._diff_D \
                = self.get_f(integrate=False)\
                *-2.*self.beta*self.r/(self.r_d**2*(1.+(self.r/self.r_d)**2))\
                *super(Moffat, self).get_diff_D(integrate=False)
        if integrate:
            return self.sum_os(self._diff_D)
        else:
            return self._diff_D

    diff_D = property(get_diff_D, None, None,
                      "The partial derivative of the PSF wrt. 'D'.")


