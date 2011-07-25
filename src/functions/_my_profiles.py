"""
_MY_PROFILES
============

PURPOSE:
--------
Several analytical profiles are provided that can be used to fill a 2dim.
array.

USAGE:
------
Each profile is represented by a class. When creating an instance of the
class, the desired array shape, the parameters of the profile must be
defined.
Note that the magnitude zeropoint is 0.0, so a profile with mag=0. will
have an integrated flux of 1!
Optionally, oversampling properties can be defined. This is necessary to
properly model profiles that have a sharply peaked core (e.g. a Sersic).
In such cases, there can be strong deviations between the integrated flux
within a pixel and the flux expected when only considering the distance
approximating the flux within a pixel at the distance of the pixel center
(wrt. the center of the profile).
The calculated array can afterwards be accessed via the attribute 'array'.

EXAMPLE FOR USAGE:
------------------
>>> sersic = _my_profiles.Sersic((100,100), 49.7, 50.3, -15.5, 4, 7.5)
## create an instance
>>> sersic_data = sersic.array
## grab calculated array


CHANGELOG:
----------
V1.0 - 2009/04/24 (c) S. Kamann
V1.1 - 2009/04/27 - parameters are automatically converted to float
V1.2 - 2009/04/28 - added Gaussian Profile
                  - fixed bug in iterative determination of kappa
V1.2.1            - corrected sign error in u & v calculation
V1.3.0 2009/11/12 - replaced numarray with numpy
V1.3.1 2009/11/20 - changed way that arguments are passed to Moffat &
                    Gaussian:
                     x_c, y_c and mag are now passed directly to draw. Thus
                     one can use the same instance for several profiles
V1.4 - 2010/03/11 - added some documentation
V2.0 - 2010/06/16 - former versions lead to deviant fluxes at the profile
                    center. This is now circumvented by temporaryly
                    supersampling the array in the central region.
                    The way of calling one of the classes from a piece of
                    code has changed (see USAGE above).
V2.1 - 2010/06/18 - fixed a bug in Moffat-class (confused x- & y-coords)
                  - fixed problem that code crashes if center of profile
                    near edge of FoV
V2.2 - 2010/07/22 - so far bugfix for profiles near edge of FoV only work
                    for Sersic-class; now works for all classes
V2.3 - 2011/07/21 - Updated documentation
                  - added option to calculate PSF profile only in a region
                    around its center instead of the whole array
                    (only Moffat so far)
                  - there are now two possible formats for the `array`-
                    attribute: Either the actual array (default) as before,
                    or three a tuple of 1dim. arrays:
                     (data, (index0, index1)).
                    This second format is very convenient for creating
                    sparse matrices and saving memory (only Moffat so far)
"""
__version__ = 2.3  ## last change: 2011/07/21
import numpy, math
from scipy import special, sparse
from ima_arith import paste_area



"""The Moffat Profile"""
class Moffat:
    """
    The functional form of the Moffat profile is:

        M(x, y) = SIGMA0*(1. + (r(x,y)/RD)**2)**-BETA)

    with: r(x,y) = SQRT(u(x,y)**2 + (v(x,y)/(1.-E))**2)
          u(x,y) = (x-XC)*SIN(THETA)-(y-YC)*COS(THETA)
          v(x,y) = (y-YC)*SIN(THETA)+(x-XC)*COS(THETA)
          RD     = FWHM /(2.*SQRT(2**(1./BETA) -1.))
          SIGMA0 = (BETA-1.)*FLUX/(PI*(RD**2)*(1.-E))
          FLUX   = -2.5*LOG10(MAG)

    The properties of the Moffat profile can be altered using the following
    set of parameters: E, THETA, FWHM, BETA (shape); XC, YC (center); MAG
    (brightness).
    Sizes are always assumed to be pixel-based (not arcsec, nm, ...)
    """


    def __init__(self, size, x_c, y_c, mag, beta, fwhm, e=0., theta=0.,
                 osfactor=10, osradius=2, maxrad=None, return_sparse=False):
        """
        Initialize an instance of the Moffat-class.

        Parameters:
        -----------
        size : tuple
            The shape of the array on which the profile will be calculated
        xc : float
            The center of the profile in x-direction (i.e. second axis in
            standard Python counting)
        yc : float
            The center of the profile in y-direction (i.e. first axis in
            standard Python counting)
        mag : float
            The integrated magnitude of the profile. mag=0. will result
            in a normalized profile.
        beta : float
            The beta parameter of the Moffat profile.
        fwhm : float
            The fwhm of the Moffat profile.
        e : float, optional
            The ellipticity of the moffat profile.
        theta : float, optional
            The position angle of the Moffat profile; given in radian and
            counted in ccw-direction. An position angle of 0deg means the
            semimajor-axis of the profile is alingned with the y-axis (i.e
            the first axis in standard Python counting.
        osfactor : integer, optional
            Each pixel within `osradius` is divided into `osfactor`x`osfactor`-
            subpixels. For each subpixel, the integrated flux is calculated,
            the flux within each parent pixel then is the sum of the flux in
            the enclosed subpixels.
        osradius : float, optional
            The 'flux-calculation by oversampling' is performed for every pixel
            within a distance <=`osradius` towards the center of the profile.
        maxrad : None or float, optional
            To calculate the profile only out to  a certain radius, set
            `maxrad` to the desired value. All other pixels will be set to
            zero. By default, the profile is calculated on the whole array.
        return_sparse : boolean, optional
            Flag indicating whether the array itself (default) or a tuple
            containing the (non-zero) data and the pixel coordinates should
            be stored. 
        """
        self.size  = size
        self.x_c   = x_c
        self.y_c   = y_c
        self.flux  = 10**(-0.4*mag)
        self.beta  = beta
        self.e     = e
        self.theta = theta
        self.osf   = osfactor
        self.osr   = osradius
        self.maxrad= maxrad
        self.sparse= return_sparse

        ## effective radius
        self.r_d  = fwhm /(2. *math.sqrt(2**(1. /beta) -1.))

        ## central surface brightness
        self.sigma_0 = (beta-1.)*self.flux/(math.pi*(self.r_d**2)*(1.-e))

        ## select sub-array around profile center (if requested)
        if self.maxrad != None:
            xdef = abs(numpy.arange(self.size[1])-self.x_c)<=self.maxrad
            ydef = abs(numpy.arange(self.size[0])-self.y_c)<=self.maxrad

            xs = xdef.argmax()
            ys = ydef.argmax()

            self._size = (ydef.sum(), xdef.sum())

        else:
            xs = 0
            ys = 0

            self._size = self.size

        ## central  coordinates on sub-array
        self._x_c  = self.x_c - xs
        self._y_c  = self.y_c - ys

        ##  calculate profile
        self._make_array()

        ## store requested output in `array`
        if (self.sparse or self.maxrad != None):
            Y, X = numpy.indices(self._size)

            profdata = (self._array.flatten(), (X.flatten()+ys, Y.flatten()+xs))

            if self.sparse:
                self.array = profdata
            else:
                self.array = numpy.array(sparse.coo_matrix(profdata).todense())

        else:
            self.array = self._array



    def _make_array(self):
        """
        Method that actually fills the array. It uses numpy's fromfunction-
        method to call the `_draw`-method. 
        If an oversampling-region is defined, the profile in it is calculated
        separately and inserted into the full array afterwards.
        """
        self._array =numpy.fromfunction(self._draw, self._size, dtype='float32')

        if self.osf != 1:
            csize = ((2*self.osr+1)*self.osf, (2*self.osr+1)*self.osf)

            x_n = int(round(self._x_c))
            y_n = int(round(self._y_c))

            self._x_c += (self.osr - round(self._x_c))
            self._y_c += (self.osr - round(self._y_c))

            self._x_c *= self.osf
            self._y_c *= self.osf

            self.x_c += 0.5*(self.osf-1.)
            self.y_c += 0.5*(self.osf-1.)

            self.r_d *= self.osf
            self.sigma_0 /= (self.osf**2)

            if self.maxrad != None:
                self.maxrad *= self.osf
            carray = numpy.fromfunction(self._draw, csize, dtype='float32')

            s1_size = (2*self.osr+1, (2*self.osr+1)*self.osf, self.osf)
            s2_size = (2*self.osr+1, 2*self.osr+1, self.osf)
            step1 = numpy.sum(numpy.reshape(carray, s1_size, 'C'), axis=2)
            step2 = numpy.sum(numpy.reshape(step1, s2_size, 'F'), axis=2)
        
            self._array = paste_area(step2, self._size, (self.osr, self.osr),
                                     (x_n, y_n), out_array=self._array)


    def _draw(self, y, x):
        """
        method to fill array with profile, called via numpy.fromfunction
        """
        u =(x-self._x_c)*math.sin(self.theta)-(y-self._y_c)*math.cos(self.theta)
        v =(y-self._y_c)*math.sin(self.theta)+(x-self._x_c)*math.cos(self.theta)

        r = numpy.sqrt(u**2 + (v/(1. - self.e))**2)
        
        return numpy.where(r>self.maxrad, 
                           0, self.sigma_0*(1. + (r/self.r_d)**2)**(-self.beta))



"""
The Sersic Profile
Formulae for Sersic profile taken from Graham & Driver (2005)
bibcode: 2005PASA...22..118G
"""
class Sersic:
    def __init__(self, size, x_c, y_c, mag, n, r_e, e=0., theta=0., osfactor=10,
                 osradius=2):

        self.size  = size
        self.x_c   = x_c
        self.y_c   = y_c
        self.n     = n
        self.r_e   = r_e
        self.e     = e
        self.theta = theta
        self.osf   = osfactor
        self.osr   = osradius


        flux = 10**(-0.4*mag)

        self._get_kappa()
        self.sigma_e = flux/(2*math.pi*(r_e**2)*math.exp(self.kappa)*n*
                             (self.kappa)**(-2.*n)*special.gamma(2.*n)*(1.-e))

        self._make_array()
        

    def _get_kappa(self):
        init = 1.9992*self.n - 0.3271
        self.kappa = self.__newton_it(init)
        

    def __newton_it(self, x0, epsilon=1e-8):
        for i in range(2000):
            x0 -= self.__gammainc(x0)[0]/self.__gammainc(x0)[1]
            if abs(self.__gammainc(x0)[0]) <= epsilon:
                break
            if i == 1999:
                print 'Warning: Iteration failed!'
        return x0

                    
    def __gammainc(self, x):
        f  = special.gammainc(2*self.n, x) - 0.5
        df = (math.exp(-x) * x**(2.*self.n - 1.))/special.gamma(2.*self.n)
        return (f, df)


    def _make_array(self):
        
        self.array = numpy.fromfunction(self._draw, self.size, dtype='float32')

        if self.osf != 1:
            csize = ((2*self.osr+1)*self.osf, (2*self.osr+1)*self.osf)

            x_n = int(round(self.x_c))
            y_n = int(round(self.y_c))

            self.x_c += (self.osr - round(self.x_c))
            self.y_c += (self.osr - round(self.y_c))

            self.x_c *= self.osf
            self.y_c *= self.osf

            self.x_c += 0.5*(self.osf-1.)
            self.y_c += 0.5*(self.osf-1.)

            self.r_e *= self.osf
            self.sigma_e /= (self.osf)**2

            carray = numpy.fromfunction(self._draw, csize, dtype='float32')
            
            s1_size = (2*self.osr+1, (2*self.osr+1)*self.osf, self.osf)
            s2_size = (2*self.osr+1, 2*self.osr+1, self.osf)
            step1 = numpy.sum(numpy.reshape(carray, s1_size, 'C'), axis=2)
            step2 = numpy.sum(numpy.reshape(step1, s2_size, 'F'), axis=2)
        
            self.array = paste_area(step2, self.size, (self.osr, self.osr),
                                    (x_n, y_n), out_array=self.array)


    def _draw(self, y, x):

        u = (x-self.x_c)*math.sin(self.theta)-(y-self.y_c)*math.cos(self.theta)
        v = (y-self.y_c)*math.sin(self.theta)+(x-self.x_c)*math.cos(self.theta)

        r = numpy.sqrt(u**2 + (v/(1. - self.e))**2)

        return self.sigma_e*numpy.exp(-self.kappa*((r/self.r_e)**(1/self.n)-1))



"""The Gaussian Profile"""
class Gaussian:

    def __init__(self, size, x_c, y_c, mag, fwhm, e=0., theta=0., osfactor=10,
                 osradius=2):
        self.size  = size
        self.x_c   = x_c
        self.y_c   = y_c
        self.flux  = 10**(-0.4*mag) 
        self.e     = e
        self.theta = theta
        self.osf   = osfactor
        self.osr   = osradius

        self.std = fwhm / math.sqrt(8.*math.log(2.))
                
        self.sigma_0 = self.flux/(2.*math.pi*(self.std**2)*(1.-e))

        self._make_array()


    def _make_array(self):
        
        self.array = numpy.fromfunction(self._draw, self.size, dtype='float32')

        if self.osf != 1:
            csize = ((2*self.osr+1)*self.osf, (2*self.osr+1)*self.osf)

            x_n = int(round(self.x_c))
            y_n = int(round(self.y_c))

            self.x_c += (self.osr - round(self.x_c))
            self.y_c += (self.osr - round(self.y_c))

            self.x_c *= self.osf
            self.y_c *= self.osf

            self.x_c += 0.5*(self.osf-1.)
            self.y_c += 0.5*(self.osf-1.)

            self.std *= self.osf
            self.sigma_0 /= (self.osf)**2

            carray = numpy.fromfunction(self._draw, csize, dtype='float32')
            
            s1_size = (2*self.osr+1, (2*self.osr+1)*self.osf, self.osf)
            s2_size = (2*self.osr+1, 2*self.osr+1, self.osf)
            step1 = numpy.sum(numpy.reshape(carray, s1_size, 'C'), axis=2)
            step2 = numpy.sum(numpy.reshape(step1, s2_size, 'F'), axis=2)
        
            self.array = paste_area(step2, self.size, (self.osr, self.osr),
                                    (x_n, y_n), out_array=self.array)


    def _draw(self, y, x):
        
        u = (x-self.x_c)*math.sin(self.theta)-(y-self.y_c)*math.cos(self.theta)
        v = (y-self.y_c)*math.sin(self.theta)+(x-self.x_c)*math.cos(self.theta)

        r = numpy.sqrt(u**2 + (v/(1. - self.e))**2)

        return self.sigma_0*numpy.exp(-(r**2)/(2.*(self.std**2)))
