"""
MY_PROFILES
============

PURPOSE:
--------
Several analytical profiles are provided that can be used to fill a 2dim.
array.

USAGE:
------
Each profile is represented by a class. When creating an instance of the
class, some basic shape parameters of the profile must be defined. The
function-call of each class can than be used to create and return profiles
with input- dependent brightness and center positions on an array of
provided size. Note that the magnitude zeropoint is 0.0, so a profile with
mag=0. will have an integrated flux of 1!
Optionally, oversampling properties can be defined when a new instance of
a class is initialized. This is necessary to properly model profiles that
have a sharply peaked core (e.g. a Sersic). In such cases, there can be
strong deviations between the integrated flux within a pixel and the flux
expected when only considering the distance approximating the flux within
a pixel at the distance of the pixel center (wrt. the center of the
profile).

EXAMPLE FOR USAGE:
------------------
>>> sersic = _my_profiles.Sersic(n=4, r_e=7.5, osfactor=8, osradius=3)
## create an instance of the Sersic-class
>>> sersic_data = sersic(size=(100,100), x_c=49.7, y_c=50.3, mag=-15.5)
## calculate and return a profile


CHANGELOG:
----------
v0.0.0.0 - 2009/04/24 (c) S. Kamann
v0.1.0.0 - 2009/04/27 - parameters are automatically converted to float
v0.2.0.0 - 2009/04/28 - added Gaussian Profile
                      - fixed bug in iterative determination of kappa
v0.2.1.0              - corrected sign error in u & v calculation
v0.3.0.0 - 2009/11/12 - replaced numarray with numpy
v0.4.0.0 - 2009/11/20 - changed way that arguments are passed to Moffat &
                        Gaussian:
                        x_c, y_c and mag are now passed directly to draw.
                        Thus one can use the same instance for several
                        profiles.
v0.4.0.1 - 2010/03/11 - added some documentation
v0.5.0.0 - 2010/06/16 - former versions lead to deviant fluxes at the
                        profilecenter. This is now circumvented by
                        temporarily supersampling the array in the central
                        region.
                        The way of calling one of the classes from a piece
                        of code has changed (see USAGE above).
v0.5.1.0 - 2010/06/18 - fixed a bug in Moffat-class (confused x- & y-coords)
                      - fixed problem that code crashes if center of profile
                        near edge of FoV
v0.5.2.0 - 2010/07/22 - so far bugfix for profiles near edge of FoV only
                        work for Sersic-class; now works for all classes
v0.6.0.0 - 2011/07/21 - Updated documentation
                      - added option to calculate PSF profile only in a
                        region around its center instead of the whole array
                        (only Moffat so far)
                      - there are now two possible formats for the `array`-
                        attribute: Either the actual array (default) as
                        before, or three a tuple of 1dim. arrays:
                        (data, (index0, index1)).
                        This second format is very convenient for creating
                        sparse matrices and saving memory (only Moffat so
                        far)
v0.7.0.0 - 2011/08/09 - now using function-calls to calculate and return
                        the actual profiles. Once an instance of a class is
                        initialized, profiles can be generated using
                        multiple function-calls.
                        ==> New package my_profiles.
                      - several updates and bugfixes wrt. sparse matrices &
                        calculation of profiles on subarrays.
                      - all classes now support sparse matrices
v0.7.1.0 - 2011/08/31 - fixed bug in class 'Gaussian': replaced 'size' with
                        '_size' when inserting oversampling-region.
v0.7.2.0 - 2011/09/01 - fixed bug in class 'Gaussian': oversampling-region was
                        inserted at wrong coordinates. 
v0.7.3.1 - 2011/09/05 - fixed 2 bugs in class 'Gaussian': calculating and
                        inserting of oversampling region was still buggy.
                        Also improved usage of provided os-region parameters.
v0.8.0.0 - 2011/09/09 - changed how shape parameters are deduced from input
                        in classes Moffat and Gaussian to allow modification
                        of the shape of the profile without creating a new
                        class-instance.
v0.8.0.1, 2011/10/13 - changed how required PampelMuse-packages are imported
v0.8.0.2, 2011/11/11 - updated calculation and inserting of oversampling region
                       for Moffat profile
v0.8.0.3, 2012/02/01 - replaced math with numpy. added logging module. Included
                       handler for beta<=1. in Moffat profile.
v0.8.1.3, 2012/05/02 - bugfix in class `Gaussian`: former versions tried to
                       modify attribute `std` in-place.
v0.9.0.0, 2012/06/19 - major update wrt. to structure of the defined classes:
                       All inherit from super-class `Profile`. Generic
                       properties  such as `e`, `theta` or the pixel-distance
                       array are defined in `Profile` so for each individual
                       PSF function only the specific shape parameters (e.g
                       beta for a `Moffat`) need to be specified.
                       Also added options to use a Lorentzian PSF-profile or a
                       combination of Moffat and Gauss.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.9.0.0"

import logging
import numpy as np
from scipy import special, sparse

## required PampelMuse-packages
## from pampelmuse.functions.ima_arith import paste_area



class Profile(object):

    def __init__(self, e=0, theta=0, osfactor=10, osradius=2, seeing=None,
                 **kwargs):
        """
        e : float, optional
            The ellipticity of the moffat profile.
        theta : float, optional
            The position angle of the Moffat profile; given in radian and
            counted from the y-axis (= first axis in 2dim. numpy arrays) in
            ccw-direction
        osfactor : integer, optional
            Each pixel within `osradius` is divided into `osfactor` times
            `osfactor` subpixels. For each subpixel, the integrated flux is
            calculated, the flux within each parent pixel then is the sum of
            the flux in the enclosed subpixels.
        osradius : float, optional
            The 'flux-calculation by oversampling' is performed for every
            pixel within a distance <=`osradius` towards the center of the
            profile.
        seeing : float, optional
            A measure for the FWHM of the profile, in spatial pixels. If none
            is provided, the routine will estimate one using the parameters
            of the chosen profile.
        """
        self.e     = abs(e)
        self.theta = theta
        self.osr   = osradius
        if not self.osr > 0:
            self.osf = 1
        else:
            self.osf = osfactor

        self._fwhm = seeing


    def get_fwhm(self):
        """
        Obtain the seeing FWHM
        """
        return self._fwhm

    def set_fwhm(self, value):
        """
        Specify the seeing FWHM

        Parameters:
        -----------
        value : float
            The FWHM in spatial pixels.
        """
        self._fwhm = value
    
    fwhm = property(get_fwhm, set_fwhm, None, "The seeing FWHM")



    def make_array(self, R, **kwargs):
        """
        Method that calculates the actual intensity profile. Note that this
        method is mainly a place-holder for the `make_array`-methods of the
        individual sub-classes.

        Parameters:
        -----------
        R : array_like
            The array of pixel distances with respect to the center of the
            profile.

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



    def __call__(self, size, x_c, y_c, mag, maxrad=np.inf,
                 return_sparse=False, as_sparse_matrix=True, **kwargs):
        """
        A function call is used to actually fill an array of given size. The
        centroid position and magnitude of the profile must be provided.
        If an oversampling-region is defined, the profile in it is
        calculated separately and inserted into the full array afterwards.

        Parameters
        ----------
        size : tuple
            The shape of the array on which the profile will be calculated.
            Note that `size` should be given as (y_size, x_size). 
        x_c : float
            The center of the profile in x-direction (i.e. second axis in
            a 2dim. numpy array)
        y_c : float
            The center of the profile in y-direction (i.e. first axis in
            a 2dim. numpy array)
        mag : float
            The integrated magnitude of the profile. mag=0. will result
            in a normalized profile.
        maxrad : None or float, optional
            To calculate the profile only out to a certain radius, set
            `maxrad` to the desired value. All other pixels will be set to
            zero. By default, the profile is calculated on the whole array.
            Note that for `e`>0, the profile will also be calculated within
            an ellipse. In this case, `maxrad` gives the semimajor axis of
            the ellipse.
        return_sparse : boolean, optional
            Flag indicating whether the full array (i.e. a dense matrix, the
            default) or only the non-zero values (i.e. a sparse matrix, see
            `as_sparse_matrix` for details) should be returned. 
        as_sparse_matrix : boolean, optional
           Only used if `return_sparse` is enabled. Flag indicates whether a
           sparse coo-matrix is returned or a tuple containing the
           (non-zero) data and the corresponding pixel coordinates.

        Returns
        -------
        array : nd_array, tuple or sparse coo-matrix
            Depending on the `return_sparse`- and `as_sparse_matrix`- flags,
            either a numpy array of shape `size`, a tuple containing the
            non-zero values and respective pixel coordinates or a sparse
            coo-matrix of same `size` is returned.
        """
        self.size     = size
        self.x_c      = x_c
        self.y_c      = y_c
        self.flux     = 10**(-0.4*mag)
        self.maxrad   = maxrad
        self.sparse   = return_sparse
        self.asmatrix = as_sparse_matrix

        ## select sub-array around profile center (if requested)
        if self.maxrad != None:
            xdef = abs(np.arange(self.size[1])-self.x_c)<=self.maxrad
            ydef = abs(np.arange(self.size[0])-self.y_c)<=self.maxrad

            xs = xdef.argmax()
            ys = ydef.argmax()

            self._size = (ydef.sum(), xdef.sum())

        else:
            xs = 0
            ys = 0

            self._size = self.size

        ## central coordinates on sub-array
        self._x_c  = self.x_c - xs
        self._y_c  = self.y_c - ys  

        Y, X = np.indices(self._size)
        sel = np.sqrt((X-self._x_c)**2+(Y-self._y_c)**2)<self.maxrad
        _X = X[sel]
        _Y = Y[sel]

        _U = (_X-self._x_c)*np.sin(self.theta)-(_Y-self._y_c)*np.cos(self.theta)
        _V = (_Y-self._y_c)*np.sin(self.theta)+(_X-self._x_c)*np.cos(self.theta)
        _R = np.sqrt(_U**2 + (_V/(1. - self.e))**2)

        self.array = self.make_array(_R, **kwargs)

        if self.osf != 1:
    
            ctrsel = np.sqrt((_Y-self._x_c)**2+(_X-self._x_c)**2)<self.osr
            
            if ctrsel.any():
                subPix = np.resize(\
                    np.linspace(-0.5+1./(2*self.osf),
                                 0.5-1./(2*self.osf),
                                 self.osf),
                (self.osf, self.osf))
                _Xos = _X[ctrsel][:,np.newaxis] + subPix.flatten()
                _Yos = _Y[ctrsel][:,np.newaxis] + subPix.T.flatten()

                _Uos = (_Xos-self._x_c)*np.sin(self.theta) \
                    - (_Yos-self._y_c)*np.cos(self.theta)
                _Vos=(_Yos-self._y_c)*np.sin(self.theta) \
                    + (_Xos-self._x_c)*np.cos(self.theta)
                _Ros= np.sqrt(_Uos**2 + (_Vos/(1. - self.e))**2)

                self.array[ctrsel] \
                    = (self.make_array(_Ros,**kwargs)/(self.osf**2)).sum(axis=1)


        if self.sparse:
            if self.asmatrix:
                return sparse.coo_matrix((self.array,(_Y+ys, _X+xs)),
                                         shape=self.size)
            else:
                return (self.array, (_Y+ys, _X+xs))
        else:
            return np.array(\
                sparse.coo_matrix((self.array, (_Y+ys, _X+xs)),
                                  shape=self.size)\
                    .todense())




"""The Moffat Profile"""
class Moffat(Profile):
    """
    The functional form of the Moffat profile is:

        M(x, y) = SIGMA0*(1. + (r(x,y)/RD)**2)**-BETA)

    with: r(x,y) = SQRT(u(x,y)**2 + (v(x,y)/(1.-E))**2)
          u(x,y) = (x-XC)*SIN(THETA)-(y-YC)*COS(THETA)
          v(x,y) = (y-YC)*SIN(THETA)+(x-XC)*COS(THETA)
          RD     = FWHM /(2.*SQRT(2**(1./BETA) -1.))
          SIGMA0 = (BETA-1.)*FLUX/(PI*(RD**2)*(1.-E))
          FLUX   = 10**(-0.4*MAG)

    The properties of the Moffat profile can be altered using the following
    set of parameters: E, THETA, FWHM, BETA (shape); XC, YC (center); MAG
    (brightness).
    Sizes are always assumed to be pixel-based (not arcsec, nm, ...)
    """
    def __init__(self, beta, Mfwhm=None, Mfrac=1, **kwargs):
        """
        Initialize an instance of the Moffat-class.

        Parameters:
        -----------
        beta : float
            The beta parameter of the Moffat profile.
        Mfwhm : float, optional
            The fwhm of the Moffat profile. For backwards compatibility, the
            parameter is optional if `kwargs` includes a key `fwhm`.
        Mfrac : float, optional.
            The fractional flux in the profile. A value !=1 makes only sense
            if the PSF is modeled from more than one profile.

        All other parameters provided as `kwargs` are passed on to the call
        of the `Profile` super-class.
        """
        self.beta  = beta

        if self.beta <= 1.:
            logging.error("Value of beta ({0:.2f}) out of valid range [>1]."\
                              .format(float(self.beta)))
            logging.error("Using beta=2.5 instead.")
            self.beta=2.5


        if Mfwhm != None:
            self.Mfwhm = Mfwhm 
        else:
            assert kwargs.has_key("fwhm"), "No FWHM for Moffat profile found."
            self.Mfwhm = kwargs["fwhm"]

        self.Mfrac = Mfrac
            
        super(Moffat, self).__init__(**kwargs)

        if super(Moffat, self).get_fwhm() == None:
            super(Moffat, self).set_fwhm(self.Mfwhm)



    ## effective radius
    @property
    def r_d(self): 
        return  self.Mfwhm /(2. *np.sqrt(2**(1. /self.beta) -1.))



    ## central surface brightness for mag=0 (i.e. flux=1)
    @property
    def Msigma_0(self):
        return self.Mfrac*(self.beta-1.)/(np.pi*(self.r_d**2)*(1.-self.e))
    


    def make_array(self, R, **kwargs):
        """
        Method that calculates the actual intensity profile of the defined
        Moffat-profile.

        Parameters:
        -----------
        R : array_like
            The array of pixel distances with respect to the center of the
            profile.

        Returns:
        --------
        result : array_like, same size as R
            The normalized intensity in each pixel according to the defined
            profile.
        """
        return super(Moffat, self).make_array(R, **kwargs) \
            + self.flux*self.Msigma_0*(1. + (R/self.r_d)**2)**(-self.beta)




"""The Sersic profile"""
class Sersic(Profile):
    """
    The functional form of the Moffat profile is:

          S(x, y) = SIGMA_E*EXP(-KAPPA*(r(x,y)/R_E)**(1/N)-1))

    with: r(x,y) = SQRT(u(x,y)**2 + (v(x,y)/(1.-E))**2)
          u(x,y) = (x-XC)*SIN(THETA)-(y-YC)*COS(THETA)
          v(x,y) = (y-YC)*SIN(THETA)+(x-XC)*COS(THETA)
          FLUX   = -2.5*LOG10(MAG)
          ... (see below)

    The formulae for Sersic profile taken from Graham & Driver (2005)
    (bibcode: 2005PASA...22..118G). The paper also demonstrates the way to
    calculate KAPPA and SIGMA_E that is used here.

    The properties of the Moffat profile can be altered using the following
    set of parameters: E, THETA, N, R_E (shape); XC, YC (center); MAG
    (brightness).
    Sizes are always assumed to be pixel-based (not arcsec, nm, ...)
    """
    def __init__(self, n, r_e, Sfrac=1, **kwargs):
        """
        Initialize an instance of the Sersic-class.

        Parameters:
        -----------
        n : float
            The Sersic index of the profile.
        r_e : float
            The effective radius of the Sersic profile.
        Sfrac : float, optional.
            The fractional flux in the profile. A value !=1 makes only sense
            if the PSF is modeled from more than one profile.

        All other parameters provided as `kwargs` are passed on to the call
        of the `Profile` super-class.
        """

        self.n     = n
        self.r_e   = r_e

        self.Sfrac = Sfrac

        super(Sersic, self).__init__(**kwargs)

        if super(Sersic, self).get_fwhm() == None:
            super(Sersic, self).set_fwhm(2.355*self.r_e)


    @property
    def Ssigma_e(self):
        return self.Sfrac/(2*np.pi*(self.r_e**2)*np.exp(self.kappa)*self.n\
                               *(self.kappa)**(-2.*self.n)\
                               *special.gamma(2.*self.n)*(1.-self.e))


    @property
    def kappa(self):
        init = 1.9992*self.n - 0.3271
        return self.__newton_it(init)
        


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
        df = (np.exp(-x) * x**(2.*self.n - 1.))/special.gamma(2.*self.n)
        return (f, df)



    def make_array(self, R, **kwargs):
        """
        Method that calculates the actual intensity profile of the defined
        Sersic-profile.

        Parameters:
        -----------
        R : array_like
            The array of pixel distances with respect to the center of the
            profile.

        Returns:
        --------
        result : array_like, same size as R
            The normalized intensity in each pixel according to the defined
            profile.
        """
        return super(Sersic, self).make_array(R, **kwargs) \
            + self.flux*self.Ssigma_e \
            *np.exp(-self.kappa*((R/self.r_e)**(1/self.n)-1))




"""The Gaussian Profile"""
class Gaussian(Profile):
    """
    The functional form of the Gaussian profile is:

        G(x, y) = SIGMA0*EXP(-(r(x,y)**2)/(2.*(STD**2)))

    with: r(x,y) = SQRT(u(x,y)**2 + (v(x,y)/(1.-E))**2)
          u(x,y) = (x-XC)*SIN(THETA)-(y-YC)*COS(THETA)
          v(x,y) = (y-YC)*SIN(THETA)+(x-XC)*COS(THETA)
          SIGMA0 = FLUX/(2.*PI*(STD**2)*(1.-E))
          STD    = FWHM/SQRT(8.*LN(2.))
          FLUX   = -2.5*LOG10(MAG)

    The properties of the Gaussian profile can be altered using the
    following set of parameters:
     E, THETA, FWHM (shape); XC, YC (center); MAG (brightness).
    Sizes are always assumed to be pixel-based (not arcsec, nm, ...)
    """
    def __init__(self, Gfwhm=None, Gfrac=1, **kwargs):
        """
        Initialize an instance of the Gaussian-class.

        Parameters:
        -----------
        Gfwhm : float, optional
            The FWHM of the Gaussian profile. For backwards compatibility, the
            parameter is optional if `kwargs` includes a key `fwhm`.
        Gfrac : float, optional.
            The fractional flux in the profile. A value !=1 makes only sense
            if the PSF is modeled from more than one profile.

        All other parameters provided as `kwargs` are passed on to the call
        of the `Profile` super-class.
        """
        if Gfwhm != None:
            self.Gfwhm  = Gfwhm
        else:
            assert kwargs.has_key("fwhm"), "No FWHM for Gaussian profile found."
            self.Gfwhm  = kwargs["fwhm"]

        self.Gfrac = Gfrac

        super(Gaussian, self).__init__(**kwargs)

        if super(Gaussian, self).get_fwhm() == None:
            super(Gaussian, self).set_fwhm(self.Gfwhm)



    ## The standard deviation
    @property
    def std(self):
        return self.Gfwhm / np.sqrt(8.*np.log(2.))
            
   
 
    ## The central surface brightness for mag=0 (i.e. flux=1)
    @property
    def Gsigma_0(self):
        return self.Gfrac/(2.*np.pi*(self.std**2)*(1.-self.e))



    def make_array(self, R, **kwargs):
        """
        Method that calculates the actual intensity profile of the defined
        Gaussian-profile.

        Parameters:
        -----------
        R : array_like
            The array of pixel distances with respect to the center of the
            profile.

        Returns:
        --------
        result : array_like, same size as R
            The normalized intensity in each pixel according to the defined
            profile.
        """
        return super(Gaussian, self).make_array(R, **kwargs) \
            + self.flux*self.Gsigma_0*np.exp(-(R**2)/(2.*(self.std**2)))

        
    def Dgfwhm(self):
        



"""The Lorentz Profile"""
class Lorentz(Profile):
    """
    The functional form of the Lorentz profile is:

        L(x, y) = SIGMA0/(r(x, y)**2 + (FWHM/2)**2)

    with: r(x,y) = SQRT(u(x,y)**2 + (v(x,y)/(1.-E))**2)
          u(x,y) = (x-XC)*SIN(THETA)-(y-YC)*COS(THETA)
          v(x,y) = (y-YC)*SIN(THETA)+(x-XC)*COS(THETA)
          SIGMA0 = FLUX/(PI*(1.-E)*LN(1+(2.*RMAX/FWHM)**2)
          FLUX   = -2.5*LOG10(MAG)

    The properties of the Lorentz profile can be altered using the
    following set of parameters:
     E, THETA, FWHM (shape); XC, YC (center); MAG (brightness).
    Sizes are always assumed to be pixel-based (not arcsec, nm, ...)
    
    NOTE: The 2dim. integral over the Lorentz function diverges. For this
          reason, a maximum radius RMAX must always be defined.
          If high spectrophotometric accuracy is required, the usage of a
          Lorentzian is not recommended.
    """
    def __init__(self, Lfwhm=None, Lfrac=1, rmax=100, **kwargs):
        """
        Initialize an instance of the Gaussian-class.

        Parameters:
        -----------
        Lfwhm : float, optional
            The FWHM of the Lorentz profile. For convenience, the
            parameter is optional if `kwargs` includes a key `fwhm`.
        Lfrac : float, optional.
            The fractional flux in the profile. A value !=1 makes only sense
            if the PSF is modeled from more than one profile.

        All other parameters provided as `kwargs` are passed on to the call
        of the `Profile` super-class.
        """
        if Lfwhm != None:
            self.Lfwhm = Lfwhm
        else:
            assert kwargs.has_key("fwhm"), "No FWHM for Lorentz profile found."
            self.Lfwhm = kwargs["fwhm"]

        if rmax == -1: self.rmax= 10.*self.Lfwhm
        else: self.rmax = rmax

        self.Lfrac =  Lfrac

        super(Lorentz, self).__init__(**kwargs)

        if super(Lorentz, self).get_fwhm() == None:
            super(Lorentz, self).set_fwhm(self.Lfwhm)


    @property
    def Lsigma_0(self):
        return self.Lfrac\
            /(np.pi*(1.-self.e)*np.log(1.+(2.*self.rmax/self.Lfwhm)**2))



    def make_array(self, R, **kwargs):
        """
        Method that calculates the actual intensity profile of the defined
        Lorentz-profile.

        Parameters:
        -----------
        R : array_like
            The array of pixel distances with respect to the center of the
            profile.

        Returns:
        --------
        result : array_like, same size as R
            The normalized intensity in each pixel according to the defined
            profile.
        """
        return super(Lorentz, self).make_array(R, **kwargs) \
            + self.flux*self.Lsigma_0/(R**2 + (self.Lfwhm/2.)**2)



"""The combined Moffat & Gaussian Profile"""
class GauMoff(Moffat, Gaussian):
    """
    Create a PSF that is a sum of a Moffat and Gaussian profile.
    Please consult the individual classes for more information on the
    functional forms of the profiles.

    A note about the inheritance of super-classes:
    `GauMoff` inherits first from `Moffat`, then from `Gaussian`. This means
    that `super`-calls are delegated first to `Moffat`, then to `Gaussian`
    and finally to `Profile` (and `object`). This assures that e.g both the
    `Moffat.make_array`- and Gaussian.make_array`-methods are called.

    """
    def __init__(self, Gfwhm, beta, Mfwhm, Mfrac, **kwargs):
        """
        Initialize an instance of the Moffat&Gaussian-class.

        Parameters:
        -----------
        Gfwhm : float
            The fwhm of the Gaussian profile
        beta : float
            The beta parameter of the Moffat profile.
        Mfwhm : float, optional
            The fwhm of the Moffat profile.
        Mfrac : float, optional.
            The fractional flux in the Moffat-profile. The fractional flux
            in the Gaussian profile will then be 1-`Mfrac`.

        All other parameters provided as `kwargs` are passed on to the call
        of the `Profile` super-class.
        """


        kwargs["Gfwhm"] = Gfwhm
        kwargs["Gfrac"] = 1.-Mfrac

        kwargs["beta"]  = beta
        kwargs["Mfwhm"] = Mfwhm
        kwargs["Mfrac"] = Mfrac

        super(GauMoff, self).__init__(**kwargs)



    def make_array(self, R, **kwargs):
        """
        Method that calculates the actual intensity profile of the combined
        Moffat- & Gaussian-profile.

        Parameters:
        -----------
        R : array_like
            The array of pixel distances with respect to the center of the
            profile.

        Returns:
        --------
        result : array_like, same size as R
            The normalized intensity in each pixel according to the defined
            profile.
        """
        return super(GauMoff, self).make_array(R, **kwargs)


