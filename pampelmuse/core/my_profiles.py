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
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.8.1.3"

import logging
import numpy as np
from scipy import special, sparse

## required PampelMuse-packages
from pampelmuse.functions.ima_arith import paste_area



"""The Moffat Profile"""
class Moffat(object):
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
    def __init__(self, beta, fwhm, e=0., theta=0., osfactor=10, osradius=2):
        """
        Initialize an instance of the Moffat-class.

        Parameters:
        -----------
        beta : float
            The beta parameter of the Moffat profile.
        fwhm : float
            The fwhm of the Moffat profile.
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
        """
        self.beta  = beta
        self.fwhm  = fwhm 
        self.e     = e
        self.theta = theta
        self.osr   = osradius
        if not self.osr > 0:
            self.osf = 1
        else:
            self.osf = osfactor

        if self.beta <= 1.:
            logging.error("Value of beta ({0:.2f}) out of valid range [>1]."\
                              .format(float(self.beta)))
            logging.error("Using beta=2.5 instead.")
            self.beta=2.5


    ## effective radius
    @property
    def r_d(self): 
        return  self.fwhm /(2. *np.sqrt(2**(1. /self.beta) -1.))


    ## central surface brightness for mag=0 (i.e. flux=1)
    @property
    def _sigma_0(self):
        return (self.beta-1.)/(np.pi*(self.r_d**2)*(1.-self.e))



    def __call__(self, size, x_c, y_c, mag, maxrad=np.inf,
                 return_sparse=False, as_sparse_matrix=True):
        """
        A function call is used to actually fill an array of given size.
        The centroid position and magnitude of the profile must be
        provided.
        Numpy's fromfunction-method is used to call the `_draw`-method. 
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
        self.size  = size
        self.x_c   = x_c
        self.y_c   = y_c
        self.flux  = 10**(-0.4*mag)
        self.maxrad = maxrad
        self.sparse = return_sparse
        self.asmatrix = as_sparse_matrix

        self.sigma_0 = self.flux*self._sigma_0

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

        self.array = self.sigma_0*(1. + (_R/self.r_d)**2)**(-self.beta)

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
                    = (self.sigma_0/(self.osf**2)\
                           *(1. + (_Ros/self.r_d)**2)**(-self.beta))\
                    .sum(axis=1)

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

 
##         ##  calculate profile
##         self.array =np.fromfunction(self._draw, self._size, dtype='float32')
##         print time.time() - t0
## 
##         if self.osf != 1: ## take care of oversampling region
##             csize = ((2*self.osr+1)*self.osf, (2*self.osr+1)*self.osf)
## 
##             x_n = int(round(self._x_c))
##             y_n = int(round(self._y_c))
## 
##             self._x_c += (self.osr - x_n)
##             self._y_c += (self.osr - y_n)
## 
##             self._x_c *= self.osf
##             self._y_c *= self.osf
## 
##             self._x_c += 0.5*(self.osf-1.)
##             self._y_c += 0.5*(self.osf-1.)
## 
##             ## modify shape parameters
##             self.r_d *= self.osf
##             self.sigma_0 /= (self.osf**2)
## 
##             self.maxrad = (self.osr+np.sqrt(0.5))*self.osf
## 
##             ## calculate profile
##             carray = np.fromfunction(self._draw, csize, dtype='float32')
## 
##             ## undo changes to shape parameters
##             self.r_d /= self.osf
##             self.sigma_0 *= (self.osf**2)
## 
##             self.maxrad = maxrad
## 
##             ## resample and insert
##             s1_size = (2*self.osr+1, (2*self.osr+1)*self.osf, self.osf)
##             s2_size = (2*self.osr+1, 2*self.osr+1, self.osf)
##             step1 = np.sum(np.reshape(carray, s1_size, 'C'), axis=2)
##             step2 = np.sum(np.reshape(step1, s2_size, 'F'), axis=2)
## 
##             Y, X = np.indices(step2.shape)
##             SEL = (np.sqrt((X-self._x_c)**2 + (Y-self._y_c)**2) <= self.osr)\
##                 &((X-self.osr+x_n)>=0)&((X-self.osr+x_n)<self._size[1])\
##                 &((Y-self.osr+y_n)>=0)&((Y-self.osr+y_n)<self._size[0])
## 
##             if SEL.any():
##                 self.array[Y[SEL]+y_n-self.osr, X[SEL]+x_n-self.osr] \
##                     = step2[Y[SEL], X[SEL]]
## 
## ##             self.array = paste_area(step2, self._size, (self.osr, self.osr),
## ##                                     (x_n, y_n), out_array=self.array)
## 
##         ## return requested output
##         if (self.sparse or self.maxrad != np.inf):
##             Y, X = np.indices(self._size)
## 
##             _valid   = (self.array.flatten()!=0.)
##             profdata = (self.array.flatten()[_valid], (Y.flatten()[_valid]+ys,
##                                                        X.flatten()[_valid]+xs))
## 
##             if self.sparse:
##                 if self.asmatrix:
##                     return sparse.coo_matrix(profdata, shape=self.size)
##                 else:
##                     return profdata
##             else:
##                 return np.array(sparse.coo_matrix(\
##                         profdata, shape=self.size).todense())
## 
##         else:
##             return self.array



    def _draw(self, y, x): ## changed x & y to "correct" numpy axis-swapping 
        """
        method to fill array with profile, called via np.fromfunction
        """
        u =(x-self._x_c)*np.sin(self.theta)-(y-self._y_c)*np.cos(self.theta)
        v =(y-self._y_c)*np.sin(self.theta)+(x-self._x_c)*np.cos(self.theta)

        r = np.sqrt(u**2 + (v/(1. - self.e))**2)

        return np.where(r>self.maxrad, 
                        0, self.sigma_0*(1. + (r/self.r_d)**2)**(-self.beta))



"""The Sersic profile"""
class Sersic(object):
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
    def __init__(self, n, r_e, e=0., theta=0., osfactor=10, osradius=2):
        """
        Initialize an instance of the Sersic-class.

        Parameters:
        -----------
        n : float
            The Sersic index of the profile.
        r_e : float
            The effective radius of the Sersic profile.
        e : float, optional
            The ellipticity of the Sersic profile.
        theta : float, optional
            The position angle of the Sersic profile; given in radian and
            counted from the y-axis (= first axis in 2dim. numpy arrays) in
            ccw-direction
        osfactor : integer, optional
            Each pixel within `osradius` is divided into `osfactor`x`osfactor`-
            subpixels. For each subpixel, the integrated flux is calculated,
            the flux within each parent pixel then is the sum of the flux in
            the enclosed subpixels.
        osradius : float, optional
            The 'flux-calculation by oversampling' is performed for every pixel
            within a distance <=`osradius` towards the center of the profile.
        """

        self.n     = n
        self.r_e   = r_e
        self.e     = e
        self.theta = theta
        self.osr   = osradius
        if not self.osr > 0:
            self.osf = 1
        else:
            self.osf = osfactor

        self._get_kappa()
        self._sigma_e = 1./(2*np.pi*(r_e**2)*np.exp(self.kappa)*n*
                            (self.kappa)**(-2.*n)*special.gamma(2.*n)*(1.-e))

        

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
        df = (np.exp(-x) * x**(2.*self.n - 1.))/special.gamma(2.*self.n)
        return (f, df)



    def __call__(self, size, x_c, y_c, mag, maxrad=np.inf,
                 return_sparse=False, as_sparse_matrix=True):
        """
        A function call is used to actually fill an array of given size. The
        centroid position and magnitude of the profile must be provided.
        Numpy's fromfunction-method is used to call the `_draw`-method. 
        If an oversampling-region is defined, the profile in it is calculated
        separately and inserted into the full array afterwards.

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
           sparse coo-matrix is returned or a tuple containing the (non-zero)
           data and the corresponding pixel coordinates.

        Returns
        -------
        array : nd_array, tuple or sparse coo-matrix
            Depending on the `return_sparse`- and `as_sparse_matrix`- flags,
            either a numpy array of shape `size`, a tuple containing the
            non-zero values and respective pixel coordinates or a sparse
            coo-matrix of same `size` is returned.
        """

        self.size  = size
        self.x_c   = x_c
        self.y_c   = y_c
        self.flux  = 10**(-0.4*mag)

        self.sigma_e = self.flux*self._sigma_e

        self.maxrad = maxrad
        self.sparse = return_sparse
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

        ##  calculate profile
        self.array =np.fromfunction(self._draw, self._size, dtype='float32')

        if self.osf != 1: ## take care of oversampling region
            csize = ((2*self.osr+1)*self.osf, (2*self.osr+1)*self.osf)

            x_n = int(round(self._x_c))
            y_n = int(round(self._y_c))

            self._x_c += (self.osr - x_n)
            self._y_c += (self.osr - y_n)

            self._x_c *= self.osf
            self._y_c *= self.osf

            self._x_c += 0.5*(self.osf-1.)
            self._y_c += 0.5*(self.osf-1.)

            ## modify shape parameters
            self.r_e *= self.osf
            self.sigma_e /= (self.osf)**2
            self.maxrad *= self.osf

            ## calculate profile
            carray = np.fromfunction(self._draw, csize, dtype='float32')

            ## undo changes to shape parameters
            self.r_e /= self.osf
            self.sigma_e *= (self.osf**2)
            self.maxrad /= self.osf

            ## resample and insert
            s1_size = (2*self.osr+1, (2*self.osr+1)*self.osf, self.osf)
            s2_size = (2*self.osr+1, 2*self.osr+1, self.osf)
            step1 = np.sum(np.reshape(carray, s1_size, 'C'), axis=2)
            step2 = np.sum(np.reshape(step1, s2_size, 'F'), axis=2)

            self.array = paste_area(step2, self._size, (self.osr, self.osr),
                                    (x_n, y_n), out_array=self.array)


        ## return requested output
        if (self.sparse or self.maxrad != np.inf):
            Y, X = np.indices(self._size)

            _valid   = (self.array.flatten()!=0.)
            profdata = (self.array.flatten()[_valid], (Y.flatten()[_valid]+ys,
                                                       X.flatten()[_valid]+xs))

            if self.sparse:
                if self.asmatrix:
                    return sparse.coo_matrix(profdata, shape=self.size)
                else:
                    return profdata
            else:
                return np.array(sparse.coo_matrix(\
                        profdata, shape=self.size).todense())

        else:
            return self.array



    def _draw(self, y, x):
        """
        method to fill array with profile, called via numpy.fromfunction
        """
        u =(x-self._x_c)*np.sin(self.theta)-(y-self._y_c)*np.cos(self.theta)
        v =(y-self._y_c)*np.sin(self.theta)+(x-self._x_c)*np.cos(self.theta)

        r = np.sqrt(u**2 + (v/(1. - self.e))**2)

        return np.where(\
            r>self.maxrad, 
            0,
            self.sigma_e*np.exp(-self.kappa*((r/self.r_e)**(1/self.n)-1)))



"""The Gaussian Profile"""
class Gaussian(object):
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
    def __init__(self, fwhm, e=0., theta=0., osfactor=10, osradius=2):
        """
        Initialize an instance of the Sersic-class.

        Parameters:
        -----------
        fwhm : float
            The FWHM of the Gaussian profile.
        e : float, optional
            The ellipticity of the Gaussian profile.
        theta : float, optional
            The position angle of the Gaussian profile; given in radian and
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
        """
        self.fwhm  = fwhm
        self.e     = e
        self.theta = theta
        self.osr   = osradius
        if not self.osr > 0:
            self.osf = 1
        else:
            self.osf = osfactor


    ## The standard deviation
    @property
    def std(self):
        return self.fwhm / np.sqrt(8.*np.log(2.))
               
 
    ## The central surface brightness for mag=0 (i.e. flux=1)
    @property
    def _sigma_0(self):
        return 1./(2.*np.pi*(self.std**2)*(1.-self.e))



    def __call__(self, size, x_c, y_c, mag, maxrad=np.inf,
                 return_sparse=False, as_sparse_matrix=True):
        """
        A function call is used to actually fill an array of given size. The
        centroid position and magnitude of the profile must be provided.
        Numpy's fromfunction-method is used to call the `_draw`-method. 
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
            Flag indicating whether the full array (i.e. a dense matrix,
            the default) or only the non-zero values (i.e. a sparse matrix,
            see `as_sparse_matrix` for details) should be returned. 
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
        self.size  = size
        self.x_c   = x_c
        self.y_c   = y_c
        self.flux  = 10**(-0.4*mag)

        self.sigma_0 = self.flux*self._sigma_0

        self.maxrad = maxrad
        self.sparse = return_sparse
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


        ##  calculate profile
        self.array = np.fromfunction(self._draw, self._size, dtype='float32')

        if self.osf != 1: ## take care of oversampling region
            csize = ((2*self.osr+1)*self.osf, (2*self.osr+1)*self.osf)

            x_n = int(round(self._x_c))
            y_n = int(round(self._y_c))

            self._x_c += (self.osr - x_n)
            self._y_c += (self.osr - y_n)

            self._x_c *= self.osf
            self._y_c *= self.osf

            self._x_c += 0.5*(self.osf-1.)
            self._y_c += 0.5*(self.osf-1.)


            ## modify shape parameters
            self.fwhm *= self.osf
            self.sigma_0 /= (self.osf)**2
            self.maxrad *= self.osf

            ##  calculate profile
            carray = np.fromfunction(self._draw, csize, dtype='float32')
            
            ## undo changes on shape parameters
            self.fwhm /= self.osf
            self.sigma_0 *= (self.osf)**2
            self.maxrad /= self.osr

            ## resample and insert profile
            s1_size = (2*self.osr+1, (2*self.osr+1)*self.osf, self.osf)
            s2_size = (2*self.osr+1, 2*self.osr+1, self.osf)
            step1 = np.sum(np.reshape(carray, s1_size, 'C'), axis=2)
            step2 = np.sum(np.reshape(step1, s2_size, 'F'), axis=2)
        
            self.array = paste_area(step2, self._size, (self.osr, self.osr),
                                    (x_n, y_n), out_array=self.array)
            

        ## return requested output
        if (self.sparse or self.maxrad != np.inf):
            Y, X = np.indices(self._size)

            _valid   = (self.array.flatten()!=0.)

            profdata = (self.array.flatten()[_valid], (Y.flatten()[_valid]+ys,
                                                       X.flatten()[_valid]+xs))

            if self.sparse:
                if self.asmatrix:
                    return sparse.coo_matrix(profdata, shape=self.size)
                else:
                    return profdata
            else:
                return np.array(sparse.coo_matrix(\
                        profdata, shape=self.size).todense())

        else:
            return self.array



    def _draw(self, y, x):
        """
        method to fill array with profile, called via np.fromfunction
        """
        u =(x-self._x_c)*np.sin(self.theta)-(y-self._y_c)*np.cos(self.theta)
        v =(y-self._y_c)*np.sin(self.theta)+(x-self._x_c)*np.cos(self.theta)

        r = np.sqrt(u**2 + (v/(1. - self.e))**2)

        return np.where(r>self.maxrad,
                        0.,
                        self.sigma_0*np.exp(-(r**2)/(2.*(self.std**2)))
                           )
