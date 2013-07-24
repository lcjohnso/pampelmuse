"""
fitting.py
==========

Provides
--------
class pampelmuse.functions.fitting.StepPolynomial

Purpose
-------
This module defines python functions or classes that deal with the
fitting of data. The current set of fitting environments defined here
includes:

(1) StepPolynomial:
Defines a 1dim. step (or broken) polynomial that can be fit to a dataset.
To this aim, the data are split up into N equally large subsets and each 
ubset is fitted by a plain polynomial of order n. To obtain a smooth fit,
the subsets overlap at their edges.

Note that in cases where the fitting is encoded in a class environment,
a @staticmethod is provided that allows one to create a class instance
when fitting the data, i.e. one does not have to initialize an instance
before one can perform a fit.

To import a function from this module (e.g. StepPolynomial), proceed as
follows:
>>> from pampelmuse.functions.fitting import StepPolynomial

Latest SVN Revision
-------------------
103, 2013/07/16
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 103

import numpy as np



class StepPolynomial(object):
    """
    The class defines a sequence of M 1dim. polynomials of order N that
    overlap at their edges. 
    
    Instances of StepPolynomial currently support the addition and
    subtraction of constant values.
    
    When data is fitted, the data are split up into M bins and
    a polynomial of order N is fitted to each bin. To obtain a smooth
    function, the individual bins overlap by a user-definable amount.
    """
    def __init__(self, order, bins, polynomial=None):
        """
        Initialize an instance of the StepPolynomial class.
        
        Parameters
        ----------
        order : int
            The order N of the polynomial used for each bin.
        bins : int
            The number of bins M that the broken polynomial will
            have.
        polynomial : array-like, opional
            The values used to initialize the polynomials, provided
            as a sequence of tuples that contain for each bin: the 
            start of the overlap region, the start of the bin, the
            end of the bin, the end of the overlap region, and the
            N+1 coefficients of the polynomial used in the bin.
        """
        
        self.order = order
        self.bins = bins
    
        dtype = [("min", "<f4"), ("start", "<f4"), ("end", "<f4"), ("max", "<f4"), ("parameters", "<f8", (self.order+1,))]
    
        if polynomial == None:
            self.polynomial = np.zeros(bins, dtype=dtype)
        else:
            self.polynomial= np.array(polynomial, dtype=dtype)
    
    
    def __add__(self, other):
        """Returns the sum."""
        if isinstance(other, (int, np.integer, float, np.float)):
            coeffs = np.copy(self.polynomial)
            for i in xrange(self.bins): coeffs["parameters"][i, -1] += other
            return self.__class__(self.order, self.bins,  polynomial=coeffs)
        else:
            return NotImplemented
    
    
    def __sub__(self, other):
        """Returns the difference."""
        if isinstance(other, (int, np.integer, float, np.float)):
            coeffs = np.copy(self.polynomial)
            for i in xrange(self.bins): coeffs["parameters"][i, -1] -= other
            return self.__class__(self.order, self.bins,  polynomial=coeffs)
        else:
            return NotImplemented
    
    
    @staticmethod
    def fit(x, y, order, bins=1, overlap=-1):
        """
        Initializes an instance of the class StepPolynomial from
        fitting a 1dim. dataset f(x)=y
        
        Parameters
        ----------
        x : ndarray
            The x values of the data.
        y : ndarray
            The y values of the data.
        order : int
            The order N of the polynomial used to fit each bin.
        bins : int
            The number of bins M that the dataset is split up into.
        overlap : int, optional
            The overlap between the adjacent bins, in units of data-
            points. If -1 is provided, the overlap region will be 
            set to 1/10 of the size of each bin.
            
        Returns
        -------
        A class SpepPolynomial instance
            The fit to the data is used to initialize a new instance of
            the class.
        """

        n = x.size
        assert y.size==n, "x and y data must have same length."
        
        if overlap == -1:
            overlap = int(0.1*len(x)//bins)
        
        binsize = (x.size//bins)+1
        if bins>1: binsize -= overlap
            
        polynomial = []
            
        for i in xrange(bins):
        
            jmin   = max(0, i*(binsize+overlap)-overlap)
            jstart = i*(binsize+overlap)
            if i!=bins-1:
                jend = min(n-1, (i+1)*(binsize+overlap)-overlap)
                jmax = min(n-1, (i+1)*(binsize+overlap))
            else:
                jend = n-1
                jmax = n-1
            polynomial.append((x[jmin], x[jstart], x[jend], x[jmax], np.polyfit(x[jmin:jmax], y[jmin:jmax], deg=order)))
        
        return StepPolynomial(order=order, bins=bins, polynomial=polynomial)
        

    def __call__(self, x):
        """
        Calculate the values of the broken polynomial for the
        provided coordinates.
        
        Parameters
        ----------
        x : ndarray
            The x values for which the polynomial should be calculated.
            
        Returns
        -------
        y : ndarray
            The polynomial values at the provided x coordinates.
        """
        if self.bins==1:
            return np.polyval(self.polynomial["parameters"][0], x)
        
        else:
            y = np.zeros(x.shape, dtype=x.dtype)
            for i in xrange(self.bins):
        
                min = self.polynomial["min"][i]
                start = self.polynomial["start"][i]
                end = self.polynomial["end"][i]
                max = self.polynomial["max"][i]
        
                weights = np.zeros(x.size, dtype=x.dtype)
                weights[(x>=start)&(x<=end)] = 1
                if ((x>min)&(x<start)).any():
                    weights[(x>min)&(x<start)] = (x[(x>min)&(x<start)]-min)/(start-min)
                if ((x>end)&(x<max)).any():
                    weights[(x>end)&(x<max)] = (max-x[(x>end)&(x<max)])/(max-end)

                y += weights*np.polyval(self.polynomial["parameters"][i], x)
            
            return y
