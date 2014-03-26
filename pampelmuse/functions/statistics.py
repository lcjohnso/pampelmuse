"""
statistics.py
=============

Provides
--------
function pampelmuse.functions.statistics.clip_sigma

Purpose
-------
This module defines python functions that deal with statistical analyses. 
Currently, the following functions are defined:

(1) clip_sigma : perform kappa-sigma clipping on an n-dimensional dataset
                 along a given axis and return a masked array where every
                 clipped datapoint is represented by a masked value.
(2) DER_SNR: calculates the S/N of a 1dim. spectrum using the code presented
             in www.spacetelescope.org/about/further_information/newsletters\
                /html/newsletter_42.html

To import a function from this module (e.g. clip_sigma), proceed as follows:
>>> from pampelmuse.functions.statistics import clip_sigma

Latest SVN revision
-------------------
162, 2014/03/26
"""

__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 162

import logging
import numpy as np


# =====================================================================================

def DER_SNR(flux):
   
# =====================================================================================
   """
   DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
               definition set forth by the Spectral Container Working Group of ST-ECF,
               MAST and CADC. 

               signal = median(flux)      
               noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
               snr    = signal / noise
               values with padded zeros are skipped

   USAGE       snr = DER_SNR(flux)
   PARAMETERS  none
   INPUT       flux (the computation is unit independent)
   OUTPUT      the estimated signal-to-noise ratio [dimensionless]
   USES        numpy      
   NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum 
               as a whole as long as
               * the noise is uncorrelated in wavelength bins spaced two pixels apart
               * the noise is Normal distributed
               * for large wavelength regions, the signal over the scale of 5 or
                 more pixels can be approximated by a straight line
 
               For most spectra, these conditions are met.

   REFERENCES  * ST-ECF Newsletter, Issue #42:
               www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
               * Software:
               www.stecf.org/software/ASTROsoft/DER_SNR/
   AUTHOR      Felix Stoehr, ST-ECF
               24.05.2007, fst, initial import
               01.01.2007, fst, added more help text
               28.04.2010, fst, return value is a float now instead of a numpy.float64
   """
   flux = np.array(flux)

   # Values that are exactly zero (padded) are skipped
   flux = np.array(flux[np.where(flux != 0.0)])
   n    = len(flux)      

   # For spectra shorter than this, no value can be returned
   if (n>4):
      signal = np.median(flux)

      noise  = 0.6052697 * np.median(np.abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))

      return float(signal / noise)  

   else:

      return 0.0



def clip_sigma(data, axis=None, kappa=3.):
    """
    Average data along given axis using sigma-clipping.
   
    Parameters
    ----------
    data : nd_array
        The array to perform kappa-sigma clipping on.
    axis : integer, optional
        Axis over which the data is average. By default `axis`
        is None, and all elements are averaged.
    kappa : float, optional
        All pixels whose values deviate from the median of the
        distribution by more than `kappa` times the sigma of the
        distribution are excluded.

    Returns
    -------
    outdata : numpy.ma.array
        An masked array that has the same shape as 'data' and contains a
        masked value on every position where a data point was clipped.
    """
    
    ## prepare input array for calculation
    if axis == None: ## create 1dim. sorted copy of input
        workdata = np.sort(data, axis=None)
    elif axis not in [0, -data.ndim]: ## move requested axis to first index and sort values along that axis
        workdata = np.sort(np.rollaxis(data, axis, 0), axis=0)
    else:
        workdata = np.sort(data, axis=0)
        
    ## view the data as a masked array (no copy!) and mask NaN pixels   
    workdata_ma = np.ma.array(workdata, mask=(workdata!=workdata))
        
    i=0 ## the iteration index
    while True: ## perform clipping
        sigma  = workdata_ma.std(axis=0)
        median = np.ma.median(workdata_ma, axis=0)

        toclip = (workdata_ma<(median-kappa*sigma))|(workdata_ma>(median+kappa*sigma))
        N = toclip.sum() ## the number of clipped data points in this iteration
        
        workdata_ma[toclip] = np.ma.masked

        logging.debug("Kappa-sigma clipping: Iteration {0}, clipped {1} points".format(i, int(N)))
        if N == 0: break

        if (workdata_ma.mask==False).sum() < 5:
            logging.warn("Kappa-sigma clipping: Iteration {0}, <5 valid pixels left.".format(i))
            break
        i += 1

    ## locate original positions of datapoints that should be clipped, again
    ## clip also NaN pixels
    if axis in [None, 0, -data.ndim]:
        outmask = (data<(median-kappa*sigma))|(data>(median+kappa*sigma))
    else:
        outmask = np.rollaxis((np.rollaxis(data, axis, 0)<(median-kappa*sigma))|(np.rollaxis(data, axis, 0)>(median+kappa*sigma)), 0, axis+1)

    ## return the output
    return np.ma.array(data, mask=(outmask|(data!=data)))

