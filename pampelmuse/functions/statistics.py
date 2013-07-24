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

To import a function from this module (e.g. clip_sigma), proceed as follows:
>>> from pampelmuse.functions.statistics import clip_sigma

Latest SVN revision
-------------------
103, 2013/07/17
"""

__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 103

import logging
import numpy as np


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
        
    ## view the data as a masked array (no copy!)    
    workdata_ma = np.ma.array(workdata)
        
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

    ## locate original positions of datapoints that should be clipped
    if axis in [None, 0, -data.ndim]:
        outmask = (data<(median-kappa*sigma))|(data>(median+kappa*sigma))
    else:
        outmask = np.rollaxis((np.rollaxis(data, axis, 0)<(median-kappa*sigma))|(np.rollaxis(data, axis, 0)>(median+kappa*sigma)), 0, axis+1)

    ## return the output
    return np.ma.array(data, mask=outmask)

