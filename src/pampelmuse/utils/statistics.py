"""
statistics.py
=============
Copyright 2013-2017 Sebastian Kamann

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
function pampelmuse.functions.statistics.der_snr
function pampelmuse.functions.statistics.clip_sigma

Purpose
-------
This module defines python functions that deal with statistical analyses. 
Currently, the following functions are defined:

(1) clip_sigma : perform kappa-sigma clipping on an n-dimensional dataset
                 along a given axis and return a masked array where every
                 clipped datapoint is represented by a masked value.
(2) der_snr: calculates the S/N of a 1dim. spectrum using the code presented
             in www.spacetelescope.org/about/further_information/newsletters\
                /html/newsletter_42.html

To import a function from this module (e.g. clip_sigma), proceed as follows:
>>> from pampelmuse.utils.statistics import clip_sigma

Latest Git revision
-------------------
2017/08/25
"""
import logging
import numpy as np


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20170825


def der_snr(flux):
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
    flux = np.array(flux[np.isfinite(flux)])

    # Values that are exactly zero (padded) or NaN are skipped
    flux = flux[(flux != 0.0) & np.isfinite(flux)]
    n = len(flux)

    # For spectra shorter than this, no value can be returned
    if n > 4:
        signal = np.median(flux)

        noise = 0.6052697 * np.median(np.abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))

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
    
    # prepare input array for calculation
    if axis is None:  # create 1dim. sorted copy of input
        workdata = np.sort(data, axis=None)
    elif axis not in [0, -data.ndim]:  # move requested axis to first index and sort values along that axis
        workdata = np.sort(np.rollaxis(data, axis, 0), axis=0)
    else:
        workdata = np.sort(data, axis=0)
        
    # view the data as a masked array (no copy!) and spectrumMask NaN pixels
    workdata_ma = np.ma.array(workdata, mask=(workdata!=workdata))
        
    i = 0  # the iteration index
    while True:  # perform clipping
        sigma = workdata_ma.std(axis=0)
        median = np.ma.median(workdata_ma, axis=0)

        toclip = (workdata_ma < (median-kappa*sigma)) | (workdata_ma > (median+kappa*sigma))
        N = toclip.sum()  # the number of clipped data points in this iteration
        
        workdata_ma[toclip] = np.ma.masked

        logging.debug("Kappa-sigma clipping: Iteration {0}, clipped {1} points".format(i, int(N)))
        if N == 0:
            break

        if (~workdata_ma.mask).sum() < 5:
            logging.warn("Kappa-sigma clipping: Iteration {0}, <5 valid pixels left.".format(i))
            break
        i += 1

    # locate original positions of datapoints that should be clipped, again clip also NaN pixels
    if axis in [None, 0, -data.ndim]:
        outmask = (data < (median-kappa*sigma)) | (data > (median+kappa*sigma))
    else:
        outmask = np.rollaxis((np.rollaxis(data, axis, 0) < (median - kappa*sigma)) | (
            np.rollaxis(data, axis, 0) > (median+kappa*sigma)), 0, axis+1)

    # return the output
    return np.ma.array(data, mask=(outmask | np.isnan(data)))


def robust_fit(fit_function, x, y, order, domain=None):
    """
    Method to implement a fit that is robust against outliers.

    The idea is to implement an iterative weighting scheme in which each
    point is weighted according to w = 1/[1 + (DY/3*MADY)^4], where DY is
    the offset between a data point and the current fit and MADY is the
    median absolute deviation between the data and the current fit.

    This scheme is similar to the one used by P. Stetson for DAOphot, with
    alpha~2 and beta=4.

    Parameters
    ----------
    fit_function : instance of numpy.polynomial.polynomial.Polynomial
        The function used to fit the data.
    x : array_like
        The x values of the data to be fitted.
    y : array_like
        The y-values of the data to be fitted.
    order : int
        The order of the polynomial fit to be used.
    domain : tuple, optional
        The domain for the poynomial fit.

    Returns
    -------
    fit : series
        A series that represents the least squares fit to the data and has
        the domain specified in the call (taken from the scipy documentation.)
    """
    fit = None
    w = np.ones_like(x)

    for i in range(3):
        fit = fit_function.fit(x, y, order, domain=domain, w=w)

        dy = y - fit(x)
        mad_dy = np.median(np.abs(dy - np.median(dy)))

        w = 1./(1. + (np.abs(dy)/(3.*mad_dy))**4)

    return fit
