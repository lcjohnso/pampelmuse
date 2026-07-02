"""
gaussian.py
===========
Copyright 2013-2026 Sebastian Kamann

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
function pampelmuse.psf.functions.gaussian.std
function pampelmuse.psf.functions.gaussian.flux
function pampelmuse.psf.functions.gaussian.pderiv_fwhm
function pampelmuse.psf.functions.gaussian.pderiv_e
function pampelmuse.psf.functions.gaussian.pderiv_r

Purpose
-------
This module implements the functions that are required to calculate a Gaussian
PSF profile. Besides the actual flux distribution, the code also provides the
partial derivatives with respect to the individual parameters of the Gaussian.

Added
-----
2018/04/22

Latest Git revision
-------------------
2026/07/02
"""
import numpy as np


__author__ = 'Sebastian Kamann (s.kamann@ljmu.ac.uk)'
__revision__ = 20260702


def std(fwhm):
    """
    Determine standard deviation from FWHM by evaluating
    STD = FWHM / sqrt(8*ln(2)).
    
    Parameters
    ----------
    fwhm : float
        The FWHM of the profile.
    
    Returns
    -------
    std : float
        The standard deviation of the profile.
    """
    return fwhm / np.sqrt(8. * np.log(2.))


def flux(r, fwhm, e):
    """
    Calculate the values of the Gaussian function for the provided radii and
    profile parameters.
        
    The Gaussian profile with unit flux is defined as follows.
     
    G(r) = 1. / (2*PI*STD^2*(1-E)) * exp(-r^2/(2*STD^2))

    The width of the Gaussian is defined through the standard deviation 'STD'
    that is related to the FWHM of the profile via

    FWHM = STD * sqrt(8*ln(2)).
    
    In this implementation of the Gaussian profile, the FWHM is used as the
    free parameter and the standard deviation is just computed internally. E
    is the ellipticity of the profile.
        
    Parameters
    ----------
    r : array_like
        The radii for which the Gaussian function should be evaluated.
    fwhm : float
        The FWHM of the Gaussian profile.
    e : float
        The ellipticity of the Gaussian profile.

    Returns
    -------
    f : array_like
        The values of the Gaussian function for the provided radii.
    """
    return 1 / (2. * np.pi * std(fwhm) ** 2 * (1. - e)) * np.exp(-r ** 2 / (2. * std(fwhm) ** 2))


def pderiv_fwhm(r, fwhm, e):
    """
    Calculate the partial derivative of the Gaussian function with respect to
    the FWHM for the provided radius/radii.
        
    Parameters
    ----------
    r : array_like
       The radii at which the partial derivative should be evaluated. 
    fwhm : float
        The FWHM of the Gaussian profile.
    e : float
        The ellipticity of the Gaussian profile.

    Returns
    -------
    pderiv_fwhm : array_like
    The partial derivative with respect to the FWHM at the provided radii.
    """
    return flux(r, fwhm, e) * ((r / std(fwhm)) ** 2 - 2.) / fwhm


def pderiv_e(r, fwhm, e, dr_de):
    """
    Calculate the partial derivative of the Gaussian function with respect to
    the ellipticity, e, for the provided radii.
        
    Parameters
    ----------
    r : array_like
        The radii at which the partial derivative should be evaluated. 
    fwhm : float
        The FWHM of the Gaussian profile.
    e : float
        The ellipticity of the Gaussian profile.
    dr_de : array_like
        The partial derivative of the provided radii with respect to the
        ellipticity. This is needed because the radius itself is a function of
        the ellipticity. The array must have the same size as 'r'.

    Returns
    -------
    pderiv_e : array_like
        The partial derivative with respect to the ellipticity at the provided
        radii.
    """
    return flux(r, fwhm, e) * (1. / (1. - e) - (r / (std(fwhm) ** 2)) * dr_de)


def pderiv_r(r, fwhm, e):
    """
    Calculate the partial derivative of the Gaussian function with respect
    to the radius, r, for the provided radii.
        
    Parameters
    ----------
    r : array_like
        The radii at which the partial derivative should be evaluated. 
    fwhm : float
        The FWHM of the Gaussian profile.
    e : float
        The ellipticity of the Gaussian profile.
        
    Returns
    -------
    pderiv_r : array_like
        The partial derivative with respect to the radius at the provided
        radii.
    """
    return -flux(r, fwhm, e) * r / (std(fwhm) ** 2)
