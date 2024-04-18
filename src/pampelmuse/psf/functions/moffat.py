"""
moffat.py
===========
Copyright 2013-2018 Sebastian Kamann

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
function moffat.psf.functions.moffat.r_d
function moffat.psf.functions.moffat.flux
function moffat.psf.functions.moffat.pderiv_beta
function moffat.psf.functions.moffat.pderiv_fwhm
function moffat.psf.functions.moffat.pderiv_e
function moffat.psf.functions.moffat.pderiv_r

Purpose
-------
This module implements the functions that are required to calculate a Moffat
PSF profile. Besides the actual flux distribution, the code also provides the
partial derivatives with respect to the individual parameters of the Gaussian.

Added
-----
2018/04/22

Latest Git revision
-------------------
2018/04/24
"""
import numpy as np


__author__ = 'Sebastian Kamann (s.kamann@ljmu.ac.uk)'
__revision__ = 20180424


def r_d(beta, fwhm):
    """
    Calculate the effective radius r_e from the beta-parameter and the FWHM
    of a Moffat.
    
    Parameters
    ----------
    beta : float
        The beta parameter of the Moffat.
    fwhm : float
        The FWHM of the Moffat.
    
    Returns
    -------
    r_d : float
        The effective radius of the profile.
    """
    return fwhm / (2. * np.sqrt(2 ** (1. / beta) - 1.))


def flux(r, beta, fwhm, e):
    """
    Calculate the values of the Moffat function for the provided radii and
    profile parameters.
        
    The Moffat profile with unit flux is defined as follows.
     
    M(r) = (beta - 1)/(PI* r_d^2 * (1 - e)) * [1 + (r/r_d)^2]^-beta

    The width of the Gaussian is defined through the effective radius 'r_d'
    while 'beta' defines the kurtosis of the profile. Smaller values of 'beta'
    corresponds to more pronounced wings. The FWHM of the Moffat profile can
    be calculated via

    FWHM = 2 * r_d* SQRT(2^(1/beta) - 1)

    In this implementation of the Moffat profile, the FWHM and 'beta' are
    used as the free parameters and the effective radius is just computed
    internally.
    
    Parameters
    ----------
    r : array_like
        The radii at which the Moffat should be evaluated.
    beta : float
        The beta parameter of the Moffat.
    fwhm : float
        The FWHM of the Moffat.
    e : float
        The ellipticity of the Moffat.

    Returns
    -------
    flux : array_like
        The flux of the Moffat at the provided radii.
    """
    return (beta - 1.) / (np.pi * (r_d(beta, fwhm) ** 2) * (1. - e)) * (1. + (r / r_d(beta, fwhm)) ** 2) ** (-beta)


def pderiv_beta(r, beta, fwhm, e):
    """
    Calculate the partial derivative of the Moffat with respect to beta.
    
    Parameters
    ----------
    r : array_like
        The radii at which the Moffat should be evaluated.
    beta : float
        The beta parameter of the Moffat.
    fwhm : float
        The FWHM of the Moffat.
    e : float
        The ellipticity of the Moffat.

    Returns
    -------
    pderiv_beta : array_like
        The partial derivative of the Moffat with respect to beta.
    """
    fa = 1. / (beta - 1.)
    fb = np.log(1. + (r / r_d(beta, fwhm)) ** 2)
    fc = np.log(2.) / (beta**2 * (1. - 2. ** (1. / beta)))
    fd = 1. + 2**(1./beta) * beta * (r / r_d(beta, fwhm)) ** 2 / (1. + (r / r_d(beta, fwhm)) ** 2)

    return flux(r, beta, fwhm, e) * (fa - fb - fc * fd)


def pderiv_fwhm(r, beta, fwhm, e):
    """
    Calculate the partial derivative of the Moffat with respect to the FWHM.
    
    Parameters
    ----------
    r : array_like
        The radii at which the Moffat should be evaluated.
    beta : float
        The beta parameter of the Moffat.
    fwhm : float
        The FWHM of the Moffat.
    e : float
        The ellipticity of the Moffat.

    Returns
    -------
    pderiv_fwhm : array_like
        The partial derivative with respect to tghe FWHM. 
    """
    return flux(r, beta, fwhm, e) / (r_d(beta, fwhm) * np.sqrt(2. ** (1. / beta) - 1.)) * (
        beta * (r / r_d(beta, fwhm)) ** 2 / (1. + (r / r_d(beta, fwhm)) ** 2) - 1.)


def pderiv_e(r, beta, fwhm, e, r_pderiv_e):
    """
    Calculate the partial derivative of the Moffat with respect to the
    ellipticity e, at the provided radii.
    
    Parameters
    ----------
    r : array_like
        The radii at which the Moffat should be evaluated.
    beta : float
        The beta parameter of the Moffat.
    fwhm : float
        The FWHM of the Moffat.
    e : float
        The ellipticity of the Moffat.
    r_pderiv_e : array_like
        The partial derivative of the provided radii with respect to the
        ellipticity. This is needed because the radius itself is a function of
        the ellipticity. The array must have the same size as 'r'.

    Returns
    -------
    pderiv_e : array_like
        The partial derivative with respect to the ellipticity.
    """
    return flux(r, beta, fwhm, e) * (
        1. / (1. - e) - 2. * beta * r / ((r_d(beta, fwhm) ** 2) * (1. + (r / r_d(beta, fwhm)) ** 2)) * r_pderiv_e)


def pderiv_r(r, beta, fwhm, e):
    """
    Calculate the partial derivative of the Moffat with respect to the radius,
    for the given radii and function parameters.
    
    Parameters
    ----------
    r : array_like
        The radii at which the Moffat should be evaluated.
    beta : float
        The beta parameter of the Moffat.
    fwhm : float
        The FWHM of the Moffat.
    e : float
        The ellipticity of the Moffat.

    Returns
    -------
    pderiv_r : array_like
        The partial derivative at the provided radii.
    """
    return flux(r, beta, fwhm, e) * -2. * beta * r / (r_d(beta, fwhm) ** 2 * (1. + (r / r_d(beta, fwhm)) ** 2))
