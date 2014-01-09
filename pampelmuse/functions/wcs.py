"""
wcs.py
======

Provides
--------
pampelmuse.functions.wcs.calc_offset

Purpose
-------
This module implements functions that deal with WCS coordinates. The
current list of functions defined in this module contains:

1) calc_offset
Given a set of sources with RA and Dec coordinates, this function calculates
the offset dRA and dDec to a reference point in acrseconds. The function is
basically a slightly modified version of the function 'calcAngSepDeg' from the
astLib.astCoords package
(see http://astlib.sourceforge.net/docs/astLib/astLib.astCoords-module.html)

Latest SVN revision
-------------------
117, 2014/01/09

"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 117

import numpy as np


def calc_offset(ref_ra, ref_dec, ra, dec):
    """
    Calculates the offset in arcseconds in RA and Dec of one or
    more sources with respect to a reference position.
    
    Parameters
    ----------
    ref_ra : float
        The right ascencion of the reference point in decimal degrees.
    ref_dec : float
        The declination of the reference point in decimal degrees.
    ra : float or array_like
        The right ascencion of the sample data given in decimal degrees.
    dec : float or array_like
        The declination of the sample data given in decimal degrees.
        
    Returns
    -------
    offset :  float or array_like, depending on type of 'ra' and 'dec'
        The calculated offsets in arcseconds.
    """
    cRA=np.radians(ref_ra)
    cDec=np.radians(ref_dec)

    gRA=np.radians(ra)
    gDec=np.radians(dec)
    dRA=cRA-gRA
    dDec=gDec-cDec
    cosC=(np.sin(gDec)*np.sin(cDec))+(np.cos(gDec)*np.cos(cDec)*np.cos(gRA-cRA))
    x=3600.*np.degrees((np.cos(cDec)*np.sin(gRA-cRA))/cosC)
    y=-3600.*np.degrees(((np.cos(gDec)*np.sin(cDec))-(np.sin(gDec)*np.cos(cDec)*np.cos(gRA-cRA)))/cosC)
    ## NOTE: the routine returns x=Ra2-Ra1 and y=Dec2-Dec1, but convert
    ##       to a pixel offset on the detector, you need to multiply x by
    ##       -1
    return x, y
