"""
pampelmuse.filters
==================

Provides
--------
dictionary pampelmuse.filters.ALIASES
function pampelmuse.filters.read_filter

Purpose
-------
This module provides an interface to the filter curves of the
photometric bandpasses that are included in PampelMuse.

Added
-----
2015/10/23 (Revision 210)

Latest SVN Revision
-------------------
237, 2015/02/11
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 237

import logging
import os
import numpy as np
from scipy import interpolate


# ALIASES is a dictionary containing for each available filter a list of alternative names
ALIASES = {"f555w": ["v", "vvega"], "f606w": ["vvega", "v"], "f814w": ["ivega", "i"], "k": ["kmag", ]}

# FWHM values of the individual filters, given in Angstroms, FWHM values for HST filters from ACS
FILTERFWHM = {"f555w": 847.79, "f606w": 1583.2, "f814w": 1541.6, "k": 3900.}


def get_filter(name):
    """
    Reads the filter curve for the provided passband name and returns it
    as an interpolator. Note that the filter curves are searched for in
    the 'data' folder that is part of this package.
    
    Parameters
    ----------
    name : string
        The name of the passband for which the filter curve should be
        returned.
        
    Returns
    -------
    f : a scipy.interpolate.UnivariateSpline instance
        The interpolator created from the filter curve.
        
    Raises
    ------
    IOError if filter curve is missing.
    """
    # get actual location of this package on disk
    __currentdir__ = os.path.abspath(os.path.dirname(__file__))
    
    # get list of available filters
    filters = os.listdir(os.path.join(__currentdir__, "data"))
    
    # check if filter is available
    if not name.upper() in filters:
        logging.critical("Cannot find filtercurve for passband {0}".format(name.upper))
        raise IOError("Missing filtercurve for passband {0}".format(name.upper()))
    
    # open filter curve
    filtercurve = np.genfromtxt(os.path.join(__currentdir__, "data", name.upper()), usecols=(0, 1),
                                dtype=[("wl", np.float32), ("tp", np.float32)])

    # return the filter curve as a scipy.interpolate.UnivariateSpline instance
    f = interpolate.UnivariateSpline(filtercurve["wl"], filtercurve["tp"], k=3, s=0)
    return f
