"""
pampelmuse.filters
==================

Provides
--------
dictionary pampelmuse.filters.ALIASES
function pampelmuse.filters.read_filter

Purpose
-------
This module provides an interface to the filtercurves of the
photometric bandpasses that are incldued in PampelMuse.

Added
-----
2014/10/23 (Revision 210)

Latest SVN Revision
-------------------
214, 2014/10/24
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 214

import logging, glob, os
import numpy as np
from scipy import interpolate


# ALIASES is a dictionary containing for each available filter a list of alternative names
ALIASES = {"f555w":["v", "vvega"], "f606w":["vvega", "v"], "f814w":["ivega", "i"], "k":["kmag", ]}


def get_filter(name):
    """
    Reads the filter curve for the provided passband name and returns it
    as an interpolator. Note that the filter curves are searched for in
    the 'data' folder that is part of this package.
    
    Parameters
    ----------
    name : string
        The name of the passband for which the filtercurve should be
        returned.
        
    Returns
    -------
    f : a scipy.interpolate.UnivariateSpline instance
        The interpolator created from the filtercurve.
        
    Raises
    ------
    IOError if filtercurve is missing.
    """
    # get actual location of this package on disk
    __currentdir__ = os.path.abspath(os.path.dirname(__file__))
    
    # get list of available filters
    filters = os.listdir(os.path.join(__currentdir__, "data"))
    
    # check if filter is available
    if not name.upper() in filters:
        logging.critical("Cannot find filtercurve for passband {0}".format(name.upper))
        raise IOError, "Missing filtercurve for passband {0}".format(name.upper())
    
    # open filtercurve
    filtercurve = np.genfromtxt(os.path.join(__currentdir__, "data", name.upper()), usecols=(0,1), dtype=[("wl", np.float32), ("tp", np.float32)])

    # return the filter curve as a scipy.interpolate.UnivariateSpline instance
    f = interpolate.UnivariateSpline(filtercurve["wl"], filtercurve["tp"], k=3, s=0)
    return f
