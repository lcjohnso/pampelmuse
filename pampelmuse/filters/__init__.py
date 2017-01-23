"""
__init__.py
===========
Copyright 2013-2016 Sebastian Kamann

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
387, 2017/01/23
"""
import logging
import os
import numpy as np
from scipy import interpolate


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 387


# ALIASES is a dictionary containing for each available filter a list of alternative names
ALIASES = {"f555w": ["v", "vvega"], "f606w": ["vvega", "v"], "f814w": ["ivega", "i"], "k": ["kmag", ]}

# FWHM values of the individual filters, given in Angstroms, FWHM values for HST filters from ACS
# FILTERFWHM = {"f555w": 847.79, "f606w": 1583.2, "f814w": 1541.6, "k": 3900.}
# not used anymore as of rev. 279


def get_filter(name):
    """
    Reads the filter curve for the provided passband name and returns it
    as an interpolator. Note that the filter curves are searched for in
    the 'data' folder that is part of this package. In addition, the routine
    checks for an environment variable PMFILTERPATH. Is it is set, then it
    will also use filter curves present in that directory.
    
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

    filterfile = None

    # check if filter is available
    # first choice: environment variable PMFILTERPATH
    pmfilterpath = os.environ.get("PMFILTERPATH")
    if pmfilterpath is not None:
        # get list of available filters
        user_filters = os.listdir(pmfilterpath)
        if name.upper() in user_filters:
            filterfile = os.path.join(pmfilterpath, name.upper())

    # second choice: the filters.data directory of PampelMuse
    if filterfile is None:
        # get list of available filters
        filters = os.listdir(os.path.join(__currentdir__, "data"))
        if name.upper() in filters:
            filterfile = os.path.join(__currentdir__, "data", name.upper())

    if filterfile is None:
        logging.critical("Cannot find transmission curve for passband {0}".format(name.upper()))
        raise IOError("Missing transmission curve for passband {0}".format(name.upper()))
    
    # open filter curve
    filtercurve = np.genfromtxt(filterfile, usecols=(0, 1), dtype=[("wl", np.float32), ("tp", np.float32)])

    # return the filter curve as a scipy.interpolate.InterpolatedUnivariateSpline instance
    # use option ext=1 so that values outside valid range are set to zero.
    return interpolate.InterpolatedUnivariateSpline(filtercurve["wl"], filtercurve["tp"], k=3, ext=1)

