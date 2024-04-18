"""
image.py
========
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
Purpose
-------
This package defines utility functions that work on image data

Added
-----
rev. 334, 2016/07/26

Latest GIT revision
-------------------
2019/06/20
"""
import logging
import numpy as np
from scipy import signal


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 20190620


logger = logging.getLogger(__name__)


def find_offset(reference, comparison, init_guess=(0, 0), max_off=None):
    """
    This function determines the offset between two images via cross-
    correlation. Note that for this routine to work both images must
    have the same physical pixel scales.

    Parameters
    ----------
    reference : ndarray
        The image used as reference in the cross-correlation.
    comparison : ndarray
        The image for which the offset with respect to the reference should
        be determined.
    init_guess : tuple, optional
        An initial guess for the offset between comparison and reference
        image. IMPORTANT: The returned offsets are relative to this point.
    max_off : int, optional
        The maximum offset (in pixels) from the initial guess position where
        the code searches for the correlation peak

    Returns
    -------
    x_off : float
        The offset in x-direction (i.e. the first axis in numpy counting) of
        the correlation peak with respect to init_guess[0].
    y_off : float
        The offset in y-direction (i.e. the zeroth axis in numpy counting) of
        the correlation peak with respect to init_guess[1].
    """
    if max_off is None:
        search_radius = max(comparison.shape)
    else:
        assert max_off > 0, 'search radius must be >0 but is {0}'.format(max_off)
        search_radius = int(max_off)

    logger.debug('Size of reference image for cross-correlation: {0}'.format(reference.shape))
    logger.debug('Size of comparison image for cross-correlation: {0}'.format(comparison.shape))

    # To avoid that the cc signal is dominated by individual bright pixels, the comparison image fluxes are
    # leveled off at the 99th percentile of the fluxes
    p = np.percentile(comparison, 99)
    if np.isfinite(p):
        comparison[comparison > p] = p
    else:
        logger.warning('Comparison image for cross-correlation contains invalid values.')

    if not np.isfinite(reference).all():
        logger.warning('Reference image for cross correlation contains {0} NaN/inf value(s)'.format(
            (~np.isfinite(reference)).sum()))
        reference[~np.isfinite(reference)] = np.nanmedian(reference)
    if not np.isfinite(comparison).all():
        logger.warning('Comparison image for cross correlation contains {0} NaN/inf value(s)'.format(
            (~np.isfinite(comparison)).sum()))
        comparison[~np.isfinite(comparison)] = np.nanmedian(comparison)

    # Data are normalized to get approx. normalized cross-corr. signal
    cc_kernel = (reference - np.nanmedian(reference)) / (np.nanstd(reference)*reference.size)
    cc_input = (comparison - np.nanmedian(comparison)) / np.nanstd(comparison)

    cc_signal = signal.fftconvolve(cc_input, cc_kernel[::-1, ::-1], mode='same')

    # The maximum is searched in an area with radius 'search_radius' with respect to the location suggested by the
    # by the init_guess parameter
    # Note that if the two arrays share the same origin, the maximum would be at coordinates
    # (0.5*(cc_kernel.shape[1] - 1), 0.5*(cc_kernel.shape[1] - 1))
    x_search_center = 0.5*(cc_kernel.shape[1] - 1) + init_guess[0]
    y_search_center = 0.5*(cc_kernel.shape[0] - 1) + init_guess[1]

    x_slice_start = max(int(x_search_center) - search_radius, 3)
    x_slice_end = min(int(x_search_center) + search_radius, cc_signal.shape[1] - 3)
    x_slice = slice(x_slice_start, x_slice_end)

    y_slice_start = max(int(y_search_center) - search_radius, 3)
    y_slice_end = min(int(y_search_center) + search_radius, cc_signal.shape[1] - 3)
    y_slice = slice(y_slice_start, y_slice_end)

    _y_max, _x_max = np.nonzero(cc_signal[y_slice, x_slice] == cc_signal[y_slice, x_slice].max())
    y_max = _y_max[0] + y_slice_start
    x_max = _x_max[0] + x_slice_start

    # Measure moments around peak pixel to get peak coordinates with subpixel accuracy
    moment_radius = 3
    peak_data = cc_signal[y_max - moment_radius:y_max + moment_radius + 1,
                          x_max - moment_radius:x_max + moment_radius + 1]
    peak_data -= peak_data.min()
    y_grid, x_grid = np.indices(peak_data.shape)
    x_grid += x_max - moment_radius
    y_grid += y_max - moment_radius
    total = peak_data.sum()
    xc = (x_grid*peak_data).sum()/total
    yc = (y_grid*peak_data).sum()/total

    # Get offsets to position suggested by initial guess
    x_off = xc - x_search_center
    y_off = yc - y_search_center

    # # for testing purposes: plot cross-correlation signal and save as FITS file
    # import pylab, matplotlib
    # from astropy.io import fits
    # fig = pylab.figure(figsize=(7, 7))
    # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    # ax.imshow(cc_signal, interpolation="nearest")
    # ax.plot(x_search_center, y_search_center, ls="None",  marker="x",  mew=2.5, c="g", ms=15)
    # rect = matplotlib.patches.Rectangle((x_search_center - search_radius, y_search_center - search_radius),
    #                                     2*search_radius, 2*search_radius, edgecolor="r", facecolor="none")
    # ax.add_patch(rect)
    # ax.plot(xc, yc, ls="None",  marker="x",  mew=2.5, c="r",  ms=15)
    # pylab.show()
    # outhdu = fits.HDUList([fits.PrimaryHDU(cc_signal), fits.ImageHDU(cc_kernel), fits.ImageHDU(cc_input)])
    # outhdu.writeto("ccsignal.fits", clobber=True)

    return x_off, y_off
