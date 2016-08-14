"""
PampelMuse
==========
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
Purpose
-------
This module provides the class 'Argus" that is designed to facilitate the
analysis of data observed with FLAMES/ARGUS at the VLT.

Latest SVN revision
-------------------
360, 2016/08/12
"""
import re
import logging
from matplotlib import transforms
import numpy as np
from .instrument import Instrument


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 360


class Argus(Instrument):
    """
    This class is designed to handle data coming from the IFU 'ARGUS' of the
    FLAMES spectrograph (Pasquini et al. 2002, The Messenger, 110, 1), mounted
    on UT2 of the ESO Very Large Telescope (VLT).
    """
    name = "ARGUS"
    _shape = (14, 22)  # numpy notation - first y, then x

    # The indices of the dedicated sky fibers
    _sky_fibers = np.array([20, 41, 63, 84, 105, 127, 148, 169, 191, 212, 233, 255, 276, 297, 319], dtype=np.int16)

    # The indices of the dedicated calibration fibers
    _cal_fibers = np.array([42, 106, 170, 234, 298], dtype=np.int16)

    def __init__(self, **kwargs):
        """
        Initialize an instance of the 'ARGUS' class. NOTE that by default, it
        is assumed that the traces of all fibres (including all 14 sky fibres
        and all 5 calibration fibres) are present in the data. This is the
        format provided by, e.g., p3d (Sandin et al. 2010, A&A, 515, A35). If
        the data have another format, this can be handled by either providing
        the coordinate transformation directly upon initialisation of an
        instance or by initializing a class instance via the method
        Argus.from_fiber_setup_hdu that takes a FITS-HDU containing the fiber
        setup (as provided by the official ESO pipeline) as input.

        Parameters
        ----------
        kwargs are passed to the initialisation os the Instrument super-class.
        Note that unless provided, the parameters 'transform' and 'bad' are
        initialized to default values taken from the ARGUS manual.

        Returns
        -------
        The newly created instance.
        """
        # The traces of the last three spaxels are entirely outside the field of view.
        if "bad" not in kwargs.keys():
            kwargs["bad"] = [297, 298, 299]

        # initialize transformation from (x,y)-coordinates to spaxels
        if "transform" not in kwargs.keys():
            kwargs["transform"] = np.vstack((np.array(
                    [8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 0, 12, 13, 13, 12, 11, 10, 9, 8, 7,
                     6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0,
                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 13, 12, 11,
                     10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 3,
                     2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                     13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2,
                     3, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                     0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6,
                     7, 8, 9, 10, 11, 12, 13, 13, 12, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8],
                    dtype=np.int16), np.array(
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5,
                     5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9,
                     9, 9, 9, 9, 9, 9, 9, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10,
                     10, 10, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 14, 14, 14,
                     14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 15, 15, 15,
                     15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16,
                     16, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17,
                     17, 17, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 18, 19, 19, 20,
                     21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20, 20, 20], dtype=np.int16))).T

        # all pixels at the edge of the fov - not used anymore: edge is determined automatically
        # if "edge" not in kwargs.keys():
        #     kwargs["edge"] = np.array(
        #         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 48, 47, 60, 99, 112, 111, 124, 123, 176,
        #          175, 188, 187, 200, 239, 252, 251, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
        #          293, 294, 295, 276, 277, 278, 225, 226, 213, 214, 161, 162, 149, 150, 137, 138, 85, 86, 73, 74, 21,
        #          22, 23], dtype=np.int16)

        super(Argus, self).__init__(**kwargs)

        self.wcs.wcs.crpix[:2] = [self.shape[1]/2., self.shape[0]/2.]

    @classmethod
    def from_fiber_setup_hdu(cls, hdu, **kwargs):
        """
        A new instance of the ARGUS class can also be initialized via a
        FITS-HDU that contains the fiber configuration of the observation.
        This HDU is routinely provided by the ESO-pipeline (I guess).

        The method uses the information to update the transformation from
        spaxels to x- & y- coordinates and to find the indices of the sky
        and calibration fibres.

        Parameters
        ----------
        hdu : a FITS binary table HDU instance
            The HDU containing the fibre setup.

        Returns
        -------
        A new instance of the ARGUS class, initialized with the detected
        fiber setup.
        """
        _sky_fibers = []
        _cal_fibers = []
        _x = []
        _y = []

        # use 'RECTRACTOR' information from table to identify fibers
        sky_match = re.compile(r"P2\-AR[NC][0-9]+\-([0-9]+|Sky)")

        for i in xrange(hdu.header["NAXIS2"]):  # loop over all rows in HDU

            match = sky_match.match(hdu.data.field("RETRACTOR")[i])

            if match is None:  # calibration fiber
                _cal_fibers.append(i)
            elif match.group(1) == "Sky":  # sky fiber
                _sky_fibers.append(i)
            else:  # science fiber
                _x.append(hdu.data.field("X")[i] - 1)
                _y.append(hdu.data.field("Y")[i] - 1)

        # some logging on setup that was found.
        logging.info("ARGUS fibre calibration (no. of fibres):")
        logging.info("                            science: {0:3d}".format(len(_x)))
        logging.info("                                sky: {0:3d}".format(len(_sky_fibers)))
        logging.info("                        calibration: {0:3d}".format(len(_cal_fibers)))

        # initialize new class instance
        kwargs['transform'] = np.vstack((_y, _x)).T
        argus = cls(**kwargs)
        argus._sky_fibers = _sky_fibers
        argus._cal_fibers = _cal_fibers

        return argus

    @property
    def nlayer(self):
        """
        Returns the number of layers of the current data set.
        """
        return self.datahdu.shape[0] if self.datahdu is not None else None

    # @property
    # def edgepath(self):
    #     """
    #     Create a matplotlib.path.Path instance that follows the edges of
    #     the field of view. This instance can be used for plotting purposes
    #     and to check whether a point is inside or outside the FoV.
    #
    #     Returns
    #     -------
    #     edgepath:  a matplotlib.path.Path instance
    #         The edges of the field of view.
    #     """
    #     # TODO: Find out why x- & y-axes have to be swapped
    #     if self._edgepath is None:
    #         t = transforms.Affine2D(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    #         self._edgepath = super(Argus, self).edgepath.transformed(t)
    #     return self._edgepath

    def is_valid(self, hdu):
        """
        This method inspects if a provided FITS HDU contains data that is
        compatible with the current Argus instance.

        Parameters
        ----------
        hdu : a FITS HDU containing some data
            The FITS HDU that should be checked.

        Returns
        -------
        validity_flag : boolean
            A flag indicating if the data in the HDU can be used with the
            current Instrument instance.
        """
        # check compatibility with base class.
        if not super(Argus, self).is_valid(hdu):
            return False
        elif hdu.header['NAXIS'] != 2:
            logging.error('HDUs containing Flames/ARGUS data must be 2dimensional.')
            return False
        elif hdu.header['NAXIS1'] != self.nfiber:
            logging.error('HDUs containing Flames/ARGUS data must have {0} fibers (NAXIS1).'.format(self.nspaxel))
            return False
        return True

    def get_spectrum(self, i, return_var=False, **kwargs):
        """
        This method returns a single spectrum (i.e. the content of a single
        spaxel) from the data.

        Parameters
        ----------
        i : int
            The index of the spaxel for which the spectrum should be returned
        return_var : boolean, optional
            A flag indicating if the variances should be returned along with
            the data.
        kwargs
            Any additional keyword argument is passed on to the call of the
            parent class.

        Returns
        -------
        spectrum : ndarray
            The spectrum in the requested spaxel. Note that if a bad-pixel
            mask is available, flagged pixels are set to NaN but still
            included in the data.
        variances : ndarray or None
            If the return_var parameter was set, this array will contain the
             variances of the spectrum in the requested spaxel.
        """
        if i < -self.nspaxel or i >= self.nspaxel:
            logging.error("Spaxel index must be <{0} but is {1}.".format(self.nspaxel, i))
            return None, None

        mask = self.maskhdu.data[:, i] if self.maskhdu is not None else np.zeros(self.wave.shape, dtype=np.bool)

        return (np.where(mask, np.nan, self.datahdu.data[:, i]),
                np.where(mask, np.nan, self.varhdu.data[:, i]) if return_var and self.varhdu is not None else None)

    def get_layer(self, i, return_residuals=False, return_grid=False, return_var=False, **kwargs):
        """
        This method returns a single layer (i.e. the content of all spaxels
        with a given wavelength) from the data.

        Parameters
        ----------
        i : int
            The index of the layer for which the content should be returned
        return_residuals : boolean, optional
            A flag indicating if the residuals from a previous analysis should
            be returned instead of the data.
        return_grid : boolean, optional
            A flag indicating if the layer should be returned as 1D or 2D
            array. The default is to return a 1D array.
        return_var : boolean, optional
            A flag indicating if the variances should be returned along with
            the data
        kwargs
            Any additional keyword argument is passed on to the call of the
            parent class.

        Returns
        -------
        values : ndarray
            The data values (ore residuals) in the requested layer. Note that
            the data will contain Argus.nspaxel pixels. Values for calibration
            or sky fibres are not returned, values of masked pixels are set to
            NaN.
        variances : ndarray or None
            If return_var was set, this array will contain the variances of
            the data values in the requested layer. It has the same shape as
            the data array.
        """
        # check if data or residuals (if any) should be returned
        if not return_residuals or self.residualshdu is None:
            data = self.datahdu.section[i][self.isscience]
        else:
            data = self.residualshdu.section[i][self.isscience]

        # check if variances should be returned
        var = self.varhdu.section[i][self.isscience] if return_var and self.varhdu is not None else None

        # check if mask is available. If so, apply it. Note that the bad pixels are not removed but set to NaN.
        if self.maskhdu is not None:
            mask = self.maskhdu.section[i][self.isscience] if self.maskhdu is not None else None
            data[mask > 0] = np.nan
            if var is not None:
                var[mask > 0] = np.nan

        if not return_grid:
            return data, var
        else:
            return self.make_image(data), self.make_image(var) if return_var else None

    def search_configuration(self, hdu, **kwargs):
        """
        This method is designed to update the configuration of the current
        Argus instance using specific header keywords from a given HDU.

        Parameters
        ----------
        hdu : a FITS HDU
            The HDU containing the header to be checked for the keywords.
        kwargs
            Used for consistency.
        """
        super(Argus, self).search_configuration(hdu)

        if 'CRVAL2' in hdu.header:
            self.wcs.wcs.crval[-1] = hdu.header['CRVAL2']
        if 'CDELT2' in hdu.header:
            self.wcs.wcs.cd[-1, -1] = hdu.header['CDELT2']
        if 'CRPIX2' in hdu.header:
            self.wcs.wcs.crpix[-1] = hdu.header['CRPIX2']
        if 'CUNIT2' in hdu.header:
            self.wcs.wcs.cunit[-1] = hdu.header['CUNIT2']
        if 'ESO ADA POSANG' in hdu.header:  # check if information on position angle that was used can be found.
            self.angle = hdu.header['ESO ADA POSANG'] + 90
        if 'ESO INS1 OPTI1 POS' in hdu.header:  # check if information on scale that was used can be found
            scalepos = hdu.header['ESO INS1 OPTI1 POS']
            if scalepos == "POS_1":
                self.scale = 0.52
            elif scalepos == "POS_1_67":
                self.scale = 0.3
            else:
                logging.error('Unknown value for pixel scale keyword ESO INS1 OPTI1 POS: {0}'.format(scalepos))

    def _make_cd(self):
        """
        This method adds a CD matrix to the WCS instance defined inside the
        Argus instance. To do so, it uses the information of the pixel
        scale and the rotation angle of the instrument (if available).
        """
        # The orientation of ARGUS on the sky is as follows. For a position angle of 0 degrees, the x axis is aligned
        # north and the y-axis is aligned east. Rotations are are counted clockwise
        cdnew = np.array([[0., self.scale/3600., 0.], [self.scale/3600., 0., 0.], [0., 0., self.wcs.wcs.cd[2, 2]]])
        if self.angle is not None and self.angle != 0:
            c, s = np.cos(np.radians(-self.angle)), np.sin(np.radians(-self.angle))
            rot = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
            self.wcs.wcs.cd = np.dot(rot, cdnew)
        else:
            self.wcs.wcs.cd = cdnew

    def slice_for_layer(self, i):
        """
        This function provides a slice that can be applied to the full data
        array of any HDU to extract an individual layer from it.

        Parameters
        ----------
        i : int
            The index of the layer.

        Returns
        -------
        slice : a slice object
            The slice that returns the layer if applied to the data.
        """
        return [i * np.ones((self.nspaxel,), dtype=np.int16), np.flatnonzero(self.isscience)]


class SimIfu(Argus):
    """
    This class was designed to handle simulated ARGUS data as it was provided
    by N. Luetzgendorf for a paper project. The data had basically the same
    format as ARGUS data, only the field of view was 50x50 spaxels.
    """
    name = "simifu"
    shape = (50, 50)

    _angle = 0  # rotation angle was fixed
    _scale = 0.52  # spaxel scale was fixed to 0.52"/spaxel

    def __init__(self, **kwargs):
        """
        Initialize an instance of the SimIfu class.

        Parameters
        ----------
        kwargs are passed on to the call of the Argus supper-class.

        Returns
        -------
        The newly created instance.
        """
        # initialize transformation from (x,y)-coordinates to spaxels
        transform = np.vstack((np.arange(2500, dtype=np.int16) // 50, np.mod(np.arange(2500, dtype=np.int16), 50))).T
        kwargs['transform'] = transform

        super(SimIfu, self).__init__(**kwargs)
