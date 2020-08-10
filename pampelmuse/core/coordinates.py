"""
coordinates.py
==============
Copyright 2013-2020 Sebastian Kamann

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
class pampelmuse.core.coordinates.Transformation

Purpose
-------
The module serves as an interface for loading, modifying, and storing the
coordinate transformation that is used by PampelMuse.

The class 'Transformation' is designed to provide a convenient way to
transform from input to IFU coordinates. It is used as a super-class by the
'Sources'-class of PampelMuse. It is a child-class of
pampelmuse.core.parameters.Parameters that defines a more general framework
for dealing with parameters within PampelMuse.

Currently the spaxel coordinates for a layer are determined from the reference
coordinates via an affine transformation including six parameters, A, B, C, D,
x0, y0:

(i)  x_spaxel = A*x_ref + C*y_ref + x0
(ii) y_spaxel = D*y_ref - B*x_ref + y0

or in matrix notation:

r_spaxel = [[A C] [-B D]] * r_ref + r0

The inverse transformation can be obtained by matrix inversion:

[[A C] [-B D]]^-1 = 1./(AD + BC) [[D -C] [B A]]

The following definitions are made:

A' = D/(AD + BC)
B' = -B/(AD + BC)
C' = -C/(AD + BC)
D' = A/(AD + BC)
x0' = -(D*x0 - C*y0)/(AD + BC)
y0' = -(B*x0 + A*y0)/(AD + BC)

This results in the inverse transformation:
(iii) x_ref = A'*x_spaxel + C'*y_spaxel + x0'
(iv)  y_ref = D'*y_spaxel - B'*x_spaxel + y0'

Latest Git revision
-------------------
2020/02/12
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
from ..core.parameters import Parameters


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20200212


logger = logging.getLogger(__name__)


class Transformation(Parameters):
    """
    The class handles the coordinate transformation from reference to IFS
    coordinates that is used by PampelMuse.
    
    The coordinates transformation (u, v) -> (x, y) is defined as
    
        x = A*u + C*v + x0,
        y =-B*u + D*v + y0,
    
    where (x, y) are the IFS pixel coordinates and (u,v) are the reference
    pixel coordinates. A, B, C, D, x0, and y0 are the parameters of the
    transformation.
    
    Note that earlier versions used a simpler version of the coordinate
    transformation, with C=B and D=A. This is still supported as input format.
    """
    PARAMETERS = ["A", "B", "C", "D", "x0", "y0"]

    def __init__(self, data=None, free=None, **kwargs):
        """
        Initialize an instance of the 'Transformation`-class.
       
        Parameters
        ----------
        data : pandas.DataFrame, optional
            The data used to initialize a new instance should be provided as a
            pandas DataFrame containing one parameter per column. By default,
            all parameters are set to NaN.
        free : array_like,
            A boolean array containing a flag for each parameter in 'data'
            indicating if the parameter is allowed to vary in a fit. By
             default, all parameters are free.
        kwargs
            Any keyword arguments that are provided are passed on to the call
            of the initialization method of the super-class.
        """
        # if no data provided, initialise all parameters to NaN
        if data is None:
            data = pd.DataFrame([[np.nan]*len(self.PARAMETERS)], columns=self.PARAMETERS)

        # call initialization method of super-class, make sure order is A, B, C, D.
        super(Transformation, self).__init__(data=data, free=free, **kwargs)

        # if only four parameters are defined, the transformation is conformal and needs to be converted
        if self.defined.sum() == 4:
            logger.warning('Initialising coordinate transformation using outdated parameters.')

            # replace C=x0 and D=y0
            self.data.rename(columns={'C': 'x0', 'D': 'y0'}, inplace=True)

            # copy B to C and A to D
            self.data[('C', 'fit')] = self.data[('B', 'fit')]
            self.data[('C', 'value')] = self.data[('B', 'value')]
            self.data[('D', 'fit')] = self.data[('A', 'fit')]
            self.data[('D', 'value')] = self.data[('A', 'value')]

            # update information on free parameters
            self.free.rename(index={'C': 'x0', 'D': 'y0'}, inplace=True)
            self.free['C'] = self.free['B']
            self.free['D'] = self.free['A']

        # check if all required parameters are available
        for name in self.PARAMETERS:
            if name not in self.names:
                logger.error("Missing required parameter '{0}' of coord. transformation.".format(name))

    @classmethod
    def from_coordinates(cls, xy_in, xy_out, fixed=None):
        """
        Initialize a new coordinate transformation using the provided input
        (reference) and output (IFS) coordinates in a least squares fit.

        Both, input and output coordinates should be provided as pandas data
        frames. The input coordinates should be formatted such that 'x' and
        'y' are individual columns of the data frame and the source
        identifiers are used as row indices. The data frame with th output
        coordinates should contain the identifiers as individual columns
        (each with x and y as second level of a pandas MultiIndex), while the
        rows encode any dispersion information. If the data frame with the
        output coordinates has more than one row, a coordinate transformation
        will be determined for each row.

        Parameters
        ----------
        xy_in : pandas.DataFrame
            The input coordinates. The rows of the data frame must use the
            source identifiers as indices and the x- and y-coordinates should
            be available as columns 'x' and 'y'.
        xy_out : pandas.DataFrame
            The output coordinates. The columns of the data frame must contain
            the source identifiers and the x- and y-coordinates as a pandas
            MultiIndex. If the data frame contains N rows, the transformation
            will be calculated for each row independently.
        fixed : pandas.DataFrame, optional
            If any parameters of the coordinate transformation should be fixed
            to given values, the fixed parameters and their values can be
            provided as a pandas DataFrame using the same indices as 'xy_out'
            and the fixed parameters as column names.

        Raises
        ------
        IOError
            If input data do not allow the determination of the coordinate
            transformation.
        """
        logger.info("Initialising coordinate transformation from provided data...")

        # perform sanity checks on input data
        if not isinstance(xy_in, pd.DataFrame):
            raise IOError('Parameter "xy_in" must be pandas DataFrame, not {0}.'.format(type(xy_in)))
        if 'x' not in xy_in.columns or 'y' not in xy_in.columns:
            raise IOError('Cannot find coordinates in input coordinates ("xy_in" misses column "x" or "y").')
        if not cls._validate_input(xy_out):
            raise IOError('Data provided for parameter xy_out is invalid. Check previous errors.')

        # find sources that overlap between input and output coordinates
        common_ids = np.intersect1d(xy_in.index, np.unique(xy_out.columns.get_level_values(0)), assume_unique=True)
        if len(common_ids) < 4:
            raise IOError('Need at least 4 sources to fit coordinate transformation.')

        # for the fitting, plain numpy arrays are used
        in_data = np.zeros((len(common_ids), 2), dtype=np.float32)
        out_data = np.zeros((xy_out.shape[0], len(common_ids), 2), dtype=np.float32)

        for i, source_id in enumerate(common_ids):
            in_data[i, 0] = xy_in.at[source_id, 'x']
            in_data[i, 1] = xy_in.at[source_id, 'y']

            out_data[:, i, 0] = xy_out[(source_id, 'x')]
            out_data[:, i, 1] = xy_out[(source_id, 'y')]

        # convert format for fixed parameters to dictionary
        _fixed = {}
        if fixed is not None:
            for parameter in fixed.columns.names:
                _fixed[parameter] = fixed[parameter].values

        # perform the actual fit and initialize the values of the parameters of the coordinate transformation.
        _data, residuals = cls.quick_fit(in_data, out_data, fixed=_fixed)

        data = pd.DataFrame(_data, index=xy_out.index, columns=cls.PARAMETERS)

        # The main results of the computation ar presented as logging.INFO messages
        logger.info("Fitted coordinate transformation for {0} image(s).".format(out_data.shape[0]))

        logger.info("Best-fit parameters:")

        for i, parameter in enumerate(cls.PARAMETERS):
            logger.info("   {0} = {1:8.3f}".format(parameter, np.nanmean(data[parameter])))

        logger.info('RMS scatter between predicted and actual coordinates: STD(dx)={0:.2g}, STD(dy)={1:.2g}'.format(
            np.nanstd(residuals[:, :, 0]), np.nanstd(residuals[:, :, 1])))

        logger.debug("Offsets of individual sources:")
        logger.debug("       ID     dx     dy")

        for i, source_id in enumerate(common_ids):
            logger.debug("   {0:6d}  {1:+5.2f}  {2:+5.2f}".format(source_id,
                                                                  np.nanmean(residuals[:, i, 0]),
                                                                  np.nanmean(residuals[:, i, 1])))

        # define which parameters are free/fixed
        free = np.ones((len(cls.PARAMETERS)), dtype=np.bool)
        if fixed is not None:
            for parameter in fixed.columns.names:
                free[cls.PARAMETERS.index(parameter)] = False

        # initialize new instance
        return cls(data=data, free=free)

    @staticmethod
    def _validate_input(data_frame):
        """
        An internal method that checks if a provided data frame has a valid
        format, i.e. is provided with MultiIndex columns containing 'x' and
        'y' as second level indices.
        
        Parameters
        ----------
        data_frame : pandas.DataFrame
           The data frame that should be checked for consistency.
        
        Returns
        -------
        bool
            Flag indicating if the provided data frame is valid.
        """
        # check if data frame
        if not isinstance(data_frame, pd.DataFrame):
            logger.error('Expected pandas DataFrame, got "{0}"'.format(type(data_frame)))
            return False
        # make sure columns are a pandas MultiIndex
        if not isinstance(data_frame.columns, pd.MultiIndex):
            logger.error('Columns of provided data frame must be pandas.MultiIndex instance.')
            return False
        # make sure columns have the right depths
        if len(data_frame.columns.levshape) < 2:
            logger.error('Expected MultiIndex with 2 levels, got {0}.'.format(data_frame.columns.levshape))
            return False
        # check if required parameters were provided
        if 'x' not in data_frame.columns.levels[1].values or 'y' not in data_frame.columns.levels[1].values:
            logger.error('Second level of MultiIndex must contain "x" and "y".')
            return False
        return True

    @staticmethod
    def quick_fit(in_data, out_data, fixed=None):
        """
        Obtain the parameters of a coordinate transformation from a least
        squares fit to the provided coordinates.
        
        The method basically solves the equation A*x=b in a least squares
        sense for x=[A, B, C, D, x0, y0]. The input coordinates are written to
        A while the output coordinates constitute b.
        
        Parameters
        ----------
        in_data : array_like, shape (N, 2)
            The input coordinates (u and v) for a sample of N>=2 sources.
        out_data : array_like, shape (N, 2) or (K, N, 2)
            The output coordinates (x and y) for the same sample of sources
            as in 'in_data'. If 'out_data' is 3dimensional, the transformation
            is determined for the K spectral layers.
        fixed : dict, optional
            If one or more of the parameters of the coordinate transformation
            should be fixed to specific values, provide them as a dictionary
            of 'parameter_name: value' pairs. Note that if 'out_data' is 3D,
            the values must be specified for each of the K spectral layers.

        Returns
        -------
        best_fit : nd_array, shape (K, 6 - n_fixed)
            The parameter values obtained in the least squares fit. Note that
            if any 'fixed' parameters were provided, they will not appear in
            the array that is returned.
            IMPORTANT: The array will be 2D even if the 'out_data' parameter
            was provided as a 2D array.
        residuals : nd_array, shape (K, N, 2)
            The offsets along x and y for each provided source.
            IMPORTANT: The array will be 2D even if the 'out_data' parameter
            was provided as a 2D array.

        Raises
        ------
        IOError
            if N <= number of requested parameters
        """
        # initialize empty dictionary if no fixed parameters were provided
        if fixed is None:
            fixed = {}

        # n is the number of provided sources, N=2n is the number of columns of the matrix A
        # (each source contributes two rows)
        n = in_data.shape[0]
        if (2*n) <= (6 - len(fixed)):
            logger.critical('Number of coordinates must be greater than number of parameters.')
            raise IOError('Not enough sources provided.')

        # fill matrix using input coordinates, the matrix equation will contain alternating equations for x and y,
        # equations involving output x coordinates in even rows (0,2,...N-2) and involving output y coordinates in
        # odd rows (1,3,...,N-1)
        in_matrix = np.zeros((2*n, 6), dtype=np.float32)
        in_matrix[0:2*n:2, 0] = in_data[:, 0]
        in_matrix[0:2*n:2, 2] = in_data[:, 1]
        in_matrix[0:2*n:2, -2] = 1.
        in_matrix[1:2*n:2, 1] = -in_data[:, 0]
        in_matrix[1:2*n:2, 3] = in_data[:, 1]
        in_matrix[1:2*n:2, -1] = 1.

        # make sure input is available as 3D numpy array (if needed, create pseudo-dispersion axis)
        if len(out_data.shape) == 2:
            out_data = out_data[np.newaxis, ...]
        # collapse last two axes (i.e. x and y)
        b = out_data.reshape((-1, out_data.shape[-1] * out_data.shape[-2]))

        # handle fixed parameters, if any:
        # The idea is to multiply each fixed parameter with the corresponding column of the matrix,
        # subtract the results from b, and exclude the column from the fit
        use_column = np.ones(in_matrix.shape[1], dtype=np.bool)
        for parameter, value in fixed.items():
            value = np.atleast_1d(value)
            if value.size != b.shape[0]:
                logger.error('Fixed parameter {0} in coord. transf. has invalid shape.'.format(parameter))
            else:
                i = Transformation.PARAMETERS.index(parameter)
                b -= value[:, np.newaxis]*in_matrix[:, i]
                use_column[i] = False

        # some versions of numpy seem to have issues if the input to np.linalg.lstsq contains NaNs.
        is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
        if is_conda or np.__version__ < '1.14' or getattr(np.__config__, 'mkl_info', {}):
            if np.isnan(b).any():
                logger.warning('Replacing {0} NaNs using linear interpolation.'.format(np.isnan(b).sum()))
                for i in range(b.shape[1]):
                    mask = np.isnan(b[:, i])
                    b[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), b[~mask, i])

        # for linear fit, the dispersion axis of b must be the last axis, so b needs to be transposed
        best_fit = np.linalg.lstsq(in_matrix[:, use_column], b.T, rcond=None)[0]
        # -> returns array with shape (6 - n_fixed, K)

        # create residuals
        # for dot-product, last dimension of 1st argument must be equal to first dimension of 2nd argument
        residuals = b - np.dot(best_fit.T, in_matrix.T[use_column])
        # (K, 6 - n_fixed) DOT (6 - n_fixed, 2*n) -> creates array with shape (K, 2*n)

        # when returning residuals, transpose residuals array and split x- and y-axes in residuals array again
        return best_fit.T, residuals.reshape((-1, n, 2))

    @property
    def z_matrix(self):
        """
        Returns
        -------
        z_matrix : numpy.array
             The matrix will have a shape of (K, 2, 2), where K is the number
             of defined layers.
        """
        # create diagonal matrix with trace [x0, y0] for each layer (index 0 is spectral one)
        z = np.zeros((self.n_dispersion, 2, 2), dtype=np.float32)
        z[:, 0, 0] = self.data[('x0', 'value' if not self.fitted['x0'] else 'fit')]  # parameter x0
        z[:, 1, 1] = self.data[('y0', 'value' if not self.fitted['y0'] else 'fit')]  # parameter y0

        return z

    @property
    def f_matrix(self):
        """
        Returns
        -------
        f_matrix : numpy.array
             The matrix will have a shape of (K, 2, 2), where K is the number
             of defined layers.
        """
        # create matrix [[A C] [-B D]] for each layer (index 0 is spectral one)
        f = np.zeros((self.n_dispersion, 2, 2), dtype=np.float32)
        f[:, 0, 0] = self.data[('A', 'value' if not self.fitted['A'] else 'fit')]
        f[:, 0, 1] = self.data[('C', 'value' if not self.fitted['C'] else 'fit')]
        f[:, 1, 0] = -self.data[('B', 'value' if not self.fitted['B'] else 'fit')]
        f[:, 1, 1] = self.data[('D', 'value' if not self.fitted['D'] else 'fit')]

        return f

    def __call__(self, xy_in, layers=None):
        """
        Transform a set of reference coordinates to IFS coordinates using the
        current values of the coordinate transformation.
        
        Parameters
        ----------
        xy_in : pandas.DataFrame
            The input coordinates. The rows of the data frame must use the
            source identifiers as indices and the x- and y-coordinates as
            should be available as columns 'x' and 'y'.
        layers : array-like or integer, optional
            The indices of the layers for which the IFS coordinates should be
            loaded. If none are provided, the coordinates are loaded for the
            whole wavelength range.

        Returns
        -------
        pandas DataFrame
            The transformed coordinates for K layers and N sources. The layers
            represent the rows of the data frame while the sources and their x
            and y-coordinates are stored as MultiIndex columns.
        """
        # perform sanity checks on input data
        if not isinstance(xy_in, pd.DataFrame):
            logger.error('Parameter "xy_in" must be pandas DataFrame, not {0}.'.format(type(xy_in)))
            return
        if 'x' not in xy_in.columns or 'y' not in xy_in.columns:
            logger.error('Cannot find coordinates in input coordinates ("xy_in" misses column "x" or "y").')
            return

        # check if requested layers are valid
        if layers is not None and not isinstance(layers, (int, np.integer, list, np.ndarray)):
            logger.error('Parameter "layers" must be integer or array-like, not {0}.'.format(type(layers)))
            return
        elif layers is None:
            layers = np.arange(self.n_dispersion)
        else:
            layers = np.atleast_1d(layers)
            # layer selection must be consecutive
            if (np.diff(layers) != 1).any():
                logger.error('Layer selection must be consecutive.')

        _xy_in = xy_in[['x', 'y']].values  # -> has shape (N, 2)
        _xy_out = np.dot(self.z_matrix[layers], np.ones(_xy_in.T.shape)) + np.dot(self.f_matrix[layers], _xy_in.T)
        # -> creates array of shape (K, 2, N)

        # when creating data frame, need to swap last to axes.
        _xy_out = _xy_out.swapaxes(-2, -1)

        # finally, x- and y-axes need to be merged again
        return pd.DataFrame(_xy_out.reshape((_xy_out.shape[0], -1)), index=self.data.index[layers],
                            columns=pd.MultiIndex.from_product([xy_in.index, ['x', 'y']]))

    def inverse(self, xy_out):
        """
        Given a set of output coordinates, the method will apply the inverse
        coordinates transformation to obtain the input coordinates

        Parameters
        ----------
        xy_out

        Returns
        -------

        """
        if not self._validate_input(xy_out):
            logger.error('Data provided for parameter xy_out is invalid. Check previous errors.')
            return

        xy_in = xy_out.copy()

        for i, row in xy_out.iterrows():
            xy_out_i = row.values.reshape((-1, 2))

            xy_in_i = np.dot(np.linalg.inv(self.f_matrix[i]), xy_out_i.T - np.dot(self.z_matrix[i],
                                                                                  np.ones(xy_out_i.shape).T))
            xy_in.loc[i] = xy_in_i.T.reshape((1, -1))
        return xy_in

    def info(self, names=None):
        """
        The method pretty-prints some information about the status of the
        coordinate transformation parameters out on the screen.
        
        names : list, optional
            The parameter can be used to restrict the output to a subset of
            parameters or to print them in a given order. By default, use the
            order defined by coordinates.PARAMETERS.
        """
        logger.info("Summary of coordinate transformation instance:")

        if names is None:
            names = self.PARAMETERS

        # call info method of super class. Provide parameter names to get order right
        super(Transformation, self).info(names=names)

    def make_hdu(self, include_fit=True, header_keys=None):
        """
        Write the current values of the parameters of the coordinate
        transformation to a fits.BinTableHDU instance.
        
        The method is basically a wrapper of the 'make_hdu' instance
        defined in the Parameters-class and adds some header keywords that
        provide further information about the coordinate transformation in
        use.
        
        Parameters
        ----------
        include_fit : boolean, optional
            Whether or not the fits to the parameters should  be included
            in the table.
        header_keys : dict, optional
            Any additional keywords that should be included in the FITS
            header of the  table.
            
        Returns
        -------
        astropy.io.fits.BinTableHDU instance
            The HDU containing the parameters.
        """
        if header_keys is None:
            header_keys = {}
        header_keys['EXTNAME'] = 'POSPARS'

        logger.info("Saving coordinate transformation in BinTableHDU '{0}'...".format(header_keys['EXTNAME']))

        return super(Transformation, self).make_hdu(include_fit=include_fit, header_keys=header_keys)
