"""
coordinates.py
==============
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
function pampelmuse.core.coordinates.my_lsq_sigma
class pampelmuse.core.coordinates.Transformation

Purpose
-------
The module serves as an interface for loading, modifying, and storing the
coordinate transformation that is used by PampelMuse.

The function 'my_lq_sigma' is designs to perform a linear least squares fit
of type [A*x = b] for x when besides the data uncertainties of the data are
provided.

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

Latest SVN revision
-------------------
418, 2017/04/12
"""
import logging
import numpy as np
import pandas as pd
from ..core.parameters import Parameters


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 418


logger = logging.getLogger(__name__)


def my_lsq_sigma(a, b, sigma=None):
    """
    The routine is a wrapper around the the numpy.linalg.lstsq-routine. It is
    designed as an extension that allows the user to also provide uncertainties
    together with the data when minimizing |Ax-b|.

    Parameters
    ----------
    a : array_like, shape (M, N)
        "Coefficient" matrix.
    b : array_like, shape (M,) or (K, M)
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    sigma : array_like, same shape as b, optional
        Uncertainties associated to the data provided in `b`.

    Returns
    -------
    x : ndarray, shape (N,) or (K, N)
        Least-squares solution.  The shape of `x` depends on the shape of
        `b`.
    """
    # implement uncertainties (if provided)
    if sigma is not None:
        _A = (1. / (sigma[..., np.newaxis] / a))
        _b = b / sigma
    else:
        _A = a
        _b = b
    slc = ~np.isnan(_b)

    if b.ndim == 1:
        if slc.sum() > _A.shape[-1]:
            x = np.linalg.lstsq(_A[slc], _b[slc])[0]
        else:
            x = np.zeros(_A.shape[-1], dtype=np.float32)
            x[:] = np.nan
    else:
        x = np.zeros((_b.shape[0], _A.shape[-1]), dtype=np.float32)
        for i in range(_b.shape[0]):
            if slc[i].sum() > _A.shape[-1]:
                if sigma is not None:
                    x[i] = np.linalg.lstsq(_A[i, slc[i]], _b[i, slc[i]])[0]
                else:
                    x[i] = np.linalg.lstsq(_A[slc[i]], _b[i, slc[i]])[0]
            else:
                x[i] = np.nan
    return x


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

    def __init__(self, data, free=None, **kwargs):
        """
        Initialize an instance of the 'Transformation`-class.
       
        Parameters
        ----------
        data : pandas.DataFrame
            The data used to initialize a new instance should be provided as a
            pandas DataFrame containing one parameter per column.
        free : array_like,
            A boolean array containing a flag for each parameter in 'data'
            indicating if the parameter is allowed to vary in a fit
        kwargs
            Any keyword arguments that are provided are passed on to the call
            of the initialization method of the super-class.
        """
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
        frames, containing the sources as individual columns (each with x and
        y as second level of a pandas MultiIndex). If the data frame with the
        output coordinates has more than one row, a coordinate transformation
        will be determined for each row.

        Parameters
        ----------
        xy_in : pandas.DataFrame
            The input coordinates. The columns of the data frame must contain
            the source identifiers and the x- and y-coordinates as a pandas
            MultiIndex. Currently only the first row of the data frame will
             be used.
        xy_out : pandas.DataFrame
            The output coordinates. Again, the columns of the data frame must
            contain the source identifiers and the x- and y-coordinates as a
            pandas MultiIndex. If the data frame contains N rows, the
            transformation will be calculated for each row independently.
        fixed : pandas.DataFrame, optional
            If any parameters of the coordinate transformation should be fixed
            to given values, the fixed parameters and their values can be
            provided as a pandas DataFrame using the same indices as 'xy_out'
            and the fixed parameters as column names.
        """
        logger.info("Initialising coordinate transformation from provided data...")

        # perform sanity checks on input data
        for name, df in {'xy_in': xy_in, 'xy_out': xy_out}:
            if not isinstance(df, pd.DataFrame):
                logger.error('Parameter {0} must be pandas DataFrame, not "{1}"'.format(name, type(xy_in)))
            # make sure data frame has right format
            if len(df.columns.levshape) != 2:
                logger.error('Invalid column format for parameter "{0}".'.format(name))
                logger.error('MultiIndex must have depth=2.')
                return
            elif df.columns.levshape[1] != 2:
                logger.error('Invalid column format for parameter "{0}".'.format(name))
                logger.error('Second level of MultiIndex must have length=2')
                return
            elif 'x' not in df.columns.levels[1].values or 'y' not in df.columns.levels[1].values:
                logger.error('Invalid column format for parameter "{0}".'.format(name))
                logger.error('Second level of MultiIndex must contain "x" and "y".')
                return

        # find sources that overlap between input and output coordinates
        common_ids = np.intersect1d(xy_in.columns.levels[0], xy_out.columns.levels[0], assume_unique=True)
        if len(common_ids) < 4:
            logger.error('Need at least 4 sources to fit coordinate transformation.')
            return

        # for the fitting, plain numpy arrays are used
        in_data = np.zeros((len(common_ids), 2), dtype=np.float32)
        out_data = np.zeros((xy_out.shape[0], len(common_ids), 2), dtype=np.float32)

        for i, source_id in enumerate(common_ids):
            in_data[i, 0] = xy_in[(source_id, 'x')]
            in_data[i, 1] = xy_in[(source_id, 'y')]

            out_data[:, i, 0] = xy_out[(source_id, 'x')]
            out_data[:, i, 1] = xy_out[(source_id, 'y')]

        # perform the actual fit and initialize the values of the parameters of the coordinate transformation.
        _data, residuals = cls.quick_fit(in_data, out_data, fixed=fixed)

        data = pd.DataFrame(_data, index=xy_out.index, columns=cls.PARAMETERS)

        # The main results of the computation ar presented as logging.INFO messages
        logger.info("Fitted coordinate transformation for {0} image(s).".format(out_data.shape[0]))

        logger.info("Best-fit parameters:")

        for i, parameter in enumerate(cls.PARAMETERS):
            logger.info("   {0} = {1:8.3f}".format(parameter, np.nanmean(data[parameter])))

        logger.info("Offsets of individual sources:")
        logger.info("       ID     dx     dy")

        for i, source_id in enumerate(common_ids):
            logger.info("   {0:6d}  {1:+5.2f}  {2:+5.2f}".format(source_id,
                                                                 np.nanmean(residuals[:, i, 0]),
                                                                 np.nanmean(residuals[:, i, 1])))

        # define which parameters are free/fixed
        free = np.ones((len(cls.PARAMETERS)), dtype=np.bool)
        if fixed is not None:
            for parameter in fixed.columns.names:
                free[cls.PARAMETERS.index(parameter)] = False

        # initialize new instance
        return cls(data=data, free=free)

    @property
    def valid(self):
        """
        Returns
        -------
        bool
           Flag indicating if all parameters of the coordinate transformation
           have valid data.
        """
        return self.defined.all()

    @staticmethod
    def quick_fit(in_data, out_data, fixed=None):
        """
        A short-cut for calculating a new coordinate transformation, without
        any checking of the input. The method basically solves the equation
        A*x=b in a least squares sense for x=[A,B,C,D,...]. The input coordinates
        are written to A while the output coordinates constitute b.
        
        NOTE: Deviating from the behaviour of the method 'fit_to_data', the parameters
        of the transformation are not re-initialized here. Instead, the result of the
        least squared fit is returned.
        
        Parameters
        ----------
        in_data : array_like, shape (N, 2)
            The input coordinates (u and v) for a sample of N>=2 sources.
        out_data : array_like, shape (N, 2) or (M, N, 2)
            The output coordinates (x and y) for the same sample of sources
            as in 'in_data'. If 'out_data' is 3dimensional, the transformation
            is determined for the M spectral layers.
        fixed : dict, optional
            If one or more of the parameters of the coordinate transformation
            should be fixed to specific values, provide them as a dictionary
            of 'parameter_name: value' pairs. Note that if outdata is 3D, the
            values must be specified for each of the M spectral layers.

        Returns
        -------
        ndarray, shape (M, 6)
            The results of the least squares fit. Consult the documentation of
            scipy.linalg.lstsq for more information on the output.

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
            logger.error('Number of coordinates must be greater than number of parameters.')
            return None

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
        out_data = np.atleast_3d(out_data)
        # collapse last two axes
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

        # for linear fit, the dispersion axis of b must be the last axis
        best_fit = np.linalg.lstsq(in_matrix[:, use_column], b.T)[0]  # returns array with shape (6, M)
        residuals = b - np.dot(best_fit, in_matrix.T)  # creates array with shape (M, 2*n)

        # call fitting function
        return best_fit.T, residuals.reshape((-1, n, 2))  # split x- and y-axes in residuals array again

    @property
    def zmatrix(self):
        """
        Returns
        -------
        zmatrix : numpy.array
             The matrix will have a shape of (K, 2, 2), where K is the number
             of defined layers.
        """
        # create diagonal matrix with trace [x0, y0] for each layer (index 0 is spectral one)
        _m = np.zeros((self.n_dispersion, 2, 2), dtype=np.float32)
        _m[:, 0, 0] = self.data[:, -2, int(self.fitted[-2])]  # parameter x0
        _m[:, 1, 1] = self.data[:, -1, int(self.fitted[-1])]  # parameter y0

        # if np.isnan(_m).any():
        #     logging.warning("Coordinate transformation contains interpolated values.")
        #     for i in xrange(2):
        #         for j in xrange(2):
        #             valid = ~np.isnan(_m[:, i, j])
        #             fct = interpolate.InterpolatedUnivariateSpline(np.flatnonzero(valid), _m[valid, i, j],  k=2)
        #             _m[~valid, i, j] = fct(np.flatnonzero(~valid))
        return _m

    @property
    def fmatrix(self):
        """
        Returns
        -------
        fmatrix : numpy.array
             The matrix will have a shape of (K, 2, 2), where K is the number
             of defined layers.
        """
        # create matrix [[A C] [-B D]] for each layer (index 0 is spectral one)
        _f = np.zeros((self.n_dispersion, 2, 2), dtype=np.float32)
        _f[:, 0, 0] = self.data[:, 0, int(self.fitted[0])]
        _f[:, 0, 1] = self.data[:, 2, int(self.fitted[2])]
        _f[:, 1, 0] = -self.data[:, 1, int(self.fitted[1])]
        _f[:, 1, 1] = self.data[:, 3, int(self.fitted[3])]

        # if np.isnan(_f).any():
        #     logging.warning("Coordinate transformation contains interpolated values.")
        #     for i in xrange(2):
        #         for j in xrange(2):
        #             valid = ~np.isnan(_f[:, i, j])
        #             fct = interpolate.InterpolatedUnivariateSpline(np.flatnonzero(valid), _f[valid, i, j],  k=2)
        #             _f[~valid, i, j] = fct(np.flatnonzero(~valid))
        return _f

    def __call__(self, xy):
        """
        
        Parameters
        ----------
        xy : array_like, shape (N, 2)
            The input coordinates for N sources.

        Returns
        -------
        numpy.array, shape (K, N, 2)
            The transformed coordinates for K layers and N sources.
        """
        temp = np.inner(self.zmatrix, np.ones(xy.shape, dtype=xy.dtype)) + np.inner(self.fmatrix, xy)

        # before returning results, need to swap last to axes.
        return temp.swapaxes(-2, -1)

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
