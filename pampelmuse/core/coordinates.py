"""
coordinates.py
==============

Provides
--------
function pampelmuse.core.coordinates.my_lsq_sigma
class pampelmuse.core.coordinates.Transformation

Purpose
-------
The module is designed as an interface for loading, modifying, and storing the
coordinate transformation that is used by PampelMuse. The class 'Transformation'
is designed to provide a convenient way to transform from input to IFU coordi-
nates. It is used as a super-class by the 'Sources'-class of PampelMuse.


The function 'my_lq_sigma' is designs to perform a linear least squares fit
of type [A*x = b] for x when besides the data uncertainties of the data are
provided.


The class 'Transformation' inherits from the class pampelmuse.core.parameters.Parameters
that defines a more general framework for dealing with parameters within PampelMuse.

Currently the spaxel coordinates for a layer are determined from the reference
coordinates via an affine transformation including six parameters, A, B, C, D, x0, y0:

(i)  x_spaxel = A*x_ref + C*y_ref + x0
(ii) y_spaxel = D*y_ref - B*x_ref + y0

# TODO: update the following
#OUTDATED#The inverse transformation is given by the following formulae:
#OUTDATED#(iii) x_ref = A'*x_spaxel + B'*y_spaxel + C'
#OUTDATED#(iv)  y_ref = A'*y_spaxel - B'*x_spaxel + D'
#OUTDATED#
#OUTDATED#where A' = A/(A*A+B*B), B' = -B/(A*A+B*B), C' = (B*D-A*C)/(A*A+B*B), and
#OUTDATED#D' = -(A*D+B*C)/(A*A+B*B)


Latest SVN revision
-------------------
249, 2015/04/15
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 249

import logging
import re

import numpy as np

# required PampelMuse modules
from pampelmuse.core.parameters import Parameters


def my_lsq_sigma(A, b, sigma=None):
    """
    The routine is a wrapper around the the numpy.linalg.lstsq-routine. It is
    designed as an extension that allows the user to also provide uncertainties
    together with the data when minimizing |Ax-b|.

    Parameters
    ----------
    A : array_like, shape (M, N)
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
        _A = (1. / (sigma[..., np.newaxis] / A))
        _b = b / sigma
    else:
        _A = A
        _b = b
    slc = ~np.isnan(_b)

    if b.ndim == 1:
        x = np.linalg.lstsq(_A[slc], _b[slc])[0]
    else:
        if sigma is not None:
            x = np.atleast_2d([np.linalg.lstsq(_A[i, slc], _b[i, slc])[0] for i in xrange(_b.shape[0])])
        else:
            x = np.atleast_2d([np.linalg.lstsq(_A[slc], _b[i, slc])[0] for i in xrange(_b.shape[0])])

    return x


class Transformation(Parameters):
    """
    The class stores information about the coordinate transformation that is
    used by PampelMuse. So far, the coordinates transformation is defined as
        x = A*u + C*v + x0
        y =-B*u + D*v + y0
    With (x,y) being the coordinates in IFU pixels and (u,v) being the input
    (or reference) coordinates. A, B, C, D, x0, and y0 are the parameters of
    the transformation.
    Note that earlier versions used a simpler version of the coordinate trans-
    formation, with C=B and D=A. This is still supported as input format.
    """
    PARAMETERS = ["A", "B", "C", "D", "x0", "y0"]

    def __init__(self, conformal=False, **kwargs):
        """
        Initialize an instance of the 'Transformation`-class.
       
        Parameters
        ----------
        kwargs
            Any keyword arguments that are provided are passed on to the call
            of the initialization method of the super-class. They might be used
            to provide initial values to the parameters of the coordinate trans-
            formation.
        """
        self.conformal = conformal

        # if no HDU is provided as a keyword argument, check if the individual parameters are provided, otherwise
        #  initialize them to NaN.
        if "hdu" not in kwargs:
            for name in self.PARAMETERS:
                if name not in kwargs:
                    kwargs[name] = np.nan

        # call initialization method of super-class, make sure order is A, B, C, D.
        super(Transformation, self).__init__(order=self.PARAMETERS, **kwargs)

        if self.defined.sum() == 4:
            self.conformal = True
            self._data[:, -2:] = self.data[:, -4:-2]
            self._data[:, -4:-2] = np.nan
            self.free[-2:] = self.free[-4:-2]
            self.fitted[-2:] = self.fitted[-4:-2]

        # check if all required parameters are available
        i = 0
        for name in self.PARAMETERS:
            if name not in self.names:
                logging.critical("Missing required parameter '{0}' of coord. transformation.".format(name))
            #     assert name in self.names, "Cannot find '{0}' among parameters of coord. transf.".format(name)
            i += 1

    @property
    def valid(self):
        return (self.fitted | self.defined).sum() >= 4

    def fit_to_data(self, in_coords, out_coords, ids=None, out_sigma=None):
        """
        Calculate a new coordinate transformation using the provided input
        (reference) and output (IFS) coordinates in a least squares fit. The
        data may be provided as plain multidimensional numpy arrays or using
        record arrays, containing at least three fields named 'id', 'x', and
        'y'. Note that it is currently not possible to use different types of
        arrays (plain/record) for input and output.
        Uncertainties associated with the output coordinates might also be
        provided.

        Parameters
        ----------
        in_coords : ndarray
            The input coordinates. If they are provided as a record array, it
            must contain at least 3 fields, named 'id', 'x', and 'y',
            containing the IDs of the sources and their reference x- and y-
            coordinates. Otherwise, a 2dim. numpy array should be provided with
            shape (N,2), and N>=2 being the number of sources.
        out_coords : ndarray
            The output coordinates. Again, if a record array is provided, it
            must contain the three fields 'id', 'x', and 'y'. The sources are
            matched to their counterparts in the input coordinates array based
            on the ID. If the array contains fields named 'x_err' and 'y_err',
            those are assumed to be the uncertainties associated with 'x' and
            'y'.
            In case a plain numpy array is provided, it must be 3dim. and of
            shape (M,N,2), with M being the number of spectral layers, and
            N>2 the number of sources. Note that the same sources in the same
            order as for the input coordinates must be provided (unless record
            arrays are used). The keyword 'out_sigma' may be used to provide
            the uncertainties of the input coordinates.
        ids : ndarray, optional
            If the coordinates are not provided as record arrays, this keyword
            parameter may be used to provide the source IDs. They are only used
            for logging.
        out_sigma : ndarray, optional
            If the coordinates are not provided as record arrays, this keyword
            parameter may be used to provided the uncertainties associated with
            the output coordinates. The array must have the same shape as the
            output coordinates array of course.
        """
        logging.info("Determining coordinate transformation using provided data...")

        # perform sanity checks on input data
        assert isinstance(in_coords, np.ndarray), "Parameter 'in_coords' must be numpy array, not {0}".format(
            re.findall(r"'(.*)'", str(type(in_coords)))[0])
        assert isinstance(out_coords, np.ndarray), "Parameter 'out_coords' must be numpy array, not {0}".format(
            re.findall(r"'(.*)'", str(type(out_coords)))[0])

        # check if record arrays or plain numpy arrays were provided.
        if in_coords.dtype.names is not None:  # deal with record arrays

            # check if provided data is valid
            for name in ['id', 'x', 'y']:
                if name not in in_coords.dtype.names:
                    logging.error("Cannot find field '{0}' in input coordinates.".format(name))
                    return
                if name not in out_coords.dtype.names:
                    logging.error("Cannot find field '{0}' in output coordinates.".format(name))
                    return

            # get dispersion information for provided output coordinates
            if out_coords["x"].shape != out_coords["y"].shape:
                logging.error("Output x- & y-coordinates. have different dimensions.")
                return
            else:
                disp = (out_coords["x"].ndim > 1)

            if disp:
                m = out_coords["x"].shape[1]
            else:
                m = 1

            # check if errors for output coordinates are provided
            if "x_err" in out_coords.dtype.names and "y_err" in out_coords.dtype.names:
                err = True
                logging.info("Found uncertainties of output coordinates.")
            else:
                err = False
                logging.info("No uncertainties of output coordinates found.")

            # find matching coordinate pairs
            in_select = []
            out_select = []

            for i in xrange(in_coords.size):
                in_id = in_coords['id'][i]
                if in_id not in out_coords['id']:
                    continue
                in_select.append(i)
                out_select.append(np.where(out_coords['id'] == in_id)[0][0])

            n = len(in_select)  # no of matching sources provided
            if n < 2:
                logging.error("Matched <2 sources between input and output coordinates.")
                return
            elif n < 4:
                logging.warning("Matched only {0} sources between input and output coordinates.".format(n))
            else:
                logging.info("Matched {0} sources between input and output coordinates.".format(n))

            # create ndarrays for fitting
            indata = np.zeros((n, 2), dtype=np.float32)
            indata[:, 0] = in_coords["x"][in_select]
            indata[:, 1] = in_coords["y"][in_select]

            outdata = np.zeros((m, n, 2), dtype=np.float32)
            if disp:
                outdata[..., 0] = out_coords["x"][out_select].swapaxes(0, 1)
                outdata[..., 1] = out_coords["y"][out_select].swapaxes(0, 1)
            else:
                outdata[..., 0] = out_coords["x"][out_select]
                outdata[..., 1] = out_coords["y"][out_select]

            if err:
                outsigma = np.zeros(outdata.shape, dtype=np.float32)
                if disp:
                    outsigma[..., 0] = out_coords["x_err"][out_select].swapaxes(0, 1)
                    outsigma[..., 1] = out_coords["y_err"][out_select].swapaxes(0, 1)
                else:
                    outsigma[..., 0] = out_coords["x_err"][out_select]
                    outsigma[..., 1] = out_coords["y_err"][out_select]
            else:
                outsigma = None

            ids = in_coords["id"][in_select]

        else:  # deal with normal (plain) numpy arrays
            # sanity checks for input coordinates
            assert in_coords.ndim == 2, "Input coordinates array must be 2dim."
            assert in_coords.shape[0] >= 2, "At least 2 sources must be provided."
            assert in_coords.shape[1] == 2, "First axis of input coordinates array must be [x y]."
            indata = in_coords

            # sanity checks for output coordinates
            assert out_coords.ndim == 3, "Output coordinates array must be 3dim."
            assert out_coords.shape[1] == in_coords.shape[
                0], "Input and output coordinates must be provided for the same sources."
            assert out_coords.shape[2] == 2, "Second axis of output coordinates array must be [x y]."
            outdata = out_coords

            # check if uncertainties provided
            if out_sigma is not None:
                assert isinstance(out_sigma, np.ndarray), "Parameter 'out_sigma' must be numpy array, not {0}".format(
                    re.findall(r"'(.*)'", str(type(out_sigma)))[0])
                assert out_sigma.shape == out_coords.shape, \
                    "Array shapes of output coordinates and their uncertainties differ."
                outsigma = out_sigma
            else:
                outsigma = None

        # perform the actual fit and initialize the values of the parameters of the coordinate transformation.
        self.set_data(self._fit_to_data(indata, outdata, outsigma=outsigma))

        # Only logging from this line to end of method
        logging.info("Determined coord. transformation for {0} image(s).".format(outdata.shape[0]))

        logging.info("Best-fit parameters:")
        for i in xrange(self.npars):
            name = self.names[i]
            logging.info("   {0} = {1:8.3f}".format(name, self.data[~self.mask[:, i, 0], i, 0].mean()))

        logging.info("Offsets of individual sources:")
        if ids is not None:
            logging.info("       ID     dx     dy")
        else:
            logging.info("   source     dx     dy")
        offset = (outdata - self.__call__(indata))[~(self.mask[..., 0].any(axis=1))]  # determine offsets
        for i in xrange(out_coords.shape[1]):
            if ids is not None:
                ID = int(ids[i])
            else:
                ID = i
            logging.info("   {0:6d}  {1:+5.2f}  {2:+5.2f}".format(ID, float(offset[:, i, 0].mean()),
                                                                  float(offset[:, i, 1].mean())))

    def _fit_to_data(self, indata, outdata, outsigma=None):
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
        indata : array_like, shape (N, 2)
           The input coordinates (u and v) for a sample of N>=2 sources.
        outdata : array_like, shape (N, 2) or (M, N, 2)
           The output coordinates (x and y) for the same sample of sources
           as in `indata`. If 'outdata' is 3dimensional, the transformation
           is determined for the M spectral layers.
        outsigma : array_like, shape (N, 2) or (M, N, 2), optional
           The uncertainties (if any) associated with the output coordinates.
           Must be of same size as 'outdata' if provided.

        Returns
        -------
        The results of the least squares fit. Consult the documentation of
        my_lsq_sigma for more information on the output.

        """
        # n is the number of provided sources, N=2n is the number of columns of the matrix A
        # (each source contributes two rows)
        n = indata.shape[0]
        N = 2 * n

        # fill matrix using input coordinates, the matrix equation will contain alternating equations for x and y,
        # equations involving output x coordinates in even rows (0,2,...N-2) and involving output y coordinates in
        # odd rows (1,3,...,N-1)
        if self.conformal:
            A = np.ones((N, 4), dtype=np.float32)
            A[slice(0, N, 2), 0] = indata[:, 0]
            A[slice(1, N, 2), 0] = indata[:, 1]
            A[slice(0, N, 2), 1] = indata[:, 1]
            A[slice(1, N, 2), 1] = -indata[:, 0]
            A[slice(1, N, 2), 2] = 0.
            A[slice(0, N, 2), 3] = 0.
        else:
            A = np.zeros((N, 6), dtype=np.float32)
            A[slice(0, N, 2), 0] = indata[:, 0]
            A[slice(0, N, 2), 2] = indata[:, 1]
            A[slice(0, N, 2), -2] = 1.
            A[slice(1, N, 2), 1] = -indata[:, 0]
            A[slice(1, N, 2), 3] = indata[:, 1]
            A[slice(1, N, 2), -1] = 1.

        # make sure input is available as numpy arrays & join x- & y-data
        outdata = np.atleast_2d(outdata)
        b = outdata.reshape((-1, outdata.shape[-1] * outdata.shape[-2]))
        # collapse last two axes, add spectral axis at index 0 (if none was provided)

        if outsigma is not None:
            outsigma = np.atleast_2d(outsigma)
            sigma = outsigma.reshape(b.shape)
        else:
            sigma = None

        # call fitting function
        return my_lsq_sigma(A, b, sigma=sigma)

    @property
    def zmatrix(self):
        # create diagonal matrix with trace [x0, y0] for each layer (index 0 is spectral one)
        _m = np.zeros((self.ndisp, 2, 2), dtype=np.float32)
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
        # create matrix [[A C] [-B D]] for each layer (index 0 is spectral one)
        _f = np.zeros((self.ndisp, 2, 2), dtype=np.float32)
        _f[:, 0, 0] = self.data[:, 0, int(self.fitted[0])]
        _f[:, 0, 1] = self.data[:, 1, int(self.fitted[1])] if self.conformal else self.data[:, 2, int(self.fitted[2])]
        _f[:, 1, 0] = -self.data[:, 1, int(self.fitted[1])]
        _f[:, 1, 1] = self.data[:, 0, int(self.fitted[0])] if self.conformal else self.data[:, 3, int(self.fitted[3])]

        # if np.isnan(_f).any():
        #     logging.warning("Coordinate transformation contains interpolated values.")
        #     for i in xrange(2):
        #         for j in xrange(2):
        #             valid = ~np.isnan(_f[:, i, j])
        #             fct = interpolate.InterpolatedUnivariateSpline(np.flatnonzero(valid), _f[valid, i, j],  k=2)
        #             _f[~valid, i, j] = fct(np.flatnonzero(~valid))
        return _f

    def __call__(self, in_coords):
        print self.conformal
        return np.inner(self.zmatrix, np.ones(in_coords.shape, dtype=in_coords.dtype)).swapaxes(-2, -1) + np.inner(
            self.fmatrix, in_coords).swapaxes(-2, -1)

    def info(self):
        """
        The method pretty-prints some information about the status of the
        coordinate transformation parameters out on the scree.
        """
        logging.info("Summary of coordinate transformation instance:")

        # call info method of super class. Provide parameter names to get order right
        super(Transformation, self).info(self.PARAMETERS)

    def make_hdu(self, **kwargs):
        """
        Write the current values of the parameters of the coordinate
        tranformation to a fits.BinTableHDU instance. The method is
        basically a wrapper of the 'make_hdu' instance defined in the
        super-class and adds some header keywords that provide further
        information about the coordinate transformation in use.
        
        Parameters
        ----------
        kwargs
            Any keyword parameters parameters passed to this method will
            be directed to the call of the 'make_hdu' method of the super-
            class.
        """
        kwargs["extname"] = "POSPARS"

        logging.info("Saving coordinate transformation in BinTableHDU '{0}'...".format(kwargs["extname"]))

        return super(Transformation, self).make_hdu(**kwargs)
