import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr as lsmr
from scipy.spatial import distance


class Background(object):
    """
    The purpose of this class is to define a background that is spatially
    variable. To this aim, it defines a background grid. For each gridpoint,
    a weighting function w(r) of the form

        w(r) ~ EXP(-0.5*(r/Rs)^n)

    is defined. r measures the distance to the gridpoint, Rs is the scale
    radius which is equal to half the distance between to adjacent gridpoints.
    In this way, a smooth but variable background is created across the field
    of view.
    """
    def __init__(self, centroids, n=2, rcut=None, **kwargs):
        """
        Initialize an instance of the Background class.

        Parameters
        ----------
        centroids : ndarray
            The central coordinates of the individual background patches, as an
            array of shape (<no. of patches>, 2).  Note that the centroids must
            be on a homogeneous grid currently.
        n : int, optional
            The power law index that controls the weighting of the spaxels with
            distance. It should currently be set to 2 (the default).
        rcut : float, optional
            The radius out to which a background patch expands, spaxels at
            larger radii are assigned zero weight. By default, it will be set
            to 3 x rscale.
        """
        self.centroids = np.asarray(centroids)
        assert self.centroids.ndim==2 and self.centroids.shape[-1]==2, "Centroids must be a 2dim. array of shape (N,2)."

        # make sure centroids are on a homogeneous grid: calculate distance to nearest neighbour, must be the same for each gridpoint
        if self.centroids.shape[0]>1:
            distances = distance.cdist(self.centroids, self.centroids)
            distances.sort(axis=0)
            assert np.unique(distances[1]).size==1, "Centroids must form homogeneous grid."

            self.rscale = distances[1][0]/2. # scale radius is set to half the distance between two adjacent centroids.
        else:
            self.rscale = np.inf

        self.n = n # the power-law index used to assign weights

        # To make the computation more efficient, each background patch is only calculated out to a given radius
        if rcut==None: self.rcut = 3*self.rscale
        else: self.rcut = rcut

    @classmethod
    def from_shape(cls, shape, rscale, **kwargs):
        """
        This method initializes an instance of the Background class using an
        array shape and a scale radius. The centroids of the background
        patches will be defined on a homogeneous grid with constant distance
        2 x scale radius between adjacent centroids.

        Parameters
        ----------
        shape : tuple
            The spatial size of the array to work on, provided as (dy, dx)
        rscale : float
            The effective radius of a background patch in units of spaxels.
            The distance between neighbouring patches will be 2 x rscale.
        kwargs : optional
            All remaining arguments are passed on to the call of the
            Background.__init__ method.
        """
        # create the grid with the centroids of the background patches
        # first get centroids along x and y
        _xcentroids = np.arange(0, shape[0], 2*rscale)
        _ycentroids = np.arange(0, shape[1], 2*rscale)
        # then make sure that distances to edges of FoV are approximately the same
        _xcentroids += 0.5*(shape[0]-1-_xcentroids.max())
        _ycentroids += 0.5*(shape[1]-1-_ycentroids.max())
        # combine x and y centroids to array of shape (2, gridsize)
        xcentroids = np.resize(_xcentroids, _xcentroids.size*_ycentroids.size)
        xcentroids.sort()
        ycentroids = np.resize(_ycentroids, _xcentroids.size*_ycentroids.size)        
        return cls(np.vstack((xcentroids, ycentroids)).T, **kwargs)

    @classmethod
    def constant(cls):
        """
        This method initializes a constant background across an arbitrary field
        of view.
        """
        return cls(np.array([[0, 0], ], dtype=np.float32))

    @property
    def gridsize(self):
        # get number of patches across field of view
        return self.centroids.shape[0]

    def __call__(self, u, v):
        """
        The __call__ method of the Background class is intended to be provided
        to the call of the scipy.distance.cdist routine. The idea is that this
        routine is used to calculate weights instead of distances between two
        points u and v.

        Parameters
        ----------
        u : array_like
            The coordinates of the first datapoint, can be N dimensional.
        v : array_like
            The coordinates of the second datapoint. Note than len(v)==len(u)
            is required.

        Returns
        -------
        weight : float
            The weight of point v with respect to point u.
        """
        if self.rscale <= 0:
            return 1
        else:
            return np.exp(-0.5*((u-v)**self.n).sum()/self.rscale**self.n) if np.sqrt(((u-v)**2).sum())<self.rcut else 0


    def get_weights(self, data):
        """
        This method performs a call to scipy.distance.cdist using the funtion
        defined by Background.__call__ as 'metric'.

        Parameters
        ----------
        data : array_like, shape (m, 2)
            The coordinates for which the weights in each background patch
            should be calculated. `m' is the number of sources.

        Returns
        -------
        weights : nd_array, shape (gridsize, m)
            The weight of each provided datapoint on each background patch.
        """
        return distance.cdist(self.centroids, data, self)

    def calculate_weights(self, data):
        """
        Basically a wrapper around the Background.get_weights method, but
        instead of returning the weights, they are stored in a class
        attribute. In contrast to the Background.get_weights method, the
        weights are also normalized by this method.
        """
        self.weights  = self.get_weights(data)
        self.weights /= self.weights.sum(axis=0)

    def prepare_sparse(self, mask=None):
        """
        This method prepares a sparse matrix using the weights as values and
        the spaxel and patch indices as column and row indices. Note that the
        weights must have been calculated with the Background.calculate_weights
        method before.

        Parameters
        ----------
        mask : nd_array, optional
            A mask indicating invalid spaxels/coordinates. Must have a length
            equal to the number of spaxels for which weights were calculated.
            The idea is that a single call to Background.calculate_weights -
            which can be quite time-consuming for large arrays - can be
            followed by many calls to this method even if some spaxels are
            flagged as invalid. If None, it is assumed that all spaxels are
            valid.

        Returns
        -------
        A : sparse coo-matrix
            The sparse matrix containing the weights.
        """
        assert hasattr(self, "weights"), "Need to calculate weights first."

        if mask==None:
            used = np.ones(self.weights.shape[1], dtype=np.bool)
        else:
            assert len(mask)==self.weights.shape[1], "Invalid mask size."
            used = (mask==False)

        coeffs   = []
        indices0 = []
        indices1 = []

        for i in xrange(self.gridsize):
            slc = used&(self.weights[i]>0)
            coeffs.append(self.weights[i][slc])
            indices0.append(np.cumsum(used)[slc]-1)
            indices1.append(i*np.ones(slc.sum(), dtype=np.int32))

        return sparse.coo_matrix((np.hstack(coeffs), (np.hstack(indices0), np.hstack(indices1))), shape=(used.sum(), self.gridsize))
