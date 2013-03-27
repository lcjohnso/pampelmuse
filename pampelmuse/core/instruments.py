"""
INSTRUMENTS
===========
The classes defined in this script are supposed to provide support in handling
instrument-specific characteristics: dead fibers, holes in FoV,
non-rectangular FoV, ...
The idea is to define a sub-class of the class 'Instruments' for every IFU that
should be supported. Everything that is instrument-specific should be handled
in those classes so that the other routines of PampelMuse do not need to be
updated every time support for a new instrument is added. Currently PMAS/Larr
and FLAMES/ARGUS are included.

CHANGELOG:
----------
2012/11/09, v0.1.0.0, preliminary version completed, support for PMAS/Larr
                      and FLAMES/ARGUS

2013/02/02, v0.1.1.0, bugfix in PMAS class: replaced int16 with np.int16
2013/03/22, v0.2.0.0, significantly updated functionality of the individual
                      classes; also added documentation.
2013/03/26, v0.2.0.1, included a more general way to determine if a source is
                      covered by the IFU footprint. Also added documentation.
2013/03/27, v0.2.0.2, changed default value of 'name'-property to "Generic IFU
                      in Instrument-class (was None)
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.2.0.2"
import logging, re
import numpy as np
from matplotlib import path
from scipy.spatial import distance
from scipy.interpolate import griddata



class Instrument(object):
    """
    The Instrument class can be used with any IFU that has quadratic spaxels
    aranged on a rectangular grid with unity filling factor. It also serves as
    super-class for all instrument specific sub-classes defined.
    """
    name = "Generic IFU"

    _transform = None ## will contain the transformation from spaxels to x&y
    _edgepix = None ## will contain the indices of the spaxels that define
                    ## the edge of the field of view.

    _sky_fibers = []  ## The indices of the sky fibers
    _cal_fibers = []  ## The indices of the calibration fibers

    spectral_axis = 1 ## The index of the spectral axis; NOTE that numpy
                      ## swaps the axes, so spectral_axis=1 corresponds
                      ## to the second axis in numpy notation
    shape = None ## will contain the spatial shape of the field of view
    
    def __init__(self, transform=None, edgepix=None, badspax=[], **kwargs):
        """
        Initialize an instance of the Instrument-class
        
        Parameters
        ----------
        transform : nd_array, optional
            Can be an array of shape (2, n_spaxel) that gives the y- and x-
            coordinates of the spaxels. Note that following the numpy counting,
            the first axis is the y-axis. If transform=None, it can be
            specified after initialization.
        edgepix : nd_array, optional
            Can be an array of integers that contains the indices of the
            spaxels that define the edge of the field of view. NOTE that the
            indices should be provided in such an order that the spaxels follow
            a loop around the field of view.
            If edgepix=None, it can be specified after initialization.
        badspax : array_like, optional
            A list containing the indices of dead spaxels.
##      NOT USED ANYMORE   maxoff : float, optional
##             The maximum distance of a position in the (x,y)-plane to the
##             centre of a spaxel that is considered inside the field of view.
##             The default value of sqrt(0.5) assumes a filling factor of unity.
        """

        self.set_transform(transform)
        
        self.set_edgepix(edgepix)

        ## Handle bad spaxels (e.g., dead fibers)
        self.badspax = badspax
        for spax in badspax:
            logging.info("Excluding spaxel #{0}.".format(spax))



    def set_transform(self, value):
        """
        Set the transformation from spaxels to coordinates on the (x,y)-plane.
        Serves as the setter-method for the 'transform'-property

        Parameters
        ----------
        value : nd_array or None
            Unless 'value' is None, an array of shape (2, n_spaxel) that gives
            the y- and x-coordinates of all spaxels. Note that following the
            numpy counting, the first axis is the y-axis.
        """
        if value == None:
            return None ## No transformation specified

        ## Perform sanity checks on provided data
        if value.ndim != 2:
            logging.critical(\
                "'transform' must be 2dim., is {0}dim.".format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."

        if value.shape[1] != 2:
            logging.critical(\
                "'transform' must contain 2 spatial coordinates, has {0}."\
                    .format(value.shape[1]))
        assert value.shape[1]==2, \
            "'transform' must contain 2 spatial coordinates."
        
        self._transform = value


    def get_transform(self):
        """
        The getter method of the 'transform'-property

        Returns
        -------
        transform : nd_array
            The current transformation from spaxels to coordinates on the
            (x,y)-plane as an array of shape (N_spaxels, 2)
        """
        return self._transform


    transform = property(get_transform, set_transform, None,
                         "The mapping of the spaxels on the 2dim. plane.")


    def set_edgepix(self, value):
        """
        Set the spaxels that define the edge of the field of view. Serves as
        the setter-method for the 'edgepix'-proeprty.

        Parameters
        ----------
        value : nd_array or None
            If 'value' is not None, a 1dim. boolean array of the same length
            as the current
        """
        if value == None:
            return None  ## edgepix not define.

        ## Perform sanity checks on provided data
        if self.transform == None:
            logging.error(\
                "Cannot set 'edgepix' unless 'transform' is specified.")
        assert self.transform is not None, \
            "No transformation from spaxels to x- & y-coordinates available."

        if value.ndim != 1:
            logging.critical("'edgepix' must be 1dim., but is {0}dim."\
                                 .format(value.ndim))
        assert value.ndim == 1, "'edgepix' must be 1dim."

        self._edgepix = value


    def get_edgepix(self):
        """
        The getter-method of the 'edgepix'-property. Note that if the spaxels
        that define edge of FoV have not yet been provided but the 'transform'-
        property  is set, the code tries to infer the spaxels on the edge from
        the provided transformation from spaxels to x- & y-coordinates.
        """
        if self._edgepix == None:
            if self.transform==None:
                logging.warning("No transf. from spaxels to x & y available.")
                return None
            
            ## assume rectangular field of view and fibres aranged row after
            ## row (like PMAS/LArr).
            return np.hstack(\
                (np.nonzero((self.y==self.y.min())&(self.x>self.x.min())),
                 np.nonzero((self.x==self.x.max())&(self.y>self.y.min())),
                 np.nonzero((self.y==self.y.max())&(self.x<self.x.max()))[::-1],
                 np.nonzero((self.x==self.x.min())&(self.y<self.y.max()))[::-1]\
                     ))

        else:
            return self._edgepix


    def del_edgepix(self):
        """
        Delete the current selection of spaxels on the edge of the FoV.
        """
        self._edgepix = None


    edgepix = property(get_edgepix, set_edgepix, del_edgepix,
                       "The spaxels that define the edge of the FoV.")


    @property
    def nspaxel(self):
        """
        Returns
        -------
        nspaxel : int
            The number of spaxels of the current instance of 'Instrument'.
        """
        return self.transform.shape[0]


    @property
    def x(self):
        """
        Returns
        -------
        x : nd_array
            The x-coordinates of the spaxels of the current instance of
            'Instrument'.
        """
        return self.transform[:,1]


    @property
    def y(self):
        """
        Returns
        -------
        y : nd_array
            The y-coordinates of the spaxels of the current instance of
            'Instrument'.
        """
        return self.transform[:,0]


    def set_sky_fibers(self, value):
        """
        Set the indices of the fibers used to observe the night sky
        simultaneously with the science data. This method serves a setter for
        the 'sky_fibers' property.

        Parameters
        ----------
        value : array_like
            A list containing the indices of the sky fibers
        """
        self._sky_fibers = value

    def get_sky_fibers(self):
        """
        The getter-method of the 'sky_fibers'-method.
        
        Returns
        -------
        sky_fibers : array_like
            A list containing the indices of the sky fibers.
        """
        return self._sky_fibers


    sky_fibers = property(get_sky_fibers, set_sky_fibers, None,
                          "The numbers of the fibers used as sky fibers.")


    def set_cal_fibers(self, value):
        """
        Set the indices of the fibers used to calibrate the data simultaneously
        with the observations of the science data. This method serves a setter
        for the 'cal_fibers' property.

        Parameters
        ----------
        value : array_like
            A list containing the indices of the calibration fibers.
        """

        self._cal_fibers = value

    def get_cal_fibers(self):
        """
        The getter-method of the 'cal_fibers'-method.
        
        Returns
        -------
        cal_fibers : array_like
            A list containing the indices of the calibration fibers.
        """        
        return self._cal_fibers


    cal_fibers = property(get_cal_fibers, set_cal_fibers, None,
                          "The numbers of the fibers used for calibration.")


    @property
    def aux_fibers(self):
        """
        Returns
        -------
        aux_fibers : array_like
            A list containing the indices of all auxiliary fibers used either
            as sky or calibration fibers.
        """
        return np.asarray(np.sort(np.hstack(\
                    (self.sky_fibers, self.cal_fibers))), dtype=np.int16)


    @property
    def naux(self):
        """
        Returns
        -------
        naux : int
            The number of auxiliary (= sky or calibration) fibers per datacube.
        """
        return len(self.aux_fibers)
        

    @property
    def isscience(self):
        """
        Returns
        -------
        isscience : array_like
            A boolean array that is True if a science fibre is present at this
            index and False if an auxiliary fibre is located there.
        """
        _isscience = np.ones((self.nspaxel+self.naux,), dtype=np.bool)
        _isscience[self.aux_fibers] = False

        return _isscience


    @property
    def spatial_axis(self):
        """
        Returns
        -------
        spatial_axis :  array_like
            The index of the spatial axis, either 1 or 2. NOTE that numpy
            swaps the axes, so spatial_axis=2 corresponds to the first axis
            in numpy notation.
        """
        if self.spectral_axis==1: return 2
        else: return 1


    def dist2spax(self, x, y, metric="euclidean"):
        """
        Calculate the distance of one or more points located at positions
        (x_i,y_i) to all spaxels of the current Instrument-instance.

        Parameters
        ----------
        x : float or array_like
            The x-position(s) of the source(s), in units of spaxels.
        y : float or array_like
            The y-position(s) of the source(s), in units of spaxels.
        metric : string
            The metric used to calculate the distances. For the different
            options and their meanings consult the documentation of
            scipy.distance.cdist. The default is 'euclidean'.

        Returns
        -------
        distances : float or array_like
            The distance(s) calculated using the provided metric and in units
            of spaxels.
        """
        return distance.cdist(self.transform, np.vstack((y, x)).T,
                              metric=metric)


    def infov(self, x, y):
        """
        Check if the provided (x,y)-positions are inside the field of view of
        the current Instrument-instance. The method uses the 'contains_point'-
        method of matplotlib.path.Path for this purpose. The this reason, the
        provided indices of 'edgepix' should form a polygon along the edge of
        the field of view.

        Parameters
        ----------
        x : float or array_like
            The x-position(s) of the source(s), in units of spaxels.
        y : float or array_like
            The y-position(s) of the source(s), in units of spaxels.

        Returns
        -------
        infov : array_like or bool
            A boolean flag for each source indicating whether or not it is
            inside the field of view.
        """
        ## initialize path instance:
        edge_path = path.Path(self.transform[self.edgepix])
        
        ## loop over provided coordinates
        if isinstance(x, (np.ndarray, list)):
            infov = np.zeros(len(x), dtype=np.bool)
            for i in xrange(infov.size):
                infov[i] = edge_path.contains_point((y[i], x[i]))
        else:
            infov = (edge_path.contains_point((y, x))==1)

        return infov


    def dist2edge(self, x, y, metric="euclidean"):
        """
        Calculate the  distance of one or more point in (x,y)-space to the
        edge of the field of view of the current Instrument instance.

        Parameters
        ----------
        x : float or array_like
            The x-position(s) of the source(s), in units of spaxels.
        y : float or array_like
            The y-position(s) of the source(s), in units of spaxels.
        metric : string
            The metric used to calculate the distances. For the different
            options and their meanings consult the documentation of
            scipy.distance.cdist. The default is 'euclidean'.

        Returns
        -------
        dist2edge : array_like or float
            The distance of each provided source to the edge of the field
            of view. Note that sources outside the field of view will have
            negative distances.
        """
        edist = distance.cdist(\
            self.transform[self.edgepix], np.vstack((y, x)).T,
            metric=metric).min(axis=0)
        edist[self.infov(x, y)==False] *= -1 ## sources outside have negative
                                             ## values
        return edist


    def make_grid(self, data, shape=None, method='linear', fill_value=np.nan):
        """
        Calculate a representation of an observed dataset of the current 
        Instrument-instance on a regular grid in the (x,y)-plane. This method
        is basically a wrapper around the 'griddata'-routine of scipy. Check
        the documentation of that routine for further details.

        Parameters
        ----------
        data : nd_array
            The input data. It must have the appropriate number of spaxels
            (either with or without auxiliary fibers). Its spatial and
            spectral axes should coincide with those of the Instrument-
            instance.
       shape : tuple, optional
            The (y,x)-shape of the regular grid to  be created. NOTE that
            following the numpy-convention, the first value gives the y-
            size. If no size is provided, the method will use the default
            size of the Instrument-instance.
        method : string, optional
            The interpolation method. Passed on to the call of the 'griddata'
            routine of scipy.
        fill_value : float, optional
            The value given to undefined spaxels in the calculated array.
            Passed on to the call of the 'griddata' routine of scipy.

        Returns
        -------
        out_data : nd_array
            The data representation on the regular grid in the (x,y)-plane.
        """
        
        ## The index of the spatial axis in numpy-counting.
        np_spatial_axis = np.mod(self.spatial_axis,2)

        ## Perform sanity checks
        ## - Dimensionality
        if np_spatial_axis == 1 and data.ndim == 1:
            ## add pseudo-spectral axis to handle 1dim. data (single layer):
            calc_data = data.reshape((1,data.shape[0]))
        elif np_spatial_axis == 0 and data.ndim == 1:
            ## add pseudo-spectral axis to handle 1dim. data (single layer):
            calc_data = data.reshape((data.shape[0],1))
        else:
            calc_data = data

        if calc_data.ndim != 2:
            logging.error("Data must be 1D or 2D, but is {0}D.".format(\
                    int(calc_data.ndim)))
        assert calc_data.ndim == 2, "Data must be 1D or 2D"

        ## - Shape
        if calc_data.shape[np_spatial_axis] != self.nspaxel:
            logging.error("Spatial axis of data has invalid length")
            logging.error("Expected {0} spaxels, but got {1}."\
                    .format(self.nspaxel,
                            float(calc_data.shape[np_spatial_axis])))
        assert calc_data.shape[np_spatial_axis] == self.nspaxel, \
            "Spatial axis of data has invalid length."

        if shape == None:
            shape = self.shape

        ## prepare regular grid on which to resample the data
        grid_y, grid_x = np.mgrid[0:shape[0]-1:1j*shape[0],
                                  0:shape[1]-1:1j*shape[1]]

        ## extract science fibers (if necessary) and run griddata
        out_data = griddata((self.y, self.x), self.get_science(calc_data),
                            (grid_y, grid_x), method=method,
                            fill_value=fill_value)
        return np.squeeze(out_data) ## remove pseudo-spectral axis if needed


    def get_science(self, data):
        """
        Extract the science signal from the provided data. Signal of fibers
        marked as sky or calibration (= auxiliary) fibers is neglected and the
        science data is returned.
        
        Parameters
        ----------
        data : array_like
            The provided data, must be 1D or 2D. If it is 2D, the order of
            spatial and spectral axis must match the one of the Instrument
            instance. The data size along the spatial axis must be either
            equal to the number of science and auxiliary fibers or to the
            number of science fibers. In the later case, the data is
            returned without modifications.

        Returns
        -------
        sci_data : nd_array
            The extracted science data.
        """    

        ## The index of the spatial axis in numpy-counting.
        np_spatial_axis = np.mod(self.spatial_axis,2)

        ## Perform sanity checks
        ## - Dimensionality
        if np_spatial_axis == 1 and data.ndim == 1:
            ## add pseudo-spectral axis to handle 1dim. data (single layer):
            calc_data = data.reshape((1,data.shape[0]))
        elif np_spatial_axis == 0 and data.ndim == 1:
            ## add pseudo-spectral axis to handle 1dim. data (single layer):
            calc_data = data.reshape((data.shape[0],1))
        else:
            calc_data = data

        if calc_data.ndim != 2:
            logging.error("Data must be 1D or 2D, but is {0}D.".format(\
                    int(calc_data.ndim)))
        assert calc_data.ndim == 2, "Data must be 1D or 2D"

        ## - Number of spaxels
        if calc_data.shape[np_spatial_axis] == self.nspaxel:
            return np.squeeze(calc_data)

        elif calc_data.shape[np_spatial_axis] == (self.nspaxel + self.naux):
            if self.spatial_axis==2:
                return np.squeeze(calc_data[self.isscience])
            else:
                return np.squeeze(calc_data[:,self.isscience])
        else:
            logging.error("Provided data has invalid dimensions.")
            logging.error("Expected {0:d} or {1:d} fibres but got {2:d}.".\
                              format(self.nspaxel, self.nspaxel+self.naux,
                                     calc_data.shape[np_spatial_axis]))
            return None



class Pmas(Instrument):
    """
    Class to handle data observed with the PMAS/LArr instrument
    (Roth et al. 2005, PASP, 117, 620), mounted at the 3.5m telescope at
    Calar Alto observatory.
    """

    name = "pmas"
    shape = (16,16)
    

    def __init__(self, **kwargs):
        """
        Initialize an instance of the 'PMAS'-class.

        Parameters
        ----------
        kwargs
            Any arguments parsed to this method are passed on to the
            parent-class 'Instrument' of 'PMAS'.
        """
        ## initialize transformation from (x,y)-coords to spaxels
        transform = np.vstack((np.arange(256, dtype=np.int16)//16,
                               np.mod(np.arange(256, dtype=np.int16), 16))).T
        
        ## all pixels at the edge of the fov 
        edgepix = np.asarray([0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,239,223,207,191,175,159,143,127,111,95,79,63,47,31,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], dtype=np.int16)

## old usage, provide x- & y-coordinate of each spaxel at edge of FoV 
##         edgepix = np.vstack((\
##                 np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15], dtype=np.int16),
##                 np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], dtype=np.int16))).T

        super(Pmas, self).__init__(transform, edgepix, **kwargs)
        

        

class Argus(Instrument):
    """
    Class to handle data coming from the IFU 'ARGUS' of the FLAMES spectrograph
    (Pasquini et al. 2002, The messenger, 110, 1), mounted on UT2 of the ESO
    Very Large Telescope (VLT).
    """
    name = "argus"
    shape = (14,22)  ## numpy notation - first y, then x
    spectral_axis = 2

    ## The indices of the dedicated sky fibers
    _sky_fibers = np.array(\
        [20,41,63,84,105,127,148,169,191,212,233,255,276,297,319],
        dtype=np.int16)

    ## The indices of the dedicated calibration fibers
    _cal_fibers = np.array([42,106,170,234,298], dtype=np.int16)


    def __init__(self, **kwargs):
        """
        Initialize an instance of the 'ARGUS' class. NOTE that by default, it
        is assumed that the traces of all fibres (including all 14 sky fibres
        and all 5 calibration fibres) are present in the data. This is the
        format provided by, e.g., p3d (Sandin et al. 2010, A&A, 515, A35). If
        the data have another format, this can be handled by either providing
        the coordinate transformation directly upon initialisation of an
        instance or calling the dedicated 'read_fiber_setup'-method that takes
        a FITS-HDU containing the fiber setup (as provided by the official ESO
        pipeline) as input.
        """

        ## The traces of the last three spaxels are entirely outside the
        ## field of view.
        if not "badspax" in kwargs.keys():
            kwargs["badspax"] = [297,298,299]

        ## initialize transformation from (x,y)-coords to spaxels
        if not "transform" in kwargs.keys():
            kwargs["transform"] = np.vstack((np.array([5,4,3,2,1,1,2,3,4,5,6,7,8,9,10,11,12,12,12,13,1,0,0,1,2,3,4,5,6,7,8,9,10,11,11,10,9,8,7,6,6,7,8,9,10,11,12,13,13,12,11,10,9,8,7,6,5,4,3,2,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,2,3,4,5,6,7,8,9,10,11,12,13,13,12,11,10,9,8,7,6,10,11,12,13,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,1,9,8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8,9,1,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,12,11,10,6,7,8,9,10,11,12,13,13,12,11,10,9,8,7,6,5,4,3,2,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,2,3,4,5,6,7,8,9,10,11,12,13,13,12,11,10,9,8,7,6,6,7,8,9,10,11,11,10,9,8,7,6,5,4,3,2,1,0,0,1,13,12,12,12,11,10,9,8,7,6,5,4,3,2,1,1,2,3,4,5],dtype=np.int16),np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,11,11,11,11,11,11,11,11,11,11,10,10,10,10,10,10,10,10,10,10,13,13,12,12,12,12,12,12,12,12,12,12,12,12,12,12,11,11,11,11,14,14,14,14,14,14,14,14,13,13,13,13,13,13,13,13,13,13,13,13,15,15,15,15,15,15,15,15,15,15,15,15,15,15,14,14,14,14,14,14,17,17,17,17,17,17,16,16,16,16,16,16,16,16,16,16,16,16,16,16,18,18,18,18,18,18,18,18,18,18,18,18,17,17,17,17,17,17,17,17,20,20,20,20,20,20,19,19,19,19,19,19,19,19,19,19,19,19,18,18,19,19,20,21,21,21,21,21,21,21,21,21,21,21,21,20,20,20,20,20], dtype=np.int16))).T

        
        ## all pixels at the edge of the fov
        if not "edgepix" in kwargs.keys():
            kwargs["edgepix"] = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,48,47,60,99,112,111,124,123,176,175,188,187,200,239,252,251,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,276,277,278,225,226,213,214,161,162,149,150,137,138,85,86,73,74,21,22,23], dtype=np.int16)
## 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0], dtype=np.bool)

## old usage,provide x- & y-coordinate of each spaxel at edge of FoV 
##         edgepix = np.vstack((\
##                 np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13], dtype=np.int16),
##                 np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0,1,2,19,20,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,1,2,19,20,21,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], dtype=np.int16))).T

        super(Argus, self).__init__(**kwargs)


        
    def read_fiber_setup(self, hdu):
        """
        A FITS-HDU can be provided that contains the fiber configuration for
        the observation. This HDU is routinely provided by the ESO-pipeline 
        (I guess). The method uses the provided information to update the
        transformation from spaxels to x- & y- coordinates and to find the
        indices of the sky and calibration fibres.

        Parameters
        ----------
        hdu : a FITS binary table HDU instance
            The HDU containing the fibre setup.
        """

        self._sky_fibers = []
        self._cal_fibers = []
        _x          = []
        _y          = []

        sky_match = re.compile(r"ASKY_([0-9]+)")
        
        for i in xrange(hdu.header["NAXIS2"]): ## loop over all rows in HDU
            if hdu.data.field("X")[i]>0 and hdu.data.field("Y")[i]>0:
                _x.append(hdu.data.field("X")[i]-1)
                _y.append(hdu.data.field("Y")[i]-1)
            else:
                if sky_match.match(hdu.data.field("OBJECT")[i]) != None:
                    self._sky_fibers.append(i)
                else:
                    self._cal_fibers.append(i)

        ## update information
        super(Argus, self).set_transform(np.vstack((_y, _x)).T)
        super(Argus, self).set_sky_fibers(self._sky_fibers)
        super(Argus, self).set_cal_fibers(self._cal_fibers)

        ## some logging on setup that was found.
        logging.info("ARGUS fibre calibration (no. of fibres):")
        logging.info("                            science: {0:3d}".format(\
                self.nspaxel))
        logging.info("                                sky: {0:3d}".format(\
                len(self.sky_fibers)))
        logging.info("                        calibration: {0:3d}".format(\
                len(self.cal_fibers)))



