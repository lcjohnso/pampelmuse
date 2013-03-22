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
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.2.0.0"
import logging, re
import numpy as np
from scipy.spatial import distance
from scipy.interpolate import griddata



class Instrument(object):
    """
    The Instrument class can be used with any IFU that has quadratic spaxels
    aranged on a rectangular grid with unity filling factor. It also serves as
    super-class for all instrument specific sub-classes defined.
    """
    name = None

    _transform = None ## will contain the transformation from spaxels to x&y
    _edgepix = None ## will contain the indices of the spaxels that define
                    ## the edge of the field of view.

    _sky_fibers = []  ## The indices of the sky fibers
    _cal_fibers = []  ## The indices of the calibration fibers

    spectral_axis = 1 ## The index of the spectral axis; NOTE that numpy
                      ## swaps the axes, so spectral_axis=1 corresponds
                      ## to the second axis in numpy notation
    shape = None ## will contain the spatial shape of the field of view
    
    def __init__(self, transform=None, edgepix=None, badspax=[],
                 maxoff=np.sqrt(0.5), **kwargs):
        """
        Initialize an instance of the Instrument-class
        
        Parameters
        ----------
        transform : nd_array
            Can be an array of shape (2, n_spaxel) that gives the y- and x-
            coordinates of the spaxels. Note that following the numpy counting,
            the first axis is the y-axis. If transform=None, it can be
            specified after initialization.
        edgepix : nd_array, optional
            Can be a boolean array with shape (n_spaxel,) that is set to True
            for each spaxel at the edge of the field of view. If edgepix=None,
            it can be specified after initialization.
        badspax : array_like, optional
            A list containing the indices of dead spaxels.
        maxoff : float, optional
            The maximum distance of a position in the (x,y)-plane to the
            centre of a spaxel that is considered inside the field of view.
            The default value of sqrt(0.5) assumes a filling factor of unity.
        """

        self.maxoff = maxoff

        self.set_transform(transform)
        
        self.set_edgepix(edgepix)

        ## Handle bad spaxels (dead fibers)
        self.badspax = badspax
        for spax in badspax:
            logging.info("Excluding spaxel #{0}.".format(spax))

        ## self.badspax was a one-dimensional boolean array with a length
        ## equal to the number of spaxel that is True if the spaxel is bad and
        ## False otherwise



        ## a 2dim. boolean array that is True if a valid pixel is located at
        ## position (x,y) and False otherwise
##         self.fovmask = np.zeros(tuple(self.transform.max(axis=0)+1),
##                                 dtype=np.bool)

        ## fill the two boolean arrays just initialized
##         for i in xrange(self.nspaxel):
##             self.fovmask[self.transform[i,0], self.transform[i,1]] = True


    def set_transform(self, value):

        if value == None:
            return None

        if value.ndim != 2:
            logging.critical("'transform' must be 2dim., is {0}dim."\
                                 .format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."

        if value.shape[1] != 2:
            logging.critical(\
                "'transform' must contain 2 spatial coordinates, is {0}."\
                    .format(value.shape[1]))
        assert value.shape[1]==2, \
            "'transform' must contain 2 spatial coordinates."
        
        self._transform = value


    def get_transform(self):
        
        return self._transform


    transform = property(get_transform, set_transform, None,
                         "The mapping of the spaxels on the 2dim. plane.")


    def set_edgepix(self, value):

        if value == None:
            return None

        if value.ndim != 2:
            logging.critical("'edgepix' must be 2dim., is {0}dim."\
                                 .format(value.ndim))
        assert value.ndim == 2, "'edgepix' must be 2dim."

        if value.shape[1] != 2:
            logging.critical(\
                "'edgepix' must contain 2 spatial coordinates, is {0}."\
                    .format(value.shape[1]))
        assert value.shape[1]==2, \
            "'edgepix' must contain 2 spatial coordinates."
        
        self._edgepix = value


    def get_edgepix(self):

        ## if pixels that define edge of FoV are not provided, try to infer
        ## them from provided transformation from spaxels to x & y
        if self._edgepix == None:
            if self.transform==None:
                logging.error("No transformation from spaxel to x & y defined.")
                assert self.transform is not None, \
                    "No transformation from spaxel to x & y defined."
            return self.transform[\
                (self.x==self.x.min())|(self.x==self.x.max())\
                    |(self.y==self.y.min())|(self.y==self.y.max())]
        
        else:
            return self._edgepix


    def del_edgepix(self):
        self._edgepix = None


    edgepix = property(get_edgepix, set_edgepix, del_edgepix,
                       "The spaxels that define the edge of the FoV.")


    @property
    def nspaxel(self):
        return self.transform.shape[0]


    @property
    def x(self):
        return self.transform[:,1]


    @property
    def y(self):
        return self.transform[:,0]


    def set_sky_fibers(self, value):

        self._sky_fibers = value

    def get_sky_fibers(self):
        
        return self._sky_fibers

    sky_fibers = property(get_sky_fibers, set_sky_fibers, None,
                          "The numbers of the fibers used as sky fibers.")


    def set_cal_fibers(self, value):

        self._cal_fibers = value

    def get_cal_fibers(self):
        
        return self._cal_fibers

    cal_fibers = property(get_cal_fibers, set_cal_fibers, None,
                          "The numbers of the fibers used for calibration.")


    @property
    def aux_fibers(self):
        return np.asarray(np.sort(np.hstack(\
                    (self.sky_fibers, self.cal_fibers))), dtype=np.int16)


    @property
    def naux(self):
        return len(self.aux_fibers)
        

    @property
    def isscience(self):
        _isscience = np.ones((self.nspaxel+self.naux,), dtype=np.bool)
        _isscience[self.aux_fibers] = 0

        return _isscience


    @property
    def spatial_axis(self):
        if self.spectral_axis==1: return 2
        else: return 1


    def dist2spax(self, x, y, metric="euclidean"):

        return distance.cdist(self.transform, np.vstack((y, x)).T,
                              metric=metric)


    def infov(self, x, y):
        
        return (self.dist2spax(x, y)<=self.maxoff).any(axis=0)


    def dist2edge(self, x, y, metric="euclidean"):

        edist = distance.cdist(self.edgepix, np.vstack((y, x)).T,
                              metric=metric).min(axis=0)
        edist[self.infov(x, y)==False] *= -1

        return edist


    def make_grid(self, data, shape=None, method='linear', fill_value=np.nan):

        assert data.shape[0] == self.nspaxel, \
            "Spatial axis of data has invalid length."

        if shape == None:
            shape = self.shape

        grid_y, grid_x = np.mgrid[0:shape[0]-1:1j*shape[0],
                                  0:shape[1]-1:1j*shape[1]]

        return griddata((self.y, self.x), data, (grid_y, grid_x),
                        method=method, fill_value=fill_value)
            


    def get_science(self, data):
    
        if data.shape[-min(data.ndim,self.spatial_axis)] == self.nspaxel:
            return data

        elif data.shape[-self.spatial_axis] == (self.nspaxel + self.naux):

            if self.spatial_axis==2 or data.ndim == 1:
                return data[self.isscience]
            else:
                return data[...,self.isscience]
        else:

            logging.error("Provided data has invalid dimensions.")
            logging.error("Expected {0:d} or {1:d} fibres but got {2:d}.".\
                              format(self.nspaxel, self.nspaxel+self.naux,
                                     data.shape[-self.spatial_axis]))
            return data



class Pmas(Instrument):

    name = "pmas"
    shape = (16,16)
    

    def __init__(self, **kwargs):

        ## initialize transformation from (x,y)-coords to spaxels
        transform = np.vstack((np.arange(256, dtype=np.int16)//16,
                               np.mod(np.arange(256, dtype=np.int16), 16))).T
        
        ## all pixels at the edge of the fov 
        edgepix = np.vstack((\
                np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15], dtype=np.int16),
                np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], dtype=np.int16))).T

        super(Pmas, self).__init__(transform, edgepix, **kwargs)
        

        

class Argus(Instrument):

    name = "argus"
    shape = (14,22)  ## numpy notation - first y, then x
    spectral_axis = 2

    _sky_fibers = np.array([20,41,63,84,105,127,148,169,191,212,233,255,276,297,319], dtype=np.int16)
    _cal_fibers = np.array([42,106,170,234,298], dtype=np.int16)


    def __init__(self, **kwargs):

        if not "badspax" in kwargs.keys():
            kwargs["badspax"] = [297,298,299]

        ## initialize transformation from (x,y)-coords to spaxels
            transform = np.vstack((np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 13, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 13, 12, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=np.int16), np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 18, 19, 19, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20, 20, 20], dtype=np.int16))).T

        
        ## all pixels at the edge of the fov 
        edgepix = np.vstack((\
                np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13], dtype=np.int16),
                np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0,1,2,19,20,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,1,2,19,20,21,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], dtype=np.int16))).T

        super(Argus, self).__init__(transform, edgepix, **kwargs)


        
    def read_fiber_setup(self, hdu):
        """
        A FITs-HDU can be provided that contains the fiber configuration for the observation.
        This HDU is routinely provided by the ESO-pipeline (I guess)
        """

        self._sky_fibers = []
        self._cal_fibers = []
        _x          = []
        _y          = []

        sky_match = re.compile(r"ASKY_([0-9]+)")
        
        for i in xrange(hdu.header["NAXIS2"]):
            if hdu.data.field("X")[i]>0 and hdu.data.field("Y")[i]>0:
                _x.append(hdu.data.field("X")[i]-1)
                _y.append(hdu.data.field("Y")[i]-1)
            else:
                if sky_match.match(hdu.data.field("OBJECT")[i]) != None:
                    self._sky_fibers.append(i)
                else:
                    self._cal_fibers.append(i)


        super(Argus, self).set_transform(np.vstack((_y, _x)).T)
        super(Argus, self).set_sky_fibers(self._sky_fibers)
        super(Argus, self).set_cal_fibers(self._cal_fibers)


        logging.info("ARGUS fibre calibration (no. of fibres):")
        logging.info("   science - {0}, sky - {1}, calibration - {2}".format(\
                self.nspaxel, len(self.sky_fibers), len(self.cal_fibers)))



