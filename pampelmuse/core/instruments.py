"""
INSTRUMENTS
===========
The classes defined in this class are supposed to provide support in handling
instrument -pecific characteristics: dead fibers, holes in FoV,
non-rectangular FoV
The idea is to define a sub-class of the class 'Instruments' for every IFU that
should be supported. Currently PMAS/Larr and FLAMES/ARGUS are included.

CHANGELOG:
----------
2012/11/09, v0.1.0.0, preliminary version completed, support for PMAS/Larr
                      and FLAMES/ARGUS

"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.0.0"
import logging
import numpy as np
from scipy.spatial import distance



class Instruments(object):

    def __init__(self, transform, edgepix, badspax=[], maxoff=np.sqrt(0.5), 
                 **kwargs):

        self.maxoff = maxoff


        self.transform = transform

        if self.transform.ndim != 2:
            logging.critical("'transform' must be 2dim., got {0}dim."\
                                 .format(self.transform.ndim))
        assert self.transform.ndim == 2, "'transform must be 2dim."

        if self.transform.shape[1] != 2:
            logging.critical(\
                "'transfrom' must contain 2 spatial coordinates, got {0}."\
                    .format(self.transform.shape[1]))
        assert self.transform.shape[1]==2, \
            "'transfrom' must contain 2 spatial coordinates"
        
        self.edgepix = edgepix


        self.nspaxel = self.transform.shape[0]

        ## check if specified bad spaxels are okay
        if len(badspax) > 0 and (np.asarray(badspax)>=self.nspaxel).any():
            logging.warning("At least one 'badspax' out of valid range:")
            logging.warning("   n_spaxel={0}, badspax={1}".format(\
                    int(self.nspaxel), badspax))

        ## A one-dimensional boolean array with a length equal to the number
        ## of spaxel that is True if the spaxel is bad and False otherwise
        self.badspax = np.zeros((self.nspaxel,), dtype=np.bool)

        ## a 2dim. boolean array that is True if a valid pixel is located at
        ## position (x,y) and False otherwise
        self.fovmask = np.zeros(tuple(self.transform.max(axis=0)+1),
                                 dtype=np.bool)

        ## fill the two boolean arrays just initialized
        for i in xrange(self.nspaxel):
            if i in badspax:
                logging.info("Excluding spaxel #{0}.".format(i))
                self.badspax[i] = True
                continue
            self.fovmask[self.transform[i,0], self.transform[i,1]] = True


    def dist2spax(self, x, y, metric="euclidean"):

        return distance.cdist(self.transform, np.vstack((x, y)).T,
                              metric=metric)


    def infov(self, x, y):
        
        return (self.dist2spax(x, y)<=self.maxoff).any(axis=0)


    def dist2edge(self, x, y, metric="euclidean"):

        edist = distance.cdist(self.edgepix, np.vstack((x, y)).T,
                              metric=metric).min(axis=0)
        edist[self.infov(x,y)==False] *= -1

        return edist


class Pmas(Instruments):

    def __init__(self, **kwargs):

        ## initialize transformation from (x,y)-coords to spaxels
        transform = np.vstack((\
                np.mod(np.arange(256, dtype=int16), 16),
                np.arange(256, dtype=int16)//16)).T

        ## all pixels at the edge of the fov 
        edgepix = np.vstack((\
                np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], dtype=np.int16),
                np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15], dtype=np.int16))).T

        super(Pmas, self).__init__(transform, edgepix, **kwargs)
        
        

class Argus(Instruments):

    def __init__(self, **kwargs):

        if not "badspax" in kwargs.keys():
            kwargs["badspax"] = [297,298,299]

        ## initialize transformation from (x,y)-coords to spaxels
        transform = np.vstack((\
                np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 18, 19, 19, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20, 20, 20], dtype=np.int16),
                np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 13, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 13, 12, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=np.int16))).T

        
        ## all pixels at the edge of the fov 
        edgepix = np.vstack((\
                np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0,1,2,19,20,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,21,0,1,2,19,20,21,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], dtype=np.int16),
                np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13], dtype=np.int16))).T

        super(Argus, self).__init__(transform, edgepix, **kwargs)

