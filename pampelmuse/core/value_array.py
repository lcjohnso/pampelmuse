"""
VALUE_ARRAY
===========
The provided class is a subclass of numpy.ndarray and is designed to allow
for a convenient I/O of fit parameters such as PSF shape parameters or source
coordinates.

The idea is that only such parameters of the ValueArray are updated that are
`free' parameters and do not have a `poly'nomial fit defined.

When obtaining parameters from the array the array will return the `poly'nomial
fit if it is available, otherwise the parameter value.

NOTE that the array must be 3dimensional with the first array axis having
length=2, containing value and polynomial fit (if any).

CHANGELOG
---------
v0.1.0.0, 2012/03/05

"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.0.0"
import logging
import numpy as np


class ValueArray(np.ndarray):

    def __new__(cls, input_array, free=False, poly=False):

        obj = np.asarray(input_array).view(cls)

        obj.free = np.resize(free, input_array.shape[1])
        obj.poly = np.resize(poly, input_array.shape[1])

        return obj



    def __array_finalize__(self, obj):

        if obj == None:  return
        
        self.free = getattr(obj, 'free', False)
        self.poly = getattr(obj, 'poly', False)



    @property
    def toPut(self):
        return (self.free)&(self.poly==False)


    
    @property
    def toGet(self):
        return np.asarray(self.poly,  dtype=np.int16)



    def __setslice__(self, i, j, value):
        
##         logging.error("Slicing is not yet implemented.")

        np.ndarray.__setslice__(self, i, j, value)



    def __setitem__(self, indices, value):

        if isinstance(indices, int):
            if self.toPut[indices]:
                _indices = (0, indices, slice(None, None, None))
                np.ndarray.__setitem__(self, _indices, value)

        elif len(indices) == 1:
            if self.toPut[indices]:
                _indices = (0, indices, slice(None, None, None))
                np.ndarray.__setitem__(self, _indices, value)

        else:
            i = indices[0]
            j = indices[1]
            if isinstance(i, int):
                if self.toPut[i]:
                    np.ndarray.__setitem__(self, (0, i, j), value)
            else:
                if i == Ellipsis or i == slice(None, None, None):
                    _indices =(0, self.toPut[i], j)
                else:
                    _indices =(0, np.arange(self.shape[1])[i][self.toPut[i]], j)
   
                value = np.asarray(value)
                if value.shape == (): _value=value
                else: _value = value[_indices[1]]

                np.ndarray.__setitem__(self, _indices, _value)


    
    def __getslice__(self, i, j):

##         logging.error("Slicing is not yet implemented.")

        return np.ndarray.__getslice__(self, i, j)



    def __getitem__(self, indices):

        if self.ndim<3:
            return np.ndarray.__getitem__(self, indices)
        if not isinstance(indices, int):
            if len(indices)==3:
                return np.ndarray.__getitem__(self, indices)

        if isinstance(indices, int):
            if indices>=0:
                indices = (self.toGet[indices], indices)
            else:
                indices = ([self.toGet[indices]], [indices])
        elif len(indices) == 1:
            indices = (self.toGet[indices], np.arange(self.shape[1])[indices])
        else:
            indices = (self.toGet[indices[0]],
                       np.arange(self.shape[1])[indices[0]],
                       indices[1])

        return np.ndarray.__getitem__(self, indices)


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)

    x = np.arange(12000).reshape(2,3,-1)
    y = np.array([-1,-2,-3])

    test = ValueArray(x, free=[True, False, True], poly=[True, False, False])

    print x
    print "----------------------------------------"

    
    t0 =  time.time()
    print "Read:"
    print test[2]

    t1 = time.time()
    print "Set:"
    test[2] = y[0]

    t2 = time.time()
    
    print test[2]
    print t1-t0, t2-t1
    print "----------------------------------------"

    print test.view(np.ndarray)[0]
