"""
TRANSCOORDS
===========
Given two arrays containing input- and output-coordiates, the routine
finds the best-fit transformation (A,B,C,D) between the two. The 
transformation formulae are:
(i)  x_out = A*x_in + B*y_in + C
(ii) y_out = A*y_in - B*x_in + D

The inverse transformation is given by the following formulae:
(iii) x_in = A'*x_out + B'*y_out + C'
(iv)  y_in = A'*y_out - B'*x_out + D'

with
A' = A/(A*A+B*B);  B' = -B/(A*A+B*B);  C' = (B*D-A*C)/(A*A+B*B);
D' = -(A*D+B*C)/(A*A+B*B)

CHANGELOG
---------
2011/09/06, v0.0.0.0
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.0.0"
import numpy as np



def trans_coords(incoords, outcoords):
    """
    Perform transformation from input- to output-coordinates using linear
    least-squares.

    Parameters
    ----------
    incoords : numpy_record_array
        The input coordinates given as a structured array. The fields
        'id', 'x' and 'y' must be present and one-dimensional. 
    outcoords :  numpy_record_array
        The output coordinates given as a structured array. Again, the
        fields 'id', 'x' and 'y' must be present. 'x' and 'y' can be 2dim.
        fields in which case the coordinate transformation is determined
        for every set of coordinates.
        
    Returns
    -------
    results : numpy_record_array
        A structured array containing the fields 'A', 'B', 'C' and 'D'
        with the values of the coordinate transformation. The size of
        the fields is controlled by the size of the outut coordinates.
        
    """
    for name in ['id', 'x', 'y']:   ## check if input is valid
        if not name in incoords.dtype.names:
            raise IOError, "No field named '{0}'".format(name)\
                +" was found in `incoords'-array"
        if not name in outcoords.dtype.names:
            raise IOError, "No field named '{0}'".format(name)\
                +" was found in `outcoords'-array"
    
    inSel  = []
    outSel = []

    ## find matching coordinate pairs
    for i in xrange(incoords.size):
        inID = incoords['id'][i]
        if not inID in outcoords['id']:
            continue
        inSel.append(i)
        outSel.append(np.where(outcoords['id']==inID)[0][0])
    

    m = len(inSel) ## No of matching sources provided
    if m<4:
        raise IOError, "Need at least 4 coordinate pairs to determine"\
            +" transformation, found only {0}".format(m)

    M = 2*m
    N = 4

    A = np.ones((M,N), dtype='float32')  ## fill matrix
    A[:m,0] = incoords['x'][inSel]
    A[m:,0] = incoords['y'][inSel]
    A[:m,1] = incoords['y'][inSel]
    A[m:,1] = -incoords['x'][inSel]
    A[m:,2] = 0.
    A[:m,3] = 0.

    b = np.ones((M,), dtype='float32')

    results = [] ## prepare output
    dtype   = [('A', 'f8'), ('B', 'f8'), ('C', 'f8'), ('D', 'f8')]
    
    if outcoords['x'].ndim == 1:
        b[:m] = outcoords['x'][outSel]
        b[m:] = outcoords['y'][outSel]
        
        results.append(tuple(np.linalg.lstsq(A, b)[0]))

    else:
        for i in xrange(outcoords['x'].shape[1]):

            b[:m] = outcoords['x'][outSel,i]
            b[m:] = outcoords['y'][outSel,i]

            results.append(tuple(np.linalg.lstsq(A, b)[0]))

    return np.array(results, dtype=dtype)
