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
2011/11/29, v0.0.0.1, added logging
2011/12/05, v0.0.1.2, add `quiet'-option to supress logging (used when
                      calling `trans_coords' during the PSF-fit of
                      `fit_array').
                      supplying a numpy integer as an argument to the `format'
                      property of a string seemed to cause problems.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.1.2"
import logging
import numpy as np



def trans_coords(incoords, outcoords, quiet=False):
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
    if not quiet:
        logging.info("Determining coordinate transformation using {0} sources."\
                         .format(m))
    if m<4:
        logging.error("Number of sources <4!")
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

    tmpResult = [] ## prepare output
    dtype     = [('A', 'f8'), ('B', 'f8'), ('C', 'f8'), ('D', 'f8')]
    
    if outcoords['x'].ndim == 1:
        b[:m] = outcoords['x'][outSel]
        b[m:] = outcoords['y'][outSel]
        
        tmpResult.append(tuple(np.linalg.lstsq(A, b)[0]))

        if not quiet:
            logging.info("Determined coordinate transformation for 1 image.")

    else:
        for i in xrange(outcoords['x'].shape[1]):

            b[:m] = outcoords['x'][outSel,i]
            b[m:] = outcoords['y'][outSel,i]

            tmpResult.append(tuple(np.linalg.lstsq(A, b)[0]))

        if not quiet:
            logging.info("Determined coordinate transformation for {0} images."\
                             .format(outcoords['x'].shape[1]))

    result = np.array(tmpResult, dtype=dtype)

    if not quiet:
        logging.info("Best-fit parameters:")
        for name in result.dtype.names:
            logging.info("   {0} = {1:8.3f}".format(name, result[name].mean()))

        logging.info("Offsets of individual sources:")
        logging.info("       ID     dx     dy")
        for i in xrange(m):
            J = inSel[i]
            K = outSel[i]
            x_off = np.mean(outcoords['x'][K] - result['A']*incoords['x'][J]\
                                - result['B']*incoords['y'][J] - result['C'])
            y_off = np.mean(outcoords['y'][K] - result['A']*incoords['y'][J]\
                                + result['B']*incoords['x'][J] - result['D'])
            logging.info("   {0:6d}  {1:+5.2f}  {2:+5.2f}".format(\
                    int(incoords['id'][J]), x_off, y_off))

    return result
