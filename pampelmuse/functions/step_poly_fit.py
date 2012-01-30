"""
STEPPOLYFIT
===========
The purpose of this routine is to fit a sequence of polynomials to a 1dim.
data array. To this aim, the data is split into a given number of bins (having
equal sizes) and each bin is fit with a polynomial of given order. The result
is given by the combination of the fits to all bins. To make sure that the
result is smooth, the bins overlap. In the overlap region, a smooth transition
from one polynomial to the other happens.

CHANGELOG:
----------
2012/01/17, v0.0.0.0
2012/01/25, v0.0.0.1, added option to provide a bad-pixel mask. Incresased
                      fit region for each polynomial out to 2*overlap.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.0.1"
import numpy as np


def StepPolyFit(data, order=3, bins=4, overlap=150, mask=None):
    """
    Fit a 1dimensional data array with a sequence of polynomials.

    Parameters
    ----------
    data : nd_array, 1dim.
        The data array that should be fitted.
    order : integer, optional
        The order of the individual polynomials used to fit the data.
    bins : integer, optional
        The number of bins into which the data array is split.
    overlap : integer, optional
        The overlap region (in pixels) used to match the individual fits.
        This region is needed to obtain a smooth fit to the overall data
        array.
    mask : nd_array, optional
        If provided, `mask' must be a boolean array with the same size as
        `data'. All data pixels that are flagged False in the mask are
        ignored in the fit.

    Returns
    -------
    result : nd_array
        The result of the fitting process.
    """
    
    if mask == None:
        mask = np.ones(data.size, dtype=np.bool)

    weights   = -1*np.ones((bins, data.size), dtype="float32")
    polyVals  = np.zeros((bins, data.size), dtype="float32")
    distToBin = np.zeros((2, data.size), dtype="float32")

    pixPerBin  = int(data[mask].size/bins)+1
    pixPerBin -= overlap
    indices    = mask.cumsum()-1
    pixCoords  = np.arange(data.size, dtype=np.int32) 

    for i in xrange(bins):
        jMin = i*(pixPerBin+overlap) 
        if i==(bins-1):
            jMax = data[mask].size
        else:
            jMax = (i+1)*pixPerBin+i*overlap

        distToBin[0] = jMin-indices
        distToBin[1] = indices-jMax
        
        distToBin_i = np.max(distToBin, axis=0)

        core = (distToBin_i<=0)
        edge = (distToBin_i>0)&(distToBin_i<2*overlap)

        weights[i, core] = 1
        weights[i, edge] = (overlap-distToBin_i[edge])/overlap

        weights[i, weights[i]<-1] = -1
        weights[i, weights[i]>1]  =  1
        
        used              = (weights[i]>-1)
        polyFit_i         = np.polyfit(pixCoords[used&mask], data[used&mask],
                                       deg=order)
        polyVals[i, used] = np.polyval(polyFit_i, pixCoords[used])

##     outFile = open("weights.dat", "w")
##     for j in xrange(data.size):
##         outFile.write("{0}".format(j))
##         for k in xrange(bins):
##             outFile.write("  {0:5.3f}".format(float(weights[k,j])))
##             outFile.write("  {0:5.3f}".format(float(polyVals[k,j])))
##         outFile.write("\n")
##     outFile.close()

    weights[weights<0] = 0

    return (weights*polyVals).sum(axis=0)
