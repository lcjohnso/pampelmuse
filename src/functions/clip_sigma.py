__version__ = 0.3 ## 2010/11/10

## 2010/11/10 - v0.2
## 2011/02/22 - v0.3 - bugfix: axes order of goodpix array did'nt match with
##                             input data         

import numpy


def clip_sigma(data, axis=None, clip=3., return_mode=False):
    """
    Average data along given axis using sigma-clipping.
   
    Parameters
    ----------
    data : nd_array
        Elements to average
    axis : integer, optional
        Axis over which the data is average. By default `axis`
        is None, and all elements are averaged.
    clip : float, optional
        All pixels whose values deviate from the median of the
        distribution by more than `clip` times the sigma of the
        distribution are excluded.
    return_mode : boolean, optional
        If true, the mode of the distribution is returned. By default,
        `return_mode` is False and the mean is returned.

    Returns
    -------
    outdata : ndarray
        An array with the same shape as `data`, with the specified
        axis removed.  If `axis` is None, a scalar is returned.
    goodpix : ndarray
        A boolean array with the same shape as `data` and clipped
        pixels set to zero.
    """
    if axis != None:
        if data.shape[axis] == 1:
                return [data.squeeze(), numpy.ones(data.shape, dtype="bool")]

    if axis == None or data.ndim == 1:
        clpdata = data.reshape(1,-1)
    else:
        tmpdata = numpy.rollaxis(data, axis, data.ndim)
        clpdata = tmpdata.reshape(-1,data.shape[axis])


    tmp_out = numpy.zeros(clpdata.shape[0], dtype='float32')
    tmp_pix = numpy.ones(clpdata.shape, dtype='bool')

    for i in xrange(clpdata.shape[0]):
        j = 0

        while True:

            mean   = clpdata[i,tmp_pix[i]].mean()
            sigma  = clpdata[i,tmp_pix[i]].std()
            median = numpy.median(clpdata[i,tmp_pix[i]])
	
            if j > 0:
	
                if tmp_pix[i].sum() in [2, hist_length]:
                    mode = 3*median - 2*mean
                    break
 	                
            hist_length = tmp_pix[i].sum()
	            
            tmp_pix[i] = numpy.logical_and(clpdata[i]>(median-clip*sigma),
                                           clpdata[i]<(median+clip*sigma))
	
            j += 1
            
        if return_mode:
            tmp_out[i] = mode
        else:
            tmp_out[i] = mean
        
    if axis == None or data.ndim == 1:
        return [tmp_out[0], tmp_pix.reshape(data.shape)]
    else:
        outdata = tmp_out.reshape(tmpdata.shape[:-1])
        goodpix = tmp_pix.reshape(tmpdata.shape)

        return [outdata, numpy.rollaxis(goodpix, data.ndim-1, axis)]



