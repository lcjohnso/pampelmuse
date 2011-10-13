import numpy



def cut_area(in_array, center, radius, output=''):
    x_i = round(center[0], 0)
    y_i = round(center[1], 0)

    shape = in_array.shape

    out_array = numpy.zeros((2*radius+1, 2*radius+1), dtype='float32')
    out_shape = out_array.shape

    xmin = max(0, int(x_i - radius))
    xmax = min(shape[0], int(x_i + radius + 1))
    ymin = max(0, int(y_i - radius))
    ymax = min(shape[1], int(y_i + radius + 1))

    xlo = max(0, int(radius - x_i))
    xhi = out_shape[0] - max(0, int(x_i + radius + 1 - shape[1]))
    ylo = max(0, int(radius - y_i))
    yhi = out_shape[1] - max(0, int(y_i + radius + 1 - shape[0]))

    out_array[xlo:xhi,ylo:yhi] = in_array[xmin:xmax,ymin:ymax]

    if output == 'full':
        filled_pix = numpy.zeros(out_shape, dtype='int16')
        filled_pix[xlo:xhi,ylo:yhi] += 1
        
        return out_array, filled_pix

    else:
        return out_array



def paste_area(in_array, out_shape, refpix_in, refpix_out, out_array=None):
    
    xpix_in = int(round(refpix_in[0], 0))
    ypix_in = int(round(refpix_in[1], 0))
                 
    xpix_out = int(round(refpix_out[0], 0))
    ypix_out = int(round(refpix_out[1], 0))

    in_shape = in_array.shape

    xmin = max(0, xpix_in - xpix_out)
    xmax = in_shape[1] - max(0, (in_shape[1]-xpix_in) - (out_shape[1]-xpix_out))
    ymin = max(0, ypix_in - ypix_out)
    ymax = in_shape[0] - max(0, (in_shape[0]-ypix_in) - (out_shape[0]-ypix_out))

    xlo = max(0, xpix_out - xpix_in)
    xhi = out_shape[1] - max(0, (out_shape[1]-xpix_out) - (in_shape[1]-xpix_in))
    ylo = max(0, ypix_out - ypix_in)
    yhi = out_shape[0] - max(0, (out_shape[0]-ypix_out) - (in_shape[0]-ypix_in))

    if out_array == None:
        out_array = numpy.zeros(out_shape, dtype='float32')
    else:
        if not out_array.shape == out_shape:
            raise IOError, "Shape of provided 'out_array' differs from "\
                +"provided 'out_shape'"
        
    if ylo < out_array.shape[0] and yhi > 0:
        if xlo < out_array.shape[1] and xhi > 0:

            out_array[ylo:yhi,xlo:xhi] = in_array[ymin:ymax,xmin:xmax]

    return out_array
