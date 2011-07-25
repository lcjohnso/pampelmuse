__version__ = 0.4 ## 2011/03/07
import numpy, pyfits, sys
from _my_profiles import Gaussian
## from scipy import interpolate - produces warnings for array sizes >(7x7)px
from matplotlib.mlab import griddata

## v0.3 - 2010/10/14
## v0.4 - 2011/03/07 - added option to provide bad-pixel mask


class SelectAnnulus:
    """
    Class to create boolean array with True entries in annulus-shaped
    region.
    """
    def __init__(self, shape, centers, r_min, r_max):
        """
        Initialize a class to select annulus from array
        
        Parameters
        ----------
        shape : tuple
            Size of the 2D array that is created
        center : tuple
            Pixel coordinates of the center of the annulus
        r_min : float
            Inner radius of the annulus
        r_max : float
            Outer radius of the annulus. If r_max=0, all pixel at
            distances > r_min are selected
        """
        centers = numpy.asarray(centers)

        if centers.ndim == 1:
            centers = centers.reshape(2, -1)

        self.annulus = numpy.ones(shape, dtype='bool')

        for i in xrange(centers.shape[1]):

            self.x_c = centers[0,i]
            self.y_c = centers[1,i]

            calc_array = numpy.fromfunction(self, shape, dtype='float32')

            if r_max != 0:
                self.annulus *= numpy.logical_and(calc_array>r_min,
                                                  calc_array<r_max)

            else:
                self.annulus *= calc_array>r_min

                
    def __call__(self, y, x):
        """
        Measure distance of every pixel to center.
        """
        r2 = (x - self.x_c)**2 + (y-self.y_c)**2
        return numpy.sqrt(r2)



class AperPhoto:
    """
    Perform aperture photometry on a 2 or 3dimensional array. Optionally,
    AperPhoto also calculates the first and second order moments of the
    sources. The centers of the sources have to be provided.
    """

    def __init__(self, data, centers, mask=None):
        """
        Initialize an instance of AperPhoto-class

        Parameters
        ----------
        data : ndarray
            The 2 or 3dim. array on which the aperture photometry should
            be performed. In case of a 3dim. array, the first axis must
            be the spectral one.
        centers : ndarray
            Initial guesses for the centers of all sources for which
            photometric measurements should be performed. The shape
            should be (2, N), giving x & y pixel-coordinates for all
            N sources.
        mask : ndarray or None
            Must be boolean array of same shape as `data` if provided. All
            pixels set to True in `mask` are replaced using 2dim. linear
            interpolation before aperture photometry is performed.
        """
        self.centers = centers
        self.mask    = mask

        self.Nsources = self.centers.shape[1]
        if len(data.shape) == 3:
            self.data    = data
            self.Nimages = self.data.shape[0]
        else:
            self.data    = data.reshape((1,data.shape[0], data.shape[1]))
            self.Nimages = 1
        self.spatpix = self.data.shape[1:]
        
        self.backval = numpy.zeros((self.Nimages,), dtype='float32')
        self.fluxes  = numpy.zeros((self.Nimages, self.Nsources),
                                   dtype='float32')
        self.moments = numpy.zeros((self.Nimages, self.Nsources, 5),
                                   dtype='float32')


        
    def aper_photo(self, r_aper=3., aper_cor=False, fwhm=3.0,
                   calc_moments=True, thresh=3.):
        """
        Method to perform aperture photometry (and moment measurements if
        requested) on every image in the data array. Prior to this, the
        method estimates a global background value for the image using a
        sigma-clipping approach using the _background method of the class.
        If moments are calculated, the source positions are updated using
        the first order moments determinedfor every source.

        Parameters
        ----------
        r_aper : float, optional
            Measurement radius; only pixels with distances <= r_aper to
            the source center are taken into account
        aper_cor : bool, optional
            Whether the calculated fluxes should be corrected for the
            limited aperture size
        fwhm : float, optional
            If aper_cor is True, a Gaussian with FWHM=fwhm is used to
            approximate the PSF and the fluxes are divided by the light
            fraction of the PSF inside the pixels considered.
        calc_moments : bool, optional
            Boolean flag that tells AperPhoto whether moments should be
            calculated
        thresh : float, optional
            Only pixels with values larger than <thresh> times the
            estimated background noise are considered in the moment
            calculation
        """
        self.r_aper  = r_aper
        self.thresh  = thresh
        self.calcmom = calc_moments

        if self.mask != None:
            X, Y = numpy.indices(self.spatpix)


        for i in xrange(self.Nimages):
            if self.mask != None and self.mask[i].any():
                mask_i = (self.mask[i] == False)
##                 interpolator = interpolate.interp2d(X[mask_i], Y[mask_i],
##                                                     self.data[i][mask_i])
##                 data_i = interpolator(X[:,0], Y[0])
                data_i = griddata(X[mask_i],Y[mask_i],self.data[i][mask_i],X,Y)
            else:
                data_i = self.data[i].copy()


            self.backval[i], self.backstd = self._background(self.data[i],
                                                             self.r_aper)
            data_i -= self.backval[i]

            for j in range(self.Nsources):

                x_j = self.centers[0,j]
                y_j = self.centers[1,j]

                sub_ima = self._extract_subima(data_i, (x_j, y_j), self.r_aper)

                ## Moment calculation:
                if self.calcmom:
                    sub_ima *= sub_ima>(self.thresh*self.backstd)
                    self.moments[i,j] = self._measure_moments(sub_ima)

                    self.moments[i,j,0] += max(round(x_j) - self.r_aper, 0)
                    self.moments[i,j,1] += max(round(y_j) - self.r_aper, 0)

                    sub_ima = self._extract_subima(data_i,(self.moments[i,j,0],
                                                           self.moments[i,j,1]),
                                                   self.r_aper)
                else:
                    self.moments[i,j,0] = x_j
                    self.moments[i,j,1] = y_j

                ## Aperture photometry:
                x_sj = min(round(self.moments[i,j,0]), self.r_aper)
                y_sj = min(round(self.moments[i,j,1]), self.r_aper)

                star_annu = SelectAnnulus(sub_ima.shape, (x_sj, y_sj), 0,
                                          self.r_aper)
                if aper_cor:
                    gauss_psf = Gaussian(self.spatpix, x_sj, y_sj, 0., fwhm)
                    corr = gauss_psf.array[star_annu.annulus].sum()
                else:
                    corr = 1.

                self.fluxes[i,j] = sub_ima[star_annu.annulus].sum()/corr


 

    def _background(self, data, r_min, clip_sigma=3.):
        """
        Estimate global background value in a 2dim. data array using
        sigma clipping.

        Returns the mode and the sigma of the background pixel
        distribution.
        
        Parameters
        ----------
        data : ndarray
            The 2dim. data array
        r_min : float
            Radius of circle around a source that is excluded from the
            calculation. The source positions are read from the center
            attribute.
        clip_sigma : float, optional
            During the sigma clipping, all pixel values deviating more
            than <clip_sigma> times the background noise from the
            median are clipped
        """
        back_annu = SelectAnnulus(data.shape, self.centers, r_min, 0)
        back_data = data[back_annu.annulus]
        back_data.sort()
        
        i = 0
        while True:
	
            sigma_bg  = back_data.std()
            median_bg = back_data[back_data.shape[0]>>1]
	
            if i > 0:
	
                if len(back_data) == hist_length:
                    mean_bg = back_data.mean()
                    mode_bg = 3*median_bg - 2*mean_bg
                    break
	                
            hist_length = len(back_data)
	            
            valid = numpy.logical_and(back_data>(median_bg-clip_sigma*sigma_bg),
                                      back_data<(median_bg+clip_sigma*sigma_bg))
	
            back_data = back_data[valid]
            i += 1
	
        return mode_bg, sigma_bg



    def _extract_subima(self, data, center, radius):
        """
        Cut out quadratic area around a central pixel from a 2dim.
        array.

        Returns the cut-out array. If the requested cutout extends
        beyond the original array, only the part lying on the array
        is returned

        Parameters
        ----------
        data : ndarray
            The 2dim. data array
        center : tuple
            The central pixel coodinates of the cutout in the
            original array
        radius : float
            Half edge length of the cutout (in pixels)
        """

        r = int(round(radius))
        x_lo = max(int(round(center[0]))-r, 0)
        y_lo = max(int(round(center[1]))-r, 0)
        x_hi = min(int(round(center[0]))+r+1, data.shape[1])
        y_hi = min(int(round(center[1]))+r+1, data.shape[0])

        return data[y_lo:y_hi, x_lo:x_hi]



    def _measure_moments(self, data):

        """
        Method measures first and second order moments of the pixel
        distribution in a 2dim. array.

        Returns the first and second order moments of pixel
        distribution: (x, y, x2, y2, xy). Note that x corresponds
        the second axis, while y corresponds to the first axis (in
        python notation).

        Parameters
        ----------
        data : ndarray
            The 2dim. data array
        """
        total = float(data.sum())
        Y, X = numpy.indices(data.shape, dtype='float32')
        x  = (X*data).sum()/total
        y  = (Y*data).sum()/total
        
        x2 = (X*X*data).sum()/total - x**2
        y2 = (Y*Y*data).sum()/total - y**2        
        xy = (X*Y*data).sum()/total - x*y
        
##         col = data[:,int(y)]
##         if col.sum() > 0.:
##             sigma_x = math.sqrt(abs(((numpy.arange(col.size())-y)**2)*col).sum()/col.sum())
##         else:
##             sigma_x = 99.
##         row = data[int(x),:]
##         if row.sum() > 0.:
##             sigma_y = math.sqrt(abs(((numpy.arange(row.size())-x)**2)*row).sum()/row.sum())
##         else:
##             sigma_y = 99.
##         fwhm = 2.355*(sigma_x+sigma_y)/2.

        return (x, y, x2, y2, xy)



## OUTDATED!!!
## def aper_photo(root_name, init_guesses, r_aper=3., moments=True, thresh=3.):
##     """
##     Function uses the methods of the AperPhoto-class to do aperture
##     photometry on a datacube (3dim. array)
## 
##     Results are stored as a FITS-table on the disk
## 
##     Parameters
##     ----------
##     root_name : string
##         The datacube must have filename <root_name>.dat.fits; the results
##         are stored under <root_name>.src.fits
##     init_guesses : string
##         Path to an ASCII-file containing approximate x- and y-coordinates
##         of the sources suited for aperture photometry in the cube. The
##         file must be readable by asciidata with x-coordinates as first
##         column and y-coordinates as second column
##     r_aper : float, optional
##         Radius for aperture photometry
##     moments : bool, optional
##         Whether moments for the sources should be calculated
##     thresh : float, optional
##         Only consider pixels with values above <thresh> times the
##         background noise in moment calculation.
##     """
## 
## 
##     ## read data cube
##     indata, head  = pyfits.getdata(root_name+'.dat.fits', header=True)
## 
##     ## get wavelength information
##     wavl = numpy.arange(indata.shape[0], dtype='float32')
##     wavl *= head['CDELT3']
##     wavl += head['CRVAL3']
## 
##     ## get center coordinates from ASCII-file <init_guesses>
##     centers = [[],[]]
##     srclist = asciidata.open(init_guesses)
##     for i in xrange(srclist.nrows):
##         centers[0].append(srclist[1][i])
##         centers[1].append(srclist[2][i])
##     centers = numpy.asarray(centers)
##     Nsources = centers.shape[1]
## 
##     results = numpy.zeros((3*Nsources+1, indata.shape[0]),
##                           dtype='float32')
## 
##     ## Source columns will be numbered consecutively
##     names = ['BACK']
##     for i in xrange(Nsources):
##         names.extend(['XPOS%i' %i, 'YPOS%i' %i, 'FLUX%i' %i])
## 
##     ## Aperture photometry for every image is performed
##     for i in xrange(indata.shape[0]):
##         ap_ima = AperPhoto(indata[i], centers)
##         ap_ima.aper_photo(moments=moments)
## 
##         results[0,i] = ap_ima.backval
## 
##         for j in xrange(Nsources):
##             results[1+3*j,i] = ap_ima.moments[j,1]  ## exchange x- & y-axis
##             results[2+3*j,i] = ap_ima.moments[j,0]
##             results[3+3*j,i] = ap_ima.fluxes[j]
## 
##         sys.stdout.write('\rProcessing image %0*d/%0*d' %(4, i+1, 4,
##                                                           indata.shape[0]))
##         sys.stdout.flush()
## 
##     print ""
## 
##     ## use median position of first source as reference
##     x_ref = numpy.median(results[1])
##     y_ref = numpy.median(results[2])
## 
##     for j in xrange(Nsources):
##         results[1+3*j] -= x_ref
##         results[2+3*j] -= y_ref
## 
##     
##     columns = []
## 
##     columns.append(pyfits.Column(name='WAVELEN', format='D', array=wavl))
## 
##     for i in xrange(results.shape[0]):
##         columns.append(pyfits.Column(name=names[i], format='D',
##                                      array=results[i]))
##     
##     tblHDU = pyfits.new_table(columns)
##     tblHDU.header.update('XREF', x_ref)
##     tblHDU.header.update('YREF', y_ref)
## 
##     priHDU = pyfits.PrimaryHDU()
## 
##     HDUList = pyfits.HDUList([priHDU, tblHDU])
##     HDUList.writeto(root_name+'.src.fits', clobber=True)
## 

