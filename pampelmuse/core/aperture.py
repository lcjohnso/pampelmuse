"""
APERTURE
========
The routine is designed to perform aperture photometry on a 2dim. image or
aperture spectrophotometry a 3dim. datacube. 

The main routine is the class `AperPhoto` that performs aperture (spectro)-
photometry for a list of input positions, provided as a structured array. A
local background around every source is estimated and subtracted before its
flux and moments are determined. Additonally the routine estimates the global
background in the data.

The class `SelectAnnulus' is called to extract doughnut-shaped regions from
the input data and is frequently called by `AperPhoto'.


CHANGELOG:
----------
2011/09/05, v0.0.0.0, significantly modified existing older routine
2011/09/08, v0.1.0.0, changed usage of PSF in code. Now a PSF instance
                      can be provided to the `aper_photo'-method. In that
                      case, an aperture correction is automatically
                      performed using the given PSF. (previously, provided
                      FWHM was used to create a Gaussian PSF)
2011/09/19, v0.1.1.0, fixed bug that the whole sky spectrum was replaced
                      with the latest estimate of the global background
2011/12/02, v0.1.1.1, included logging-module
2011/12/05, v0.1.2.1, bugfixed logging
2012/03/16, v0.4.0.0, modified the code to use the dedicated `PSF`-class
                      defined by PampelMuse. Also adapted the current
                      version number to the version number of PampelMuse.
2012/03/19, v0.4.0.1, removed a remaining print-command used for debugging.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.4.0.1"
import logging, sys
import numpy as np, pyfits
from matplotlib.mlab import griddata

## required PampelMuse-packages
## None required, instance of the `PSF`-class provided



class SelectAnnulus:
    """
    Class to create boolean array with True entries in annulus-shaped
    region.
    """
    def __init__(self, shape, centers):
        """
        Initialize a class to select annulus from array
        
        Parameters
        ----------
        shape : tuple
            Size of the (2dim.) array to work on.
        center : numpy_record_array
            Must contain the fields 'id', 'x' and 'y'. 'x' and 'y' are
            pixel coordinates of the sources in the array.
        """

        self.shape   = shape
        self.centers = centers

        for name in ['id', 'x', 'y']:
            if not name in self.centers.dtype.names:
                logging.critical(\
                    "Field '{0}' not found in 'centers'-array".format(name))


    def __call__(self, ids, r_min, r_max):
        """
        Select pixels in an annulus around one or more sources.

        Parameters
        ----------
        ids : integer or array_like
            The ID(s) of the source(s).
        r_min : float
            Inner radius of the annulus
        r_max : float
            Outer radius of the annulus. If r_max=0, all pixel at
            distances > r_min are selected

        Returns
        -------
        selected_pixel : array_like
            A 2dim. boolean array of with shape `shape` and all pixels set to
            True that:
                (a) are at a distance >=r_min  wrt to all selected sources.
               [(b) are at a distance <=r_max to ONE selected source.]
        """

        self.IDs = np.asarray([ids]).flatten()
        selected = [np.where(self.centers['id']==ID)[0][0] for ID in self.IDs]

        dst = np.zeros((self.IDs.size, self.shape[0], self.shape[1]),
                       dtype='float32')

        j = 0
        for i in selected:
            self.x_c = self.centers['x'][i]
            self.y_c = self.centers['y'][i]

            dst[j] = np.fromfunction(self._calc, self.shape, dtype='float32')
            j += 1

        minDst = dst.min(axis=0)

        if r_max != 0:
            outArray = (minDst>=r_min)&(minDst<=r_max)
        else:
            outArray = (minDst>=r_min)

        if not outArray.any():
            logging.warn("Annulus does not contain any pixels.")

        return outArray


    def _calc(self, y, x):
        """
        Measure distance of every pixel to center. Method is called
        automatically by '__call__'.
        """

        r2 = (x - self.x_c)**2 + (y - self.y_c)**2
        return np.sqrt(r2)



class AperPhoto:
    """
    Perform aperture (spectro)photometry on 2 or 3dimensional data using
    a provided list of sources. The calculation of the first and second order
    moments for the provided sources  is optional. Prior to any flux or moment
    measurement, a background (global or local if  requested) is determined
    and subtracted. 
    """

    def __init__(self, data, centers, mask=None):
        """
        Initialize an instance of AperPhoto-class

        Parameters
        ----------
        data : array_like of pyfits HDU-instance
            The 2 or 3dim. array on which the aperture photometry should
            be performed. In case of a 3dim. array, the first axis must
            be the spectral one. Also, in the case of a 3dim. array a pyfits
            HDU-instance can be provided that will then be loaded into memory
            image-by-image (useful for large data-cubes).
        centers : numpy_record_array
            Initial guesses for the pixel coordinates of all sources for which
            photometric measurements should be performed. The record array must
            contain fields named 'id', 'x' and 'y'.
        mask :  array_like of pyfits HDU-instance
            Must be boolean array of same shape as `data` if provided. All
            pixels set to True in `mask` are replaced using 2dim. linear
            interpolation before aperture photometry is performed.
            
        Returns
        -------
        The newly created instance
        """
        logging.info("Starting aperture photometry, validating input.")

        ## reading information about sources
        self.centers = centers
        for name in ['id', 'x', 'y']: ## check if input is valid
            if not name in self.centers.dtype.names:
                logging.critical(\
                    "Field '{0}' not found in `centers'".format(name))

        self.Nsources = self.centers.size
        logging.info("Found {0} valid sources.".format(self.Nsources))

        ## prepare data array, check whether array or HDU provided
        if len(data.shape) == 3:
            self.data    = data
            if isinstance(self.data, (pyfits.PrimaryHDU, pyfits.ImageHDU)):
                self.dataFITS = True
            else:
                self.dataFITS = False
            self.Nimages = self.data.shape[0]
        else:
            self.data     = data.reshape((1,data.shape[0], data.shape[1]))
            self.dataFITS = False
            self.Nimages  = 1
        self.spatpix = self.data.shape[1:]
        logging.info("Input data is {0}D, with {1} spectral pixels."\
                         .format(2+(self.Nimages>1),  self.Nimages))

        ## prepare mask array (if any), check whether array or HDU provided
        self.mask = mask
        if isinstance(self.mask, (pyfits.PrimaryHDU, pyfits.ImageHDU)):
            self.maskFITS = True
        else:
            self.maskFITS = False
        if self.mask != None:
            if self.mask.shape != self.data.shape:
                logging.critical("Data and mask have different shapes.")

        ## prepare record array that will contains output.
        self.results = np.zeros(self.Nsources+1, dtype=[\
                ('id',np.int32),
                ('spectrum', np.float32, (self.Nimages,)),
                ('x', np.float32, (self.Nimages,)),
                ('y', np.float32, (self.Nimages,)),
                ('x2', np.float32, (self.Nimages,)),
                ('y2', np.float32, (self.Nimages,)),
                ('xy', np.float32, (self.Nimages,))
                ])
        

        
    def aper_photo(self, rAper=3., psf=None, calcMoments=True, thresh=3.,
                   localBG=True):
        """
        Method performs the actual aperture (spectro)photometry (and
        measures the moments if requested). An estimate of the global
        background is determined using a sigma-clipping approach.
        Optionally, a local background around every soource can be
        determined.

        Note that the aperture (spectro)photometry is always performed
        around the input coordinates, NOT around the first order moments
        (even if they are calculated).

        Parameters
        ----------
        rAper : float, optional
            Measurement radius; only pixels with distances <= rAper to
            the source center are taken into account
        psf : a PSF instance, optional
            An instance of a PSF-class can be provided. In that case,
            the provided PSF is used to perform an aperture correction.
        calcMoments : bool, optional
            Boolean flag that tells AperPhoto whether moments should be
            calculated
        thresh : float, optional
            Only pixels with values larger than <thresh> times the
           estimated background noise are considered in the moment
            calculation.
        localBG : boolean, optional
            Flag indicating whether a local background for every source
            should be estimated.
            
        Returns
        -------
        results : numpy_record_array
            For every source, the flux/spectrum is provided in field
            'spectrum' and moments are provided in fields 'x', 'y', 'x2',
            'y2' and 'xy'.
            Note that the global background flux/spectrum is stored in the
            'spectrum' slot of a hypothetical source with 'id' -1.
        """
        self.rAper   = rAper
        self.thresh  = thresh
        self.calcMom = calcMoments
        logging.info("Aperture photometry - starting calulation with:")
        logging.info("   Aperture radius (rAper) = {0:.1f}pixel".format(rAper))
        logging.info("   Pixel threshold (thresh) = {0}sigma".format(thresh))
        logging.info("   Moment calculation? - {0}".format(calcMoments))
        logging.info("   Local background calculation? - {0}".format(localBG))

        if psf != None:
            logging.info("   Aperture correction with `{0}'-PSF.".format(\
                    psf.psftype))
        else:
            logging.info("   No aperture correction applied.")

        if self.mask != None or self.calcMom:
            Y, X = np.indices(self.spatpix)

        self.results['id'][:self.Nsources] = self.centers['id']
        self.results['id'][self.Nsources]  = -1 ## the average sky spectrum

        self.annuSelect = SelectAnnulus(self.spatpix, self.centers)

        for i in xrange(self.Nimages):
            logging.info("Aperture spectrophotometry - image #{0}/{1})".format(\
                    i+1, self.Nimages))

            if self.mask !=  None:
                if self.maskFITS: mask_i = (self.mask.section[i]==0)
                else: mask_i  = (self.mask[i] == 0)

            if self.dataFITS: data_i = self.data.section[i]
            else:  data_i = np.copy(self.data[i])

            ##  if mask provided, interpolate masked values
            if self.mask != None:
                logging.debug("   Implementing mask.")
                data_i = griddata(Y[mask_i], X[mask_i], data_i[mask_i], Y, X)

            ##  estimate global background value.
            logging.debug("   Determining global background")
            self.backval, self.backstd = self._estGlobalBG(data_i)
            self.results['spectrum'][self.Nsources,i] = self.backval
            logging.debug("   Value={0:.3f}/pixel".format(self.backval))

            for j in xrange(self.Nsources):
                curID = self.centers['id'][j]
                logging.debug("   Source #{0}/{1} (ID={2}):".format(\
                        j, self.Nsources, curID))

                ## estimate local background (if requested)
                if localBG:
                    backval_ji,backstd_ji = self._estLocalBG(data_i,curID=curID)
                    logging.debug("      Background: {0}/pixel (local)".format(\
                            backval_ji))

                ## otherwise use global background
                else:
                    backval_ji,backstd_ji = self.backval,self.backstd
                    logging.debug("      Background: {0}/pixel (glob.)".format(\
                            backval_ji)) 

                ## Get array for calculations
                selPix_ji  = self.annuSelect(ids=curID, r_min=0,
                                             r_max=self.rAper)
                selData_ji = np.copy(data_i[selPix_ji]) - backval_ji

                ## Moment calculation:
                if self.calcMom:
                    aboveCut = (selData_ji>(self.thresh*backstd_ji))
                                  
                    if aboveCut.sum() > 3:
                        logging.debug("      Moments:")
                        print X[selPix_ji][aboveCut], Y[selPix_ji][aboveCut]
                        moments = self._measureMoments(selData_ji[aboveCut],
                                                       X=X[selPix_ji][aboveCut],
                                                       Y=Y[selPix_ji][aboveCut])
                        k = 0
                        for moment in ['x', 'y', 'x2', 'y2', 'xy']:
                            self.results[moment][j,i]  = moments[k]
                            logging.debug("       {0}={1}".format(moment,
                                                                  moments[k]))
                            k += 1
                    else:
                        logging.error(\
                            "Too few pixels (<4) to calculate moments.")
                        self.results['x'][j][i] = self.centers['x'][j]
                        self.results['y'][j][i] = self.centers['y'][j]

                else:
                    self.results['x'][j][i] = self.centers['x'][j]
                    self.results['y'][j][i] = self.centers['y'][j]

                ## Aperture photometry:
                if psf != None:
                    psfProf = psf(index=i*psf.disp) 
                    psf_ji  = psfProf(size=self.spatpix,
                                      x_c=self.centers['x'][j],
                                      y_c=self.centers['y'][j],
                                      mag=0., maxrad=15, return_sparse=False)

                    corr =  psf_ji[selPix_ji].sum()
                else:
                    corr = 1.

                self.results['spectrum'][j][i] = selData_ji.sum()/corr
                logging.debug("Flux: {0}counts".format(\
                        self.results['spectrum'][j][i]))

        return self.results
 


    def _estGlobalBG(self, data):
        """
        Estimate global background value in a 2dim. data array using
        sigma clipping.
        
        Parameters
        ----------
        data : ndarray
            The 2dim. data array

        Returns
        -------
            The mode and the sigma of the background pixel
            distribution.
        """
        backPix = self.annuSelect(ids=self.centers['id'], r_min=self.rAper,
                                  r_max=0)
        backData = data[backPix]
        backData.sort()
        
        if backData.size < 20:
            logging.warn("Using < 20 pixels to estimate global background.")

        return self.__clipSigma(backData)



    def _estLocalBG(self, data, curID):
        """
        Estimate local background value  for source 'curID' using
        sigma clipping. The background is calculated in an annulus with
        inner radius 'self.rAper' and outer radius 'self.rAper'+5.

        Returns the mode and the sigma of the background pixel
        distribution.
        
        Parameters
        ----------
        data : ndarray
            The 2dim. data array
        curID : integer
            The ID of the investigated source.

        Returns
        -------
            The mode and the sigma of the background pixel
            distribution.
        """
        backPix = self.annuSelect(ids=curID, r_min=2*self.rAper,
                                  r_max=4*self.rAper)
        backData = data[backPix]
        if backData.size < 10:
            logging.warn("Using < 10 pixels to estimate local background.")
        backData.sort()
        
        return self.__clipSigma(backData)



    def __clipSigma(self, data, thresh=3.):
        """
        Perform sigma-cliipping on 1dim. array

        Parameters
        ----------
        data : nd_array
            The array containing the data
        thresh : float, optional
            During the sigma clipping, all pixel values deviating more
            than 'thresh' times the background noise from the
            median are clipped

        Returns
        -------
        mode : float
            The sigma-clipped mode of the data
        sigma : float
            The sigma-clipped standard deviation of the data
        """

        i = 0
        while True:
	
            sigma  = data.std()
            median = data[data.shape[0]>>1]
	
            if i > 0:
	
                if len(data) == hist_length:
                    mean = data.mean()
                    mode = 3*median - 2*mean
                    break
	                
            hist_length = len(data)
	            
            valid = np.logical_and(data>(median-thresh*sigma),
                                   data<(median+thresh*sigma))

            if valid.sum() < 5:
                logging.warn("Sigma-clipping left <5 valid pixels.")
	
            data = data[valid]
            i += 1
	
        return mode, sigma



    def _measureMoments(self, data, X, Y):

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

        if data.size < 5:
            logging.warn("Less than 5 pixels used to calculate moments.")

        total = float(data.sum())
        x  = (X*data).sum()/total
        y  = (Y*data).sum()/total
        
        x2 = (X*X*data).sum()/total - x**2
        y2 = (Y*Y*data).sum()/total - y**2        
        xy = (X*Y*data).sum()/total - x*y
        
        if x2<0:
            logging.error("Source has unphysical moments: <x^2> < 0")
            x2=0

        if y2<0:
            logging.error("Source has unphysical moments: <y^2> < 0")
            y2=0

        if xy**2 >= x2*y2:
            logging.error("Source has unphysical moments: <xy>^2 >= <x^2><y^2>")
            xy = 0

##         col = data[:,int(y)]
##         if col.sum() > 0.:
##             sigma_x = math.sqrt(abs(((np.arange(col.size())-y)**2)*col).sum()/col.sum())
##         else:
##             sigma_x = 99.
##         row = data[int(x),:]
##         if row.sum() > 0.:
##             sigma_y = math.sqrt(abs(((np.arange(row.size())-x)**2)*row).sum()/row.sum())
##         else:
##             sigma_y = 99.
##         fwhm = 2.355*(sigma_x+sigma_y)/2.

        return (x, y, x2, y2, xy)
