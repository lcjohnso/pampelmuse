"""
aperture.py
===========

Provides
--------
class pampelmuse.core.aperture.AperPhoto

Purpose
-------
This module implements  the class 'AperPhoto' that is designed to perform
aperture photometry on a 2dim. image or aperture spectrophotometry a 3dim.
datacube. An instance of the class can be initialized from IFS data and a
cources catalogue. To this aim, a valid instance of pampelmuse.core.cube.Cube
and a valid instance of pampelmuse.core.sources.Sources must be provided.
The aperture photometry is performed for each layer of the IFS data indepen-
dently. A local background around every source is estimated and subtracted
before its flux and moments are determined. If requested, the routine also
estimates the global background in each layer of the data.

Provides
--------
function pampelmuse.core.aperture.run_ap


Latest SVN revision
-------------------
182, 2014/06/30
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 182

import itertools, logging, sys
from multiprocessing import cpu_count, Pool
import numpy as np
from scipy import spatial

## required PampelMuse-packages
from pampelmuse.core import Cube
from pampelmuse.core.psf import Profile
from pampelmuse.functions.statistics import clip_sigma



def run_ap(centroids, pixpos, data, background="global", r_aper=3., calc_moments=False, thresh=3.):
    """
    This function performs aperture photometry on one or more images around
    a set of input coordinates. Each image must be a 1dim. array, and the
    positions of the individual spaxels must be provided with the data.
    For the purpose of parallelizing the code, this function is defined out-
    side of the AperPhoto class. The parallelization via multithreading
    seems not to work if a method of the same class is called.
    
    Parameters
    ----------
    centroids : array_like, shape (N,2)
        The N coordinates around which aperture photometry should be performed
    pixpos : array_like, shape (M,2)
        The x- and y-positions of the M individual pixels in the 'data' array
    data : array_like, shape (M,) or (P,M)
        The data on which to perform aperture photometry. It should be provided
        with the spatial dimensions collapsed, i.e. a single image should be
        provided as a 1D array and P images should be provided as a 2D array
        with the first axis counting the images. Note that the data can have NaN
        values for invalid pixels. They are maske dout before the calculation.
    background : str, optional
        Can be either set to 'global' or to 'local'. In the former case, a global
        background for each layer is obtained, while in the latter case, the
        background is calulated locally in an annulus around each source.
    r_aper : float, optional
        Measurement radius; only pixels with distances <= r_aper to a centroid
        are taken into account.
    calc_moments : bool, optional
        Boolean flag that tells AperPhoto whether moments should be calculated.
    thresh : float, optional
        Only pixels with values larger than <thresh> times the estimated back-
        ground noise are considered in the moment calculation.
    
    Returns
    -------
    results : nd_array, shape (P,N,1) or (P,N,6)
        The calculated values are returned as a 3D array: first axis counts images,
        the second counts centroids, and the third counts the measured properties.
        If calc_moments is set to False, the only property will be the measured
        flux, otherwise, the flux will be returned together with the first and
        second order spatial moments.
    background : ndarray, shape (P,), optional
        The spectrum of the globbal baackground is only returned if the parameter
        background is set to 'global'.
    """
    ## data can either be an individual layer (1D) or a sequence of layers (2D). In the latter case, the first axis of the array must
    ## denote the layer number. The code performs a loop over the first axis, so it must be ensured that data is always 2D.
    data = np.atleast_2d(data)
    
    ## prepare output array
    results = np.zeros((data.shape[0],centroids.shape[0],1+5*calc_moments), dtype=np.float32)
    
    ## calculate distance of centroid of each source to each spaxel. Note that the shape of the returned array is (N_spaxels, N_sources)
    offsets = spatial.distance.cdist(pixpos, centroids)
    
    ## in the next steps, we define properties that do not change from one layer to another
    ## 1) global background pixels, defined as having distances >r_aper to all sources
    if background == "global":
        backpix = (offsets>r_aper).all(axis=1)
        if backpix.sum() < 20: logging.warning("Using < 20 pixels to estimate global background.")
 
        backvalues = np.zeros((data.shape[0],), dtype=np.float32) ## also create empty array for background values

    ## 2) local background pixels, defined as having distances >2*r_aper and <4*r_aper to one source
    else:
        backpix = (offsets>2*r_aper)&(offsets<=4*r_aper)
    
    ## 3) flux pixels, defined a having distances <=r_aper to one source.
    fluxpix  = offsets<=r_aper
    
    ## estimate global background value if requested. For this, all spaxels within a distance r_aper to any source centroid are masked out
    for i in xrange(data.shape[0]):

        valid = (data[i]==data[i])
        data[i,valid==False] = 0
    
        if not valid.any():
            logging.error("Layer contains only masked pixels.")
            results[i] = np.nan
            continue
    
        if background=="global":
            logging.debug("Determining global background...")
        
            ## perform kappa-sigma clipping to determine background level and noise
            backdata = clip_sigma(data[backpix.all(axis=1)&valid], axis=None)
            backval  = backdata.mean()
            backstd  = backdata.std()

            backvalues[i] =  backval
            logging.debug("...global background is {0:.2g}/pixel".format(backval))

        ## loop over all centroids
        for j in xrange(centroids.shape[0]):

            logging.debug("Processing source [{0:{1}d}/{2}]...".format(i+1, len(str(centroids.shape[0])), centroids.shape[0]))

            ## estimate local background in annulus between 2*r_aper and 4*r_aper (if requested, otherwise global background is used)
            if background=="local":
        
                backdata = clip_sigma(data[i,backpix[:,j]&valid], axis=None)
                backval = backdata.mean()
                backstd = backdata.std()
                logging.debug("...local background is {0:.2g}+/-{1:0.2g}".format(backval, backstd))

            ## Get array for calculations & subtract background estimate
            fluxdata = data[i,fluxpix[:,j]&valid] - backval
            results[i,j,0] = fluxdata.sum()
            logging.debug("...flux inside aperture is {0:.3g}".format(results[i,j,0]))

            ## Moment calculation, only if requested:
            if calc_moments:
                mompix  = fluxpix[:,j]&valid&(data[i]>(backval+thresh*backstd))
                momdata = data[i,mompix]
                logging.debug("...{0} pixel(s) available for moment calculation".format(mompix))
                if mompix.sum() > 3:
                    X = pixpos[mompix, 1]
                    Y = pixpos[mompix, 0]
                        
                    total = float(momdata.sum())
                    x  = (X*momdata).sum()/total
                    y  = (Y*momdata).sum()/total
                    x2 = (X*X*momdata).sum()/total - x**2
                    y2 = (Y*Y*momdata).sum()/total - y**2        
                    xy = (X*Y*momdata).sum()/total - x*y
                     
                    ## if unphysical moments measured or centroid outside measurement area, mask out all results for this source
                    if x2<=0 or y2<=0 or xy**2>=(x2*y2) or x<X.min() or x>X.max() or y<Y.min() or y>Y.max():
                        x = np.nan
                        y = np.nan
                        x2 = np.nan
                        y2 = np.nan
                        xy = np.nan
                    logging.debug("...calculated moments: x={0:.1f}, y={1:.1f}, x2={2:.1f}, y2={3:.1f}, xy={4:.1f}".format(x, y, x2, y2, xy))
                else:
                    logging.debug("...too few pixels - skipping moment calculation.")
                    x = np.nan
                    y = np.nan
                    x2 = np.nan
                    y2 = np.nan
                    xy = np.nan
                results[i,j,1:6] = [x,y,x2,y2,xy]

        return [results, backvalues] if background=="global" else results


class AperPhoto:
    """
    Perform aperture (spectro)photometry on 2 or 3dimensional data using
    a provided list of sources. The calculation of the first and second order
    moments for the provided sources  is optional. Prior to any flux or moment
    measurement, a background (global or local if  requested) is determined
    and subtracted. 
    """

    def __init__(self, cube, psf_instances):
        """
        Initialize an instance of AperPhoto-class

        Parameters
        ----------
        cube : an instance of pampelmuse.core.cube.Cube
            All the information about the data that should  be analysed should
            be provided here. For further information on how to do this, check
            the documentation included in the class 'Cube'.
        psf_instances : a list of instances of pampelmuse.psf.profiles.Profile
            The PSF instances should contain the information about the sources
            for which aperture photometry should be performed. At least the
            properties 'xc' and 'yc' must be available for each instance and set
            to the central coordinates of the source on the cube.
        """
        logging.info("Initializing aperture photometry, validating input...")

        ## Checking 'cube'
        assert isinstance(cube, Cube), "Parameter 'cube' must be an instance of pampelmuse.core.cube.Cube, not{0}.".format(cube.__class__)
        self.cube = cube

        ## Checking PSF instances
        if not hasattr(psf_instances, "__iter__"):
            psf_instances = [psf_instances]
        for psf in psf_instances:
            assert isinstance(psf, Profile), "Entries of 'psf_instances' must be instances of pampelmuse.core.psf.profiles.Profile, not {0}.".format(psf.__class__)
            if not psf.transform.shape == cube.instrument.transform.shape and (psf.transform==cube.instrument.transform).all():
                psf.transform = cube.instrument.transform
                logging.warning("PSF array of source #{0} did not match data, modified it.".format(psf.src_id))
        self.psf_instances = psf_instances

        self.nsources = len(self.psf_instances)
        logging.info("Prepared aperture spectrophotometry for {0} source[s].".format(self.nsources))


    def __call__(self, r_aper=3., layer_range=(0, -1), calc_moments=True, thresh=3., background="global", ncpu=1):
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
        r_aper : float, optional
            Measurement radius; only pixels with distances <= r_aper to
            the source center are taken into account
        layer_range : array_like
            The indices of the first and the last layer of the data cube
            that should be included in the calculation. The default is to
            use the entire cube. 
        calc_moments : bool, optional
            Boolean flag that tells AperPhoto whether moments should be
            calculated
        thresh : float, optional
            Only pixels with values larger than <thresh> times the
           estimated background noise are considered in the moment
            calculation.
        background : str, optional
            Can be either set to 'global' or to 'local'. In the former case,
            a global background for each layer is obtained, while in the
            latter case, the background is calulated locally in an annulus
            around each source.
        ncpu : integer, optional
            The number of CPUs used in the calculation.
            
        Returns
        -------
        results : numpy_record_array
            For every source, the flux/spectrum is provided in field
            'spectrum' and moments are provided in fields 'x', 'y', 'x2',
            'y2' and 'xy'.
            Note that if the parameter 'background' is set to 'global' the
            spectrum of the background will be stored in the 'spectrum'
            field of a hypothetical source with ID=-1.
        """
        self.r_aper = r_aper
        self.start, self.stop = layer_range
        if self.start < 0: self.start = self.cube.ndisp - self.start
        if self.stop < 0: self.stop = self.cube.ndisp + self.stop
        self.nlayer = self.stop-self.start+1
        self.thresh = thresh
        self.calc_moments = calc_moments
        self.ncpu = ncpu
        
        if background not in ["local", "global"]:
            logging.critical("Unknown value '{0}' for parameter 'background'. Must be either 'local' or 'global'.".format(background))
        assert background in ["local", "global"],  "Parameter 'background' must be  either 'local' or 'global', not '{0}'.".format(background)
        
        logging.info("Starting calulation with the following setup:")
        logging.info("...aperture radius [r_ape]: {0:.1f}pixel".format(r_aper))
        logging.info("...used layers [layer_range]: {0}-{1}".format(self.start, self.stop))
        logging.info("...pixel threshold [thresh]: {0}sigma".format(thresh))
        logging.info("...calculating moments? [calc_moments]: {0}".format(calc_moments))
        logging.info("...background estimation [background]: {0}".format(background))
        logging.info("...number of CPUs: {0}".format(ncpu))
    
        ## prepare record array that will contains output.
        logging.info("...preparing array to hold results...")
        dtype = [("id", np.int32), ("flux", np.float32, (self.nlayer,))]
        if calc_moments:
            for name in ["x", "y", "x2", "y2", "xy"]: dtype.append((name, np.float32, (self.nlayer,)))
        self.results = np.zeros(self.nsources+(background=="global"), dtype=dtype)
    
        ## collect centroid coordinates for all sources and add their IDs to output array:
        centroids = np.zeros((self.nsources, 2), dtype=np.float32)
        i=0
        for psf in self.psf_instances:
            centroids[i] = psf.yc, psf.xc
            self.results["id"][i] = psf.src_id
            i+=1 

        ## use ID=-1 for a global background if it is used
        if background == "global":
            self.results["id"][-1] = -1
            for name in ["x", "y", "x2", "y2", "xy"]: self.results[name][-1] = np.nan

        ## loop over layers, assign layers to CPUs if parallelization should be used, and perdorm aperture photometry        
        _results = []

        i=self.start
        while i<=self.stop:
            
            if ncpu>1:
                pool    = Pool(ncpu)
                threads = []
                
            for nthreads in xrange(ncpu):
                logging.info("...processing layer [{0:{1}d}/{2}]".format(i+1-self.start, len(str(self.nlayer)), self.nlayer))
                
                ## load data
                data = self.cube.data.layer(i)
                if self.cube.mask!=None:
                    mask = self.cube.mask.layer(i)
                    data[mask] = np.nan
            
                if ncpu>1:
                    threads.append(pool.apply_async(run_ap, args=(centroids, self.cube.instrument.transform, data, background, r_aper, calc_moments, thresh)))
                else:
                    ## put result in list to make output comparable with multi-CPU case
                    _results.append(run_ap(centroids, self.cube.instrument.transform, data, background, r_aper, calc_moments, thresh))
                i += 1
                if i>self.stop:
                    break
                    
            
            if ncpu>1:
                pool.close()
                pool.join()
                _results.extend([threads[n].get() for n in xrange(nthreads+1)])
            
        if background=="global":
            results = np.vstack([r[0] for r in _results])
            self.results["flux"][-1] = np.asarray([r[1] for r in _results])
        else:
            results = np.vstack(_results)

        self.results["flux"][:results.shape[1]] = results[...,0].T
        if calc_moments:
            i=1
            for name in ["x", "y", "x2", "y2", "xy"]:
                self.results[name][:results.shape[1]] = results[...,i].T
                i+=1

        return self.results
                    
                


#class AperPhoto:
#    """
#    Perform aperture (spectro)photometry on 2 or 3dimensional data using
#    a provided list of sources. The calculation of the first and second order
#    moments for the provided sources  is optional. Prior to any flux or moment
#    measurement, a background (global or local if  requested) is determined
#    and subtracted. 
#    """
#
#    def __init__(self, cube, sources, min_status=3):
#        """
#        Initialize an instance of AperPhoto-class
#
#        Parameters
#        ----------
#        cube : an instance of pampelmuse.core.cube.Cube
#            All the information about the data that should  be analysed should
#            be provided here. For further information on how to do this, check
#            the documentation included in the class 'Cube'.
#        sources : an instance of pampelmuse.core.sources.Sources
#            All the information about the sources that are present in the data
#            cube should be provided here. For further information on how to
#            properly set up  an instance of 'Sources', check the documentation
#            included in the definition of that class.  
#        min_status : int, optional
#            The minimum value of the 'Sources.status' property a source should
#            have to be considered in the aperture spectrophotometry.
#    
#        Returns
#        -------
#        The newly created instance
#        """
#        logging.info("Starting aperture photometry, validating input...")
#
#        ## Checking 'cube'
#        assert isinstance(cube, Cube), "Parameter 'sources' must be an instance of pampelmuse.core.cube.Cube."
#        self.cube = cube
#
#        ## Checking 'sources'
#        assert isinstance(sources, Sources), "Parameter 'sources' must be an instance of pampelmuse.core.sources.Sources."
#        self.sources = sources
#        assert sources.x != None, "Initial guesses for the x coordinates of the sources must be available."
#        assert sources.y != None, "Initial guesses for the y coordinates of the sources must be available."
#        assert sources.status != None, "Property Sources.status must have been set."
#
#        ## Checking which sources should be included in the calculation
#        self.used = (self.sources.status>=min_status)&(self.sources.status<4)
#        logging.info("Will carry out aperture spectrophotometry for {0} source[s] with status>={1}.".format(self.used.sum(), min_status))
#
#
#    def aper_photo(self, r_aper=3., layer_range=(0, -1), PSF=None, calc_moments=True, thresh=3., background="global", fill_value=np.nan):
#        """
#        Method performs the actual aperture (spectro)photometry (and
#        measures the moments if requested). An estimate of the global
#        background is determined using a sigma-clipping approach.
#        Optionally, a local background around every soource can be
#        determined.
#
#        Note that the aperture (spectro)photometry is always performed
#        around the input coordinates, NOT around the first order moments
#        (even if they are calculated).
#
#        Parameters
#        ----------
#        r_aper : float, optional
#            Measurement radius; only pixels with distances <= r_aper to
#            the source center are taken into account
#        layer_range : array_like
#            The indices of the first and the last layer of the data cube
#            that should be included in the calculation. The default is to
#            use the entire cube. 
#        PSF : a class pampelmuse.core.psf.profiles.Profile
#            A PSF class can be provided here,  containing an instance for
#            every source. In that case, the provided PSFs are used to
#            perform an aperture correction. NOT YET IMPLEMENTED!
#        calc_moments : bool, optional
#            Boolean flag that tells AperPhoto whether moments should be
#            calculated
#        thresh : float, optional
#            Only pixels with values larger than <thresh> times the
#           estimated background noise are considered in the moment
#            calculation.
#        background : str, optional
#            Can be either set to 'global' or to 'local'. In the former case,
#            a global background for each layer is obtained, while in the
#            latter case, the background is calulated locally in an annulus
#            around each source.
#        fill_value : float, optional
#            The value written to the output array if the number of pixels is
#            too small to calculate moments. If set to None, the routine will
#            use the input coordinates.
#            
#        Returns
#        -------
#        results : numpy_record_array
#            For every source, the flux/spectrum is provided in field
#            'spectrum' and moments are provided in fields 'x', 'y', 'x2',
#            'y2' and 'xy'.
#            Note that if the parameter 'background' is set to 'global' the
#            spectrum of the background will be stored in the 'spectrum'
#            field of a hypothetical source with ID=-1.
#        """
#        self.r_aper = r_aper
#        self.start, self.stop = layer_range
#        if self.start < 0: self.start = self.cube.ndisp - self.start
#        if self.stop < 0: self.stop = self.cube.ndisp + self.stop
#        self.Nlayer = self.stop-self.start+1
#        self.PSF = PSF
#        self.thresh = thresh
#        self.calc_moments = calc_moments
#        
#        if background not in ["local", "global"]:
#            logging.critical("Unknown value '{0}' for parameter 'background'. Must be either 'local' or 'global'.".format(background))
#        assert background in ["local", "global"],  "Parameter 'background' must be  either 'local' or 'global', not '{0}'.".format(background)
#        
#        logging.info("Aperture photometry - starting calulation, using the following setup:")
#        logging.info("- Aperture radius [r_ape]: {0:.1f}pixel".format(self.r_aper))
#        logging.info("- Included layers [layer_range]: {0}-{1}".format(self.start, self.stop))
#        logging.info("- Pixel threshold [thresh]: {0}sigma".format(self.thresh))
#        logging.info("- Moment calculation [calc_moments]: {0}".format(self.calc_moments))
#        logging.info("- Background estimation: [background]: {0}".format(background))
#        logging.info("- Aperture correction via PSF estimate [PSF]: {0}.".format(self.PSF!=None))
#
#        ## prepare record array that will contains output.
#        logging.info("Preparing array to contain results...")
#        dtype = [("ID",np.int32), ("flux", np.float32, (self.Nlayer,))]
#        if calc_moments: dtype.extend([("x", np.float32, (self.Nlayer,)), ("y", np.float32, (self.Nlayer,)), ("x2", np.float32, (self.Nlayer,)),
#                                       ("y2", np.float32, (self.Nlayer,)), ("xy", np.float32, (self.Nlayer,))])
#        self.results = np.zeros(self.used.sum()+(background=="global"), dtype=dtype)
#        
#        if background == "global":
#            self.results["ID"][:-1] = self.sources.ID[self.used]
#            self.results["ID"][-1] = -1 ## last entry in the results array is reserved for the sky estimate 
#        else:
#            self.results["ID"][:] = self.sources.ID[self.used]
#
#        ## calculation is started on the central layer of the requested range and moves to both sides simultaneously
#        for i in xrange(self.Nlayer):
#            jj = (self.Nlayer>>1) - 1 - int(i/2) + i*np.mod(i, 2) + np.mod(self.Nlayer, 2)
#            kk = jj + self.start
#            
#            logging.info("Aperture photometry - processing layer #{0} [{1}/{2}]".format(kk, i+1, self.Nlayer))
#
#            ## collect information for current layer
#            ## data & mask
#            _data = self.cube.data.layer(kk)
#            if self.cube.mask!=None:
#                curMask = self.cube.mask.layer(kk)&(_data!=_data)
#            else:
#                curMask = (_data!=_data)
#            curData = _data[curMask==0]
#
#            if not (curMask==0).any():
#                logging.error("[@{0}] This layer contains no unmasked voxels.".format(kk))
#                continue
#
#            ## spaxel coordinates
#            spaxels = self.cube.instrument.transform[curMask==0]
#
#            ## load IFU coordinates for the requested sources
#            logging.debug("[@{0}] Calculating distances between source centroids and spaxels for all sources...".format(kk))
#            self.sources.load_ifs_coordinates(ids=self.sources.ID[self.used], layers=jj)
#            centroids = self.sources.ifs_coordinates[0,...,0]
#
#            ## for this layer, calculate distance of centroid of each source to each spaxel
#            ## Note that the shape of the returned array is (N_spaxels, N_sources)
#            self.offsets = self.cube.instrument.dist2spax(centroids[:,0], centroids[:,1])[curMask==0]
#
#            ##  estimate global background value. For this, all spaxels within a distance r_aper to any source centroid are masked out
#            if background=="global":
#                logging.debug("[@{0}] Determining global background...".format(kk))
#                backpix = (self.offsets>self.r_aper).all(axis=1)
#
#                if backpix.sum() < 20: logging.warn("[@{0}] Using < 20 pixels to estimate global background.".format(kk))
#            
#                ## perform kappa-sigma clipping to determine background level and noise
#                backdata = clip_sigma(curData[backpix], axis=None)
#                backval = backdata.mean()
#                backstd = backdata.std()
#                self.results["flux"][-1, jj] = backval
#                logging.debug("[@{0}] Global background is {1:.2f}/pixel".format(kk, backval))
#
#            ## a loop over all sources (index n), but calculations are only performed if source n is included in the
#            ## list of sources for which the aperture spectrophotometry should be performed. N is the index for those
#            ## sources.
#            N = -1
#            for n in xrange(self.sources.nsrc):
#                if not self.used[n]: continue
#
#                N += 1
#
#                logging.debug("[@{0}] Processing source #{0} [{1}/{2}]:".format(kk, self.sources.ID[n], N+1, self.used.sum()))
#
#                ## estimate local background (if requested, otherwise use global background)
#                if background=="local":
#                    backpix = (self.offsets[:, N]>2*self.r_aper)&(self.offsets[:, N]<=4*self.r_aper)
#
#                    if backpix.sum() < 10: logging.warn("[@{0}, #{1}] Using < 10 pixels to estimate local background.".format(kk, self.sources.ID[n]))
#        
#                    backdata = clip_sigma(curData[backpix], axis=None)
#                    backval = backdata.mean()
#                    backstd = backdata.std()
#
#                    logging.debug("[@{0}, #{1}] Local background: {2:.2f}/pixel".format(kk, self.sources.ID[n], backval))
#
#
#                ## Get array for calculations & subtract background estimate
#                fluxpix  = self.offsets[:, N]<=self.r_aper
#                fluxdata = curData[fluxpix] - backval
#
#                ## Moment calculation, only if requested:
#                if self.calc_moments:
#                    mompix  = fluxpix&(curData>(backval+self.thresh*backstd))
#                    momdata = curData[mompix]                                  
#                    if mompix.sum() > 3:
#                        if mompix.sum() < 5: logging.warn("[@{0}, #{1}] Less than 5 pixels used to calculate moments.".format(kk, self.sources.ID[n]))
#                        moments = self.measure_moments(momdata, X=spaxels[mompix, 1], Y=spaxels[mompix, 0])
#                        
#                        ## perform sanity check on second order moments and add them to results:
#                        if moments[2]<0:
#                            logging.error("[@{0}, #{1}] Source has unphysical moments: <x^2> < 0".format(kk, self.sources.ID[n]))
#                            moments[2]=0
#                        if moments[3]<0:
#                            logging.error("[@{0}, #{1}] Source has unphysical moments: <y^2> < 0".format(kk, self.sources.ID[n]))
#                            moments[3]=0
#                        if moments[4]**2 >= moments[2]*moments[3]:
#                            logging.error("[@{0}, #{1}] Source has unphysical moments: <xy>^2 >= <x^2><y^2>".format(kk, self.sources.ID[n]))
#                            moments[4] = 0
#                            
#                        self.results["x"][N, jj] = moments[0]
#                        self.results["y"][N, jj] = moments[1]
#                        self.results["x2"][N, jj] = moments[2]
#                        self.results["y2"][N, jj] = moments[3]
#                        self.results["xy"][N, jj] = moments[4]
#                        logging.debug("[@{0}, #{1}] Calculated moments: <x>={2:.1f} <y>={3:.1f}, <x^2>={4:.1f}, <y^2>={5:.1f}, <xy>={6:.1f}".format(\
#                            kk, self.sources.ID[n], moments[0], moments[1],  moments[2], moments[3], moments[4]))
#                    else:
#                        logging.error("[@{0}, #{1}] Too few pixels (<4) to calculate moments.".format(kk, self.sources.ID[n]))
#
#                ## Aperture photometry:
#                corr = 1.
## TODO: implement usage of PSF for aperture correction
##                if psf != None:
##                    corr =  psf.f[fluxpix].sum()/psf.f.sum()
##                else:
##                    corr = 1.
#
#                flux = fluxdata.sum()/corr
#                self.results["flux"][N, jj] = flux
#                logging.debug("[@{0}, #{1}] Calculated aperture flux: {2:.2f} counts".format(kk, self.sources.ID[n], float(flux)))
#
#        return self.results
#
#
#
#    def measure_moments(self, indata, X, Y):
#
#        """
#        Method measures first and second order moments of the pixel
#        distribution in a 2dim. array.
#
#        Returns the first and second order moments of pixel
#        distribution: (x, y, x2, y2, xy). Note that x corresponds
#        the second axis, while y corresponds to the first axis (in
#        python notation).
#
#        Parameters
#        ----------
#        indata : ndarray
#            The data, provided as a 1dim. array. Note that the data should
#            be background subtracted, otherwise the second order moments will
#            be overestimated.
#        X : ndarray
#            The x-coordinates of the pixels in 'data'. Must of course be of the
#            same shape as 'data'.
#        Y : ndarray
#            The y-coordinates of the pixels in 'data'. Must of course be of the
#            same shape as 'data'.
#            
#        Returns
#        -------
#        x : float
#            The first order moment of the data in x-direction.
#        y : float
#            The first order moment of the data in y-direction.
#        x2 : float
#            The second order moment (variance) of the data in x-direction.
#        y2 : float
#            The second order moment (variance) of the data in y-direction.
#        xy : float
#            The covariance between x and y.
#        """
#        total = float(indata.sum())
#        x  = (X*indata).sum()/total
#        y  = (Y*indata).sum()/total
#        
#        x2 = (X*X*indata).sum()/total - x**2
#        y2 = (Y*Y*indata).sum()/total - y**2        
#        xy = (X*Y*indata).sum()/total - x*y
#
#        return [x, y, x2, y2, xy]
