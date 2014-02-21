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

Latest SVN revision
-------------------
129, 2014/02/21
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 129

import logging, sys
import numpy as np
from scipy import spatial

## required PampelMuse-packages
from pampelmuse.functions.statistics import clip_sigma
from pampelmuse.core import Cube, Sources
from pampelmuse.core.psf import Profile




class AperPhoto:
    """
    Perform aperture (spectro)photometry on 2 or 3dimensional data using
    a provided list of sources. The calculation of the first and second order
    moments for the provided sources  is optional. Prior to any flux or moment
    measurement, a background (global or local if  requested) is determined
    and subtracted. 
    """

    def __init__(self, cube, sources, min_status=3):
        """
        Initialize an instance of AperPhoto-class

        Parameters
        ----------
        cube : an instance of pampelmuse.core.cube.Cube
            All the information about the data that should  be analysed should
            be provided here. For further information on how to do this, check
            the documentation included in the class 'Cube'.
        sources : an instance of pampelmuse.core.sources.Sources
            All the information about the sources that are present in the data
            cube should be provided here. For further information on how to
            properly set up  an instance of 'Sources', check the documentation
            included in the definition of that class.  
        min_status : int, optional
            The minimum value of the 'Sources.status' property a source should
            have to be considered in the aperture spectrophotometry.
    
        Returns
        -------
        The newly created instance
        """
        logging.info("Starting aperture photometry, validating input...")

        ## Checking 'cube'
        assert isinstance(cube, Cube), "Parameter 'sources' must be an instance of pampelmuse.core.cube.Cube."
        self.cube = cube

        ## Checking 'sources'
        assert isinstance(sources, Sources), "Parameter 'sources' must be an instance of pampelmuse.core.sources.Sources."
        self.sources = sources
        assert sources.x != None, "Initial guesses for the x coordinates of the sources must be available."
        assert sources.y != None, "Initial guesses for the y coordinates of the sources must be available."
        assert sources.status != None, "Property Sources.status must have been set."

        ## Checking which sources should be included in the calculation
        self.used = (self.sources.status>=min_status)&(self.sources.status<4)
        logging.info("Will carry out aperture spectrophotometry for {0} source[s] with status>={1}.".format(self.used.sum(), min_status))


    def aper_photo(self, r_aper=3., layer_range=(0, -1), PSF=None, calc_moments=True, thresh=3., local_background=True):
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
        PSF : a class pampelmuse.core.psf.profiles.Profile
            A PSF class can be provided here,  containing an instance for
            every source. In that case, the provided PSFs are used to
            perform an aperture correction. NOT YET IMPLEMENTED!
        calc_moments : bool, optional
            Boolean flag that tells AperPhoto whether moments should be
            calculated
        thresh : float, optional
            Only pixels with values larger than <thresh> times the
           estimated background noise are considered in the moment
            calculation.
        local_background : boolean, optional
            Flag indicating whether a local background for every source
            should be estimated.
            
        Returns
        -------
        results : numpy_record_array
            For every source, the flux/spectrum is provided in field
            'spectrum' and moments are provided in fields 'x', 'y', 'x2',
            'y2' and 'xy'.
            Note that the global background flux/spectrum is stored in the
            'spectrum' slot of a hypothetical source with 'ID' -1.
        """
        self.r_aper = r_aper
        self.start, self.stop = layer_range
        if self.start < 0: self.start = self.cube.ndisp - self.start
        if self.stop < 0: self.stop = self.cube.ndisp + self.stop
        self.Nlayer = self.stop-self.start+1
        self.PSF = PSF
        self.thresh = thresh
        self.calc_moments = calc_moments
        
        logging.info("Aperture photometry - starting calulation, using the following setup:")
        logging.info("+Aperture radius (r_aper) = {0:.1f}pixel".format(self.r_aper))
        logging.info("+Included layers (layer_range): {0}-{1}".format(self.start, self.stop))
        logging.info("+Pixel threshold (thresh) = {0}sigma".format(self.thresh))
        logging.info("+Moment calculation? (calc_moments) - {0}".format(self.calc_moments))
        logging.info("+Local background calculation? (local_background) - {0}".format(local_background))
        logging.info("+Aperture correction via PSF estimate? (PSF) - {0}.".format(self.PSF!=None))

        ## prepare record array that will contains output.
        logging.info("Preparing array to contain results...")
        self.results = np.zeros(self.used.sum()+1, dtype=[("ID",np.int32), ("flux", np.float32, (self.Nlayer,)), ("x", np.float32, (self.Nlayer,)),
                                                          ("y", np.float32, (self.Nlayer,)), ("x2", np.float32, (self.Nlayer,)), ("y2", np.float32, (self.Nlayer,)),
                                                          ("xy", np.float32, (self.Nlayer,))])
        self.results["ID"][:-1] = self.sources.ID[self.used]
        self.results["ID"][-1] = -1 ## last entry in the results array is reserved for the sky estimate 

        ## calculation is started on the central layer of the requested range and moves to both sides simultaneously
        for i in xrange(self.Nlayer):
            jj = (self.Nlayer>>1) - 1 - int(i/2) + i*np.mod(i, 2) + np.mod(self.Nlayer, 2)
            kk = jj + self.start
            
            logging.info("Aperture spectrophotometry - processing layer #{0} [{1}/{2}]".format(kk, i+1, self.Nlayer))

            ## collect information for current layer
            ## data & mask
            _data = self.cube.data.layer(kk)
            if self.cube.mask!=None:
                curMask = self.cube.mask.layer(kk)&(_data!=_data)
            else:
                curMask = (_data!=_data)
            curData = _data[curMask]

            if not (curMask==0).any():
                logging.error("[@{0}] This layer contains no unmasked voxels.".format(kk))
                continue

            ## spaxel coordinates
            spaxels = self.cube.instrument.transform[curMask==0]

            logging.debug("[@{0}] Calculating distances between source centroids and spaxels for all sources...".format(kk))
            ## load IFU coordinates for the requested sources
            self.sources.load_ifs_coordinates(ids=self.sources.ID[self.used], layers=jj)
            centroids = self.sources.ifs_coordinates[0,...,0]

            ## for this layer, calculate distance of centroid of each source to each spaxel
            ## Note that the shape of the returned array is (N_spaxels, N_sources)
            self.offsets = self.cube.instrument.dist2spax(centroids[:,0], centroids[:,1])[curMask==0]

            ##  estimate global background value. For this, all spaxels within a distance r_aper to any source centroid are masked out
            logging.debug("[@{0}] Determining global background...".format(kk))
            backpix = (self.offsets>self.r_aper).all(axis=1)

            if backpix.sum() < 20: logging.warn("[@{0}] Using < 20 pixels to estimate global background.".format(kk))
            
            ## perform kappa-sigma clipping to determine background level and noise
            backdata = clip_sigma(curData[backpix], axis=None)
            backval = backdata.mean()
            backstd = backdata.std()
            self.results["flux"][-1, jj] = backval
            logging.debug("[@{0}] Global background is {1:.2f}/pixel".format(kk, backval))

            ## a loop over all sources (index n), but calculations are only performed if source n is included in the
            ## list of sources for which the aperture spectrophotometry should be performed. N is the index for those
            ## sources.
            N = -1
            for n in xrange(self.sources.nsrc):
                if not self.used[n]: continue

                N += 1

                logging.debug("[@{0}] Processing source #{0} [{1}/{2}]:".format(kk, self.sources.ID[n], N+1, self.used.sum()))

                ## estimate local background (if requested, otherwise use global background)
                if local_background:
                    backpix = (self.offsets[:, N]>2*self.r_aper)&(self.offsets[:, N]<=4*self.r_aper)

                    if backpix.sum() < 10: logging.warn("[@{0}, #{1}] Using < 10 pixels to estimate local background.".format(kk, self.sources.ID[n]))
        
                    backdata = clip_sigma(curData[backpix], axis=None)
                    backval = backdata.mean()
                    backstd = backdata.std()

                    logging.debug("[@{0}, #{1}] Local background: {2:.2f}/pixel".format(kk, self.sources.ID[n], backval))


                ## Get array for calculations & subtract background estimate
                fluxpix  = self.offsets[:, N]<=self.r_aper
                fluxdata = curData[fluxpix] - backval

                ## Moment calculation, only if requested:
                if self.calc_moments:
                    mompix  = fluxpix&(curData>(backval+self.thresh*backstd))
                    momdata = curData[mompix]                                  
                    if mompix.sum() > 3:
                        if mompix.sum() < 5: logging.warn("[@{0}, #{1}] Less than 5 pixels used to calculate moments.".format(kk, self.sources.ID[n]))
                        moments = self.measure_moments(momdata, X=spaxels[mompix, 1], Y=spaxels[mompix, 0])
                        
                        ## perform sanity check on second order moments and add them to results:
                        if moments[2]<0:
                            logging.error("[@{0}, #{1}] Source has unphysical moments: <x^2> < 0".format(kk, self.sources.ID[n]))
                            moments[2]=0
                        if moments[3]<0:
                            logging.error("[@{0}, #{1}] Source has unphysical moments: <y^2> < 0".format(kk, self.sources.ID[n]))
                            moments[3]=0
                        if moments[4]**2 >= moments[2]*moments[3]:
                            logging.error("[@{0}, #{1}] Source has unphysical moments: <xy>^2 >= <x^2><y^2>".format(kk, self.sources.ID[n]))
                            moments[4] = 0
                            
                        self.results["x"][N, jj] = moments[0]
                        self.results["y"][N, jj] = moments[1]
                        self.results["x2"][N, jj] = moments[2]
                        self.results["y2"][N, jj] = moments[3]
                        self.results["xy"][N, jj] = moments[4]
                        logging.debug("[@{0}, #{1}] Calculated moments: <x>={2:.1f} <y>={3:.1f}, <x^2>={4:.1f}, <y^2>={5:.1f}, <xy>={6:.1f}".format(\
                            kk, self.sources.ID[n], moments[0], moments[1],  moments[2], moments[3], moments[4]))
                    else:
                        logging.error("[@{0}, #{1}] Too few pixels (<4) to calculate moments.".format(kk, self.sources.ID[n]))

                ## Aperture photometry:
                corr = 1.
# TODO: implement usage of PSF for aperture correction
#                if psf != None:
#                    corr =  psf.f[fluxpix].sum()/psf.f.sum()
#                else:
#                    corr = 1.

                flux = fluxdata.sum()/corr
                self.results["flux"][N, jj] = flux
                logging.debug("[@{0}, #{1}] Calculated aperture flux: {2:.2f} counts".format(kk, self.sources.ID[n], float(flux)))

        return self.results



    def measure_moments(self, indata, X, Y):

        """
        Method measures first and second order moments of the pixel
        distribution in a 2dim. array.

        Returns the first and second order moments of pixel
        distribution: (x, y, x2, y2, xy). Note that x corresponds
        the second axis, while y corresponds to the first axis (in
        python notation).

        Parameters
        ----------
        indata : ndarray
            The data, provided as a 1dim. array. Note that the data should
            be background subtracted, otherwise the second order moments will
            be overestimated.
        X : ndarray
            The x-coordinates of the pixels in 'data'. Must of course be of the
            same shape as 'data'.
        Y : ndarray
            The y-coordinates of the pixels in 'data'. Must of course be of the
            same shape as 'data'.
            
        Returns
        -------
        x : float
            The first order moment of the data in x-direction.
        y : float
            The first order moment of the data in y-direction.
        x2 : float
            The second order moment (variance) of the data in x-direction.
        y2 : float
            The second order moment (variance) of the data in y-direction.
        xy : float
            The covariance between x and y.
        """
        total = float(indata.sum())
        x  = (X*indata).sum()/total
        y  = (Y*indata).sum()/total
        
        x2 = (X*X*indata).sum()/total - x**2
        y2 = (Y*Y*indata).sum()/total - y**2        
        xy = (X*Y*indata).sum()/total - x*y

        return [x, y, x2, y2, xy]
