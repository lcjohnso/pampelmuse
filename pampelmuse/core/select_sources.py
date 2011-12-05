"""
SELECT_SOURCES
==============
Given an input catalog of sources in a field, the routine checks if the
sources fulfil certain criteria. Based on these checks, a subsample of
the sources in the catalogue can then be selected.

CHANGELOG:
----------
2011/09/06, v0.0.0.0
2011/09/08, v0.1.0.0, changed usage of PSF in code. Now a PSF instance
                      must be provided when a new instance is created.
                      (previously, provided FWHM was used to create a
                      Gaussian PSF)
2011/11/29, v0.1.0.1, when an instance is initialized, a `collapsed' image
                      of the datacube must be provided, ideally created by
                      convolving the cube with the filter-curve of the pass-
                      band of the provided magnitude. This image is then used
                      to perform a photometric calibration of the data.
                      The S/N is now only determined inside the FITRAD to avoid
                      selecting sources far outside the FoV.
2011/12/05, v0.1.1.1, using a string's format-property with numpy integers or
                      floats seems to cause problems sometimes. Therefore the
                      return-value of the `howIsolated'-method is now
                      explicitely converted to a float.
                      
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.1.1"
import logging
import numpy as np
## import my_profiles


class SelectSources:
    """
    The class is designed to calculate certain characteristics of the
    sources in a catalog. Based on the results of this calculation, a
    subsample of sources fulfilling certain criteria can be selected.
    Possible characteristics are:
    (i)  `howIsolated' - returns the fractional flux of neighboring sources
                         within a given radius around the  source
    (ii) `howCloseToEdge' - returns the distance [in pixels] of the source
                            to the closest edge of the FoV. A negative
                            value is returned if the source is outside
                            the FoV.
    (iii) `whatSNR'- Based on a rough estimate of the PSF and the expected
                     flux of the source, its expected S/N is returned.
    (iv) `howFarFromBright' - the distance [in pixels] to the nearest
                              brighter source is calculated. It is returned
                              together with the ID of the bright source.
    """
    def __init__(self, sourcelist, bbImage, psf, pospars=[1.,0.,0.,0.]):
        """
        Initialize an instance of the `SelectSources'-class

        Parameters
        ----------
        sourcelist : numpy_record_array
            The catalogue of sources. The structured array must contain the
            following fields: 'id', 'x', 'y' and 'mag'
        shape : tuple
            The shape of the considered FoV [in pixels]
        psf : a PSF instance
            An Instance of one of the PSF-classes defined in my_profiles.
            It is used to check criteria `howIsolated' and `and whatSNR'.
        pospars = list of float, optional
            The parameters A,B,C,D of the coordinate transformation. By
            default, no transformation is done and the input coordinates
            are supposed to be given on the considered FoV.
        """
        self.sourcelist = sourcelist

        A,B,C,D = pospars

        self.x   = A*self.sourcelist['x'] + B*self.sourcelist['y'] + C
        self.y   = A*self.sourcelist['y'] - B*self.sourcelist['x'] + D
        self.mag = self.sourcelist['mag']

        self.shape = bbImage.shape

        ## perform photometric calibration
        logging.info("Performing photmetric calibration of data.")
        sumFlux = bbImage.sum()
        expFlux = 0.
        for i in xrange(self.sourcelist.size):
            if self.x[i]>0 and self.x[i]<self.shape[1]:
                if self.y[i]>0 and self.y[i]<self.shape[0]:
                    expFlux += 10**(-0.4*self.mag[i])
        self.mZ = -2.5*np.log10(expFlux/sumFlux)
        logging.info("   Estimated photometric zeropoint: {0:.2f}".format(\
                self.mZ))

        self.sorted = self.sourcelist['mag'].argsort()

        self.PSF   = psf


    def __call__(self, criteria, magcut=99.9, **kwargs):
        """
        Perform the calculation.

        Parameters
        ----------
        criteria : list of strings
            Contains all the charactersistics to be calculated. Valid
            options are: `howIsolated', `howCloseToEdge', `whatSNR',
            `howFarFromBright'
        magcut : float, optional
            Can be specified to restrict calculation to brighter sources
            while skipping all sources fainter `magcut'
        kwargs
            Parameters to be passed to the individual `criteria'-routines.
    
        Returns
        -------
        results :  numpy_record_array
            For every source above `magcut', the results of the calculation
            are returned as a structured array containing a field with the
            source IDs and depending on `criteria' one or more of the
            following fields:
                'howIsolated', 'howCloseToEdge', 'whatSNR', 'brightNeighbor',
                'howFarFromBright'
            Note that 'brightNeighbor' contains the ID of the nearest brighter
            source
        """

        results = []
        dtype   = [('id', 'i8')]

        logging.info("Determining following properties for sources with m<{0}:"\
                         .format(magcut))
        logging.info("   {0}".format(', '.join(criteria)))

        j = 0

        for self.I in self.sorted:

            logging.debug("Source {0}:".format(self.sourcelist['id'][self.I]))

            results_j = [self.sourcelist['id'][self.I]]

            ## insert criteria to check source against
            if 'howIsolated' in criteria:
                results_j.append(self.howIsolated(**kwargs))
                if j == 0: dtype.append(('howIsolated', 'f8'))
                logging.debug("   howIsolated: {0:.3f}".format(results_j[-1]))

            if 'howCloseToEdge' in criteria:
                results_j.append(self.howCloseToEdge())
                if j == 0: dtype.append(('howCloseToEdge', 'f8'))
                logging.debug("   howCloseToEdge: {0:.1f}pixel".format(\
                        results_j[-1]))

            if 'whatSNR' in criteria:
                results_j.append(self.whatSNR(**kwargs))
                if j == 0: dtype.append(('whatSNR', 'f8'))
                logging.debug("   whatSNR: {0:.2f}".format(results_j[-1]))

            if 'howFarFromBright' in criteria:
                bright_neighbor, dst = self.howFarFromBright()

                results_j.append(bright_neighbor)
                if j == 0: dtype.append(('brightNeighbor', 'i8'))
                logging.debug("   brightNeighbor: {0}".format(results_j[-1]))

                results_j.append(dst)
                if j == 0: dtype.append(('howFarFromBright', 'f8'))
                logging.debug("   howFarFromBright: {0:.2f}pixel".format(\
                        results_j[-1]))

                ## correct SNR if next brighter source within distance  of
                ## 0.75*FWHM
                if dst < 0.75*self.PSF.fwhm and 'whatSNR' in criteria:
                    results_j[-3] *= dst/(0.75*self.PSF.fwhm)


            results.append(tuple(results_j))
            j += 1

            if self.mag[self.I] > magcut:
                break

        return np.array(results, dtype=dtype)



    def howIsolated(self, rad, maglim=99., **kwargs):
        """
        Check how much flux bright nearby sources contribute relative to
        the flux of the source.

        Parameters
        ----------
        rad : float
            The radius within which the calculation is carried out
        maglim : float, optional
            Magnitude of faintest neighbor whose contribution is
            considered.

        Returns
        -------
        frac : float
            The result of the calculation, i.e the fractional flux
            contribution of nearby source within `rad'
        """

        ## only consider sources within distance of 2*`rad'
        nearby = (abs(self.x-self.x[self.I])<2*rad)\
            &(abs(self.y-self.y[self.I])<2*rad)
        nearby[self.I] = False

        ## only consider sources brighter maglim:
        nearby[self.mag>maglim] = False

        ## define array for calculation
        tmpData = np.zeros((2*rad+1,2*rad+1), dtype='float32')
        validPix = np.fromfunction(\
            lambda x,y: np.sqrt((x-rad)**2+(y-rad)**2)<=rad, tmpData.shape) 

        ##  carry out calculation
        for j in xrange(nearby.sum()):
            x_c = rad + self.x[nearby][j] - self.x[self.I]
            y_c = rad + self.y[nearby][j] - self.y[self.I]

            tmpData += self.PSF(size=tmpData.shape, x_c=x_c, y_c=y_c,
                                mag=(self.mag[nearby][j]-self.mag[self.I]))

        frac = tmpData[validPix].sum()
        frac /= self.PSF(size=tmpData.shape,
                         x_c=rad, y_c=rad, mag=0)[validPix].sum()
        return float(frac)



    def howCloseToEdge(self):
        """
        Caculate distance of source to closest edge of FoV.

        Returns
        -------
        result : float
            The distance of the source to the closest edge; negative for
            sources outside FoV
        """
        return min(self.x[self.I],
                   self.y[self.I],
                   self.shape[1]-1-self.x[self.I],
                   self.shape[0]-1-self.y[self.I])
    


    def whatSNR(self, sigma, psfrad, **kwargs):
        """
        Estimate S/N of the source

        Parameters
        ----------
        sigma : nd_array
            The sigma-array of the observation. Must be of same shape as
            considered FoV (i.e no datacube).

        Returns
        -------
        result : float
            The calculated S/N
        """

        psf = self.PSF(size=self.shape, x_c=self.x[self.I], y_c=self.y[self.I],
                       mag=0., maxrad=psfrad, return_sparse=True,
                       as_sparse_matrix=True)

        ## get flux
        flx = 10**(-0.4*(self.mag[self.I]-self.mZ))

        ## PSF array must be converted to a numpy array before calculation
        return flx*np.sqrt(((np.asarray(psf.todense())/sigma)**2).sum())



    def howFarFromBright(self):
        """
        Determine distance to nearest brighter source and the ID of that
        source
        
        Returns
        -------
        result : tuple
            The ID of the bright source and the distance in pixels.
        """
        if self.I == self.sorted[0]:
            return self.I, 99.9

        else:
            j = (self.sorted==self.I).argmax()

            dst = np.sqrt((self.x[self.I]-self.x[self.sorted[:j]])**2 +\
                              (self.y[self.I]-self.y[self.sorted[:j]])**2)
            
            return self.sourcelist['id'][self.sorted[dst.argmin()]], dst.min()
