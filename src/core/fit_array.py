import numpy as np,  sys
from scipy.sparse.linalg import lsqr
from scipy import sparse, optimize

##  own routines
from my_profiles import Gaussian, Moffat
from trans_coords import trans_coords

"""
The program contains `StarFit', the core-routine of PampelMuse. The
routine is optimized to handle large datacubes, e.g MUSE-like data.
For the analysis of smaller datacubes, e.g PMAS-like data,  some features
still have to be added from older routines.

v.0.0.0.0, 2011/09/30, (c). S. Kamann
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.0.0"

### some old code not used anymore
## from clip_sigma import clip_sigma
## from ima_arith import paste_area

## def shift_image(data, dx, dy):
##     
##     tmp_data = resample1d.shift_array(data, dx, axis=1, output='full')
##     
##     return resample1d.shift_array(tmp_data, dy, axis=0, output='full')


class StarFit:
    """
    `StarFit' is the core-routine of PampelMuse.  It uses a catalog of input
    sources, created using the `INITFIT'-package to analyse a complete
    datacube in an image-by-image manner. The fitting process itself is
    split into to parts: 
            (i) PSF & positions (if requested): A non-linear least squares fit
                    is performed for individual suited sources. The results
                    from the individual fits are combined afterwards.
            (ii) Fluxes: A linear least-squares fit is performed for all the
                    resolved sources simultaneously.
    Note that the routine first analyzes a central image and than alternating
    proceedes to the red and blue. This is done because usually the highest
    signal is expected near the center of the observed wavelength range.
    """
    def __init__(self, dataFits, sigmaFits, maskFits=None,
                  verbose=1, Ntrack=10):
## Options to include later on: makemod=False, makeres=False, clip=5.):
        """
        Initialize an instance of the `StarFit'-class.
    
        Parameters
        ----------
        dataFits : pyfits HDU object
            The data to analyze. Note that instead of the pure data, the HDU 
            containing the data must be provided. This is necessary since the
            routine will only load the image that is currently analyzed into
            memory.
        sigmaFits : pyfits HDU object
            The HDU containing the sigma-values of the data.
        maskFits : pyfits HDU object, optional
            The HDU containing the bad-pixel mask (if any)
        verbose : integer, optional
            The amount of output printed on the screen.
        Ntrack :  integer, optional
            In case the PSF and/or the source positions are fitted, the 
            routine will update the initial guesses of the fitted 
            parameters prior to fitting an image. It uses the results from
            the fits to the `Ntrack' neighboring images  for this purpose.
            If you do not want this behavior, set `Ntrack' to zero.


        Returns
        -------
            The newly created instance
        """

        self.dataFits  = dataFits
        self.sigmaFits = sigmaFits
        self.maskFits  = maskFits

        self.spatShape = (self.dataFits.header['NAXIS2'],
                          self.dataFits.header['NAXIS1'])
        self.spatPix   = self.dataFits.header['NAXIS1']\
            *self.dataFits.header['NAXIS2']
        self.specPix   = self.dataFits.header['NAXIS3']

        if self.maskFits != None:
            self.hasMask = True
        else:
            self.hasMask = False

        self.vb        = verbose ## 0-quiet, 1,2-some info, 3-lot of info
        
        self.Ntrack         = Ntrack
        self.lastFittedImas = []

        
    def provideSources(self, sources, spectra, coordTrans):
        """
        This method should be used to provide the instance with the necessary
        information about the sources in the field.
        All of those information is usually provided by the `INITFIT'-package.
        
        Parameters
        ----------
        sources : numpy record array
            The list of sources that are used in the analysis. The array must
            contain the fields 'id', 'x', 'y', 'mag', 'fitopt' and 'primary'.
            The specific meaning of the individual fields is described in the
            documentaion of `INITFIT'.
        spectra : nd_array
            A 2dim. array containing a spectrum for every independent component
            defined in `sources'.
        coordTrans : numpy record array
            The parameters of the coordinate transformation from input to pixel
            coordinates. The array must contain the fields `name', `value' and
            `fit', providing the name, the initial value(s) and a fit flag for
            every parameter.
            Note that `value' can either contain single values or an array with
            one value per spectral pixel.
        """
        self.sources    = sources
        self.spectra    = spectra
        self.coordTrans = coordTrans

        ## no. of fitted components
        self.Ncmp = self.sources['specrow'].max()+1

        self.specMap = {}
        for i in xrange(self.Ncmp):
            self.specMap[i] = np.where(self.sources['specrow']==i)[0]

        self.lastFitChange = np.ones((self.Ntrack, self.Ncmp), dtype='float32')

        self.posMap = [np.where(self.coordTrans['name']==name)[0][0] \
                           for name in ['A','B','C','D']]

        self.posFit = self.coordTrans['fit'].any()

    
    def providePSF(self, psfType, psfParameters, psfRad,
                   osFactor=6, osRadius=2):
        """
        This method should be used to provide the instance with the necessary
        information about the PSF that is used in the fitting process.
        
        Parameters
        ----------
        psfType : string
            The name of the  analytical  profile used to model the PSF.
            Currently either `moffat' or `gauss' can be used.
        psfParameters : numpy record array
            The initial values of the PSF parameters. The array must contain
            the fields `name', `value' and `fit', providing the name, the
            initial value(s) and a fit flag for every parameter.
            Note that `value' can either contain single values or an array
            with one value per spectral pixel.
        psfRad : float
            The radius out to which the PSF is defined around a source
        osFactor : integer, optional
            The oversampling factor of the central PSF
        osRadius : float, optional
            The radius out to which the PSF is oversampled. 
        """
        self.psfType = psfType
        self.psfPars = psfParameters
        self.psfRad  = psfRad

        self.psfFit = self.psfPars['fit'].any()

        ## collect stars suitable for PSF determination
        if self.psfFit or self.posFit:
            self.psfStars = np.where(self.sources['status']==3)[0]

        ## prepare function to generate PSF
        if self.psfType == "gauss": ## TheGaussian PSF
            self.psfMap = [np.where(self.psfPars['name']==name)[0][0] \
                               for name in ['fwhm', 'e', 'theta']]

            self.makePSF = lambda psfPars: Gaussian(fwhm=psfPars[0],
                                                    e=psfPars[1],
                                                    theta=psfPars[2],
                                                    osfactor=osFactor,
                                                    osradius=osRadius)

        elif self.psfType == "moffat": ## The Moffat PSF
            self.psfMap = [np.where(self.psfPars['name']==name)[0][0] \
                               for name in ['beta', 'fwhm', 'e', 'theta']]

            self.makePSF = lambda psfPars: Moffat(beta=psfPars[0],
                                                  fwhm=psfPars[1],
                                                  e=psfPars[2],
                                                  theta=psfPars[3],
                                                  osfactor=osFactor,
                                                  osradius=osRadius)
        


    def __call__(self, fitRad, returnModel=False, returnResiduals=False,
                 clip=5.):
        """
        Execute the fitting process. For each image of the datacube, the 
        following steps are performed:
            [1) Update the initial values]
             2) Initialize the PSF model for the analyzed image
             3) Create a PSF for every resolved source/component
            [4) Fit the PSF and/or  source positions]
             5) Fit the source fluxes
        Steps in square brackets are optional.
        
        Parameters
        ----------
        fitRad : float
            The radius within which the PSF is fitted to a source
        returnModel : boolean, optional
            Flag indicating whether an array containing the best-fit model
            should be returned. Not implemented yet.
        returnResdiduals : boolean, optional
            Flag indicating whether an array containing the residuals of the
            fit should be returned. Not implemented yet.
        clip : float, optional
            All pixels whose values devite by more than `clip' times SIGMA are
            masked. SIGMA is the standard deviation of the residuals between
            model and data.
        """
        self.makeMod = max(returnModel, returnResiduals) ## create model array
        self.makeRes = returnResiduals                   ## and residuals?
        self.clip    = clip    ## efficiency of bad pixel masking

        self.fitRad  = fitRad


        self.vprint(" "*17+"_"*42+'\n', 1)
        self.vprint(" "*17+"|Progress |Image |Current fit |Source ID |\n", 1)


        ## the fitting process
        for i in xrange(self.specPix):

            self.jj = (self.specPix>>1) - 1 - int(i/2) + i*np.mod(i,2) \
                + np.mod(self.specPix, 2)
            ## jj is a counter that starts at the central spectral pixel
            ## and proceeds simultaneoulsy to the red and blue end.

            self.vprint("\r"+" "*17+"|", 1)
            self.vprint("{0:8.2f}%|".format(100.*float(i)/self.specPix), 1)
            self.vprint("{0:6d}|".format(self.jj), 1)

            ## initialize PSF:
            psfPars  = self.psfPars['value'][self.psfMap, self.jj]
            self.PSF = self.makePSF(psfPars)

            ##  determine pixel coordinates:
            self.getPixCoords()


            ## load data
            ## np.ndarray.flatten -> returns copy
            self.data  = self.dataFits.section[self.jj].flatten()
            self.sigma = self.sigmaFits.section[self.jj].flatten()
            if self.hasMask:
                self.mask = (self.maskFits.section[self.jj].flatten()==0)
            else:
                self.mask = np.ones(self.data.shape, dtype='bool')

            ## create source matrix
            self.fillMatrix(radius=self.psfRad)

            ## fit PSF/positions if requested
            if self.psfFit:
                self.vprint(" "*9+"PSF|", 1)

                ## initial it of fluxes
                _A = self.A.tocsc().transpose()
                self.spectra[:,self.jj] += lsqr(_A, self.current)[0]
                
                self.updatePars()
                self.fitPSF()

            ## fit fluxes
                self.vprint("\b"*12, 1)
            self.vprint(" "*5+"Fluxes|", 1)

            _A = self.A.tocsc().transpose()
            self.spectra[:,self.jj] += lsqr(_A, self.current)[0]


            self.lastFittedImas.append(self.jj)
            if len(self.lastFittedImas)>2*self.Ntrack:
                self.lastFittedImas.remove(self.lastFittedImas[0])

## Only for debugging
##             if i == 30:
##                 import pylab
##                 pylab.imshow(self.current.reshape((100,100)), cmap=pylab.cm.hot,
##                              interpolation='nearest')
##                 pylab.show()
##                 
##                 break
#####################

        self.vprint("\n"+" "*17+"|"+"_"*40+'|\n\n', 1)


    @property
    def model(self):
        """Create the current best-fit model"""
        return self.A.transpose()*sparse.dia_matrix(\
                    (self.spectra[:,self.jj],[0]),
                    shape=(self.Ncmp, self.Ncmp)
                    )

    @property
    def current(self):
        """Determine the current residuals"""
        ## self.model.sum(axis=1) returns numpy.matrixlib.defmatrix-object
        ## `getA1' returns a flattened numpy.ndarray
        return self.data - self.model.sum(axis=1).getA1()


    def getPixCoords(self):
        """
        Calculate the pixel coordinates from the provided input coordinates
        and the best-fit parameters of the coordinate transformation.
        """
        if self.sources['x'].ndim > 1:
            self.x = self.sources['x'][:,self.jj]
            self.y = self.sources['y'][:,self.jj]
        else:
            if self.coordTrans['value'].ndim > 1:
                A, B, C, D = self.coordTrans['value'][self.posMap, self.jj]
            else:
                A, B, C, D = self.coordTrans['value'][self.posMap]
                
            self.x = A*self.sources['x']+B*self.sources['y']+C
            self.y = A*self.sources['y']-B*self.sources['x']+D



    def fillMatrix(self, radius):
        """
        Define the source matrix. The source matrix is a sparse matrix that
        contains the spatial profiles of all resolved components, one per
        column.
        
        Parameters
        ----------
        radius : float
            The radius out to which the profile is calculated around a point
            source.
        """
        self.A =sparse.lil_matrix((self.Ncmp, self.spatPix))

        for i in xrange(self.Ncmp):

            indices_i = []
            flux_i    = []

            for j in self.specMap[i]:

                if self.sources['id'][j] == -1:
                    self.A[i,:] = 1.
                else:
                    psf_ij = self.PSF(size=self.spatShape, x_c=self.x[j],
                                      y_c=self.y[j], mag=0., maxrad=radius,
                                      return_sparse=True,
                                      as_sparse_matrix=False)


                    indices_i.append(psf_ij[1][1] \
                                         + self.spatShape[1]*psf_ij[1][0])
                    flux_i.append(psf_ij[0])

                    if len(indices_i) > 1:
                        indices_i.append(np.union1d(indices_i[0], indices_i[1]))

                        flux_i.append(np.zeros(indices_i[-1].shape,
                                               dtype='float32'))
                        flux_i[-1][np.in1d(indices_i[-1], indices_i[0])]\
                            = flux_i[0]
                        flux_i[-1][np.in1d(indices_i[-1], indices_i[1])]\
                            = flux_i[1]

                        for z in xrange(2):
                            indices_i.remove(indices_i[0])
                            flux_i.remove(flux_i[0])

                    self.A[i, indices_i[-1]] = flux_i[-1] 



    def fitPSF(self):
        """
        Perform the PSF fit. For every PSF-source, a non-linear least-squares
        fit is performed. Depending on the respective fit-flags, either the
        PSF-parameters and/or source coordinates are fitted.
        The results of the individual fits are then used to update the PSF and/
        or the coordinates of the parameter transformation.
        """
        nPars = self.psfPars.size+3

        allPars = np.zeros((nPars, self.psfStars.size), dtype='float32')

        fitFlags = np.zeros((nPars,), dtype='bool')
        fitFlags[:2] = self.coordTrans['fit'].any()
        fitFlags[2:-1] = self.psfPars['fit']
        fitFlags[-1] = True

        ## create working copy of `current'-array. Required since the
        ## definition of `current' (using an @property-method) does not
        ## work with in-place operations like `+=' or `-='.
        self._current = self.current.copy()

        ## loop over all PSF sources; add source to residuals, fit resulting
        ##  array and subtract source again.
        j = 0
        for i in self.psfStars:
            kk = self.sources['specrow'][i]
            self.vprint("{0:10d}|".format(int(self.sources['id'][i])), 1)

            allPars[0,j]    = self.x[i]
            allPars[1,j]    = self.y[i]
            allPars[2:-1,j] = self.psfPars['value'][self.psfMap, self.jj]
            allPars[-1,j]   = self.spectra[kk,self.jj]

            initPSF =  np.asarray(self.A[kk,:].todense(), dtype="float32")[0]
            self._current += initPSF*self.spectra[kk,self.jj]

            allPars[fitFlags,j] = optimize.fmin(\
                func=self.fitOnePSF,
                x0=allPars[fitFlags,j],
                args=(allPars[fitFlags==False,j],fitFlags),
                disp=False)

            self._current -= initPSF*self.spectra[kk,self.jj]
            j += 1

            self.vprint("\b"*11, 1)

        if self.coordTrans['fit'].any(): ## update coordinate transformation
            newCoords = np.array(zip(self.sources['id'][self.psfStars],
                                     allPars[0], allPars[1]),
                                 dtype=[('id', 'i4'), ('x', 'f8'), ('y', 'f8')])

            newTrans = trans_coords(self.sources, newCoords)
            for prm in newTrans.dtype.names:
                index = np.where(self.coordTrans['name']==prm)[0][0]
                self.coordTrans['value'][index, self.jj] = newTrans[prm][0]

            self.getPixCoords()

        if self.psfPars['fit'].any(): ## update PSF
            self.psfPars['value'][self.psfMap, self.jj] \
                = allPars[2:-1,:].mean(axis=1)

            self.PSF = self.makePSF(self.psfPars['value'][self.psfMap, self.jj])

        self.fillMatrix(radius=self.psfRad) ## recalculate source matrix



    def fitOnePSF(self, x0, fixedPars, fitFlags):
        """
        The optimization routine that is passed to the `fmin'-routine of the
        `scipy.optimize'-package. During each iteration, a PSF array is created,
        scaled and subtracted from the data. The chi^2-value is returned.
        
        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted in the
            following order: x, y, psfPar_0,.., psfPar_i, flux
        fixedPars : array_like
            The values of the parameters that are not fitted in the following
            order: x, y, psfPar_0,.., psfPar_i, flux
        fitFlags : array_like
            A boolean flag for each parameter needed to create the PSF
            indicating whether it is fitted. The number of True-values must be
            equal to the size of `x0' and the number of False-values must be
            equal to the size of `fixedPars'.
        
        Returns
        -------
        chi2 : float
            The chi^2 value of the fit.
        """
        
        psfPars = np.zeros(fitFlags.shape, dtype='float64')
        psfPars[fitFlags] = x0
        psfPars[fitFlags==False] = fixedPars

        thePSF   = self.makePSF(psfPars[2:-1])

        psfArray = thePSF(size=self.spatShape, x_c=psfPars[0],
                          y_c=psfPars[1], mag=0., maxrad=self.fitRad,
                          return_sparse=True, as_sparse_matrix=False)

        usedPix  = psfArray[1][1] + self.spatShape[1]*psfArray[1][0]

        err  = (self._current[usedPix] - psfPars[-1]*psfArray[0])
        err *= self.mask[usedPix]

        return ((err*err)/(self.sigma[usedPix]*self.sigma[usedPix])).sum()




    def updatePars(self):
        """
        Method is used to update the initial guesses of the parameters o
        the PSF and the coordiate transformation prior to a PSF fit.
        The new initial guesses are determined by calculating the mean
        of the results of the `Ntrack'/2 nearest images that have been
        fitted so far.
        """
        if self.Ntrack == 0:
            ##  no updating requested
            return

        if len(self.lastFittedImas) < 2:
            ## nothing to update yet
            return
        elif len(self.lastFittedImas) < self.Ntrack:
            ## use all images that have been fitted so far
            LASTFITS = self.lastFittedImas
        else:
            ## get every second of the recently fitted images
            LASTFITS = [self.lastFittedImas[i] for i in xrange(\
                    1, len(self.lastFittedImas), 2)]

        if self.coordTrans['fit'].any():
            self.coordTrans['value'][self.coordTrans['fit'], self.jj] \
                = self.coordTrans['value'][self.coordTrans['fit']][:,LASTFITS]\
                .mean(axis=-1)


        if self.psfPars['fit'].any(): ## update PSF
            self.psfPars['value'][self.psfPars['fit'], self.jj] \
                = self.psfPars['value'][self.psfPars['fit']][:,LASTFITS]\
                .mean(axis=-1)



    def vprint(self, string, vb_min):
        """
        Method used to print information on the screen.
        
        Parameters
        ----------
        string : string
            The string to be printed.
        vb_min : integer
            The minimum verbosity at which the string is printed.
        """
    
        if self.vb >= vb_min:
            sys.stdout.write(string)
            sys.stdout.flush()


class OldData:
    
    ## update the data and sigma array (mask array only if provided)
    ## should be called for every image in a data cube
    def update_data(self, data, sigma, inmask=None):

        self.data[:]  = deepcopy(data.reshape(-1))
        self.sigma[:] = deepcopy(sigma.reshape(-1))

        if inmask != None:
            self.inmask[:] = (deepcopy(inmask.reshape(-1))==0)



    ## Methods that update the arrays containing the fit parameters
    ## Note: if the method(s) are not called for a new dataset (image)
    ##       the code will use the best-fit values of the last fit as
    ##       initial guesses.
    ## force can be used to update all parameters (by default, only
    ## the fixed parameters are updated (except if self.Nima=0))
    def update_fluxes(self, fluxes, broadband=None, force=False):
        if self.Nima == 0 or force:
            update = np.ones((self.Nflx,), dtype="bool")
        else:
            update = np.zeros((self.Nflx,), dtype="bool")
            if self.fluxfit == 0:
                update[:-2] = True
            if self.skyfit == 0:
                update[-2] = True
            if self.unresfit == 0:
                update[-1] = True

        self.fluxes[:,update] = deepcopy(fluxes)

        if broadband != None:
            self.bbfl = broadband
            for i in xrange(self.Nrsv): 
                ## det. relative contribution of merged sources
                total = self.bbfl[self.map==self.map[i]].sum()
                self.frac[i] = self.bbfl[i]/total


    def update_positions(self, xpos, ypos, force=False):
        if self.Nima == 0 or force or self.posfit < 3:
            self.xpos[:] = deepcopy(xpos)
            self.ypos[:] = deepcopy(ypos)



    def update_psfpars(self, psfpars, force=False):
        if self.Nima == 0 or force:
            self.psfpars[:] = deepcopy(psfpars)
        elif self.psffit.min() == 0:
            self.psfpars[:, self.psffit==False] \
                = deepcopy(psfpars[self.psffit==False])



    def update_pospars(self, pospars, force=False):
        if self.Nima == 0 or force:
            self.pospars[:]= deepcopy(pospars)
        elif self.pptofit.min() == 0:
            self.pospars[1, self.pptofit==False] \
                = deepcopy(pospars[self.pptofit==False])



    ###### The routine to perform an (iterative) fit ######
    ## If more than one property (e.g, PSF, positions, fluxes) are fitted, an
    ## iterative process will  be performed
    ## After the fit has converged to a solution, the method will perform an
    ## F-test to check whether all sources are needed in the fit and remove
    ## the ones that are not needed
    def exec_fit(self):    

        self.Nima     += 1
        self.psf_maker = self.PSF()
        self.chisq     = []
        self.infit     = np.array([True]*self.Nflx)  ## to keep track of 
                                                        ## fitted fluxes
        self.newpsf    = True


        iteration = 1

        for i in xrange(2): ## Caluclate true pixel positions
            self.xpix[i] =  self.pospars[i,0]*self.xpos[i] \
                + self.pospars[i,1]*self.ypos[i] + self.pospars[i,2]
            self.ypix[i] = -self.pospars[i,1]*self.xpos[i] \
                + self.pospars[i,0]*self.ypos[i] + self.pospars[i,3]

        self.sources[:] = 0.
        for i in xrange(self.Nrsv): ## create PSF for every source
            j = self.map[i]
            self.sources[:,j] += \
                self.frac[i]*self.psf_maker(self.xpix[1,i], self.ypix[1,i],
                                            self.psfpars[1]).reshape(-1)

            if self._reject and self.fluxes[1,j] <= 0.:
                self.fluxes[j] = 0.
                self.infit[j] = False

        self.sources[:,-2:] = 1.

        self._unres_comp()  ## create array with contribution of unresolved
                            ## sources (entries in 'all', not in 'resolved')
        self._unres_norm()  ## add 'resolved' sources w\ flux <= 0 (only if
                            ## self._reject) and normalize

        self.infit[-2] = self.skyfit
        self.infit[-1] = self.unresfit

        ## estimate chi^2 for the initial guesses (& reject bad pixels)
        self._mask(clip=self.clip, chisq_compute=True)

        ## Calculate noise for every source
        self.noise = 1./np.sqrt(((self.sources/self.sigma[:,np.newaxis])**2)[self.mask].sum(axis=0))

        ## aperture photometry for sources with flux=0
##         if (self.infit==False).sum() > 0:
##             self.aper_photo()


        while True:  ## Iteratively find best solution for all stars
            Nfits = 0
            self.vprint("\r|%7i|%11i" %(self.Nima, iteration), 1)

            if self.infit.sum() > 0:
                Nfits += 1
                self.vprint("|%13s" %"fluxes", 1)
                self.fit_fluxes()
                self.vprint("|%7.1f" %self.chisq[-1] + "\b" * 22, 1)

            if self.posfit > 0:
                Nfits += 1
                self.vprint("|%13s" %"positions", 1)
                self.fit_positions()
                self.vprint("|%7.1f" %self.chisq[-1] + "\b" * 22, 1)

            if self.psfvar > 0:
                Nfits += 1
                self.vprint("|%13s" %"PSF", 1)
                self.fit_psf()
                self.vprint("|%7.1f" %self.chisq[-1]+ "\b" * 22, 1)

            self.deleted = (self.fluxes[1,:-2]==0.).sum()

            ## convergence check
            if Nfits > 1:
                dchisq = (self.chisq[0]-self.chisq[-1])/self.chisq[-1] 
                self.vprint('|'+13*' '+'|%7.1f|%14.1e|%9i|' \
                                %(self.chisq[-1], dchisq, self.deleted), 1)

                if dchisq < 1e-3:
                    Nfits = 0
                else:
                    self.chisq = []

            iteration += 1

            if Nfits < 2:
                if Nfits == 1:
                    self.vprint('|'+13*' '+'|%7.1f|'%self.chisq[-1]+14*' '+\
                                    '|%9i|' %self.deleted, 1)
                break


##         if self.fluxfit > 0:
##             while True: ## Second iteration: Check significance of detections
## 
##                 self.vprint("\r|%7i|%11i" %(self.Nima, iteration), 1)
## 
##                 self.chisq = [self.chisq[-1]]
## 
##                 self.snr = self.fluxes[1]/self.noise
##                 min_snr  = self.snr[self.infit].min()
##                 i_ex     = abs(self.snr-min_snr).argmin()
## 
##                 self.infit[i_ex] = False
##                 
##                 self.vprint("|%13s" %"fluxes", 1)
##                 self.fit_fluxes()
## 
##                 p1 = self.infit.sum()  ## perform F-test
##                 p2 = p1+1
##                 n  = self.mask.sum()
##                 F = ((self.chisq[1]-self.chisq[0]))/(self.chisq[0]/(n-p2))
## 
##                 if special.fdtri(p2-p1, n-p2, 0.95) < F:
##                     self.vprint("|%7.1f|%14s|%9i|" %(self.chisq[0],
##                                                      "F test: exit",
##                                                      self.deleted), 1)
##                     self.fluxes[1, self.infit] = self.fluxes[0, self.infit]
##                     self.infit[i_ex] = True
##                     self.chisq = [self.chisq[0]]
##                     break
##                 else:
##                     self.deleted += 1
##                     self.fluxes[:, i_ex] = 0.
##                     self.vprint('|%7.1f|%14s|%9i|' %(self.chisq[0],
##                                                      "F test: remove",
##                                                      self.deleted), 1)
##                     iteration += 1

        self.outmask = (self.mask.reshape(self.size)==0)
        
        if self.makemod:

            model_1d  = (self.fluxes[1] * self.sources).sum(axis=1)
            self.model = model_1d.reshape(self.size)
            if self.makeres:
                residuals_1d = self.data - model_1d
                self.residuals = residuals_1d.reshape(self.size)

        self.vprint("\n", 4)



    def _unres_comp(self):

        ## only update if unresolved component included and PSF or
        ## positions have changed
        if self.hasunres and self.newpsf:

            self.unres_faint = np.zeros(self.size, dtype="float32")

            for i in xrange(self.Ntot):
                if self.map[i] == -1:

                    psf_i = self.psf_maker(self.xpix[1,i], self.ypix[1,i],
                                           self.psfpars[1])
                    self.unres_faint += self.bbfl[i]* psf_i
                        
            self.newpsf=False
            

            

    def _unres_norm(self):
        
        if self.hasunres:
            self.sources[:,-1] = self.unres_faint.reshape((-1,))

            if not self.infit[:-2].min():
                for i in xrange(self.Nrsv):
                    j = self.map[i]
                    if not self.infit[j]:
                        self.sources[:,-1] +=self.bbfl[i]*self.sources[:,j]

            mean_faint = self.sources[:,-1].mean()
            if mean_faint != 0:
                self.sources[:,-1] /= mean_faint


                
    def _mask(self, clip, chisq_compute=False):
        
        ## calculate difference between model and data
        err = self.data - (self.fluxes[1]*self.sources).sum(axis=1)
        err[self.inmask==0] = np.median(err) 

        ## find bad pixels via sigma-clipping
        clip_result = clip_sigma((err/self.sigma), clip=clip)

        ## update mask
        self.mask = np.logical_and(self.inmask, clip_result[1])

##         print "\n", self.mask.sum()

        ## if requested, add chi^2 value
        if chisq_compute:
            err *= self.mask

            self.chisq.append(((err*err)/(self.sigma*self.sigma)).sum())


            
    def fit_positions(self):

        if self.posfit == 3:
            isfitted = np.zeros((self.Ntot,), dtype="bool")
            for i in xrange(self.Nrsv):
                if self.infit[self.map[i]]: isfitted[i] = True
            fitpars = np.concatenate((self.xpos[1,isfitted],
                                         self.ypos[1,isfitted])) 
        else:
            fitpars  = self.pospars[1,self.pptofit]
            isfitted = np.ones((self.Nrsv,), dtype='bool')

        self.workcopy  = self.data.copy()
        self.workcopy -= (self.fluxes[1,-2:]*self.sources[:,-2:]).sum(axis=1)

        results = optimize.fmin(self._fit_newpsf, fitpars,
                                args=(False,self.xpos[1],self.ypos[1],
                                      self.psfpars[1],self.pospars[1],isfitted),
                                full_output=1, disp=False)

        if self.posfit == 3:
            self.xpos[0, isfitted] = self.xpos[1, isfitted]
            self.ypos[0, isfitted] = self.ypos[1, isfitted]

            self.xpos[1, self.infit[:-2]] = results[0][:isfitted.sum()]
            self.ypos[1, self.infit[:-2]] = results[0][isfitted.sum():]

        else:
            self.pospars[0, self.pptofit] = self.pospars[1, self.pptofit]
            self.pospars[1, self.pptofit] = results[0]

        for i in xrange(2):
            self.xpix[i] =  self.pospars[i,0]*self.xpos[i] \
                + self.pospars[i,1]*self.ypos[i] + self.pospars[i,2]
            self.ypix[i] = -self.pospars[i,1]*self.xpos[i] \
                + self.pospars[i,0]*self.ypos[i] + self.pospars[i,3]

        self._mask(clip=self.clip)
        self.chisq.append(results[1])
        self.newpsf = True



    def fit_psf(self):
        
        fitpars = self.psfpars[1,self.psffit]

        self.workcopy  = self.data.copy()
        self.workcopy -= (self.fluxes[1,-2:]*self.sources[:,-2:]).sum(axis=1)

        isfitted = np.ones((self.Nrsv,), dtype='bool')

        results = optimize.fmin(self._fit_newpsf, fitpars,
                                args=(True,self.xpos[1],self.ypos[1],
                                      self.psfpars[1],self.pospars[1],isfitted),
                                full_output=1, disp=False)

        self.psfpars[0,self.psffit] = self.psfpars[1, self.psffit]
        self.psfpars[1,self.psffit] = results[0]

        self._mask(clip=self.clip)
        self.chisq.append(results[1])
        self.newpsf = True


    def _fit_newpsf(self, fitpars, newshape, x, y, psfpars, pospars, isfitted):


        if newshape:
            psfpars[self.psffit] = fitpars
        else:
            if self.posfit == 3:
                x[isfitted] = np.asarray(fitpars[:isfitted.sum()],
                                            dtype="float32")
                y[isfitted] = np.asarray(fitpars[isfitted.sum():],
                                            dtype="float32")
            else:
                pospars[self.pptofit] = fitpars


        xpix =  pospars[0]*x + pospars[1]*y + pospars[2]
        ypix = -pospars[1]*x + pospars[0]*y + pospars[3]

        self.sources[:,:-2] = 0. 
        for i in xrange(self.Nrsv):
            if isfitted[i]:
                j = self.map[i]
                self.sources[:,j] += self.frac[i]*self.psf_maker(xpix[i], ypix[i], psfpars).reshape(-1)


        err  =self.workcopy-(self.fluxes[1,:-2]*self.sources[:,:-2]).sum(axis=1)
        err *=self.mask

        return ((err*err)/(self.sigma*self.sigma)).sum()


    
    def aper_photo(self, cut=2.0, r_aper=2.5):

        res  = self.data - (self.fluxes[1] * self.sources).sum(axis=1)

        for i in xrange(self.Nind):
            if not self.infit[i]:
                nearpix = np.fromfunction(self._select_pix, self.size, cid=i,
                                             r_in=r_aper)
                flx = res[nearpix*self.mask].sum()

                if flx > cut*self.noise[i]:
                    self.fluxes[1,i] = flx
                    self.infit[i] = True


    
    def _select_pix(self, x, y, sid, r_in, r_out=None):

        r = np.sqrt((x-self.ypix[1,sid])**2 + (y-self.xpix[1,sid])**2)

        if r_out == None: return (r<=r_in)
        else: return np.logical_and(r>r_in, r<=r_out)



    def vprint(self, string, vb_min):
        if self.vb >= vb_min:
            sys.stdout.write(string)
            sys.stdout.flush()



    def fit_fluxes_smt(self):

        self.fluxes[0] = self.fluxes[1]

        self.workcopy  = self.data.copy()
        self.workcopy -=(self.fluxes[1,self.infit==False]*self.sources[:,self.infit==False]).sum(axis=1)

        if self.newpsf:
            self.noise = 1./np.sqrt(((self.sources/self.sigma[:,np.newaxis])**2).sum(axis=0))
            self._unres_comp()

        deleted = (self.infit[:-2]==False).sum()

        while True:

            self._unres_norm()

            a = (self.sources/self.sigma[:,np.newaxis])[self.mask][:,self.infit]
            b = (self.workcopy/self.sigma)[self.mask]

            results = np.linalg.lstsq(a,b)

            self.fluxes[1, self.infit] = results[0]

            if self._reject:
                for i in xrange(self.Nind):
                    if self.infit[i] and self.fluxes[1,:-2] < self.noise:
                        self.infit[i] = False
                        self.fluxes[1,i] = 0.
                        ## no need to update workcopy since flux=0


            if (self.infit[:-2]==False).sum() == deleted:
                break
            else:
                deleted = (self.infit[:-2]==False).sum()

        self._mask(clip=self.clip)
        self.chisq.append(results[1])



    def fit_fluxes_hyb(self):

        ## update noise values if PSF/positions have changed
        if self.newpsf:
            self.noise = 1./np.sqrt(((self.sources/self.sigma[:,np.newaxis])**2)[self.mask].sum(axis=0))
        
        insrcfit = self.infit.copy()

        ## create work copy and subtract all components
        self.workcopy  = self.data.copy()
        self.workcopy -=(self.fluxes[1]*self.sources).sum(axis=1)

        ## Create "valid pixel array" for every source 
        fitapers = np.zeros(self.sources.shape, dtype="bool")
        for i in xrange(self.Nrsv):
            j = self.map[i]
            if insrcfit[j]:
                sel_i = np.fromfunction(self._select_pix,self.size,
                                           sid=i, r_in=self.fitrad).reshape(-1)
                fitapers[:,j] = np.logical_or(fitapers[:,j], sel_i)
        fitapers[:,-2] = insrcfit[-2]
        fitapers[:,-1] = insrcfit[-1]
        
        for j in xrange(self.Nind):
            if not np.logical_and(self.mask, fitapers[:,j]).any():
                self.fluxes[:,j] = 0.
                fitapers[:,j]    = False
                insrcfit[j]      = False

        
        iteration = 0
        converged = 0
        while True: ## BEGINNING OF ITERATIVE PART

            self.fluxes[0] = self.fluxes[1]

            ## fitting of sources/background
            fitpix = np.logical_and(self.mask, np.any(fitapers,axis=1))

            A = ((self.sources*fitapers)[:,insrcfit]/self.sigma[:,np.newaxis])[fitpix]
            b = (self.workcopy/self.sigma)[fitpix]
            
            results = np.linalg.lstsq(A,b)
            dflux   = results[0]

            self.fluxes[1,insrcfit] += dflux

            ## update workcopy using difference between the two fits
            self.workcopy -= (dflux*self.sources[:,insrcfit]).sum(axis=1)
            

            if iteration >= 4: ## check if sources have converged
                j = 0
                for i in xrange(self.Nind):
                    if insrcfit[i]:
                        cc_flux  = self.flux_cv*self.fluxes[1,i]
                        cc_sigma = self.sigma_cv*self.noise[i]
                        if dflux[j] < max(cc_flux, cc_sigma):
                            ## if so, remove them from list of fitted sources
                            ## and subtract them completely from workcopy
                            converged += 1
                            insrcfit[i] = False
                        j += 1

            if converged > 0 and insrcfit[-2:].any():
                ## fix background to current value
                insrcfit[-2:] = False
                    
            if insrcfit.any() == False or iteration == 10:
                ## max. no of iterations = 10
                break
            else:
                iteration += 1

        self._mask(clip=self.clip, chisq_compute=True)



    def fit_fluxes_itt(self):

        ## update noise values if PSF/positions have changed
        if self.newpsf:
            self.noise = 1./np.sqrt(((self.sources/self.sigma[:,np.newaxis])**2)[self.mask].sum(axis=0))
        
        insrcfit = self.infit.copy()

        ## create workcopy and subtract all components
        self.workcopy  = self.data.copy()
        self.workcopy -=(self.fluxes[1]*self.sources).sum(axis=1)


        ## Create "valid pixel array" for every source 
        fitapers = np.zeros(self.sources.shape, dtype="bool")
        for i in xrange(self.Nrsv):
            j = self.map[i]
            if insrcfit[j]:
                sel_i = np.fromfunction(self._select_pix,self.size,
                                           sid=i, r_in=self.fitrad).reshape(-1)
                fitapers[:,j] = np.logical_or(fitapers[:,j], sel_i)
        fitapers[:,-2] = insrcfit[-2]
        fitapers[:,-1] = insrcfit[-1]
        
        for j in xrange(self.Nind):
            if not np.logical_and(self.mask, fitapers[:,j]).any():
                self.fluxes[:,j] = 0.
                fitapers[:,j]    = False
                insrcfit[j]      = False


        ## sort sources by descending fluxes
        self.fsorted = list(self.fluxes[1,:-2].argsort())
        self.fsorted.reverse()

        ## background components that are fitted:
        bf = np.concatenate([np.zeros(self.Nind, dtype='bool'),
                                self.infit[-2:]])

        iteration = 0
        converged = 0
        while True: ## BEGINNING OF ITERATIVE PART

            self.fluxes[0] = self.fluxes[1]

            if bf.any() and converged == 0:
                a = (self.sources[:,bf]/self.sigma[:,np.newaxis])[self.mask]
                b = (self.workcopy/self.sigma)[self.mask]

                results = np.linalg.lstsq(a,b)

                self.fluxes[1,bf] += results[0]
                self.workcopy -= (results[0]*self.sources[:,bf]).sum(axis=1)

            for i in self.fsorted: ## fit the individual sources
                if insrcfit[i]:

                    fitpix = np.logical_and(self.mask, fitapers[:,i])

                    a = (self.sources[:,i]/self.sigma)[fitpix]
                    A = a.reshape((-1,1))
                    b = (self.workcopy/self.sigma)[fitpix]

                    results = np.linalg.lstsq(A,b)
                    dflux   = results[0][0]
                    
                    self.fluxes[1,i] += dflux
                    self.workcopy    -= (dflux*self.sources[:,i])
            
                    if iteration >= 4: ## check if source has converged
                        cc_flux  = self.flux_cv*self.fluxes[1,i]
                        cc_sigma = self.sigma_cv*self.noise[i]
                        if dflux < max(cc_flux, cc_sigma):
                            self.fsorted.remove(i)
                            converged += 1

            if self.fsorted == [] or iteration == 10: ## max. no of iterations
                break
            else:
                iteration += 1

        self._mask(clip=self.clip, chisq_compute=True)


##     def _fit_samepsf(self, fitpars):
##         last = 0
## 
##         err = self.workcopy.copy()
## 
##         if self.skyfit > 1:
##             last -= 1
##             err -= fitpars[last]*self.unres_array
##         if self.skyfit > 0:
##             last -= 1
##             err -= fitpars[last]
##         if self.fluxfit > 0:
##             err  -= (fitpars[:last, np.newaxis, np.newaxis] \
##                          *self.sources[self.infit]).sum(axis=0)
## 
##         err *= self.mask
## 
##         return ((err*err)/(self.sigma*self.sigma)).sum()






## class OldFit:
##     def __init__(self, data, sigma, mask=None):
## 
##         self.data  = data
##         self.sigma = sigma  
## 
##         self.size  = self.data.shape
## 
##         self.mask  = np.ones(self.size, dtype='int16')
##         if mask != None:
##             self.mask -= (mask>0.)
## 
## 
##         self.pdict   = {}  ## will contain all fit parameters and a boolean fit
##                            ## option for each parameter
## 
##         self.psfvar = False ## is the PSF (parameters, star positions)
##                             ##  variable?
## 
##         self.fluxfit = False
##         
## 
## 
##     def init_psf(self, type, lut=None, data=None, params=[], tofit=[]):
##         """
##         valid PSF types: 'gauss', 'moffat' or 'array'
##         params and tofit are 1D arrays containing the PSF parameters and for 
##         each parameter a (boolean) fit option (used if type is not 'array')
##         NOTE: The flux is automatically set to 1
##         data contains PSF (used if type is 'array')
##         """
## 
##         if sum(tofit) > 0:
##             self.psfvar = True
## 
##         Nparams = {'array':1,'gauss':3, 'moffat':4}
## 
##         p = np.asarray(params)
##         f = np.asarray(tofit)
## 
##         self.psf = type
## 
##         self.lut = lut
## 
##         if self.psf == 'array':
##             self.pdict['psf'] = data
##             self.Nppsf = 0
## 
##         elif Nparams.has_key(type):
##             self.Nppsf = Nparams[type]
## 
##             if p.shape[0] == self.Nppsf:
##                 parpsf = np.zeros((2, Nparams[type]), dtype='float32')
##                 parpsf[0] = params
##                 parpsf[1] = tofit
##                 self.pdict['psf'] = parpsf
##             else:
##                 raise IOError('Invalid number of parameters for PSF-type "'+\
##                                   type+'"! Must be '+str(Nparams[type])+'!')
## 
##         else:
##             raise IOError('Unknown PSF option: %s' %self.psf)
##             
## 
## ## !CHANGED    
##     def init_sources(self, positions, fluxes, posfit, fluxfit):
##         """
## 	<positions> is an array containing the information about the positions
## 	of the sources. To simplify the usage of highres a priori information,
## 	it is constructed in the following way: The first 4 entries contain the 
## 	transformation parameters A, B, C and D, followed by the (not
## 	transformed) x- and y coordinates for every star. The positions on the
## 	fitted array are determined via: x_fit =  A*x + B*y + C
## 	                                 y_fit = -B*x + A*y + D 
## 	<fluxes> is an (Nsources,) shaped array containing the fluxes of every
## 	source. 
## 
##         <posfit> can be either 0, 1, 2 or 3. If 0, the positions are fixed; if
## 	1, only C and D are free; if 2, A to D are free; if 3, individual
## 	positions are fitted.
##         <fluxfit> determines whether the fluxes are fitted (<fluxfit>=1) or not
## 	(<fluxfit>=0).
##         """
## 
##         if posfit > 0:
##             self.psfvar = True
##         if fluxfit > 0:
##             self.fluxfit = True
## 
## ##         self.x_ref = refpos[0]
## ##         self.y_ref = refpos[1]
## 
##         self.ns      = fluxes.shape[0]        
##         self.sources = np.arange(self.ns)  ## contains all sources still in
##                                               ## the fit
## 
##         parpos = np.zeros((2, 2*self.ns+4), dtype='float32')
##         parflx = np.zeros((2, self.ns), dtype='float32')
## 
## 
##         if posfit == 1:
##             parpos[1,2:4] = 1
## 	elif posfit == 2:
## 	    parpos[1,:4] =1
##             
##         parpos[0] = positions
##         parflx[0] = fluxes
## 
##         if posfit == 3:
##             parpos[1,2:] = 1
##         if fluxfit == 1:
##             parflx[1] = 1
## 
##         self.pdict['pos'] = parpos
##         self.pdict['flx'] = parflx
## 
##         self.objects = np.zeros((self.ns,self.size[0],self.size[1]),
##                                    dtype='float32')
## 
## 
## 
## 
##     def init_background(self, value=0., tofit=0):
##         """
##         If background should be fitted, set tofit=1. Value than acts as initial
##         guess
##         """
##         parbck = np.zeros((2,1), dtype='float32')
##         parbck[0] = value
##         parbck[1] = tofit
##         self.pdict['bck'] = parbck
## 
## 
## ## !CHANGED
##     def exec_fit(self, disp=True, model=False, residuals=False):
## 
##         ## write all fit parameters into one array
##         if self.psf == 'array':
##             self.fitparams = np.concatenate((self.pdict['pos'][0],
##                                                 self.pdict['flx'][0],
##                                                 self.pdict['bck'][0]))
##             self.isfitted  = np.concatenate((self.pdict['pos'][1],
##                                                 self.pdict['flx'][1],
##                                                 self.pdict['bck'][1]))
##         else:
##             self.fitparams = np.concatenate((self.pdict['psf'][0],
##                                                 self.pdict['pos'][0],
##                                                 self.pdict['flx'][0],
##                                                 self.pdict['bck'][0]))
##             self.isfitted  = np.concatenate((self.pdict['psf'][1],
##                                                 self.pdict['pos'][1],
##                                                 self.pdict['flx'][1],
##                                                 self.pdict['bck'][1]))
## 
##         self.Npix = self.mask.sum()
## 
##         if not self.fluxfit:
##             select_sources = True
##             stop = True
##         else:
##             select_sources = False
##             stop = False
## 
##         iter = 0
##         self.chisq = 0.
##         while True:
##             
##             if select_sources:
##                 fluxes = self.fitparams[self.Nppsf + 4 + 2*self.ns:-1]
## 
##                 bad    = (fluxes <= 0.)
##                 self.fitparams[self.Nppsf + 4 + 2*self.ns:-1][bad] = 0.
##                 self.isfitted[self.Nppsf + 4 + 2*self.ns:-1][bad] = False
## 
##                 self.sources = np.arange(self.ns)[fluxes>0]
## 
##             self.Ndof = self.isfitted.sum()
## 
##             free_params = self.fitparams[self.isfitted==1]
## 
##             if not self.psfvar:
##                 self.make_psf()
## 
##             result = optimize.fmin(self._fit_res, free_params, full_output=1,
##                                    disp=False)
## 
##             if iter > 0: ## using an F test to see whether the fit quality
##                          ## decreased significantly
## 
##                 p2 = self.Ndof+1
##                 p1 = self.Ndof
##                 n  = self.mask.sum()
## 
##                 F = ((result[1]-self.chisq)/(p2-p1))/(self.chisq/(n-p2))
## 
##                 if special.fdtri(p2-p1, n-p2, 0.95) < F: 
##                     ## if so (by more than 2 sigma): stop
##                 
##                     self.fitparams = deepcopy(old_fp)
##                     self.snr       = deepcopy(old_snr)
##                     self.sources   = deepcopy(old_src)
##                     stop = True
## 
##             else:
##                 self.fitparams[self.isfitted==1] = np.array(result[0],
##                                                                dtype='float32')
##                 self.chisq = result[1]
## 
## 
##             if stop:
##                 break
## 
##             select_sources = True
##             
##             self.snr  = (self.objects*self.mask).sum(axis=2).sum(axis=1)
##             self.snr *= self.fitparams[self.Nppsf + 4 + 2*self.ns:-1]
##             self.snr /= np.sqrt(1./((self.objects/self.sigma)**2).sum(axis=2).sum(axis=1))
## 
##             fluxes = self.fitparams[self.Nppsf + 4 + 2*self.ns:-1]
##             bad    = (fluxes<0).sum()
## 
##             if bad==0:
##                 if len(self.sources) > 0:
##                     iter += 1
##                     old_fp  = deepcopy(self.fitparams)
##                     old_snr = deepcopy(self.snr)
##                     old_src = deepcopy(self.sources)
## 
##                     obj_snrmin=self.sources[self.snr[self.sources].argmin()]
##                     
##                     ind_snrmin = self.Nppsf + 4 + 2*self.ns + obj_snrmin
##                     
##                     self.fitparams[ind_snrmin] = 0.
##                 else:
##                     break
## 
##  
##         self.bestfit_psf = self.fitparams[:self.Nppsf]
##         self.bestfit_pos = self.fitparams[self.Nppsf:self.Nppsf+4+2*self.ns]
##         self.bestfit_flx = self.fitparams[self.Nppsf+4+2*self.ns:-1]
##         self.bestfit_bck = self.fitparams[-1]
##         self.red_chisq   = self.chisq/(self.mask.sum() - self.Ndof)
## 
##         self.deleted = self.ns - self.sources.shape[0]
## 
##         if model or residuals:
##             self.model = self.fitparams[-1]*np.ones(self.size,
##                                                        dtype='float32')
##             for i in self.sources:
##                 j = self.Nppsf + 4 + 2*self.ns + i
##                 self.model += self.fitparams[j]*self.objects[i]
## 
##                 if residuals:
##                     self.residuals = np.array(self.data - self.model,
##                                                  dtype='float32')
## 
## 
## 
## ## !CHANGED
##     def make_psf(self):
## 
##         A, B, C, D = self.fitparams[self.Nppsf:self.Nppsf+4]
## 
##         for i in self.sources:
## 
##             j = 2*i + self.Nppsf + 4
## 
##             xpos =  A*self.fitparams[j] + B*self.fitparams[j+1] + C
##             ypos = -B*self.fitparams[j] + A*self.fitparams[j+1] + D
## 
## 
##             if self.psf == 'gauss':
##                 PSF_i = Gaussian(self.size, xpos, ypos, 0., self.fitparams[0],
##                                  self.fitparams[1], self.fitparams[2])
##         
##             elif self.psf == 'moffat':
##                 PSF_i = Moffat(self.size, xpos, ypos, 0., self.fitparams[0],
##                                self.fitparams[1], e=self.fitparams[2],
##                                theta=self.fitparams[3])
## 
##             self.objects[i] = PSF_i.array
## 
##             if self.lut != None:
##                 lut_size = (self.lut.shape[0]+1,self.lut.shape[1]+1)
##                 lut_arr  = np.zeros(lut_size, dtype='float32')
## 
##                 dx = xpos - int(xpos)
##                 dy = ypos - int(ypos)
## 
##                 if dx >= 0.5:
##                     if dy >= 0.5:
##                         lut_arr[:-1,:-1] = self.lut
##                     else:
##                         lut_arr[1:,:-1] = self.lut
##                 else:
##                     if dy >= 0.5:
##                         lut_arr[:-1,1:] = self.lut
##                     else:
##                         lut_arr[1:,1:] = self.lut
##                             
##                 tmp_lut = ndimage.zoom(lut_arr, 0.5, order=0)
##                 
##                 xcen = (tmp_lut.shape[1]>>1)
##                 ycen = (tmp_lut.shape[0]>>1)
## 
##                 lut = paste_area(tmp_lut, self.size, (xcen, ycen),
##                                  (xpos, ypos))
##                 
##                 self.objects[i] += lut
## 
## 
## 
## 
## ##     def _draw_psf(self):
## ## 
## ##         if self.psf == 'array':
## ##             psf_data = self.pdict['psf']
## ## 
## ##             x_c = psf_data.shape[1]>>1
## ##             y_c = psf_data.shape[0]>>1
## ##             for i in xrange(self.ns):
## ##                 j = 3*i + self.Nppsf
## ##                 dx = self.fitparams[j] - x_c
## ##                 dy = self.fitparams[j+1] - y_c
## ##                 
## ##                 psf_j = shift_image(psf_data, dx, dy)
## ## 
## ##                 max_x = min(psf_j.shape[1], self.size[1])
## ##                 max_y = min(psf_j.shape[0], self.size[0])
## ## 
## ##                 self.objects[i,:max_y,:max_x] = psf_j
## ## not updated yet!!!
## 
## 
## 
## ## !CHANGED
##     def _fit_res(self, pfit):
## 
##         pfit = np.asarray(pfit, dtype='float32')
## 
##         self.fitparams[self.isfitted==1] = pfit
## 
##         err = self.data - self.fitparams[-1]
## 
##         if self.psfvar:
##             self.make_psf()
## 
##         for i in self.sources:
##             j = self.Nppsf + 4 + 2*self.ns + i
##             err -= self.fitparams[j]*self.objects[i]
## 
##         err *= self.mask
## 
##         return ((err*err)/(self.sigma*self.sigma)).sum()
## 



