"""
The program contains `StarFit', the core-routine of PampelMuse. Its purpose
is to extract the spectra of the sources and to fit their positions and/or
the PSF of the observed data-cube. The routine is designed to handle both large
(e.g MUSE-like data) and small (e.g PMAS-like data) data-sets.

The analysis is started in the central slice of the provided datacube. After
the analysis of this slice is completed, the routine moves on to the next blue
resp. red slice in an alternating manner.

In case an the PSF and/or positions are fitted, their fit is preceded by an
initial fit of the source fluxes using the initial guesses for PSF and/or
positions. A final fit of the source fluxes and the estimation of the flux
uncertainties complete the analysis of each slice.

The whole fitting process can be controlled via a Qt-Gui that is automatically
launched at the beginning of the fit. It is also possible to pause the fit
after an arbitrary slice and continue later on.


v0.0.0.0, 2011/09/30, (c). S. Kamann
v0.0.0.1, 2011/10/13, changed how required PampelMuse-packages are imported
v0.1.0.0, 2011/10/27, modified creation and usage of fitting matrix
                      included GUI to control fitting process (experimental)
                      added option to continue previously stopped fit
v0.1.1.1, 2011/12/14, fixed bug that neither mask nor sigma cube were used.
                      included PSF fitting routine suited for small IFUs.
                      removed `verbose'-option, using `logging'-module instead.
v0.1.2.1, 2012/01/07, fixed bug that wrong images where used to update initial
                      guesses if a previously stopped analysis was continued.
v0.1.2.2, 2012/01/12, minor update about the information that is logged.
v0.1.3.2, 2012/01/12, corrected minor bug that wrong fits were used to update
                      initial guesses for first images.
v0.1.4.3, 2012/01/13, some bugfixing on previous change, updated logging,
                      changed usage of sky component (now dividing by number
                      of spatial pixels)
v0.1.4.4, 2012/01/16, changed how residuals are calculated in fitting routine
                      for positions/PSF since the old approach was unstable.
                      Some updates on logging.
v0.1.4.5, 2012/01/16, Modified logging of the flux changes to avoid 'divide by
                      zero' warning.
v0.1.4.6, 2012/01/16, added option to specify name of stored residual images. 
v0.1.4.7, 2012/01/17, modified code to handle new option of providing IFU
                      pixel coordinates for every source.
v0.2.0.0, 2012/01/30, modified code so that it can handle the updated
                      structure of the prm-files. Mainly updates on how
                      positions are handled. Also started adding possibility
                      to fit individual source position. This is not working
                      yet.
v0.2.0.1, 2012/02/01, not using the `fitRad'-variable anymore since in certain
                      situation is caused an unstable behavior of the routine.
                      The `psfRad'-variable is used for all fits instead. 
v0.2.0.2, 2012/02/13, added parameter `jumpStart' to allow starting a fit
                      remotely.
v0.3.0.0, 2012/02/24, major update on how PSF and source positions are used.
                      The now have to be provided as instances of dedicated
                      classes `Positions' and `PSF'.
v0.3.1.1, 2012/02/28, some minor updates and bugfixing wrt new usage of PSF
                      and positions.
v0.3.1.2, 2012/03/01, disabled option to store residuals. Instead, the skript
                      SUBTRES of PampelMuse can be used for this purpose.
                      Implemented usage of PampelMuse's `Spectra'-class.
v0.3.2.2, 2012/03/02, some severe bugfixing wrt to implementation of
                      `Spectra'-class.
v0.4.0.0, 2012/03/16, completed implementation of the dedicated PampelMuse-
                      classes `Sources` and `PSF`. The updated routine has
                      been successfully tested using different data sets.
v0.4.1.0, 2012/03/17, fixed major bug with regard to preparation of sparse
                      matrix, in some cases sources were put into the wrong
                      columns.
v0.4.1.1, 2012/07/19, added option to iteratively improve the solution also
                      for individual layers.
v0.4.1.2, 2012/07/19, changed the convergence criterion for the individual
                      layers to use the median instead of the max. flux change.
v0.4.1.3, 2012/07/23, modified counting of iterations so that they start from
                      one instead of zero.
v0.4.1.4, 2012/07/24, included handling of missing PyQt4-library.
v0.5.0.0, 2012/07/24, moved `QStarFit` into a separate file because otherwise
                      the class-dependencies were always checked (even if only
                      `StarFit` was loaded) and the code crashed if matplotlib
                      and/or PyQt4 were missing.
                      Also fixed a bug in the teration scheme on individual
                      layers: Earlier the absolute value of the median flux
                      changes was used instead of the median of the absolute
                      flux changes.
"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.5.0.0"

import logging, os, sys, time
import numpy as np, pyfits
from scipy.sparse.linalg import lsqr
from scipy import sparse, optimize
from multiprocessing import cpu_count
from multiprocessing import Pool



class StarFit(object):
    """
    `StarFit' is the core-routine of PampelMuse.  It uses a catalog of input
    sources, created e.g. using the dedicated `INITFIT`-script to analyse a
    complete datacube in an image-by-image manner. The fitting process itself
    is split into to parts: 
            (i) PSF & positions (if requested): A non-linear least squares fit
                    is performed for individual suited sources or the ensemble
                    of resolved sources. In case individual PSF-stars are used,
                    the results from the individual fits are combined
                    afterwards.
            (ii) Fluxes: A linear least-squares fit is performed for all the
                    resolved sources simultaneously.
    Note that the routine first analyzes a central image and than alternating
    proceedes to the red and blue. This is done because usually the highest
    signal is expected near the center of the observed wavelength range.
    """
    def __init__(self, dataFits, sigmaFits, maskFits=None, **kwargs):
        """
        Initialize an instance of the `StarFit'-class.
    
        Parameters
        ----------
        dataFits : a pyfits ImageHDU- or PrimaryHDU-instance
            The data to analyze. Note that instead of the pure data, the HDU 
            containing the data must be provided. This is necessary since the
            routine will only load the image that is currently analyzed into
            memory.
        sigmaFits : a pyfits ImageHDU- or PrimaryHDU-instance
            The HDU containing the sigma-values of the data.
        maskFits : a pyfits ImageHDU- or PrimaryHDU-instance, optional
            The HDU containing the bad-pixel mask (if any).
        parent : optional
            Is provided as input to the `FittingGui`-superclass.

        Returns
        -------
            The newly created instance
        """

        super(StarFit, self).__init__(**kwargs)

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
            logging.info("Bad-pixel mask is used.")
        else:
            self.hasMask = False
            logging.info("No bad-pixel mask is used.")

        


    def provideSources(self, sources, fullPosFit=False):
        """
        This method should be used to provide the instance with the necessary
        information about the sources in the field. This should be done by
        providing a valid instance of the dedicated `Sources`-clas that is
        part of the PampelMuse-package.
        A convenient way to create a valid `Sources`-instance is to use the  of
        `INITFIT'-script of PampelMuse.
        
        Parameters
        ----------
        sources : a pampelmuse.core.sources.Sources-instance
            To provide all necessary information asbout the sources, their
            spectra and positions, an instance of PampelMuse's Sources-class
            must be provided. Check the documentation of that routine for
            further information about how to initialize a valid instance.
        fullPosFit : boolean, optional
            A flag indicating whether individual source positions should be
            fitted. If `fullPosFit` is False (default), the routine will
            assume that a coordinate trasnsformation is used or polynomial
            fits to the individual source positions are available.
        """
        self.sources    = sources

        ##  if fullPosFit-Flag is True, memory for all positions on all slices
        ## must be provided right away. Otherwise, it is sufficient to allocate
        ## memory only for the positions of the currently analysed slice. 
        self.fullPosFit = fullPosFit
        if self.fullPosFit:
            self.sources.loadIFUCoords()


    
    def providePSF(self, psf, psfRad, osFactor=5, osRadius=2):
        """
        This method should be used to provide the instance with the necessary
        information about the PSF that is used in the fitting process.
        
        Parameters
        ----------
        psf : a pampelmuse.core.psf.PSF-instance
            The psf should be prepared using the dedicated PampelMuse `PSF`-
            class. For further documentation on the PSF, check the
            documentation of the class.
        psfRad : float
            The radius out to which the PSF is defined around a source
        osFactor : integer, optional
            The oversampling factor of the central PSF
        osRadius : float, optional
            The radius out to which the PSF is oversampled. 
        """
        self.psf    = psf
        self.psfRad = psfRad

        self.osFactor = osFactor
        self.osRadius = osRadius
        

    def __call__(self, fitRad=None, Ntrack=10, lastFitIndex=None,
                 asSmallIFU=False, ncpu=1, maxiter=20, clip=5., **kwargs):
        """
        Execute (i.e start or continue) the fitting process. For each image of
        the datacube, the following steps are performed:
            [1) Update the initial values]
             2) Initialize the PSF model for the analyzed image
             3) Create a PSF for every resolved source/component
            [4) Fit the PSF and/or  source positions]
             5) Fit the source fluxes
        Steps in square brackets are optional.
        
        Parameters
        ----------
        fitRad : float
            The radius within which the PSF is fitted to a source. DEPRECATED!
        Ntrack :  integer, optional
            In case the PSF and/or the source positions are fitted, the 
            routine will update the initial guesses of the fitted 
            parameters prior to fitting an image. It uses the results from
            the fits to the `Ntrack' neighboring images  for this purpose.
            If you do not want this behavior, set `Ntrack' to zero.
        lastFitIndex : integer or None, optional
            In case a previous fit was stopped and the results were stored,
            you can continue the fit by providing the last image that was
            fitted.
        asSmallIFU : boolean, optional
            Flag indicating whether the data will be handled as comming from
            a small or a large IFU. This parameter controls how the PSF and/
            or positions are fitted.
        maxiter : int, optional
            The maximum number of iterations per image. The default is 20.
        clip : float, optional
            All pixels whose values devite by more than `clip' times SIGMA are
            masked. SIGMA is the standard deviation of the residuals between
            model and data. WARNING: NOT IMPLEMENTED YET!
        jumpStart : boolean, optional
            In case the fit should start directly (i.e. without pressing
            `START'-button), enable this flag.
        """
        self.clip    = clip    ## efficiency of bad pixel masking
                               ## WARNING: Has no effect yet.
        if fitRad != None:
            logging.warning("The `fitRad'-variable is not used anymore.")
            logging.warning("Using `psfRad' instead to define fit regions.")
##             self.fitRad  = fitRad

        if lastFitIndex == None:
            self.i = 0    ## iterator over all images
        else:
            self.i = lastFitIndex

        self.Ntrack         = Ntrack

        self.maxiter        = int(maxiter)
        if self.psf.Fit==False and self.sources.Fit==False:
            self.maxiter = 1

        if self.maxiter > 1:
            logging.info("Maximum iterations per layer: {0}".format(\
                    self.maxiter))
        else:
            logging.info("No iteration on individual layers performed.")
            self.maxiter = 1

        self.lastFittedImas = []

        ## recover last processed images from previous fit (if any)
        j = self.i-2*self.Ntrack
        while j < self.i:
            if j >= 0:
                self.lastFittedImas.append(\
                    (self.specPix>>1) - 1 - int(j/2) + j*np.mod(j, 2)\
                        + np.mod(self.specPix, 2)
                    )
            j += 1

        self.asSmallIFU = asSmallIFU

        if ncpu == None: self.ncpu = cpu_count()
        else: self.ncpu = int(ncpu)
        logging.info("Starting analysis using {0} CPU(s).".format(self.ncpu))

        self.timeout()


    
    def timeout(self):

        while self.i < self.specPix:

            self.jj = (self.specPix>>1) - 1 - int(self.i/2) \
                + self.i*np.mod(self.i, 2) + np.mod(self.specPix, 2)
            ## jj is a counter that starts at the central spectral pixel
            ## and proceeds simultaneoulsy to the red and blue end.

            self.prefixString = "[{0:0{1}d}/{2}]".format(\
                self.i, int(np.log10(self.specPix))+1, self.specPix)

            logging.info("{0} Starting fit.".format(self.prefixString))
            self.iterFitImage()

            self.i += 1



    def iterFitImage(self):
        """
        The method is called for every slice of the datacube by the `timeout`-
        method. It is basically a wrapper that calls all the necessary
        routines required to analyze an individual slice from the datacube.
        It is also used to load the data section from the HDUs that is
        currently analysed, update the information in the GUI and calculate
        the uncertainties associated with the fluces once the analysis is
        completed.
        """

        ## update parameter initial guesses (if any):
        self.updatePars() 

        ## initialize PSF:
        self.PSF = self.psf(index=self.jj,  osradius=self.osRadius,
                            osfactor=self.osFactor)
 
        ## update coordinate data if necessary
        if not self.fullPosFit:
            self.sources.loadIFUCoords(index=self.jj)

        ## load data
        ## np.ndarray.flatten -> returns copy
        self.data  = self.dataFits.section[self.jj].flatten()
        self.sigma = self.sigmaFits.section[self.jj].flatten()
        if self.hasMask:
            self.mask = (self.maskFits.section[self.jj].flatten()==0)
            self.pixMap = self.mask.cumsum()-1 ## maps input coordinates to
            self.pixMap[self.mask==False] = -1 ## coordinates on subarray of
                                               ## unmasked pixels

            logging.debug("[@{0}] {1} masked pixels".format(\
                    self.jj, (self.mask==False).sum()))
            
        else:
            self.mask = np.ones(self.data.shape, dtype='bool')


        n_iter = 1 ##  Number of iterations
            
        while True: ## Start iteration
            
            if self.maxiter > 1:
                logging.info("[@{0}] Iteration {1} (of max. {2})".format(\
                        self.jj, n_iter, self.maxiter))

            ## create model matrix & update noise values
            self.Apsf = self.fillMatrix(radius=self.psfRad)
            self.sources.sigma[...,self.jj] = \
                1./np.sqrt(self.Apsf.multiply(self.Apsf).sum(axis=0))

##      Test whether not subtracting sources before fit makes a difference
##         bestFit = lsqr(self.Apsf, (self.data/self.sigma)[self.mask])
##         self.sources.data[:,self.jj] = bestFit[0]
            
            ## initial fit of fluxes
            bestFit = lsqr(self.Apsf, self.current)
            self.sources.data[:,self.jj] += bestFit[0]

            ## determine relative change in fluxes
            delta_flux = bestFit[0]/self.sources.data[:,self.jj]

            ## some logging
            logging.debug("[@{0}] Chi**2 of current fit: {1:.1f}".format(\
                    self.jj, bestFit[4]**2))
            logging.debug("[@{0}] Relative changes in fluxes: {1}".format(\
                    self.jj, delta_flux))

            if np.median(abs(delta_flux)) < 1e-3:
                break

            ## fit PSF/positions if requested            
            if self.psf.Fit or self.sources.Fit:

                ## PSF/positions fit
                if self.asSmallIFU: self.fitPSF_smallIFU()
                else: self.fitPSF_largeIFU()

##             else:
##                 break

            n_iter += 1
            
            if n_iter > self.maxiter:
                if self.maxiter > 1:
                    logging.warning("[@{0}] Max. number of iterations reached."\
                                        .format(self.jj))
                    break
                 
           
        ## update inventory of last fits
        self.lastFittedImas.append(self.jj)
        if len(self.lastFittedImas)>2*self.Ntrack:
            self.lastFittedImas.remove(self.lastFittedImas[0])
 


    @property
    def model(self):
        """Create the current best-fit model"""
        return self.Apsf\
            *sparse.dia_matrix((self.sources.data[:,self.jj],[0]),
                               shape=(self.sources.ncmp, self.sources.ncmp))



    @property
    def current(self):
        """Determine the current residuals"""
        ## self.model.sum(axis=1) returns numpy.matrixlib.defmatrix-object
        ## `getA1' returns a flattened numpy.ndarray

        return (self.data/self.sigma)[self.mask]-self.model.sum(axis=1).getA1()



    def fillMatrix(self, radius, calcUnres=True, doLog=True):
        """
        Define the source matrix. The source matrix is a sparse matrix that
        contains the spatial profiles of all resolved components, one per
        column.
        The method initially creates empty arrays to hold the source (PSF)
        coefficient, their (1dim.) coordinates on the data array (the row
        indices in the PSF coefficient matrix) and the source indices (the
        column indices in the PSF coeffficient matrix). It then loops over
        all components in the `Sources`-instance, calculates the PSF for the
        component and adds the data to the three arrays. FInally, the three
        arrays are combined into a single sparse matrix.
        
        Parameters
        ----------
        radius : float
            The radius out to which the profile is calculated around a point
            source.
        calcUnres : boolean, optional
            If False, the matrix column containing the unresolved stellar
            component (if included) is not computed, but left empty.
        doLog : boolean, optional
            If True, the routine method will print a logging.ERROR-message
            every time a source whose PSF is completely outside the FoV is
            encountered.
        
        Returns
        -------
        a scipy.sparse.coo_matrix-instance
            This is  the PSF coefficient matrix that is used in the linear
            least-squares fit to determine the source fluxes.
        """

        ## Initialize parallelization (if requested)
        if self.ncpu > 1:

            pool    = Pool(self.ncpu)
            threads = []

            for i in xrange(self.sources.ncmp):
                if i == self.sources.unresRow and not calcUnres: continue
                if i == self.sources.skyRow: continue

                for j in xrange(len(self.sources.indexmap[i])):
                    k = (self.sources.ifuSelect==self.sources.indexmap[i][j])\
                        .argmax()

                    x_c=self.sources.xvalues[k, self.jj*self.fullPosFit]
                    y_c=self.sources.yvalues[k, self.jj*self.fullPosFit]
                    mag=self.sources.weights[i][j]

                    threads.append(pool.apply_async(\
                            self.PSF,
                            args=(self.spatShape, x_c, y_c, mag, radius, 1, 0)))

            pool.close()
            pool.join()

        coeffs   = np.array([], dtype='float32')  ## PSF coefficents
        indices0 = np.array([], dtype='int32')    ## row indices
        indices1 = np.array([], dtype='int32')    ## column indices

        n = 0

        for i in xrange(self.sources.ncmp):
            if i==self.sources.unresRow and not calcUnres:
                ## if unresolved component should  not be recalculated,
                ## add existing one to data.
                nValues = self.unresCoeffs.size

                coeffs.resize(coeffs.size+nValues)
                indices0.resize(indices0.size+nValues)
                indices1.resize(indices1.size+nValues)

                coeffs[-nValues:]   = self.unresCoeffs
                indices0[-nValues:] = self.unresIndices
                indices1[-nValues:] = i*np.ones((nValues,), dtype='int32')
                continue

            if i == self.sources.skyRow: ## add constant sky component
                nValues = self.spatPix

                coeffs.resize(coeffs.size+nValues)
                indices0.resize(indices0.size+nValues)
                indices1.resize(indices1.size+nValues)

                coeffs[-nValues:]   = \
                    np.ones(nValues, dtype=np.float32)/(nValues*self.sigma)
                indices0[-nValues:] = np.arange(nValues, dtype=np.int32)
                indices1[-nValues:] = i*np.ones(nValues, dtype=np.int32)
                continue

            nValuesCmp = 0                
            for j in xrange(len(self.sources.indexmap[i])):

                if self.ncpu > 1:
                    psf_ij = threads[n].get()
                    n += 1
                else:
                    k = (self.sources.ifuSelect==self.sources.indexmap[i][j])\
                        .argmax()
                    x_c=self.sources.xvalues[k, self.jj*self.fullPosFit]
                    y_c=self.sources.yvalues[k, self.jj*self.fullPosFit]
                    mag=self.sources.weights[i][j]

                    psf_ij = self.PSF(\
                          size=self.spatShape, x_c=x_c, y_c=y_c, mag=mag,
                          return_sparse=True, as_sparse_matrix=False)

                nValues    = psf_ij[0].size
                newCoeffs  = psf_ij[0]
                newIndices = psf_ij[1][1] + self.spatShape[1]*psf_ij[1][0]

                if nValues > 0: ## resize arrays and add new data
                    coeffs.resize(coeffs.size+nValues)
                    indices0.resize(indices0.size+nValues)
                    indices1.resize(indices1.size+nValues)

                    ## divide by flux uncertainties before adding coefficents:
                    coeffs[-nValues:]   = newCoeffs/self.sigma[newIndices]

                    indices0[-nValues:] = newIndices
                    indices1[-nValues:] = i*np.ones((nValues,), dtype='int32')

                    nValuesCmp += nValues

            if nValuesCmp == 0 and doLog:
                print self.dataFits.name, self.dataFits.fileinfo()
                logging.error(\
                    "[@{0}] Component #{1} entirely outside FoV.".format(\
                        self.jj, self.sources.ID[self.sources.indexmap[i][0]]))
                    
        if self.sources.hasUnres and calcUnres:
            self.unresIndices = indices0[indices1==self.sources.unresRow]
            self.unresCoeffs  = coeffs[indices1==self.sources.unresRow]

        if self.hasMask: ## implement mask if it has been provided
            mappedIndices0 = self.pixMap[indices0]
            used           = (mappedIndices0 > -1)

            return sparse.coo_matrix(\
                (coeffs[used], (mappedIndices0[used], indices1[used])),
                shape=(self.mask.sum(), self.sources.ncmp)\
                    ).tocsc()

        else:
            return sparse.coo_matrix(\
                (coeffs, (indices0, indices1)),
                shape=(self.spatPix, self.sources.ncmp)\
                    ).tocsc()



    def fitPSF_largeIFU(self):
        """
        Perform the PSF fit. For every PSF-source, a non-linear least-squares
        fit is performed. Depending on the respective fit-flags, either the
        PSF-parameters and/or source coordinates are fitted.
        The results of the individual fits are then used to update the PSF and/
        or the coordinates of the parameter transformation.
        """
        nPars = self.psf.npars+3

        allPars = np.zeros((nPars, self.sources.psfIndices.size),
                           dtype='float32')

        fitFlags       = np.zeros((nPars,), dtype='bool')
        fitFlags[:2]   = self.sources.Fit
        fitFlags[2:-1] = self.psf.free&(self.psf.poly==False)
        fitFlags[-1]   = True

        ## Initialize parallelization (if requested)
        if self.ncpu > 1:
            pool    = Pool(self.ncpu)
            threads = []

        ## Loop over all PSF sources; add source to residuals, fit resulting
        ## array and subtract source again.
        j    = 0
        kPSF = np.nonzero(\
            self.sources.ifuSelect[:,np.newaxis]==self.sources.psfIndices)[0]

        ## Fill parameter array for fit
        for i in self.sources.psfIndices:
            k = kPSF[j]  ## Index of PSF source in IFU coordinate arrays

            allPars[0,j]    = self.sources.xvalues[k, self.jj*self.fullPosFit]
            allPars[1,j]    = self.sources.yvalues[k, self.jj*self.fullPosFit]
            allPars[2:-1,j] = self.psf.values[:,self.jj]
            allPars[-1,j]   = self.sources.data[self.sources.specrow[i],self.jj]

            if self.ncpu > 1: threads.append(pool.apply_async(\
                    self.multiWrap,
                    args=(np.copy(allPars[:,j]), fitFlags)))
            j += 1

        if self.ncpu > 1: ## only for parallelization
            pool.close()
            pool.join()

        ## Fit sources
        j = 0
        for i in self.sources.psfIndices:
            if self.ncpu > 1: allPars[fitFlags,j] = threads[j].get()
            else: allPars[fitFlags, j] = self.multiWrap(allPars[:,j], fitFlags)

            j += 1            


        if self.sources.Fit: ## update coordinates/coordinate transformation

            if self.fullPosFit:
                for k in kPSF:
                    self.sources.xvalues[k, self.jj] = allPars[0,k]
                    self.sources.yvalues[k, self.jj] = allPars[1,k]

            else:
                self.sources.ctValues[...,self.jj] = self.sources.quickFit(\
                    [self.sources.x[self.sources.psfIndices],
                     self.sources.y[self.sources.psfIndices]],
                    allPars[:2],
                    outSigma=None)[0]


        if self.psf.Fit: ## update PSF
            self.psf.values[...,self.jj] = allPars[2:-1,:].mean(axis=1)
            self.PSF = self.psf(index=self.jj, osradius=self.osRadius,
                                osfactor=self.osFactor)



    def multiWrap(self, allPars, fitFlags):

        ## get current PSF estimate of source and add it to residuals
        psf_i = self.PSF(size=self.spatShape, x_c=allPars[0], y_c=allPars[1],
                         mag=0., maxrad=self.psfRad, return_sparse=True,
                         as_sparse_matrix=False)
        psfIndices = psf_i[1][1] + self.spatShape[1]*psf_i[1][0]
        psfCoeffs  = psf_i[0]/self.sigma[psfIndices]
        _current   = np.copy(self.current)
        if self.hasMask:
            mappedIndices = self.pixMap[psfIndices]
            used          = (mappedIndices > -1)
            _current[mappedIndices[used]] += psfCoeffs[used]*allPars[-1]
        else:
            _current[psfIndices] += psfCoeffs*allPars[-1]

        ## fit PSF 
        self.BAD = False ## a flag indicating if a PSF parameter was
                         ## fitted with an invalid value (e.g. beta<=1)

        bestFit_j = optimize.fmin(\
            func=self.fitOnePSF,
            x0=allPars[fitFlags],
            args=(allPars[fitFlags==False], fitFlags, _current),
            disp=False)

        if not self.BAD: ##  only update data if fit was valid
            return bestFit_j
        else:
            return allPars[fitFlags]



    def fitOnePSF(self, x0, fixedPars, fitFlags, data):
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

        if self.psf.psftype == "moffat":
            if psfPars[2]<=1 or psfPars[2]>20:
                logging.error("Moffat beta not in valid range (1,20].")
                self.BAD = True
                return 1e12

        self.psf.values[...,self.jj] = psfPars[2:-1]
        thePSF = self.psf(index=self.jj, osradius=self.osRadius,
                          osfactor=self.osFactor)

        psfArray = thePSF(size=self.spatShape, x_c=psfPars[0],
                          y_c=psfPars[1], mag=0., maxrad=self.psfRad,
                          return_sparse=True, as_sparse_matrix=False)

        psfIndices = psfArray[1][1] + self.spatShape[1]*psfArray[1][0]
        psfCoeffs  = psfArray[0]/self.sigma[psfIndices]
        
        if self.hasMask:
            mappedIndices = self.pixMap[psfIndices]
            used          = (mappedIndices > -1)

            err = data[mappedIndices[used]]-psfCoeffs[used]*psfPars[-1]
        else:
            err = data[psfIndices]-psfCoeffs*psfPars[-1]

        return (err*err).sum()



    def fitPSF_smallIFU(self):
        """
        Perform the positions/PSF fit on the whole array. A non-linear least-
        squares fit is performed using all resolved sources. This fitting
        procedure is recommended only for IFUs with a small number of spatial
        pixels.
        The results of the fit is used to update the PSF and/or the
        parameters of the coordinate transformation.
        """

        nPars = self.sources.npars+self.psf.npars

        fitFlags = np.zeros((nPars,), dtype='bool')
        fitPars  = np.zeros((nPars,), dtype='float64')

        fitPars[:self.sources.npars] = self.sources.ctValues[...,self.jj]
        fitPars[self.sources.npars:] = self.psf.values[...,self.jj]

        NCPU = self.ncpu
        self.ncpu = 1

        ## Fit position parameters
        if self.sources.Fit:
            
            fitFlags[:self.sources.npars] \
                = self.sources.ctFree&(self.sources.ctPoly==False)

##             self.sBar.showMessage(\
##                 "{0} Fitting positions".format(self.prefixString))

            fitPars[fitFlags] =  optimize.fmin(\
                func=self.fitMultiPSF,
                x0=fitPars[fitFlags],
                args=(fitPars[fitFlags==False], fitFlags),
                disp=False)
            
            self.sources.ctValues[...,self.jj] = fitPars[:self.sources.npars]

        ## Fit PSF parameters
        if self.psf.Fit:

            fitFlags[:self.sources.npars] = False
            fitFlags[self.sources.npars:] \
                = self.psf.free&(self.psf.poly==False)
            
##             self.sBar.showMessage("{0} Fitting PSF".format(self.prefixString))

            fitPars[fitFlags] = optimize.fmin(\
                func=self.fitMultiPSF,
                x0=fitPars[fitFlags],
                args=(fitPars[fitFlags==False], fitFlags),
                disp=False)

            logging.debug("[@{0}] Fitted PSF model: {1}".format(\
                    self.jj, fitPars[self.sources.npars:]))

            self.psf.values[..., self.jj] = fitPars[self.sources.npars:]
            self.PSF = self.psf(index=self.jj, osradius=self.osRadius,
                                osfactor=self.osFactor)

        self.ncpu = NCPU


    def fitMultiPSF(self, x0, fixedPars, fitFlags):
        """
        The optimization routine that is passed to the `fmin'-routine of the
        `scipy.optimize'-package. During each iteration, a PSF array is
        prepared for every resolved source, the PSFs are scaled and subtracted
        from the data. The chi^2-value is returned.
        
        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted in the
            following order:
                coordTransPar0, .., coordTransPar_i, psfPar_0, .., psfPar_j
        fixedPars : array_like
            The values of the parameters that are not fitted in the following
            order same order as `x0`. Each parameter of the coordinate transf.
            or the PSF must occur either in `x0` OR in `fixedPars`.
        fitFlags : array_like
            A boolean flag for each parameter needed to create the PSF
            indicating whether it is fitted. The number of True-values must be
            equal to the size of `x0` and the number of False-values must be
            equal to the size of `fixedPars`.
        
        Returns
        -------
        chi2 : float
            The chi^2 value of the fit.
        """
        allPars = np.zeros(fitFlags.shape, dtype='float64')

        allPars[fitFlags] = x0
        allPars[fitFlags==False] = fixedPars

        if fitFlags[:self.sources.npars].any(): ## recalc pixel positions
            self.sources.ctValues[...,self.jj] = allPars[:self.sources.npars]
            self.sources.loadIFUCoords(index=self.jj)

        if fitFlags[self.sources.npars:].any(): ## recalc PSF
            self.psf.values[...,self.jj] = allPars[self.sources.npars:]
            self.PSF = self.psf(index=self.jj, osradius=self.osRadius,
                                osfactor=self.osFactor)

        Aprm = self.fillMatrix(radius=self.psfRad, calcUnres=False, doLog=False)

        model =  Aprm*sparse.dia_matrix((self.sources.data[...,self.jj],[0]),
                                        shape=(self.sources.ncmp,
                                               self.sources.ncmp))

        err =  (self.data/self.sigma)[self.mask]-model.sum(axis=1).getA1()


## !!! Old residual calculation, unstable(?)
##         Aprm = self.fillMatrix(radius=self.fitRad, calcUnres=False,
##                                doLog=False)
## 
##         Y, X  = Aprm.nonzero()
## 
##         Aoff  = Aprm - sparse.csr_matrix((self.Apsf[Y, X].getA1(), (Y, X)),
##                                          shape=Aprm.shape)  
##         model = Aoff*sparse.dia_matrix((self.spectra[:,self.jj],[0]),
##                                        shape=(Aprm.shape[1], Aprm.shape[1]))
## 
##         err   = self.current - model.sum(axis=1).getA1()

        return (err*err).sum()

    

    def updatePars(self):
        """
        Method is used to update the initial guesses of the parameters of
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
                    len(self.lastFittedImas)-2, -1, -2)]
        logging.debug("[@{0}] Previous fits in focus: {1}".format(\
                self.jj, LASTFITS))

        if self.sources.Fit:
            
            if not self.fullPosFit:  ## update coordinate transformation
                self.sources.ctValues[...,self.jj] \
                    = self.sources.ctData[0,:][...,LASTFITS].mean(axis=-1)
            else: ## update coordinates
                self.sources.xvalues[...,self.jj] \
                    = self.sources.xdata[0,:][...,LASTFITS].mean(axis=-1)
                self.sources.yvalues[...,self.jj] \
                    = self.sources.ydata[0,:][...,LASTFITS].mean(axis=-1)

        if self.psf.Fit: ## update PSF
            self.psf.values[...,self.jj] \
                = self.psf.data[0,:][...,LASTFITS].mean(axis=-1)

            logging.debug("[@{0}] Initial PSF model: {1}".format(\
                    self.jj, self.psf.values[...,self.jj]))



