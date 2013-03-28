"""
The program contains `StarFit', the core-routine of PampelMuse. Its purpose
is to extract the spectra of the sources and to fit their positions and/or
the PSF of the observed datacube. The routine is designed to handle both large
(e.g MUSE-like) and small (e.g PMAS-like) data sets.

The analysis is started in the central layer of the provided datacube. After
its analysis is completed, the routine moves on to the next blue
or red layer, respectively, in an alternating manner.

In case an the PSF and/or positions are fitted, their fit is preceded by an
initial fit of the source fluxes using the initial guesses for PSF and/or
positions. A final fit of the source fluxes and the estimation of the flux
uncertainties complete the analysis of each layer.


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
v0.5.0.1, 2012/09/07, included break-statement in iterative fit loop to leave
                      loop after initial flux fit if both PSF and positions
                      remain fixed.
v0.5.1.1, 2012/10/09, bugfix: code still used iteration even if `maxiter`=1
v0.6.0.0, 2013/03/20, major updates to the code so that it is compatible with
                      the new way of handling the PSF during the analysis via
                      the 'pampelmuse.core.new_profiles'-class.
v0.6.1.0, 2013/03/22, some bugfixing wrt. handling of unresolved component in
                      the PSF/positions fit.
v0.6.1.1, 2013/03/26, included an option to fit only a subset of all layers in
                      method __call__
v0.6.2.1, 2013/03/27, fixed a bug when calculating the Jacobi matrix (the
                      values were not divided by the uncertainties); also
                      modified usage of option to fit subset of all layers of
                      a cube.
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.6.2.1"

import logging, os, sys, time
import numpy as np, pyfits
from scipy.sparse.linalg import lsqr
from scipy import sparse, optimize
from multiprocessing import cpu_count
from multiprocessing import Pool


## required PampelMuse-packages
from pampelmuse.core.new_profiles import Gaussian, Moffat


class StarFit(object):
    """
    `StarFit' is the core-routine of PampelMuse. It uses a catalog of input
    sources, created e.g. using the dedicated `INITFIT`-script, to analyse a
    complete datacube in a layer-by-layer manner. The fitting process itself
    is split into three parts: 
            (i) Fluxes: A linear least-squares fit is performed for all the
                    resolved sources simultaneously. This is done using a
                    sparse-matrix representation of the individual sources.
            (ii) PSF: The PSF is modelled currently as an analytical profile,
                    usually assuming a Moffat-shaped PSF. The individual
                    parameters that govern the shape of the profile can be
                    optimized during the fit.
            (iii) Source positions: Usually, the source positions are derived
                    from a set of reference coordinates via a coordinate
                    formation that is defined by several parameters. These
                    parameters can be optimized during the fit.
                    The routine also supports direct fitting of source
                    coordinates. However, this  is only recommended for 
                    individual stars instead of crowded stellar fields.
    Steps (ii) and (iii) are performed via a non-linear least-squares fit based
    on a Levenberg-Marquard approach.

    Note that the routine first analyzes a central image and than alternating
    proceedes to the red and blue. This is done because usually the highest
    signal is expected near the center of the observed wavelength range.
    """
    def __init__(self, dataFits, sigmaFits, sources, psf_attributes,
                 maskFits=None, maxrad=np.inf, osfactor=1, osradius=0,
                 **kwargs):
## fullPosFit=False, osFactor=5, osRadius=2
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
        sources : a pampelmuse.core.sources.Sources-instance
            The required information (positions, magnitudes,...) about the
            sources that should be included in the optimization.
        psf :  a pampelmuse.core.psf.psf_attributes-instance
            The required information about the PSF to be used in the opti-
            mization, i.e., type of PSF, initial guesses of shape parameters,
            ...
        maskFits : a pyfits ImageHDU- or PrimaryHDU-instance, optional
            The HDU containing the bad-pixel mask (if any).
        maxrad : float, optional
            The radius out to which each PSF will be defined, measured in
            pixels and from the centre of the profile.
        osfactor : The improve the accuracy of the PSF modelling close to the
            centre of a profile, the central pixels can be supersampled by a
            factor 'osfactor' along x and y, i.e. each pixel is split into
            'osfactor'^2 subpixels. Default is no supersampling.
        osradius : All pixels with distances <'osradius' to the centre of a
            PSF profile are supersampled by a factor 'osfactor'. Default is
            no supersampling.

        Returns
        -------
            The newly created instance
        """

        super(StarFit, self).__init__(**kwargs)

        self.dataFits  = dataFits
        self.sigmaFits = sigmaFits
        self.maskFits  = maskFits

        self.sources = sources
        self.psf_attributes = psf_attributes

        ## Needs to be updated!
        self.fullPosFit = False
        
        ## Supersampling of central pixels
        self.osfactor = osfactor
        self.osradius = osradius

        ## Get basic PSF model: Moffat or Gaussian
        if self.psf_attributes.psftype == "gauss":
            self.PSF = Gaussian
        elif self.psf_attributes.psftype == "moffat":
            self.PSF = Moffat
        else:
            logging.critical("Invalid PSF type: '{0}'".format(\
                    self.psf_attributes.psftype))
            raise IOError, "PSF type not understood."


        ## The dimensionality of the data (2D or 3D)
        self.ndim = self.dataFits.header["NAXIS"]
        
        if self.sources.instrument == None:
            assert self.ndim == 3, \
                "If instrument is not known, data must be provided as cube."

        if self.ndim == 3:
            self.spatShape = (self.dataFits.header['NAXIS2'],
                              self.dataFits.header['NAXIS1'])
            self.specPix   = self.dataFits.header['NAXIS3']
        else:
            self.spatShape = (self.sources.instrument.nspaxel,)
            self.specPix   = self.dataFits.header['NAXIS{0}'.format(\
                    self.sources.instrument.spectral_axis)]


        ## if Instrument instance is provided, feed PSF with x- & and y-
        ## positions of spaxels, otherwise assume rectangular field of
        ## view.
        if self.sources.instrument.name != None:
            self.PSF.set_transform(self.sources.instrument.transform)
        else:
            self.PSF.set_size(self.spatShape)

        ## check if mask available
        if self.maskFits != None:
            self.hasMask = True
            logging.info("Bad-pixel mask is used.")
        else:
            self.hasMask = False
            logging.info("No bad-pixel mask is used.")


        ## initialize PSF-class for every source included
        for i in xrange(self.sources.nsrc):
            self.PSF(Xc=self.sources.x[i], Yc=self.sources.y[i],
                mag=sources.weights[i], maxrad=maxrad, osfactor=osfactor,
                     osradius=osradius)

            
        ##  if fullPosFit-Flag is True, memory for all positions on all slices
        ## must be provided right away. Otherwise, it is sufficient to allocate
        ## memory only for the positions of the currently analysed slice. 

        

    def __call__(self, layerRange=(0,-1), Ntrack=10, lastFitIndex=None,
                 asSmallIFU=False, ncpu=1, maxiter=20, clip=5., **kwargs):
        """
        Execute the fitting process. If a previous analysis of the same data
        was interrupted, it might be resumed. For each layer of the provided
        datacube, the following steps are performed:
        [1) Update the initial guesses or values of the parameters of the
            PSF or the coordinate transformation.]
         2) Initialize the PSF model.
         3) Calculate a PSF profile for every source included
        [4) Fit the parameters of the coordinate transformation or the
            indiviudal source coordinates.]
        [5) Fit the shape parameters of the PSF]
         6) Fit the source fluxes
        Steps in square brackets are optional and only executed if requested.
        
        Parameters
        ----------
        layerRange : tuple
            The indices of the first and last layer of the spectral range to
            be analysed. By default, the whole cube is analysed.
        Ntrack :  integer, optional
            In case the PSF and/or the source positions are fitted, the 
            routine will update the initial guesses of the parameters that
            should be fitted prior to fitting a layer. To this aim, it uses
            the results from the fits to the 'Ntrack' neighboring layers. If
            you do not want this, set `Ntrack' to zero. Default is 10.
        lastFitIndex : integer or None, optional
            In case a previous fit was stopped and the results were stored,
            you can continue the fit by providing the index of the last layer
            that was fitted. Default is None, meaning to start from scratch.
        asSmallIFU : boolean, optional
            Flag indicating whether the data will be handled as comming from
            a small or a large IFU. This parameter controls how the PSF and/
            or the positions are fitted. Note that if individual source
            coordinates should be determined, 'asSmallIFU' will automatically
            be set to False. Default is False.
        maxiter : int, optional
            The maximum number of iterations per image. The default is 20.
        clip : float, optional
            All pixels whose values devite by more than `clip' times SIGMA are
            masked. SIGMA is the standard deviation of the residuals between
            model and data. WARNING: NOT IMPLEMENTED YET!
        """
        self.clip    = clip    ## efficiency of bad pixel masking
                               ## WARNING: Has no effect yet.

        self.start, self.stop = layerRange
        self.Nlayer = self.stop - self.start + 1
        
        logging.info("Requested range to be analysed: {0}-{1}".format(\
                self.start,self.stop))

        ## check where to start
        if lastFitIndex == None:
            self.i = 0    ## iterator over all layers
        else:
            self.i = lastFitIndex

        self.Ntrack = Ntrack

        ## Number of iterations per layer. Note that no iteration is needed
        ## if only fluxes are fitted
        self.maxiter        = int(maxiter)
        if self.psf_attributes.Fit==False and self.sources.Fit==False:
            self.maxiter = 1

        if self.maxiter > 1:
            logging.info("Maximum iterations per layer: {0}".format(\
                    self.maxiter))
        else:
            logging.info("No iteration on individual layers performed.")
            self.maxiter = 1

        ## list that will contain the indices of the last 'Ntrack' images that
        ## were analysed.
        self.lastFittedImas = []

        ## recover last processed layers in previous fit (if any)
        j = self.i-2*self.Ntrack
        while j < self.i:
            if j >= 0:
                self.lastFittedImas.append(\
                    (self.Nlayer>>1) - 1 - int(j/2) \
                        + j*np.mod(j, 2) + np.mod(self.Nlayer, 2)
                    )
            j += 1

        self.asSmallIFU = asSmallIFU

        if ncpu == None: self.ncpu = cpu_count()
        else: self.ncpu = int(ncpu)
        logging.info("Starting analysis using {0} CPU(s).".format(self.ncpu))

        ## execute fit
        self.timeout()


    
    def timeout(self):
        """
        The caller for the actual fit routine. The idea of this routine is to
        facilitate the adaption of the fitting process to a graphic-supported
        analysis using PyQt. Check the documentation of the QStarFit-class for
        further details.
        """

        ## loop over all layers in a datacube
        while self.i < self.Nlayer:

            ## jj is a counter that starts at the central spectral pixel
            ## and proceeds simultaneoulsy to the red and blue end.
            self.jj = (self.Nlayer>>1) - 1 - int(self.i/2) \
                + self.i*np.mod(self.i, 2) + np.mod(self.Nlayer, 2)

            ## A common prefix used for all logging messages related to the
            ## analysis of the current layer.
            self.prefixString = "[{0:0{1}d}/{2}]".format(\
                self.i, int(np.log10(self.Nlayer))+1, self.Nlayer)

            ## call fitting routine
            logging.info("{0} Starting fit.".format(self.prefixString))
            self.iterFitImage()

            self.i += 1



    def iterFitImage(self):
        """
        The method is called for every slice of the datacube from the
        'timeout'-method. It performs the following steps.
        (i) Initilization of the PSF;
        (ii) load the data and sigma arrays and apply the mask if one is
             provided;
        (iii) run the iterative scheme used to fit the fluxes, PSF parameters
              and coordinate transformation;
        (iv) estimate the uncertainty of the individual flux of each
             component.
        """

        ## update parameter initial guesses (if any):
        self.updatePars() 


        ## update PSF instances
        i = 0 
        for prm in self.psf_attributes.parameters:
            getattr(self.PSF, "set_{0}".format(prm))(\
                self.psf_attributes.values[i, self.jj])
            i += 1
        
        i = 0
        for prm in self.sources.parameters:
            getattr(self.PSF, "set_{0}".format(prm))(\
                self.sources.ctValues[i, self.jj])
            i += 1


        self.kk = self.jj + self.start ## The layer of the cube to be loaded

        ## load mask if present, otherwise create zero mask
        if self.hasMask:
            if self.ndim == 3:
                self.mask = self.maskFits.section[self.kk].flatten()
            elif self.sources.instrument.spatial_axis==1:
                self.mask = self.sources.instrument.get_science(\
                    self.maskFits.section[:,self.kk])
            else:
                self.mask = self.sources.instrument.get_science(\
                    self.maskFits.section[self.kk])            
        else:
            self.mask = np.zeros(self.spatShape, dtype='bool')
            
        logging.debug("[@{0}] {1} masked pixels".format(\
                self.kk, (self.mask==True).sum()))


        ## load data, flatten array if necessary and apply mask; if data is
        ## 2dimensional, also extract science fibers.
        ## Note: np.ndarray.flatten -> returns copy
        if self.ndim == 3:
            self.data  = self.dataFits.section[self.kk].flatten()[self.mask==0]
            self.sigma = self.sigmaFits.section[self.kk].flatten()[self.mask==0]
        elif self.sources.instrument.spectral_axis==1:
            self.data  = self.sources.instrument.get_science(\
                self.dataFits.section[:,self.kk])[self.mask==0]
            self.sigma = self.sources.instrument.get_science(\
                self.sigmaFits.section[:,self.kk])[self.mask==0]
        else:
            self.data  = self.sources.instrument.get_science(\
                self.dataFits.section[self.kk])[self.mask==0]
            self.sigma = self.sources.instrument.get_science(\
                self.sigmaFits.section[self.kk])[self.mask==0]

            
        ## update transformation in PSF class to include mask
        self.PSF.set_transform(self.sources.instrument.transform[self.mask==0])

        n_iter = 1 ##  Number of iterations
            
        while True: ## Start iteration
            
            if self.maxiter > 1:
                logging.info("[@{0}] Iteration {1} (of max. {2})".format(\
                        self.kk, n_iter, self.maxiter))

            ## create model matrix & update noise values
            self.Apsf = self.fillMatrix()
            self.sources.sigma[...,self.jj] = \
                1./np.sqrt(self.Apsf.multiply(self.Apsf).sum(axis=0))

            ## initial fit of fluxes
            bestFit = lsqr(self.Apsf, self.current)
            self.sources.data[:,self.jj] += bestFit[0]

            ## determine relative change in fluxes
            delta_flux = bestFit[0]/self.sources.data[:,self.jj]

            ## some logging
            logging.debug("[@{0}] Chi**2 of current fit: {1:.1f}".format(\
                    self.kk, bestFit[4]**2))
            logging.debug("[@{0}] Relative changes in fluxes: {1}".format(\
                    self.kk, delta_flux))

            ## convergence criterion: median relative change in fluxes <1*10^-3
            if np.median(abs(delta_flux)) < 1e-3:
                break

            ## fit PSF/positions if requested            
            if self.psf_attributes.Fit or self.sources.Fit:

                ## PSF/positions fit
                if self.asSmallIFU: self.fitPSF_smallIFU()
                else: self.fitPSF_largeIFU()

            else:
                break

            n_iter += 1
            
            if n_iter > self.maxiter:
                if self.maxiter > 1:
                    logging.warning("[@{0}] Max. number of iterations reached."\
                                        .format(self.kk))
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
        
        return (self.data/self.sigma)-self.model.sum(axis=1).getA1()


    
    def fillMatrix(self, calcUnres=True, doLog=True):
        """
        Define the source matrix. The source matrix is a sparse matrix that
        contains the spatial profiles of all resolved components, one per
        column.
        The method initially creates three empty arrays to hold
        (a) the source (PSF) coefficients, divided by the pixel uncertainties
        (b) the (1dim.) coordinates of the coefficients on the data array
            (they are the row indices of the final PSF coefficient matrix)
        (c) the component to which each source belongs
            (they are the column indices of the final PSF coeffficient matrix)
        It then loops over all sources in the 'Sources'-instance, obtains
        their PSF and adds it to the respective column of the array. After-
        wards, the arrays are combined into a single sparse matrix.
        
        Parameters
        ----------
        calcUnres : boolean, optional
            If False, the matrix column containing the unresolved stellar
            component (if included) is not recomputed, but taken from a
            previous calculation.
        doLog : boolean, optional
            If True, the routine method will print a logging.ERROR-message
            every time a source whose PSF is completely outside the FoV is
            encountered.
        
        Returns
        -------
        a scipy.sparse.coo_matrix-instance
            This is the PSF coefficient matrix that is used in the linear
            least-squares fit to determine the source fluxes.
        """

        ## Initialize parallelization (if requested)
        if self.ncpu > 1:

            pool    = Pool(self.ncpu)
            threads = []
            
            i = -1
            for psf in self.PSF:
                i += 1
                if self.sources.specrow[i] == self.sources.unresRow \
                        and not calcUnres: continue
                if self.sources.specrow[i] == self.sources.skyRow: continue

                threads.append(pool.apply_async(psf.get_flux))

            pool.close()
            pool.join()

        coeffs   = np.array([], dtype='float32')  ## PSF coefficents
        indices0 = np.array([], dtype='int32')    ## row indices
        indices1 = np.array([], dtype='int32')    ## column indices

        n = 0   ## iterator over threads (only if parallelization enabled)
        i = -1  ## iterator over all sources
        unres_added = False  ## flag indicating if unresolved component has
                             ##  already been added

        ## loop over all sources
        for psf in self.PSF: 
            i += 1
            j  = self.sources.specrow[i]

            if j == self.sources.unresRow and not calcUnres: 
                if not unres_added:
                ## if unresolved component should not be recalculated,
                ## add existing one to data.
                    nValues = self.unresIndices.size

                    coeffs.resize(coeffs.size+nValues)
                    indices0.resize(indices0.size+nValues)
                    indices1.resize(indices1.size+nValues)
                    
                    coeffs[-nValues:]   = self.unresCoeffs
                    indices0[-nValues:] = self.unresIndices
                    indices1[-nValues:] = j*np.ones(nValues, dtype=np.int32)
                    
                    unres_added = True

                continue

            if j == self.sources.skyRow: ## add constant sky component
                nValues = self.sources.instrument.nspaxel

                coeffs.resize(coeffs.size+nValues)
                indices0.resize(indices0.size+nValues)
                indices1.resize(indices1.size+nValues)

                coeffs[-nValues:]   = \
                    np.ones(nValues, dtype=np.float32)/(nValues*self.sigma)
                indices0[-nValues:] = np.arange(nValues, dtype=np.int32)
                indices1[-nValues:] = j*np.ones(nValues, dtype=np.int32)
                continue

            if self.ncpu > 1:
                spaxels, fluxes = threads[n].get()
                n += 1
            else:
                spaxels, fluxes = psf.get_flux()

            nValues    = psf.used.sum()
            
            if nValues > 0: ## resize arrays and add new data
                coeffs.resize(coeffs.size+nValues)
                indices0.resize(indices0.size+nValues)
                indices1.resize(indices1.size+nValues)

                ## divide by flux uncertainties before adding coefficents:
                coeffs[-nValues:]   = fluxes/self.sigma[psf.used]
                
                indices0[-nValues:] = spaxels
                indices1[-nValues:] = j*np.ones((nValues,), dtype=np.int16)
                
            elif doLog:
                logging.error("[@{0}] Source #{1} entirely outside FoV."\
                                  .format(self.kk, self.sources.ID[i]))
                
        ## save unresolved component for later recovery
        if self.sources.hasUnres and calcUnres:
            self.unresIndices = indices0[indices1==self.sources.unresRow]
            self.unresCoeffs  = coeffs[indices1==self.sources.unresRow]
                              
        return sparse.coo_matrix(\
            (coeffs, (indices0, indices1)),
            shape=(self.sources.instrument.nspaxel, self.sources.ncmp)\
                ).tocsc()

                              
                              
    ## Not updated yet!!!
    def fitPSF_largeIFU(self):
        """
        Perform the PSF fit for an IFU with many spaxels. In this case,
        a non-linear least-squares fit is performed for every PSF source
        independently. Depending on the respective fit-flags, either the
        parameters of the PSF and/or the source coordinates are fitted.
        The results of the individual fits are afterwards used to update
        the PSF and/or the coordinate transformation.
        """
        
        ## fit source coordinates (if requested)
        if self.sources.Fit:

            ## list will contain results of all fits
            allPars = []
            
            parNames = ["xc", "yc"]

            ## Initialize parallelization (if requested)
            if self.ncpu > 1:
                pool    = Pool(self.ncpu)
                threads = []

            ## Loop over all PSF sources; add source to residuals, fit
            ## resulting array and subtract source again.
            i=0
            for psf in self.PSF: ## ignore sources unless they have been
                                 ## flagged  as proper PSF candidates
                if not i in self.sources.psfIndices: continue

                x0   = [self.sources.xvalues[i,self.jj],
                        self.sources.yvalues[i, self.jj]]
                flux = self.sources.data[i,self.jj]

                if self.ncpu > 1:
                    threads.append(pool.apply_async(\
                            self.wrapper_fitOnePSF,
                            args=(x0, parNames, psf, flux)))
                else:
                    allPars.append(self.wrapper_fitOnePSF(\
                            x0=x0, names=parNames, psf=psf, flux=flux))
                
                i += 1

            if self.ncpu > 1:
                pool.close()
                pool.join()
                for i in xrange(self.sources.psfIndices.size):
                    allPars.append(threads[j].get())

            ## update parameters
            if self.fullPosFit: ## if actual  source coordinates should be
                                ## fitted,  update them directly
                for i in xrange(len(allPars)):
                    self.sources.xvalues[self.sources.psfIndices[i], self.jj] \
                        = allPars[i][0]
                    self.sources.yvalues[self.sources.psfIndices[i], self.jj] \
                        = allPars[i][1]

            else: ## otherwise, update coordinate transformation
                self.sources.ctValues[...,self.jj] = self.sources.quickFit(\
                    [self.sources.x[self.sources.psfIndices],
                     self.sources.y[self.sources.psfIndices]],
                    [[allPars[i][0] for i in xrange(len(allPars))],
                     [allPars[i][1] for i in xrange(len(allPars))]],
                    outSigma=None)[0]
                
                for i in xrange(len(self.sources.parameters)):
                    ## updating the parameters in the PSF-class overwrites
                    ## the (direct) source positions used for the fit
                    getattr(\
                        self.PSF,
                        "set_{0}".format(self.sources.parameters[i]))(\
                        self.sources.ctValues[i, self.jj])


        if self.psf_attributes.Fit: ## fit PSF

            allPars = []
            
            ## collect names and initial values of free PSF parameters
            parNames = []
            x0       = []
            for i in xrange(len(self.psf_attribbutes.parameters)):
                if self.psf_parameters.free[i] \
                        and not self.psf_attributes.poly[i]:
                    x0.append(self.psf_attributes.values[i,self.jj])
                    parNames.append(self.psf_attributes.parameters[i])

            ## Initialize parallelization (if requested)
            if self.ncpu > 1:
                pool    = Pool(self.ncpu)
                threads = []

            ## Loop over all PSF sources; add source to residuals, fit
            ## resulting array and subtract source again.
            i=0
            for psf in self.PSF: ## ignore sources unless they have been
                                 ## flagged  as proper PSF candidates
                if not i in self.sources.psfIndices: continue
            
                flux = self.sources.data[i,self.jj]

                if self.ncpu > 1:
                    threads.append(pool.apply_async(\
                            self.wrapper_fitOnePSF,
                            args=(x0, parNames, psf, flux)))
                else:
                    allPars.append(self.wrapper_fitOnePSF(\
                            x0=x0, names=parNames, psf=psf, flux=flux))
                
                i += 1

            if self.ncpu > 1: ## only parallelization
                pool.close()
                pool.join()
                for i in xrange(self.sources.psfIndices.size):
                    allPars.append(threads[j].get())
                
            ## update parameter values in the attributes of the PSF and in
            ## the actual PSF profiles
            i = 0
            for par in parNames:
                ii = np.where(self.psf_attributes.parameters==par)[0][0]
                
                self.psf_attributes.values[ii,self.jj] =\
                    np.mean([allPars[k,i] for k in xrange(len(allPars))])
                
                getattr(self.PSF, "set_{0}".format(par))(\
                    self.psf_attributes.values[ii,self.jj])
                
                i += 1
                

    def wrapper_fitOnePSF(self, x0, names, psf, flux):
        """
        This wrapper around the actual fitting routine 'fitOnePSF' is included
        to facilitate the parallelizaion of the code and a simple checking
        for the usefulness of the fitted parameters.
        It works by adding the current estimate of the PSF and the flux of the
        source to the data, calling the fit routine 'fitOnePSF' and providing
        the results of the fit (after a sanity check).

        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted.
        names :  array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        psf : an instance of class pampelmuse.core.new_profiles.Profile or
              one of its subclasses
            The PSF model for the source.
        flux : float
            The flux of the source.

        Returns
        -------
        bestfit : nd_array
            The values of the free parameters obtained in the fit.
        """

        ## add current PSF estimate of source to residuals
        _current   = np.copy(self.current[psf.used])
        _current  += psf.f/self.sigma[psf.used]*flux

        self.BAD = False ## a flag indicating if a PSF parameter was
                         ## fitted with an invalid value (e.g. beta<=1)

        ## call non-linear least squares fit
        bestFit_j = optimize.leastsq(\
            self.fitOnePSF, x0=x0, args=(names, psf, flux, _current))[0]

        if not self.BAD: ##  only update data if fit was valid
            return bestFit_j
        else:
            return x0


    def fitOnePSF(self, x0, names, psf, flux, data):
        """
        This is the optimization routine that is passed to the 'leastsq'-
        routine of the 'scipy.optimize'-package. During each iteration, a
        PSF array is created, scaled and subtracted from the data. The
        residuals of the fit are returned.
        
        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted.
        names :  array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        psf : an instance of class pampelmuse.core.new_profiles.Profile or
              one of its subclasses
            The PSF model for the source.
        flux : float
            The flux of the source.
        data : nd_array
            The data array used to fit the parameters. Note that the pixel
            values should be normalized by their respective uncertainties.

        Returns
        -------
        bestfit : nd_array
            The values of the free parameters obtained in the fit.
        """
        ## update PSF using current parameter estimates
        for i in xrange(len(x0)):
            setattr(psf, names[i], x0[i])
            ## the attribute is only set in the 'psf'-instance of the class,
            ## not in all instances! This is different from the behaviour of
            ## getattr(self.PSF, "set_<PAR>")(<VALUE>)

        ## return residuals
        return (data - psf.f/self.sigma[psf.used]*flux)



    def fitPSF_smallIFU(self):
        """
        Perform the positions/PSF fit on the whole array. A non-linear least-
        squares fit that uses Levenberg-Marquard optimization is performed
        with all resolved sources. This fitting procedure is recommended only
        for IFUs with a small number of spatial pixels. The results of the fit
        are used to update the PSF and/or the parameters of the coordinate
        transformation.
        """

        NCPU = self.ncpu ## no parallelization for this part yet
        self.ncpu = 1    ## so set ncpu=1 and save its original value

        ## Fit position parameters
        if self.sources.Fit:
            
            ## collect free parameters
            fitPars  = []
            parNames = []
            for i in xrange(len(self.sources.parameters)):
                if self.sources.ctFree[i] and not self.sources.ctPoly[i]:
                    fitPars.append(self.sources.ctValues[i,self.jj])
                    parNames.append(self.sources.parameters[i])

            ## call Levenberg-Marquard solver from scipy.optimize
            bestfit = optimize.leastsq(\
                self.fitMultiPSF, x0=fitPars,
                Dfun=self.jacobiMultiPSF, col_deriv=1, ## provide Jacobian
                args=(parNames,))[0]
            
            ## update parameters using fit results
            for i in xrange(len(parNames)):
                ii = self.sources.parameters.index(parNames[i])
                self.sources.ctValues[ii,self.jj] = bestfit[i]

            ## some logging
            logging.debug("[@{0}] Fitted coordinate transf.:".format(self.kk))
            for i in xrange(len(self.sources.parameters)):
                logging.debug("         {0}={1:.2f}".format(\
                        self.sources.parameters[i],
                        float(self.sources.ctValues[i,self.jj])))


        ## Fit PSF parameters
        if self.psf_attributes.Fit:

            ## collect free parameters
            fitPars  = []
            parNames = []
            for i in xrange(len(self.psf_attributes.parameters)):
                if self.psf_attributes.free[i] \
                        and not self.psf_attributes.poly[i]:
                    fitPars.append(self.psf_attributes.values[i,self.jj])
                    parNames.append(self.psf_attributes.parameters[i])

            ## call Levenberg-Marquard solver from scipy.optimize
            bestfit = optimize.leastsq(\
                self.fitMultiPSF, x0=fitPars, ## provide Jacobian
                Dfun=self.jacobiMultiPSF, col_deriv=1,
                args=(parNames,))[0]

            ## update parameters using fit results
            for i in xrange(len(parNames)):
                ii = self.psf_attributes.parameters.index(parNames[i])
                self.psf_attributes.values[ii,self.jj] = bestfit[i]

            ## some logging
            logging.debug("[@{0}] Fitted PSF model:".format(self.kk))
            for i in xrange(len(self.psf_attributes.parameters)):
                logging.debug("         {0}={1:.2f}".format(\
                        self.psf_attributes.parameters[i],
                        float(self.psf_attributes.values[i,self.jj])))

        self.ncpu = NCPU ## recover original number of CPUs to be used



    def fitMultiPSF(self, x0, names):
        """
        The optimization routine that is passed to the 'leastsq'-routine of
        the 'scipy.optimize'-package. During each iteration, a PSF array is
        prepared for every resolved source, the PSFs are scaled and subtracted
        from the data. The residuals are returned.
        
        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted.
        names : array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        
        Returns
        -------
        residuals : nd_array
            The residuals of the fit.
        """
        ## update parameters in PSF-class
        for i in xrange(len(x0)):
            getattr(self.PSF, "set_{0}".format(names[i]))(x0[i])

        ## calculate new model
        Aprm = self.fillMatrix(calcUnres=False, doLog=False)
        model =  Aprm*sparse.dia_matrix((self.sources.data[...,self.jj],[0]),
                                        shape=(self.sources.ncmp,
                                               self.sources.ncmp))

        ## return residuals
        return (self.data/self.sigma) - model.sum(axis=1).getA1()



    def jacobiMultiPSF(self, x0, names):
        """
        Routine that calculates the Jacobi matrix of the residuals. This
        function is called by the Levenberg-Marquard routine that performs
        the non-linear least squares fit.

        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted
        names : array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        
        Returns
        -------
        jacob : nd_array
            The Jacobi matrix with respect to the model parameters included
            in 'x0'.
        """
        jacob = np.zeros((len(x0),self.sources.instrument.nspaxel),
                         dtype=np.float64)

        i=-1
        for psf in self.PSF:
            i += 1
            ii = self.sources.specrow[i]
            if ii==self.sources.skyRow: continue
            if ii==self.sources.unresRow: continue
            for j in xrange(len(x0)):
                jacob[j, psf.used] -= self.sources.data[ii,self.jj]\
                    *getattr(psf, "diff_{0}".format(names[j]))\
                    /self.sigma[psf.used]
            
        return jacob



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
                self.kk, LASTFITS))

        if self.sources.Fit:
            
            if not self.fullPosFit:  ## update coordinate transformation
                self.sources.ctValues[...,self.jj] \
                    = self.sources.ctData[0,:][...,LASTFITS].mean(axis=-1)
            else: ## update coordinates
                self.sources.xvalues[...,self.jj] \
                    = self.sources.xdata[0,:][...,LASTFITS].mean(axis=-1)
                self.sources.yvalues[...,self.jj] \
                    = self.sources.ydata[0,:][...,LASTFITS].mean(axis=-1)

        if self.psf_attributes.Fit: ## update PSF
            self.psf_attributes.values[...,self.jj] \
                = self.psf_attributes.data[0,:][...,LASTFITS].mean(axis=-1)

            logging.debug("[@{0}] Initial PSF model: {1}".format(\
                    self.kk, self.psf_attributes.values[...,self.jj]))



