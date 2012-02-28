"""
The program contains `StarFit', the core-routine of PampelMuse. The
routine is optimized to handle large datacubes, e.g MUSE-like data.
For the analysis of smaller datacubes, e.g PMAS-like data,  some features
still have to be added from older routines.

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
v0.2.0.1, 2012/02/01, not usign the `fitRad'-variable anymore since in certain
                      situation is caused an unstable behavior of the routine.
                      The `psfRad'-variable is used for all fits instead. 
v0.2.0.2, 2012/02/13, added parameter `jumpStart' to allow starting a fit
                      remotely.
v0.3.0.0, 2012/02/24, major update on how PSF and source positions are used.
                      The now have to be provided as instances of dedicated
                      classes `Positions' and `PSF'.
v0.3.1.1, 2012/02/28, some minor updates and bugfixing wrt new usage of PSF and
                      positions.
"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.3.1.1"

import logging, os, sys, time
import numpy as np, pyfits
from scipy.sparse.linalg import lsqr
from scipy import sparse, optimize

##  required PampelMuse-routines
from pampelmuse.core.my_profiles import Gaussian, Moffat
from pampelmuse.functions.trans_coords import trans_coords
from pampelmuse.graphics.fitting_gui import FittingGui




class UserInterrupt(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class StarFit(FittingGui):
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
    def __init__(self, dataFits, sigmaFits, maskFits=None, parent=None):
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

        Returns
        -------
            The newly created instance
        """

        super(StarFit, self).__init__(parent)

        self.BUSY     = False ## possible states of the fitting loop 
        self.FINISHED = False ## not defined in FittingGui-class


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

        

        
    def provideSources(self, sources, spectra, positions, specSigma=None):
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
        positions : a pampelmuse.core.positions.Positions-instance
            To provide all the information about source positions needed by
            this routine, an instance of the Positions-class defined in
            `positions.py' must be provided. Check the documentation of that
            routine for furtehr information.
        specSigma : nd_array or None, optional
            Uncertainties associated with the provided spectra. They are not
            used in the fit, but (re)estimated afterwards. Providing
            `specSigma' only has an effect if not the whole cube is fitted.
        """
        self.sources   = sources
        self.spectra   = spectra
        self.positions = positions


        ## create array to store uncertainties of fitted spectra
        ## (if none provided)
        if specSigma == None:
            self.specSigma = np.zeros(self.spectra.shape, dtype='float32')
        else:
            self.specSigma = specSigma

        ## no. of fitted components
        self.Ncmp = self.sources['specrow'].max()+1

        self.specMap = {}
        self.relMags = {}
        for i in xrange(self.Ncmp): ## use `specrow' as reference for sources
            self.specMap[i]  = np.where(self.sources['specrow']==i)[0]

            self.relMags[i]  = self.sources['mag'][self.specMap[i]]
            self.relMags[i] -= (-2.5*np.log10(np.sum(\
                        [10**(-0.4*m) for m in self.relMags[i]]
                        )))

        for comp, spec in self.specMap.iteritems(): 
            logging.debug("   {0:5d} - {1}".format(comp, spec))

        logging.debug("Relative magnitudes of spectra in components:")
        for comp, mags in self.relMags.iteritems():
            logging.debug("   {0:5d} - {1}".format(comp, mags))


        ## look for unresolved stellar component
        unRes = self.sources['status']==0
        if unRes.any():
            self.unResCmp = self.sources['specrow'][np.where(unRes)[0][0]]
        else:
            self.unResCmp = -1


    
    def providePSF(self, psf, psfRad, osFactor=5, osRadius=2):
        """
        This method should be used to provide the instance with the necessary
        information about the PSF that is used in the fitting process.
        
        Parameters
        ----------
        psf : a pampelmuse.core.psf.PSF-instance
            The psf should be prepared using the PSF-class of the `psf.py'-
            routine. For further documentation on the PSF, check the
            documentation of that routine
        psfRad : float
            The radius out to which the PSF is defined around a source
        osFactor : integer, optional
            The oversampling factor of the central PSF
        osRadius : float, optional
            The radius out to which the PSF is oversampled. 
        """
        self.psf    = psf
        self.psfRad = psfRad

        ## collect stars suitable for PSF determination
        if self.psf.Fit or self.positions.Fit:
            self.psfStars = np.where(self.sources['status']==3)[0]
        


    def __call__(self, fitRad=None, Ntrack=10, lastFitIndex=None,
                 asSmallIFU=False, returnModel=False, returnResiduals=False,
                 prefix=None, clip=5., jumpStart=False):
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
        returnModel : boolean, optional
            Flag indicating whether an array containing the best-fit model
            should be returned. Not implemented yet.
        returnResdiduals : boolean, optional
            Flag indicating whether an array containing the residuals of the
            fit should be returned.
        prefix : string, optional
            In case that the fitted model and/or the residuals should be
            stored, `fileRoot' allows to provide a prefix used to store the
            individual images. This is needed in case that several fits are
            started from the same folder.
        clip : float, optional
            All pixels whose values devite by more than `clip' times SIGMA are
            masked. SIGMA is the standard deviation of the residuals between
            model and data. Not implemented yet.
        jumpStart : boolean, optional
            In case the fit should start directly (i.e. without pressing
            `START'-button), enable this flag.
        """
        self.makeMod = max(returnModel, returnResiduals) ## create model array
        self.makeRes = returnResiduals                   ## and residuals?
        if prefix == None:
            self.prefix = ""
        else:
            self.prefix = prefix

        self.clip    = clip    ## efficiency of bad pixel masking

        if fitRad != None:
            logging.warning("The `fitRad'-variable is not used anymore.")
            logging.warning("Using `psfRad' instead to define fit regions.")
##             self.fitRad  = fitRad

        if lastFitIndex == None:
            self.i = 0    ## iterator over all images
        else:
            self.i = lastFitIndex

        self.Ntrack         = Ntrack
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

        self.show() ## start GUI that controls fitting process

        if jumpStart:
            self.startIt()

    
    def timeout(self):

        if not self.STARTED:
            return

        if self.BUSY:
            return

        if self.ABORTED:
            self.timer.stop()
            self.close()
            raise UserInterrupt('Aborted on user request')
    
        if self.PAUSED:
            self.sBar.showMessage("PAUSED")
            return

        self.jj = (self.specPix>>1) - 1 - int(self.i/2) \
            + self.i*np.mod(self.i, 2) + np.mod(self.specPix, 2)
        ## jj is a counter that starts at the central spectral pixel
        ## and proceeds simultaneoulsy to the red and blue end.

        if self.i == self.specPix:
            self.timer.stop()
            self.save = True
            self.close()
        else:
            self.prefixString = "[{0:0{1}d}/{2}]".format(\
                self.i, int(np.log10(self.specPix))+1, self.specPix)
            self.sBar.showMessage("{0} Analyzing slice #{1}".format(\
                    self.prefixString, self.jj))

            self.iterFitImage()
            self.i += 1



    def iterFitImage(self):

        ## the fitting process
        self.BUSY = True

        ## update parameter initial guesses (if any):
        self.updatePars() 

        ## initialize PSF:
        self.PSF = self.psf(index=self.jj)
 
        ## determine pixel coordinates:
        self.x, self.y = self.positions.getIfuCoords(\
            index=self.jj, forceTrans=False)

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
 
        self.dataCanvas.addImage(self.data.reshape(self.spatShape))

        ## fit PSF/positions if requested
        if self.psf.Fit or self.positions.Fit:

            ## create model matrix
            self.Apsf = self.fillMatrix(radius=self.psfRad)
            ## create fit matrix
##             self.Afit = self.fillMatrix(radius=self.fitRad)

            ## initial fit of fluxes
            bestFit = lsqr(self.Apsf, self.current)
            
            self.spectra[:,self.jj] += bestFit[0]
            logging.debug("[@{0}] Chi**2 of initial fit: {1:.1f}".format(\
                    self.jj, bestFit[4]**2))
            logging.debug("[@{0}] Relative changes in fluxes: {1}".format(\
                    self.jj, bestFit[0]/self.spectra[:,self.jj]))

            ## PSF fit
            if self.asSmallIFU:
                self.fitPSF_smallIFU()
            else:
                self.fitPSF_largeIFU()
 
        ## fit fluxes
        self.sBar.showMessage("{0} Fitting fluxes".format(self.prefixString))

        ## (re)calculate matrices 
        self.Apsf = self.fillMatrix(radius=self.psfRad)
##         self.Afit = self.fillMatrix(radius=self.fitRad)

        ##  fit
        bestFit = lsqr(self.Apsf, self.current)
        self.spectra[:,self.jj] += bestFit[0]
        logging.debug("[@{0}] Chi**2 of final fit: {1:.1f}".format(\
                self.jj, bestFit[4]**2))
        logging.debug("[@{0}] Relative changes in fluxes: {1}".format(\
                self.jj, bestFit[0]/self.spectra[:,self.jj]))

        ## get uncertainties
        for i in xrange(self.Ncmp):
            self.specSigma[i,self.jj] = 1./np.sqrt(\
                self.Apsf[:,i].multiply(self.Apsf[:,i]).sum()
                )

        ## create 2dim. residuals for plotting (& saving if requested)
        if self.hasMask:
            resi1d = np.zeros((self.spatPix,), dtype='float32')
            resi1d[self.pixMap>-1] = self.current
            self.resi2d = (resi1d*self.sigma).reshape(self.spatShape)
        else:
            self.resi2d = (self.current*self.sigma).reshape(self.spatShape)
            
        self.resiCanvas.addImage(self.resi2d)

        if self.makeRes:
            outHDU = pyfits.PrimaryHDU(self.resi2d)
            outHDU.writeto("{0}.res{1}.fits".format(self.prefix, self.i),
                           clobber=True)
        ## to save memory, the residuals are currently stored as a set of
        ## individual images. To create a datacube, they must be combined
        ## afterwards manually.

            
        ## update inventory of last fits
        self.lastFittedImas.append(self.jj)
        if len(self.lastFittedImas)>2*self.Ntrack:
            self.lastFittedImas.remove(self.lastFittedImas[0])
 
        self.pBar.setValue(100.*float(self.i+1)/self.specPix)

        self.BUSY = False



    @property
    def model(self):
        """Create the current best-fit model"""
        return self.Apsf*sparse.dia_matrix((self.spectra[:,self.jj],[0]),
                                           shape=(self.Ncmp, self.Ncmp))

    @property
    def current(self):
        """Determine the current residuals"""
        ## self.model.sum(axis=1) returns numpy.matrixlib.defmatrix-object
        ## `getA1' returns a flattened numpy.ndarray

        return (self.data/self.sigma)[self.mask]-self.model.sum(axis=1).getA1()



    def fillMatrix(self, radius, calcUnRes=True, doLog=True):
        """
        Define the source matrix. The source matrix is a sparse matrix that
        contains the spatial profiles of all resolved components, one per
        column.
        
        Parameters
        ----------
        radius : float
            The radius out to which the profile is calculated around a point
            source.
        calcUnRes : boolean, optional
            If False, the matrix column containing the unresolved stellar
            component (if included) is not computed, but left empty.
        doLog : boolean, optional
            If True, the routine method will print a logging.ERROR-message
            every time a source whose PSF is completely outside the FoV is
            encountered.
        """

        coeffs   = np.array([], dtype='float32')
        indices0 = np.array([], dtype='int32')
        indices1 = np.array([], dtype='int32')

        for i in xrange(self.Ncmp):

            if i==self.unResCmp and not calcUnRes:
                nValues = self.unResCoeffs.size

                coeffs.resize(coeffs.size+nValues)
                indices0.resize(indices0.size+nValues)
                indices1.resize(indices1.size+nValues)

                coeffs[-nValues:]   = self.unResCoeffs
                indices0[-nValues:] = self.unResIndices
                indices1[-nValues:] = i*np.ones((nValues,), dtype='int32')
                continue

            k = -1

            nValuesCmp = 0
            for j in self.specMap[i]:

                k += 1
                if self.sources['id'][j] == -1:
                    nValues    = self.spatPix
                    newCoeffs  = np.ones(self.spatPix, dtype="float32")
                    newCoeffs /= self.spatPix
                    newIndices = np.arange(self.spatPix, dtype="int32")

                else:
                    psf_ij = self.PSF(size=self.spatShape, x_c=self.x[j],
                                      y_c=self.y[j], mag=self.relMags[i][k],
                                      maxrad=radius, return_sparse=True,
                                      as_sparse_matrix=False)

                    nValues    = psf_ij[0].size
                    newCoeffs  = psf_ij[0]
                    newIndices = psf_ij[1][1] + self.spatShape[1]*psf_ij[1][0]

                if nValues > 0:
                    coeffs.resize(coeffs.size+nValues)
                    indices0.resize(indices0.size+nValues)
                    indices1.resize(indices1.size+nValues)

                    coeffs[-nValues:]   = newCoeffs/self.sigma[newIndices]
                    indices0[-nValues:] = newIndices
                    indices1[-nValues:] = i*np.ones((nValues,), dtype='int32')

                    nValuesCmp += nValues
                    
            if nValuesCmp == 0 and doLog:
                logging.error(\
                    "[@{0}] Component #{1} entirely outside FoV.".format(\
                        self.jj, self.sources['id'][self.specMap[i][0]]))
                    

        if self.unResCmp > -1 and calcUnRes:
            self.unResIndices = indices0[indices1==self.unResCmp]
            self.unResCoeffs  = coeffs[indices1==self.unResCmp]

        if self.hasMask:
            mappedIndices0 = self.pixMap[indices0]
            used           = (mappedIndices0 > -1)

            return sparse.coo_matrix((coeffs[used],
                                      (mappedIndices0[used], indices1[used])),
                                     shape=(self.mask.sum(), self.Ncmp)).tocsc()

        else:
            return sparse.coo_matrix((coeffs, (indices0, indices1)),
                                     shape=(self.spatPix, self.Ncmp)).tocsc()



    def fitPSF_largeIFU(self):
        """
        Perform the PSF fit. For every PSF-source, a non-linear least-squares
        fit is performed. Depending on the respective fit-flags, either the
        PSF-parameters and/or source coordinates are fitted.
        The results of the individual fits are then used to update the PSF and/
        or the coordinates of the parameter transformation.
        """
        nPars = self.psf.npars+3

        allPars = np.zeros((nPars, self.psfStars.size), dtype='float32')

        fitFlags       = np.zeros((nPars,), dtype='bool')
        fitFlags[:2]   = self.positions.Fit
        fitFlags[2:-1] = self.psf.toUpdate
        fitFlags[-1]   = True

        ## create working copy of `current'-array. Required since the
        ## definition of `current' (using an @property-method) does not
        ## work with in-place operations like `+=' or `-='.
        self._current = np.copy(self.current)

        ## loop over all PSF sources; add source to residuals, fit resulting
        ##  array and subtract source again.
        j = 0
        for i in self.psfStars:
            kk = self.sources['specrow'][i]

            self.sBar.showMessage(\
                "{0} Fitting PSF to Source #{1} ({2}/{3})".format(\
                    self.prefixString, self.sources['id'][i], j+1,
                    len(self.psfStars)))

            allPars[0,j]    = self.x[i]
            allPars[1,j]    = self.y[i]
            allPars[2:-1,j] = self.psf.viewValue(index=self.jj)
            allPars[-1,j]   = self.spectra[kk, self.jj]

            ## get current PSF estimate of source and add it to residuals
            psf_i = self.PSF(size=self.spatShape, x_c=self.x[i], y_c=self.y[i],
                             mag=0., maxrad=self.psfRad, return_sparse=True,
                             as_sparse_matrix=False)
            psfIndices = psf_i[1][1] + self.spatShape[1]*psf_i[1][0]
            psfCoeffs  = psf_i[0]/self.sigma[psfIndices]

            if self.hasMask:
                mappedIndices = self.pixMap[psfIndices]
                used          = (mappedIndices > -1)
                self._current[mappedIndices[used]] \
                    += psfCoeffs[used]*self.spectra[kk,self.jj]
            else:
                self._current[psfIndices] \
                    += psfCoeffs*self.spectra[kk,self.jj]

            
            ## fit PSF
            allPars[fitFlags,j] = optimize.fmin(\
                func=self.fitOnePSF,
                x0=allPars[fitFlags,j],
                args=(allPars[fitFlags==False,j],fitFlags),
                disp=False)


            ## subtract PSF again
            if self.hasMask:
                self._current[mappedIndices[used]] \
                    -= psfCoeffs[used]*self.spectra[kk,self.jj]
            else:
                self._current[psfIndices] \
                    -= psfCoeffs*self.spectra[kk,self.jj]
            j += 1            

        if self.positions.Fit: ## update coordinate transformation

            self.positions.putIfuCoords(values=allPars[:2], index=self.jj,
                                        ids=self.sources['id'][self.psfStars])
            if self.positions.verifyTrans:
                self.positions.updateTransformation(index=self.jj)

            [self.x, self.y] = self.positions.getIfuCoords(\
                index=self.jj, forceTrans=self.positions.verifyTrans)


        if self.psf.Fit: ## update PSF
            self.psf.putValue(data=allPars[2:-1,:].mean(axis=1), index=self.jj)
            self.PSF = self.psf(index=self.jj)



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

        self.psf.putValue(data=psfPars[2:-1], index=self.jj)
        thePSF = self.psf(index=self.jj)

        psfArray = thePSF(size=self.spatShape, x_c=psfPars[0],
                          y_c=psfPars[1], mag=0., maxrad=self.psfRad,
                          return_sparse=True, as_sparse_matrix=False)

        psfIndices = psfArray[1][1] + self.spatShape[1]*psfArray[1][0]
        psfCoeffs  = psfArray[0]/self.sigma[psfIndices]
        
        if self.hasMask:
            mappedIndices = self.pixMap[psfIndices]
            used          = (mappedIndices > -1)

            err = self._current[mappedIndices[used]]-psfCoeffs[used]*psfPars[-1]
        else:
            err  = self._current[psfIndices]-psfCoeffs*psfPars[-1]

        return (err*err).sum()



    def fitPSF_smallIFU(self):
        """
        Perform the positions/PSF fit on the whole array. A non-linear least-
        squares fit is performed using all resolved sources. This fitting
        procedure is recommended only for IFUs with a small number of spatial
        pixels.
        The results of the fits is used to update the PSF and/or the
        parameters of the coordinate transformation.
        """

        nPars = self.positions.npars+self.psf.npars

        fitFlags = np.zeros((nPars,), dtype='bool')
        fitPars  = np.zeros((nPars,), dtype='float64')

        fitPars[:self.positions.npars] = self.positions.viewValue(self.jj)
        fitPars[self.positions.npars:] = self.psf.viewValue(self.jj)


        ## Fit position parameters
        if self.positions.toUpdate.any():
            
            fitFlags[:self.positions.npars] = self.positions.toUpdate

            self.sBar.showMessage(\
                "{0} Fitting positions".format(self.prefixString))

            fitPars[fitFlags] =  optimize.fmin(\
                func=self.fitMultiPSF,
                x0=fitPars[fitFlags],
                args=(fitPars[fitFlags==False], fitFlags),
                disp=False)
            
            self.positions.putValue(fitPars[:self.positions.npars],
                                     index=self.jj)

        ## Fit PSF parameters
        if self.positions.toUpdate.any():

            fitFlags[:self.positions.npars] = False
            fitFlags[self.positions.npars:] = self.psf.toUpdate
            
            self.sBar.showMessage("{0} Fitting PSF".format(self.prefixString))

            fitPars[fitFlags] = optimize.fmin(\
                func=self.fitMultiPSF,
                x0=fitPars[fitFlags],
                args=(fitPars[fitFlags==False], fitFlags),
                disp=False)

            logging.debug("[@{0}] Fitted PSF model: {1}".format(\
                    self.jj, fitPars[self.positions.npars:]))

            self.psf.putValue(fitPars[self.positions.npars:], index=self.jj)
            self.PSF = self.psf(index=self.jj)



    def fitMultiPSF(self, x0, fixedPars, fitFlags):

        allPars = np.zeros(fitFlags.shape, dtype='float64')

        allPars[fitFlags] = x0
        allPars[fitFlags==False] = fixedPars

        if fitFlags[:self.positions.npars].any(): ## recalc pixel positions
            self.positions.putValue(allPars[:self.positions.npars],
                                     index=self.jj)
            self.x, self.y = self.positions.getIfuCoords(index=self.jj)

        if fitFlags[self.positions.npars:].any(): ## recalc PSF
            self.psf.putValue(allPars[self.positions.npars:], index=self.jj)
            self.PSF = self.psf(index=self.jj)

        Aprm = self.fillMatrix(radius=self.psfRad, calcUnRes=False, doLog=False)

        model =  Aprm*sparse.dia_matrix((self.spectra[:,self.jj],[0]),
                                        shape=(self.Ncmp, self.Ncmp))

        err =  (self.data/self.sigma)[self.mask]-model.sum(axis=1).getA1()

## !!! Old residual calculation, unstable(?)
##         Aprm = self.fillMatrix(radius=self.fitRad, calcUnRes=False,
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

        if self.positions.Fit: ## update coordinate transformation
            if self.positions.ifu != None:
                lastPos = self.positions.getIfuCoords(index=LASTFITS,
                                                      forceTrans=False)
                self.positions.putIfuCoords(values=lastPos.mean(axis=-1),
                                            index=self.jj)
            if self.positions.verifyTrans:
                lastTrans = self.positions.viewValue(index=LASTFITS)
                self.positions.putValue(data=lastTrans.mean(axis=0),
                                         index=self.jj)

        if self.psf.Fit: ## update PSF
            lastPSF = self.psf.viewValue(index=LASTFITS)
            self.psf.putValue(data=lastPSF.mean(axis=0), index=self.jj)

            logging.debug("[@{0}] Initial PSF model: {1}".format(\
                    self.jj, self.psf.viewValue(index=self.jj)))




