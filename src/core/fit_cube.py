__version__ = 0.8 ## last modified: 2011/02/22


## CHANGELOG:
## v0.3 - 2010/11/05 - instead of several files (*.src.fits, *.psf.fits,
##                     *.all.dat), a single file (*.prm.fits) is now used
##                     to import the fit parameters and store thema again
##                     after the fit.
## v0.3.1 - 2010/11/08 - minor changes and updated documentation
## v0.4 - 2010/11/09 - updated documentation
##                     fixed problem that ActiveModel-extension was not updated
##                      properly if column 'CHISQ' had to be created
##                     using deepcopy to read columns of PRM-file
## v0.4.1 - 2010/11/24 - included options to specify how fluxes are fitted in
##                       the FitArray-routine
## v0.5 - 2010/12/07 - updated options to fit fluxes in the FitArray-routine
## v0.6 - 2010/12/16 - updated routine according to changes in create_prm-file.
##                     Both components of 'binary' sources are kept but they are
##                     fit with a common flux value. A `map`-array is used to
##                     connect the sources to the fluxes. Moreover, instead of
##                     creating a dictionary for the unresolved sources, 
##                     positions and magnitudes for all sources are put into
##                     one array. 
##                   - No IDs are passed to the FitArray-routine anymore. 
## v0.7 - 2011/02/04 - corrected several severe bugs (code didn't work
##                     properly if posfit was set to 3, instead of residuals
##                     or the best fit-model, the mask cube was saved,...)
## v0.8 - 2011/02/22 - added option to fit positions/PSF using the average
##                     of N>1 frames. The number of frames to coadd must be
##                     provided in the fit_cube-method. Default is 1.

import datetime, numpy, pyfits, os, sys, time
from copy import deepcopy
from scipy import special
from fit_array import StarFit



class FitCube:

    """
    Class to deblend the different components (sources, background, sky)
    in an IFU datacube of a stellar field using PSF fitting techniques.
    The class is wrapped around the FitArray-class, which performs the
    actual fitting processes.
    """

    def __init__(self, rootname, model=False, residuals=False, update_mask=True,
                 verbose=2, interact=False):
        """
        Initialize an instance of the FitCube-class

        Parameters
        ----------
        rootname : string
            The part of the filenames that is shared by all files FitCube
            requires to analyze a datacube. The following files must be
            present:
            `root`.prm.fits - A FITS-table information about the model to
                              be fitted to the data.
            `root`.dat.fits - The datacube to be fitted
            `root`.err.fits - The corresponding sigma datacube
            `root`.msk.fits - A datacube containing a bad-pixel mask
                              (optional)
        model : boolean, optional
            If True, the routine will create a datacube containing the
            best-fit model, named `root`.mod.fits, upon completion of the
            fit.
        residuals : boolean, optional
            If True, the routine will create a datacube containing the
            the residuals between the best-fit model and the data, named
            `root`.res.fits, upon completion of the fit.
        update_mask : boolean, optional
            If True, the mask will be updated with the results of the
            bad-pixel rejection performed by FitArray. If mask cube does
            not exist, it will be created, otherwise, it will be replaced.
        verbose : integer, optional
            The amount of information printed out during the analysis. `0`
            correspond to no information at all, `3` to a lot.
        interact : integer, optional
            If `interact`>0, the routine will ask before overwriting an
            existing file. Else, existing files willbe automatically
            overwritten.
        """
        self.root  = rootname
        self.vb    = verbose
        self.ia    = interact
        self.today = datetime.datetime.now()

        self._vprint("\n", 1)
        self._vprint(" "*18+"***   FitCube, version %s   ***\n" \
                         %str(__version__), 1)
        self._vprint("\n", 1)

        self._vprint("#"*20+"INITIALIZING FITTING PROCESS"+"#"*20+"\n", 2)

        ## Opening FITS cubes (data, sigma; mask if provided)
        """
        First axis (in python) of cubes must be spectral axis
        (corresponding to third axis in ds9, QFitsView etc.)
        """

        self._vprint("Reading input data:\n", 2)

        self._vprint(" %s.prm.fits..." %self.root, 2)
        self.prmfile = pyfits.open("%s.prm.fits" %self.root, mode="update")
        self.prmfile[0].header.add_history(self.today.ctime()+" - Processed using FitCube, version "+str(__version__))
        self.actHDU  = self.prmfile.index_of("ActiveModel")
        self._vprint("OK\n", 2)

        datacube = self.root+'.dat.fits'
        self._vprint(" %s..." %datacube, 2)
        self.data, header = pyfits.getdata(datacube, header=True)
        self.size = self.data.shape
        if self.size[0] != self.prmfile[self.actHDU].data.size:
            raise IOError, "Unmatched number of spectral pixels"
        self._vprint("OK\n", 2)

        errorcube = self.root+'.err.fits'
        self._vprint(" %s..." %errorcube, 2)
        self.sigma = pyfits.getdata(errorcube)
        if self.sigma.shape != self.size: ## check if size okay
            raise IOError, 'Sigma and data cubes must have same shapes'
        self._vprint("OK\n", 2)

        self.maskname = self.root+'.msk.fits'
        if os.path.exists(self.maskname):
            self._vprint(" %s..." %self.maskname, 2)
            self.mask = pyfits.getdata(self.maskname)
            if self.mask.shape != self.size: ## check if size okay
                raise IOError, 'Mask and data cubes must have same shapes'
            self._vprint("OK\n", 2)
        else:
            self.mask = numpy.zeros(self.size, dtype='int16')
  
        time.sleep(0.2)

        self.Nimages = self.size[0]

        start = header['CRPIX3'] - 1 ## get wavelength information
        self.wavl = numpy.arange(start, self.Nimages+start, dtype='float32')
        self.wavl *= header['CDELT3']
        self.wavl += header['CRVAL3']
        self._vprint("Wavelength interval:\n", 2)
        self._vprint(" %.1f - %.1f \AA\n" %(self.wavl[0], self.wavl[-1]), 2)
        self._vprint("Wavelength stepsize:\n", 2)
        self._vprint(" %.1f \AA\n" %header['CDELT3'], 2)

        self.chisq = numpy.zeros((2,self.Nimages), dtype='float32')


        ## Check if additional output is requested
        self.outcube = numpy.zeros((3,), dtype='bool') ## Model, Residuals, Mask

        self.modelname = self.root+'.mod.fits'
        if model:
            self.outcube[0] = True
            self.model = numpy.zeros(self.size, dtype='float32')

        self.resiname = self.root+'.res.fits'
        if residuals:
            self.outcube[1] = True
            self.resi = numpy.zeros(self.size, dtype='float32')

        if update_mask:
            self.outcube[2] = True


        
    def init_psf(self, isfitted, lut=None, **kwargs):
        """
        Method to initialize the PSF model used in the fit. Model can
        currently be either a Moffat or a Gaussian. Unfortunately, no
        Look-up tables are currently supported (will change soon!).
        The method will search the PRM-file for information about the
        PSF:
        * Keyword 'PSFTYPE' in the header of the PrimaryHDU, whose
          value must be either 'moffat' or 'gauss'.
        * Columns named 'PSFPAR<I>' in the ActiveModel-extension, with
          <I> = 2,3,4 for a gaussian and 1,2,3,4 for a moffat.
        If one or more of those informations are missing, they must be
        provided using the kwargs.

        Parameters
        ----------
        isfitted : array_like
            A boolean fit parameter for every parameter of the requested
            PSF-model.
        lut : None, optional
            The look-up table to be used (if requested).
            NOT YET SUPPORTED.
        kwargs : optional
            If PRM-file misses PSF-informations.
        """

        self._vprint("*"*30+"PSF MODEL"+"*"*29+"\n", 2)

        ## valid PSF types and lists of required arguments (PSFPAR<X>)
        ## 3 (& 4) are always ellipticity and position angle
        self.need = {'moffat':[1,2,3,4], 'gauss':[2,3,4]}


        psfargs = {}
        for u, v in kwargs.iteritems(): ## Set keys to uppercase
            psfargs[u.upper()] = v
        

        ## check if PRM-file contains info about type of PSF:
        self._vprint("Searching PRM-file for PSF information...\n", 2)
        self._vprint(" ...looking for keyword PSFTYPE...", 2)

        if not self.prmfile[0].header.has_key('PSFTYPE'):
            self._vprint("FAILED\n", 2)
            self._vprint("    Using keyword-argument:", 2)             
            
            self._update_hdr("PSFTYPE", psfargs['PSFTYPE'],
                             "Analytical PSF profile")
        else:
            self._vprint("OK\n   ", 2)


        self.psftype = self.prmfile[0].header['PSFTYPE']
        if not self.need.has_key(self.psftype):
            raise IOError, "Unknown PSF type: '%s'" %self.psftype
        else:
            self._vprint(" PSFTYPE='%s'\n" %self.psftype, 2)

        self.Nppsf   = len(self.need[self.psftype])
        self.psffit  = numpy.zeros((self.Nppsf,), dtype='bool')
        self.psfpars = numpy.zeros((self.Nimages, self.Nppsf), dtype='float32')

        ## check if PRM-file contains columns with required PSF-parameters
        psfcols = []
        for i in self.need[self.psftype]:
            j = self.need[self.psftype].index(i)

            self._vprint(" ...looking for column PSFPAR%i..." %i, 2)
            
            if not "PSFPAR%i" %i in self.prmfile[self.actHDU].data.names:

                self._vprint("FAILED\n", 2)
                self._vprint("    ...checking FITS-header...", 2)

                if self.prmfile[0].header.has_key('PSFPAR%i' %i):
                    self._vprint("OK:", 2)

                    psfpar_i = self.prmfile[0].header['PSFPAR%i' %i]

                else:
                    self._vprint("FAILED\n", 2)
                    self._vprint("    Using keyword-argument:", 2)
        
                    psfpar_i = psfargs['PSFPAR%i' %i]

                self._vprint(" PSFPAR%i=%.3f\n" %(i, psfpar_i), 2)

                self.psfpars[:,j] = numpy.resize(psfpar_i, self.Nimages)

                psfcols.append(pyfits.Column(name='PSFPAR%i' %i, format='E',
                                             array=self.psfpars[:,j]))

            else:
                self._vprint("OK\n", 2)
                self.psfpars[:,j] = \
                    deepcopy(self.prmfile[self.actHDU].data.field('PSFPAR%i'%i))

            self.psffit[j] = isfitted[i]

        self._vprint(" Fit PSF parameters?\n", 2, history=True)
        for i in self.need[self.psftype]:
            self._vprint("  PSFPAR%i - %s\n" %(i, isfitted[i]), 2, history=True)


        if not psfcols == []:
            old_head = self.prmfile[self.actHDU].header 

            new_cols = self.prmfile[self.actHDU].columns+pyfits.ColDefs(psfcols)
            self.prmfile[self.actHDU] = \
                pyfits.new_table(new_cols, header=old_head)



        self.lut = lut     ## `None` or (2D-)array containing LUT
                           ## !!! NEEDS TO BE UPDATED !!!

        if self.lut == None:
            self._vprint("No look-up table provided.\n", 2)
        else:
            self._vprint("Using look-up table.\n", 2)
        self._vprint("\n", 2)


        
    def init_sources(self, posfit, fluxfit, varpos=False):
        """
        Method to prepare the catalogue of resolved sources.
        The method uses the PRM-file to obtain the IDs and initial
        guesses of positions and spectra of the sources. Actually,
        each source for which a column 'FLUX<SOURCE_ID>' exists in
        the ActiveModel-extension is considered. Unless `varpos` is
        set to True, the method looks for an extension AllSources,
        a FITS-table giving IDs and positions of all the sources.

        To allow a straightforward implementation of apriori knowledge
        about the source positions (e.g HST imaging), the true pixel
        positions of the sources are calculated as follows:
            x_pix =  A*x_in + B*y_in + C
            y_pix = -B*x-in + A*y_in + D
        Unless `posfit` is set to 3 or `varpos` is activated, the
        columns 'PARA', 'PARB', 'PARC' and 'PARD' must be present in
        the ActiveModel-extension.

        Parameters
        ----------
        posfit : integer
            posfit can be either 0, 1, 2 or 3 and determines how the
            source positions are varied during the fit:
            * 0 - do not fit positions
            * 1 - fit C & D (global shift)
            * 2 - fit A,B,C and D (global shift & scaling/rotation)
            * 3 - fit individual positions;
            If `posfit`=3 or `varpos`=True, A is automatically set to 1
            while B, C and D are set to 0. No need to provide A,B,C & D
            vin the PRM-file.
        fluxfit : boolean
            Whether the fluxes of the sources are fitted.
        varpos : boolean, optional
            In case individual source positions have been pre-
            determined as a function of wavelngth, set `varpos` to
            True. In this case, the routine will serach for columns
            XPOS<SOURCE_ID> and YPOS<SOURCE_ID> in the ActiveModel-
            extension instead of looking for an extension AllSources.
        """

        self._vprint("*"*31+"SOURCES"+"*"*30+"\n", 2)
        
        self.fluxfit    = fluxfit
        self.posfit     = posfit


        ## Get all resolved sources in fit
        self._vprint(" Grabbing 'resolved' sources from PRM-file...", 2)
        self.Nind    = 0   ## The number of independent resolved components
        self.sources = {}  ## The ID for every source
        sourcestring = ""
        for name in self.prmfile[self.actHDU].data.names: 
            if name[:4] == 'FLUX':
                source_id = int(name[4:])
                self.sources[source_id] = source_id
                self.Nind += 1

                sourcestring += "%9i" %source_id
                if numpy.mod(self.Nind,5) == 0:
                    sourcestring += "\n"
        self.Nflx = self.Nind + 2

        self._vprint("OK\n", 2)
        self._vprint("  %i source(s) found:\n" %self.Nind, 2)
        self._vprint("%s\n" %sourcestring, 2)


        ## getting spectra of resolved sources from prm-file
        self._vprint(' Grabbing source spectra...', 2)
        self.fluxes = numpy.zeros((2, self.Nimages, self.Nflx), dtype='float32')

        ## sort independent components by increasing ID
        self.sorted_ind = sorted(self.sources.keys())

        i = 0
        for j in self.sorted_ind:
            self.fluxes[:,:,i] = \
                deepcopy(self.prmfile[self.actHDU].data.field('FLUX%i' %j))
            i += 1

        self._vprint("OK\n", 2)


        ## getting positions of resolved sources from prm-file
        self._vprint(' Grabbing source coordinates...', 2)

        self.varpos = False              ## Individual source positions might be
        if self.posfit == 3 or varpos:   ## determined as fct of wavelength,
            self.varpos=True             ## but not fitted

        if self.varpos:
            self.xpos =numpy.zeros((self.Nimages,self.Nind),dtype='float32')
            self.ypos =numpy.zeros((self.Nimages,self.Nind),dtype='float32')

            i = 0
            for j in self.sorted_ind:
                self.xpos[:,i] = \
                    deepcopy(self.prmfile[self.actHDU].data.field('XPOS%i' %j))
                self.ypos[:,i] = \
                    deepcopy(self.prmfile[self.actHDU].data.field('YPOS%i' %j))
                i += 1
                
            self.pospars = numpy.array([1,0,0,0], dtype='float32')
            self.map     = numpy.arange(self.Nind, dtype="int32")
            self.bbfl    = None

        else:            
            self.pospars = numpy.zeros((self.Nimages, 4), dtype='float32')
            i = 0
            for pospar in ['PARA', 'PARB', 'PARC', 'PARD']:
                self.pospars[:,i] = \
                    deepcopy(self.prmfile[self.actHDU].data.field(pospar))
                i += 1

            ## get coordinates of all 'resolved' and 'merged' sources
            self.srcHDU = self.prmfile.index_of("AllSources")

            id_true = self.prmfile[self.srcHDU].data.field("ID").tolist()
            id_comb = self.prmfile[self.srcHDU].data.field("ID_COMB")

            self.xp_all = self.prmfile[self.srcHDU].data.field("XPOS")
            self.yp_all = self.prmfile[self.srcHDU].data.field("YPOS")
            self.fl_all = self.prmfile[self.srcHDU].data.field("FLUX")
            self.fitopt = self.prmfile[self.srcHDU].data.field("FITOPT")

            for i in xrange(self.fitopt.size): ## add 'merged' sources to dict.
                if self.fitopt[i] == 1:
                    self.sources[id_true[i]] = id_comb[i]
            self.Npos = len(self.sources)

            self.xpos = numpy.zeros((self.Npos,), dtype="float32")
            self.ypos = numpy.zeros((self.Npos,), dtype="float32")
            self.bbfl = numpy.zeros((self.Npos,), dtype="float32")
            self.map  = numpy.zeros((self.Npos,), dtype="int32")

            i = 0
            for j in sorted(self.sources.keys()):
                self.xpos[i] = self.xp_all[id_true.index(j)]
                self.ypos[i] = self.yp_all[id_true.index(j)]
                self.bbfl[i] = self.fl_all[id_true.index(j)]
                self.map[i]  = self.sorted_ind.index(self.sources[j])
                i += 1
            
        self._vprint("OK\n", 2)

        self._vprint(" Fit fluxes? - %s\n" %bool(self.fluxfit), 2, history=True)
        self._vprint(" Fit global offset? - %s\n" %(self.posfit in [1,2]), 2,
                     history=True)
        self._vprint(" Fit global rotation? - %s\n" %(self.posfit == 2), 2,
                     history=True)
        self._vprint(" Fit individual positions? - %s\n\n" %(self.posfit==3), 2,
                     history=True)




    def init_background(self, skyfit, unresfit=None):
        """
        Method to initialize the background model that is used in the fit.
        Again, the method makes use of the PRM-file, which must provide an
        initial guess of the sky spectrum in the ActiveModel extension.
        The method also searches for a keyword 'HASUNRES' in the header of
        the primary extension, indicating whether an unresolved stellar
        component should be included. If absent, the routine does not
        include such a component.
        If 'HASUNRES' is set to 1, an unresolved stellar component is
        included. In that case, an intial guess of its spectrum must be 
        provided in the ActiveModel extension. Also, an extension named
        AllSources must be present (same as required by method
        `init_sources`), containing IDs, positions and fluxes (in arbitrary
        units, only relative values matter) of all sources in the unresolved
        component in a FITS-table.
        
        Parameters
        ----------
        skyfit : boolean
            A flag telling the routine whether spectrum of the sky background
            is a free parameter in the fit.
        unresfit : boolean, optional
            A flag telling the routine whether the spectrum of the unresolved
            stellar component is a free parameter in the fit. Only needed if
            unresolved component is  available.
        """
        self._vprint("*"*29+"BACKGROUND"+"*"*29+"\n", 2)
         
        self.skyfit  = skyfit
        self.unresfit = unresfit

        ## Getting sky spectrum
        self._vprint(" Grabbing sky spectrum from PRM-file...", 2)
        self.fluxes[:,:,-2] = \
            deepcopy(self.prmfile[self.actHDU].data.field("SKY"))
        self._vprint("OK\n", 2)
        self._vprint(" Fit sky spectrum? - %s" %bool(self.skyfit), 2,
                     history=True)
        self._vprint("\n", 2)
        

        ## Getting unresolved component (if available)
        self._vprint(" Searching for unresolved component in PRM-file...", 2)

        if not self.prmfile[0].header.has_key("HASUNRES"):
            self.prmfile[0].header.update("HASUNRES", 0)

        if not self.prmfile[0].header["HASUNRES"]:
            self._vprint("FAILED\n", 2)
        elif self.prmfile[0].header["HASUNRES"] and self.varpos:
            self.prmfile[0].header.update("HASUNRES", 0)
            self._vprint("OK\n", 2)
            self._vprint("***WARNING: Cannot include unresolved component***\n",
                         0)

        if self.prmfile[0].header["HASUNRES"]:

            self.fluxes[:,:,-1] = \
                deepcopy(self.prmfile[self.actHDU].data.field("UNRESOLVED"))
            
            Nunrsv = (self.fitopt==0).sum()

            self.xpos.resize((self.Npos+Nunrsv,))
            self.ypos.resize((self.Npos+Nunrsv,))
            self.bbfl.resize((self.Npos+Nunrsv,))
            self.map.resize((self.Npos+Nunrsv,))

            self.xpos[-Nunrsv:] = self.xp_all[self.fitopt==0]
            self.ypos[-Nunrsv:] = self.yp_all[self.fitopt==0]
            self.bbfl[-Nunrsv:] = self.fl_all[self.fitopt==0]
            self.map[-Nunrsv:]  = -1

            self._vprint("OK\n", 2)
            self._vprint(" Fit unresolved component? - %s\n" \
                             %bool(self.unresfit), 2, history=True)

        else:
            self.unresfit = False
            self.all      = None
            self._vprint(" Unresolved component not included in fit.\n", 2)

        self._vprint("\n", 2)



    def fit_cube(self, ftest=False, fsnrcut=3., update_guesses=10,
                 lsreject=False, clip=5., method=0, fitrad=-1,
                 flux_cv=0.0001, sigma_cv=0.05, ncoadd=1):
        """
        Method to fit the complete data cube. It embeds the FitArray-class,
        of which an instance is initialized prior to the fit. Several
        routines of the class are then called to provide the instance with
        initial guesses of the fit-parameters. The exec-fit routine of
        FitArray is then used to fit every image in the cube.
        The method can also check the significance of low S/N sources by
        doing an F test (!!!this part needs to be updated!!!)

        Parameters
        ----------
        ftest : boolean, optional
            Flag that tells the routine whether it should perform an F-test
            for the low S/N sources
            (!!!this part needs to be updated!!!). 
        fsnrcut : float, optional
            If the `ftest`-flag is enabled, an F-test is performed for all
            sources whose estimated S/N is below `fsnrcut`
            (!!!this part needs to be updated!!!)
        update_guesses : integer, optional
            Specifies whether the initial guesses are modified before they
            are fed to the FitArray.exec_fit-routine. If `0`, the guesses
            are not modified. If `>0`, the last `update_guesses` fits are
            used to modify the initial guess. See _update_guesses-method
            for details.
        lsreject : boolean, optional
            A flag that tell the routine whether fluxes with low S/N should
            be automatically rejected.
            (!!!this part needs to be updated!!!)
        clip : float, optional
            Determines efficiency of the bad-pixel cleaning. All pixel that
            deviate by more than `clip` sigma are rejected. Sigma is the
            weighted rms deviation between data and model.
        method : int, optional
            Defines how the fluxes are fitted. If `method` is set to 0 or 1,
            all fluxes are fit simultaneously, if `method` is set to 2, the
            sources are fitted individually.
        fitrad : float, optional
            The radius (in spaxels) out to which the sources are fitted if 
            `method` is set to 1 or 2. If `method` is set to 0, sources are
            fitted over the whole FoV.
        flux_cv : float, optional
            Convergence criterion used if `method` is set to 1 or 2. If the
            flux of a source varies by less than `flux_cv` times the fitted
            value, the source is considered converged.
        sigma_cv : float, optional
            Convergence criterion used if `method` is set to 1 or 2. If the
            flux of a source varies by less than `sigma_cv` times its noise,
            the source is considered converged.
        ncoadd : int, optional
            Instead of fitting the PSF and/or the source positions in every
            slice, they can be fitted using the average of `ncoadd` slices.
        """
        self._vprint(26*"#"+"FITTING DATACUBE"+26*"#"+"\n", 1)

        if update_guesses == 0:
            force = False
        else:
            force = True

        self._vprint(" No. of images in fit: %i\n" %self.Nimages, 2)
        self._vprint(" Rejection of low S/N-sources? - %s\n" %lsreject, 2)
        self._vprint(" Performing F-test? - %s\n" %ftest, 2)

        self.ncoadd = ncoadd

        ## Create an instance of the fitting routine:
        fitter = StarFit(self.size[1:], self.map, self.psftype, self.psffit,
                         self.fluxfit, self.posfit, self.skyfit, self.unresfit,
                         lowflux_reject=lsreject, verbose=self.vb,
                         makemod=self.outcube[0], makeres=self.outcube[1],
                         clip=clip, method=method, fitrad=fitrad,
                         flux_cv=flux_cv, sigma_cv=sigma_cv)


        ## Fit all images in the data cube:
        for i in xrange(self.Nimages):

            if force:
                ## determine new initial guesses
                self._update_guesses(i, update_guesses)
  
            if self.ncoadd > 1 and numpy.mod(i, self.ncoadd) == 0:

                fitter.psfvar = self.psffit.any()
                fitter.posfit = self.posfit

                subdata  = self.data[i:i+self.ncoadd]
                subsigma = self.sigma[i:i+self.ncoadd]

                weights  = (1/subsigma)**2
                weights[self.mask[i:i+self.ncoadd]==1] = 1e-9

                mean_data  =numpy.average(subdata, weights=weights, axis=0)
                mean_sigma =numpy.sqrt(1./weights.sum(axis=0))
                mean_mask  =numpy.all(self.mask[i:i+self.ncoadd], axis=0)
                mean_fluxes=numpy.mean(self.fluxes[1,i:i+self.ncoadd], axis=0)

                fitter.update_data(mean_data, mean_sigma, mean_mask)                
                fitter.update_fluxes(mean_fluxes,broadband=self.bbfl,force=True)
                if i == 0:
                    self.bbfl = None

                if self.varpos:
                    fitter.update_pospars(self.pospars, force=False)
                    fitter.update_positions(self.xpos[i], self.ypos[i],
                                            force=force)
                else:
                    fitter.update_pospars(self.pospars[i], force=force)
                    fitter.update_positions(self.xpos, self.ypos, force=False)
                fitter.update_psfpars(self.psfpars[i], force=force)

                fitter.exec_fit()
                
                if self.psffit.sum() > 0:
                    self.psfpars[i:i+self.ncoadd, self.psffit] \
                        = fitter.psfpars[1,self.psffit]
                if self.posfit == 1:
                    self.pospars[i:i+self.ncoadd,2:] \
                        = fitter.pospars[1,2:]
                if self.posfit == 2:
                    self.pospars[i:i+self.ncoadd] \
                        = fitter.pospars[1]
                if self.posfit == 3:
                    self.xpos[i:i+self.ncoadd] = fitter.xpos[1]
                    self.ypos[i:i+self.ncoadd] = fitter.ypos[1]

                fitter.psfvar = False
                fitter.posfit = 0
                fitter.Nima -= 1

            ## update data:
            fitter.update_data(self.data[i], self.sigma[i], self.mask[i])

            ## provide initial guesses for fitted pars./update fixed pars.:
            fitter.update_fluxes(self.fluxes[1,i], broadband=self.bbfl,
                                 force=force)
            if i == 0:
                self.bbfl = None

            if self.varpos:
                fitter.update_pospars(self.pospars, force=False)
                fitter.update_positions(self.xpos[i], self.ypos[i], force=force)
            else:
                fitter.update_pospars(self.pospars[i], force=force)
                fitter.update_positions(self.xpos, self.ypos, force=False)
            fitter.update_psfpars(self.psfpars[i], force=force)

            ## fit image:
            fitter.exec_fit()
               
            ## get results of fit:
            self.fluxes[1,i] = fitter.fluxes[1]

            if self.psffit.sum() > 0:
                self.psfpars[i, self.psffit] = fitter.psfpars[1,self.psffit]
            if self.posfit == 1: self.pospars[i,2:] = fitter.pospars[1,2:]
            if self.posfit == 2: self.pospars[i] = fitter.pospars[1]
            if self.posfit == 3:
                self.xpos[i] = fitter.xpos[1]
                self.ypos[i] = fitter.ypos[1]

            self.chisq[1,i] = fitter.chisq[-1]
            self.mask[i]  = fitter.outmask


            ## get residuals and model image if requested
            if self.outcube[0]: self.model[i] = fitter.model
            if self.outcube[1]: self.resi[i] = fitter.residuals

        fitter.__del__()

        self._vprint('\n', 1)
              

## F test part is not up-to-date
        if ftest: ## Perform F test if requested
            self._vprint(12*" "+"*** Performing F test for low-S/N sources ***"\
                             + 11*" "+"\n", 1)

            self.calc_snr()
            self.deleted = 0
            infit        = numpy.ones((len(self.sources+2),), dtype='bool')

            Nlow = (self.snratios<=fsnrcut).sum()

            sorted_indices = self.snratios.argsort()

            for k in xrange(Nlow):

                i  = sorted_indices[k]
                id = self.sources[i]
                
                infit[i] = False

                self._vprint("\nSource #%i (S/N=%i):\n" %(id,self.snratios[i]),1)

                if self.fluxes[1,:,i].max() > 0:
                    sources = list(numpy.asarray(self.sources)[infit])

                    self.fluxes[0] = self.fluxes[1]
                    self.chisq[0]  = self.chisq[1]

                    ## create copies of residual/model cube (if requested)
                    if self.outcube[0]: tmp_model = self.model.copy()
                    if self.outcube[1]: tmp_resi  = self.resi.copy()

                    ## Start fitting routine:
                    ## UPDATE: New order of provided parameters!!!
                    Ffitter = StarFit(self.size[1:], self.map, self.psftype,
                                      1, 0, self.skyfit, 0*self.psffit,
                                      all=self.all, verbose=self.vb,
                                      makemod=self.outcube[0],
                                      makeres=self.outcube[1])

                    for j in xrange(self.Nimages):

                        ## using previously fitted parameters as initial
                        ## guesses, _update_guesses not executed

                        ## update data arrays:
                        Ffitter.update_data(self.data[j], self.sigma[j],
                                            self.mask[j])

                        ## update initial guesses/fixed parameters:
                        Ffitter.update_fluxes(self.fluxes[1,j,infit],force=True)
                        Ffitter.update_positions(self.xpos[j,infit],
                                                 self.ypos[j,infit])
                        Ffitter.update_pospars(self.pospars[j])
                        Ffitter.update_psfpars(self.psfpars[j])

                        ## fit image:
                        Ffitter.exec_fit()             
               
                        ## get fit results:
                        self.fluxes[1,j,infit] = Ffitter.fluxes[1]
                        
                        if self.outcube[0]: tmp_model[j] = Ffitter.model
                        if self.outcube[1]: tmp_resi[j]  = Ffitter.residuals
                        self.chisq[1,j]  = Ffitter.chisq[-1]
    

                    Ffitter.__del__()

                    ## perform F test:
                    p2 = 1200*(self.Nind - self.deleted) + self.skyfit
                    n  = self.data.size - self.mask.sum()
                    F  = ((self.chisq[1].sum()-self.chisq[0].sum())/self.Nimages)/(self.chisq[0].sum()/(n-p2))


                    if special.fdtri(1200., n-p2, 0.95) > F:

                        self.deleted += 1
                    
                        self._vprint("F test: Source #%i rejected\n" %id, 1)

                        self.fluxes[1,:,i] = 0
                        
                        if self.outcube[0]: self.model = tmp_model.copy()
                        if self.outcube[1]: self.resi = tmp_resi.copy()

                    else:
                        self._vprint("F test: Source #%i accepted\n" %id, 1)
                        
                        self.fluxes[1,:,infit] = self.fluxes[0,:,infit]
                        self.chisq[1]  = self.chisq[0]
                       
                        infit[i] = True

                else:
                    self._vprint("F test: Source #%i undetected\n"  %id, 1)
                    self.deleted += 1



    def _update_guesses(self, i, n_comp=10):
        """
        Method uses the results of the last fits to update the parameters
        for the next image of the cube that is processed. 
        The general idea is to take the weighted mean of the last `n_comp`
        images and to use the chi2-values as weights. Different ways are
        used to update certain parameters:
        * object spectra: multiply by ratio of initial guess & best-fit
                          value to keep information about spectral features
                          present in initial guesses.
        * sky spectrum:   add difference between best-fit value & initial
                          guess to keep intensity of sky lines.
        * others:         use weighted mean of last n_comp images

        Parameters
        ----------
        i : integer
            index of the image that is processed next. The routine must
            be called prior to the fit of image `i`.
        n_comp : integer, optional
            The last `n_comp` fits are used to update the initial guess
            of image `i`.
        """

        ## select images that are used in the update process. If `i`<`n_comp`,
        ## the first `n_comp`-images of the cube are considered.
        b = max(0, i-n_comp)
        e = b+n_comp 

        images  = numpy.arange(self.Nimages)
        select  = numpy.logical_and(images>=b, images<e)

        ## if `i` is smaller than `n_comp`, no weigthing is performed
        if i>n_comp:
            weights = 1./self.chisq[1,select]
        else:
            weights = numpy.ones((n_comp,), dtype='float32')

        ## update source fluxes if they are fitted
        ## datapoints with zero input flux are masked out
        if self.fluxfit:
            mask   = numpy.array(2*[self.fluxes[0,select,:-2]==0])
            selflx = numpy.ma.array(self.fluxes[:,select,:-2], mask=mask)
            fct    = numpy.ma.average(selflx[1]/selflx[0], weights=weights,
                                      axis=0)
            self.fluxes[1,i,:-2] *= numpy.ma.filled(fct, 1.)

        ## update sky spectrum if it is fitted
        if self.skyfit > 0:
            self.fluxes[1,i,-2] += \
                numpy.average(self.fluxes[1,select,-2]-self.fluxes[0,select,-2],
                              weights=weights)
            
        ## update unresolved spectrum if it is fitted
        if self.unresfit > 0:
            self.fluxes[1,i,-1] *= \
                numpy.average(self.fluxes[1,select,-1]/self.fluxes[0,select,-1],
                              weights=weights)
        
        ## update source positions; update procedure depends on way in which
        ## positions are fitted.
        if self.posfit == 3 and numpy.mod(i, self.ncoadd) == 0:
            self.xpos[i] = \
                numpy.average(self.xpos[select], axis=0, weights=weights)
            self.ypos[i] = \
                numpy.average(self.ypos[select], axis=0, weights=weights)
        elif self.posfit == 2 and numpy.mod(i, self.ncoadd) == 0:
            self.pospars[i] = \
                numpy.average(self.pospars[select], axis=0, weights=weights)
        elif self.posfit == 1 and numpy.mod(i, self.ncoadd) == 0:
            self.pospars[i,2:] = \
                numpy.average(self.pospars[select, 2:], axis=0, weights=weights)

        ## update PSF parameters that are fitted.
        if self.psffit.max() > 0 and numpy.mod(i, self.ncoadd) == 0:
            self.psfpars[i,self.psffit] = \
                numpy.average(self.psfpars[select][:,self.psffit], axis=0,
                              weights=weights)



    def calc_snr(self, winsize=200):
        """
        Method estimates the S/N of the source spectra by dividing the median
        by the stddev in windows of len(`winsize`). The maximum value is used
        as the S/N.

        Parameters
        ----------
        winsize : integer, optional
           The size of the windows in which the S/N is calculated
        """
        self.snratios = numpy.zeros((self.Nind,), dtype='float32')

        for i in xrange(self.Nind):
            spec_i = self.fluxes[1,:,i].copy()

            j_min  = 0
            j_max  = winsize
            snrmax = 0.
            while j_max < self.Nimages:
                
                flux  = numpy.median(spec_i[j_min:j_max])
                sigma = spec_i[j_min:j_max].std()

                if flux/sigma > snrmax:
                    snrmax = flux/sigma

                j_min += 1
                j_max += 1

            self.snratios[i] = snrmax
 

 
    def write_results(self, backup=True):
        """
        Method saves the results of the fiting process in the PRM-file.
        If requested, a backup of the input values is stored in an
        extension named BackupModel (which is created if absent, otherwise)
        it is updated.
        If requested, the routine also saves/updates datacubes containing
        the masked pixels, the residuals and the best-fit model.

        Parameters
        ----------
        backup : boolean, optional
            Flag that indicates whether a backup of the input values should
            be created.
        """

        self._vprint(27*"#"+"SAVING RESULTS"+27*"#"+"\n", 2)        


        ## Updating PRM-file
        self._vprint("Writing fitted parameters to PRM-file...\n", 2)

        if backup: ## check if backup extension already exists,
                   ## otherwise append it.
            try:
                self.bupHDU = self.prmfile.index_of('BackupModel')
                self._vprint(" ...using extension 'BackupModel' ", 2)
            except KeyError:
                self._vprint(" ...created extension 'BackupModel' ", 2)
                self.prmfile.append(self.prmfile[self.actHDU].copy())
                self.bupHDU = len(self.prmfile)-1
                self.prmfile[self.bupHDU].header.update('EXTNAME','BackupModel')
            self._vprint("to backup input parameters.\n", 2)
        else:
            self._vprint(" ...no backup of input parameters requested.", 2)



        ## update column containing chi2-values
        self._write_column("CHISQ", self.chisq[1], backup)



        ## update columns with PSF parameters
        for i in self.need[self.psftype]:
            j    = self.need[self.psftype].index(i)
            name = "PSFPAR%i" %i
            if self.psffit[j]:
                self._write_column("PSFPAR%i" %i, self.psfpars[:,j], backup)


        update_pos = numpy.zeros((4,), dtype='bool') 
        if self.posfit == 1:   update_pos[2:] = True
        elif self.posfit == 2: update_pos[:]  = True

        i = 0
        for par in ['PARA', 'PARB', 'PARC', 'PARD']:
            if update_pos[i]:
                self._write_column(par, self.pospars[:,i], backup)
            i += 1
        
            
        ## update columns with source fluxes (& positions if fitted)
        i = 0
        for sid in self.sorted_ind:
            self._write_column('FLUX%i' %sid, self.fluxes[1,:,i], backup)
            if self.posfit == 3:
                self._write_column('XPOS%i' %sid, self.xpos[:,i], backup)
                self._write_column('YPOS%i' %sid, self.ypos[:,i], backup)
            i += 1

        ## update columns with background fluxes
        if self.skyfit:
            self._write_column('SKY', self.fluxes[1,:,-2], backup)
        if self.unresfit:
            self._write_column('UNRESOLVED', self.fluxes[1,:,-1], backup)


        ## save PRM-file
            if self.ncoadd > 1:
                self.prmfile[0].header.update("NCOADD", self.ncoadd,
                                              "No of images per PSF and pos bin")

        self.prmfile[self.actHDU].update()
        if backup:
            self.prmfile[self.bupHDU].update()
        self.prmfile.flush()
        self._vprint(" ...OK\n", 2)


        ## Saving datacube with masked pixels if requested
        if self.outcube[2]:
            self._vprint("Writing mask to file...", 2)
            maskHDU = pyfits.PrimaryHDU(self.mask)
            if os.path.exists(self.maskname):
                ow = ''
                self._vprint("EXISTS\n", 2)
                if self.ia>0:
                    while not ow in ['y', 'n']:
                        ow = raw_input(" Overwrite '%s' [y/n]? " %self.maskname)
                else:
                    self._vprint(" Overwrite '%s'[y/n]? y\n" %self.maskname, 2)
                    ow = 'y'
                if ow == 'y':
                    os.system('rm %s' %self.maskname)
                    maskHDU.writeto(self.maskname)
            else:
                maskHDU.writeto(self.maskname)
                self._vprint("OK\n", 2)


        ## Saving datacube with best-fit model if requested
        if self.outcube[0]:
            self._vprint("Writing model to file...", 2)
            modHDU = pyfits.PrimaryHDU(self.model)
            if os.path.exists(self.modelname):
                ow = ''
                self._vprint("EXISTS\n", 2)
                if self.ia>0:
                    while not ow in ['y', 'n']:
                        ow = raw_input(" Overwrite '%s' [y/n]? "%self.modelname)
                else:
                    self._vprint(" Overwrite '%s'[y/n]? y\n" %self.modelname, 2)
                    ow = 'y'
                if ow == 'y':
                    os.system('rm %s' %self.modelname)
                    modHDU.writeto(self.modelname)
            else:
                modHDU.writeto(self.modelname)
                self._vprint("OK\n", 2)


        ## Saving datacube with residuals if requested
        if self.outcube[1]:
            ow = False
            self._vprint("Writing residuals to file...", 2)
            resHDU = pyfits.PrimaryHDU(self.resi)
            if os.path.exists(self.resiname):
                ow = ''
                self._vprint("EXISTS\n", 2)
                if self.ia>0:
                    while not ow in ['y', 'n']:
                        ow = raw_input(" Overwrite '%s' [y/n]? "%self.resiname)
                else:
                    self._vprint(" Overwrite '%s'[y/n]? y\n" %self.resiname, 2)
                    ow = 'y'
                if ow == 'y':
                    os.system('rm %s' %self.resiname)
                    resHDU.writeto(self.resiname)
            else:
                resHDU.writeto(self.resiname)
                self._vprint("OK\n", 2)

        self._vprint("\n", 2)



    def _vprint(self, string, vb_min, history=False):
        """
        Method prints `string` if verbose-level is at least equal to
        `vb_min`.
        
        Parameters
        ----------
        string : string
            The string to be printed.
        vb_min : integer
            The verbose level required to trigger the printing of `string`
        history : boolean, optional
            If True, the string is also written into the history of the
            PRM-file header
        """
        if self.vb >= vb_min:
            sys.stdout.write(string)
            sys.stdout.flush()

        if history:
            self.prmfile[0].header.add_history(string.rstrip("\n"))



    def _update_hdr(self, key, value, comment=None):
        """
        Update primary header uf the PRM-file. See pyfits.Header.update
        for details.
        """
        self.prmfile[0].header.update(key, value, comment=comment)



    def _write_column(self, name, newdata, backup, format="E", vprint=3):
        """
        Update a column in the PRM-file.
        
        Parameters
        ----------
        name : string
            The name of the column
        newdata : array_like
            The updated column data
        backup : boolean
            If True, a backup of the old column data is created.
        format : string, optional
            Format of a newly created column
        vprint : integer, optional
            If verbose level is at least `vprint`, information about
            the success of the update process ifs printed out on the
            screen.
        """

        ACTCOLS = list(self.prmfile[self.actHDU].data.names)
        if backup:
            BUPCOLS = list(self.prmfile[self.bupHDU].data.names)

        ## Create new column if column `name` does not exist
        if not name in ACTCOLS:
            self._vprint(" ...creating Column '%s'..." %name, vprint)

            newcol =pyfits.Column(name=name, format=format, array=newdata)

            ## update active HDU of FITS-file
            oldhead =self.prmfile[self.actHDU].header
            newcols =self.prmfile[self.actHDU].columns+pyfits.ColDefs([newcol])
            self.prmfile[self.actHDU]=pyfits.new_table(newcols, header=oldhead)

        else:
            self._vprint(" ...updating Column '%s' " %name, vprint)
            self._vprint((43-len(name))*".", vprint)
            if backup: 
                if not name in BUPCOLS: 
                    ## Create new column if column `name` does not exist
                    pos    =ACTCOLS.index(name)
                    format =self.prmfile[self.actHDU].data.formats[pos]
                    array  =self.prmfile[self.actHDU].data.field(name)
                    bupcol =pyfits.Column(name=name, format=format, array=array)

                    ## update backup HDU of FITS-file
                    buphead =self.prmfile[self.bupHDU].header
                    bupcols =self.prmfile[self.bupHDU].columns \
                        + pyfits.ColDefs([bupcol])
                    self.prmfile[self.bupHDU] \
                        = pyfits.new_table(bupcols, header=buphead)
                else:
                    self.prmfile[self.bupHDU].data.field(name)[:] \
                        = self.prmfile[self.actHDU].data.field(name)

            self.prmfile[self.actHDU].data.field(name)[:] = newdata
        self._vprint("OK\n", vprint)
