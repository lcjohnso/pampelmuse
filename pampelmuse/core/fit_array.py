"""
fit_array.py
============

Provides
--------
class pampelmuse.core.fit_array.StarFit
function pampelmuse.core.fit_array.printoptions
function pampelmuse.core.fit_array.sum_by_group
function pampelmuse.core.fit_array.get_psf_data

Purpose
-------
The module implements the class 'StarFit', the core-routine of PampelMuse. Its
purpose is to extract the spectra of the sources and to fit their positions
and/or the PSF in the observed IFS data. The routine is designed to handle both
large (e.g MUSE-like) and small (e.g PMAS-like) data sets.

The analysis is started in the central layer of the provided datacube. After
its analysis is completed, the routine moves on to the next blue or red layer,
respectively, in an alternating manner.

In case an the PSF and/or positions are fitted, their fit is preceded by an
initial fit of the source fluxes using the initial guesses for PSF and/or
positions. A final fit of the source fluxes and the estimation of the flux
uncertainties complete the analysis of each layer.

Latest SVN revision
-------------------
152, 2014/03/05
""" 
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 152

import contextlib, itertools, logging, os, sys, time
import numpy as np, pyfits, scipy
from multiprocessing import cpu_count, Pool
from scipy import sparse, optimize
if int(scipy.__version__.split(".")[1])<10:
    logging.warning("Version of SciPy is <0.10, using sparse.linalg.lsqr instead of faster scipy.linalg.lsmr")
    from scipy.sparse.linalg import lsqr as lsmr
else:
    from scipy.sparse.linalg import lsmr
## lsmr does the same as lsqr but faster (i.e. the Max Power way), it will only work for scipy versions >=0.10

## required PampelMuse-packages
from pampelmuse.core.psf import Gaussian, Moffat


## for a nicer printing of numpy arrays, following suggestion from
## http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)


def sum_by_group(values, groups):
    """
    This is a fast algorithm for summing up the data in one
    array based on the groups provided in the second array.
    It is taken from http://stackoverflow.com/questions\
    /4373631/sum-array-by-number-in-numpy?lq=1
    
    Parameters
    ----------
    values : array_like, 1dim.
        The data that should be summed up.
    groups : array_like, 1dim.
        The groups according to which the data in 'values'
        should be summed. It mustt have the same length as
        'values'
    
    Returns
    -------
    values : nd_array
        The summed up data. Note that the entries are sorted
        according to the provided group numbers.
    groups : nd_array
        A sorted list of all the groups provided as input.
    """
    order = np.argsort(groups)
    groups = np.asarray(groups)[order]
    values = np.asarray(values)[order]
    values.cumsum(out=values)
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups


def get_psf_data(psf_instances, sigma):
    """
    For a list of PSF instances, the function calculates
    the relative intensities of the PSF profile, divides
    them by the provided uncertainties and returns them
    together with the spaxels for which the PSF profile
    is defined.
    NOTE that this method is supposed to be replaced by
    the instance StarFit.get_psf_data at some point. The
    function is also used by the SUBTRES routine.
    
    Parameters
    ----------
    psf_instances : list
        A list containing instances of a class defined in
        the package pampelmuse.core.psf
    sigma : nd_array
        The uncertainties of the spaxels for which the PSF
        profiles should be calculated.
        
    Returns
    -------
    results : list 
        For each PSF instance, the routine returns a list of len=3,
        containing the spaxels on which the PSF profile is defined,
        the relative intensities, normalized by the uncertainties
        of each spaxel, and the integrated profile flux over the whole
        field of view.
    """
    results = []
    for psf in psf_instances:

        if psf==None: ## sky component
            results.append([np.arange(sigma.size, dtype=np.int32),1./sigma, 1.])

        else: ## stellar component
            spaxels, fluxes = psf.get_flux()
            n = spaxels.size
            if n==0: logging.warning("Source #{0} entirely outside FoV.".format(psf.src_id))
            intflux = fluxes.sum()
            results.append([spaxels, fluxes/sigma[psf.used], intflux])

    return results



class StarFit(object):
    """
    The class pampelmuse.core.fit_array.StarFit implements the core-routine
    of PampelMuse. It uses a catalog of input sources, created e.g. using
    the dedicated subroutine 'INITFIT', to analyse a complete datacube in a
    layer-by-layer manner. The fitting process itself is split into three
    parts:
    
        (i) Fluxes:
        A linear least-squares fit is performed for all the resolved sources
        simultaneously. This is done using a sparse-matrix representation of
        the individual sources.

        (ii) PSF:
        The PSF is modelled currently as an analytical profile, usually
        assuming a Moffat-shaped PSF. The individual parameters that govern
        the shape of the profile can be optimized during the fit.

        (iii) Source positions:
        Usually, the source positions are derived from a set of reference
        coordinates via a coordinate formation that is defined by several
        parameters. These parameters can be optimized during the fit. The
        routine also supports direct fitting of source coordinates. However,
        this  is only recommended for individual stars instead of crowded
        stellar fields.
        
    Steps (ii) and (iii) are performed via a non-linear least-squares fit based
    on a Levenberg-Marquard approach.

    Note that the routine first analyzes a central image and than alternating
    proceedes to the red and blue. This is done because usually the highest
    signal is expected near the center of the observed wavelength range.
    """
    def __init__(self, ifs_data, sources, psf_attributes, maxrad=np.inf, osfactor=1, osradius=0, **kwargs):
        """
        Initialize an instance of the `StarFit'-class.
    
        Parameters
        ----------
        ifs_data : a pampelmuse.core.cube.Cube instance
            The data to be analyzed, provided as an instance of the dedicated
            'Cube' class of PampelMuse. Check that routines documentation for
            further information. 
        sources : a pampelmuse.core.sources.Sources-instance
            The required information (positions, magnitudes,...) about the
            sources that should be included in the optimization.
        psf : a pampelmuse.core.psf.psf_attributes-instance
            The required information about the PSF to be used in the opti-
            mization, i.e., type of PSF, initial guesses of shape parameters
            (i.e. variables), ...
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

        self.ifs_data = ifs_data

        self.sources = sources
        self.psf_attributes = psf_attributes
        
        ## Supersampling of central pixels
        self.osfactor = osfactor
        self.osradius = osradius

        ## Get basic PSF model: Moffat or Gaussian
        if self.psf_attributes.profile == "gauss":
            self.PSF = Gaussian
        elif self.psf_attributes.profile == "moffat":
            self.PSF = Moffat
        else:
            logging.critical("Invalid PSF type: '{0}'".format(self.psf_attributes.psftype))
            raise IOError, "PSF type not understood."


        ## initialize PSF-class for every source included
        self.PSF.set_transform(self.ifs_data.instrument.transform)
        
        self.psf_instances = []
        for i in xrange(self.sources.nsrc):
            if self.sources.status[i] == 4: self.psf_instances.append(None)
            else: self.psf_instances.append(self.PSF(Xc=self.sources.x[i], Yc=self.sources.y[i], mag=sources.weights[i], src_id=self.sources.ID[i], maxrad=maxrad,
                                                     osfactor=osfactor, osradius=osradius))

        ## Check if individual source positions should be fitted
        self.full_coordinates_fit = (self.sources.ifs_coordinates_free&(self.sources.ifs_coordinates_fitted==False)).any()
        if self.full_coordinates_fit:
            logging.warning("Fit of individual source positions requested.")
            self.update_coordinates = (self.sources.ifs_coordinates_fitted==False)
        else:
            self.update_coordinates = (self.sources.status==3)&(self.sources.ifs_coordinates_fitted==False)


    def __call__(self, layerRange=(0,-1), Ntrack=10, lastFitIndex=None, asSmallIFU=False, ncpu=1, maxiter=20, clip=5., **kwargs):
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
        self.clip    = clip    ## efficiency of bad pixel masking - WARNING: Has no effect yet.

        self.start, self.stop = layerRange
        self.Nlayer = self.stop - self.start + 1
        
        logging.info("Requested range to be analysed: {0}-{1}".format(self.start,self.stop))

        ## check where to start
        if lastFitIndex == None:
            self.i = 0 ## iterator over all layers
        else:
            self.i = lastFitIndex

        self.Ntrack = Ntrack

        ## Number of iterations per layer. Note that no iteration is needed if only fluxes are fitted
        self.maxiter        = int(maxiter)
        if self.psf_attributes.contains_free_parameters==False and self.sources.contains_free_coordinates==False:
            self.maxiter = 1

        if self.maxiter > 1:
            logging.info("Maximum iterations per layer: {0}".format(self.maxiter))
        else:
            logging.info("No iteration on individual layers performed.")
            self.maxiter = 1

        ## whole array with spectra needs to be loaded
        self.sources.load_spectra()

        ## if parameters (of the PSF or the coordinate transformation) are fitted, it must be ensured that the instances
        ## have correct dispersion information        
        if self.psf_attributes.ndisp != self.Nlayer:
            self.psf_attributes.resize(self.Nlayer)
            
        if self.sources.transformation.valid and self.sources.transformation.ndisp != self.Nlayer:
            self.sources.transformation.resize(self.Nlayer)

        ## list that will contain the indices of the last 'Ntrack' images that were analysed.
        self.last_fitted_layers = []

        ## recover last processed layers in previous fit (if any)
        j = self.i-2*self.Ntrack
        while j < self.i:
            if j >= 0:
                self.last_fitted_layers.append((self.Nlayer>>1) - 1 - int(j/2) + j*np.mod(j, 2) + np.mod(self.Nlayer, 2))
            j += 1

        self.asSmallIFU = asSmallIFU

        ## for large IFS data, get indices of sources used to fit PSF/positions
        if not self.asSmallIFU: self.psf_indices = np.union1d(np.flatnonzero(self.update_coordinates), self.sources.psf_indices)

        ## if individual source positions should be fitted, we need to load the initial guesses for the positions of all sources on all layers
        ## into memory from the start
        if self.full_coordinates_fit:
            self.sources.load_ifs_coordinates()

        ## Check how many CPU's should be used
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
            self.jj = (self.Nlayer>>1) - 1 - int(self.i/2) + self.i*np.mod(self.i, 2) + np.mod(self.Nlayer, 2)

            ## The layer of the cube to be loaded
            self.kk = self.jj + self.start

            ## A common prefix used for all logging messages related to the
            ## analysis of the current layer.
            self.prefixString = "[{0:0{1}d}/{2}]".format(self.i, int(np.log10(self.Nlayer))+1, self.Nlayer)

            ## call fitting routine
            logging.info("{0} Starting fit for layer #{1}.".format(self.prefixString, self.kk))
            self.iter_fit_layer()

            self.i += 1


    def iter_fit_layer(self):
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
        self.update_parameters() 

        ## load IFU coords for current layer into memory unless individual
        ## positions are fitted
        if not self.full_coordinates_fit:
            self.sources.load_ifs_coordinates(layers=self.jj)

        ## update PSF instances
        ## -with PSF parameters
        i = 0 
        for prm in self.psf_attributes.names:
            getattr(self.PSF, "set_{0}".format(prm))(self.psf_attributes.data[self.jj, i, int(self.psf_attributes.fitted[i])])
            i += 1
        
        ## -with parameters of coordinate transformation (if available)
        if self.sources.transformation.valid:
            i = 0
            for prm in self.sources.transformation.names:
                getattr(self.PSF, "set_{0}".format(prm))(self.sources.transformation.data[self.jj, i, int(self.sources.transformation.fitted[i])])
                i += 1

        ## -with direct positions if required
        i  = 0
        ll = self.jj*self.full_coordinates_fit ## spectral index in sources.ifs_coordinates
        for psf in self.psf_instances:
            if psf==None: continue
            if self.sources.ifs_coordinates_fitted[i]:
                psf.set_xc(self.sources.ifs_coordinates[ll,i,0,1])
                psf.set_yc(self.sources.ifs_coordinates[ll,i,1,1])
            elif self.sources.ifs_coordinates_free[i]:
                psf.set_xc(self.sources.ifs_coordinates[ll,i,0,0])
                psf.set_yc(self.sources.ifs_coordinates[ll,i,1,0])
            i += 1

        ## WORK ON IFS DATA
        data  = self.ifs_data.data.layer(self.kk)
        sigma = self.ifs_data.sigma.layer(self.kk)

        ## check if mask is available, otherwise only NaN pixels will be masked
        if self.ifs_data.mask==None:
            mask = (data!=data)
        else:
            mask = self.ifs_data.mask.layer(self.kk)&(data!=data)
            
        self.data = data[mask==0]
        if self.ifs_data.is_var:
            self.sigma = np.sqrt(sigma[mask==0])
        else:
            self.sigma = sigma[mask==0]
            
        logging.debug("[@{0}] {1} masked pixels".format(self.kk, (mask==1).sum()))

        ## Set spaxel to (x,y) transformation of the PSF-instances, using the transformation
        ## of the Instrument instance and the mask
        self.PSF.set_transform(self.ifs_data.instrument.transform[mask==0])

        ## Reset possibly excisting unresolved component from last layer
        self.unresolved_psf = None

        ## Start iteration
        n_iter = 1
        while True: 
            
            if self.maxiter > 1:
                logging.info("[@{0}] Iteration {1} (of max. {2})".format(self.kk, n_iter, self.maxiter))

            ## create model matrix & update noise values
            self.build_matrix()
            self.sources.data[self.jj,self.valid_components,1] = (1./np.sqrt(self.Apsf.multiply(self.Apsf).sum(axis=0)).getA1())/self.fluxfactors[self.valid_components]

            ## initial fit of fluxes
            bestFit = lsmr(self.Apsf, self.current)
            self.sources.data[self.jj,self.valid_components,0] += bestFit[0]/self.fluxfactors[self.valid_components]

#            indata = np.zeros((self.ifs_data.instrument.transform.shape[0],), dtype=np.float32)
#            indata[mask==False] = np.asarray(self.current, dtype=np.float32)
#            outdata = self.ifs_data.instrument.make_grid(indata)
#            outHDU = pyfits.PrimaryHDU(outdata)
#            outHDU.writeto("{0}_{1}.fits".format(self.jj, n_iter), clobber=True)
            
            ## determine relative change in fluxes
            delta_flux = (bestFit[0]/self.fluxfactors[self.valid_components])/self.sources.data[self.jj,self.valid_components,0]

            ## some logging
            logging.info("[@{0}] Chi**2 of current fit: {1:.1f}".format(self.kk, bestFit[4]**2))
            with printoptions(precision=4, suppress=True):
                logging.debug("[@{0}] Relative changes in fluxes: {1}".format(self.kk, delta_flux))

            ## convergence criterion: median relative change in fluxes <1*10^-3
            if np.median(abs(delta_flux)) < 1e-3:
                break

            ## fit PSF/positions if requested            
            if self.psf_attributes.contains_free_parameters or self.sources.contains_free_coordinates:

                ## PSF/positions fit
                if self.asSmallIFU: self.parameter_fit_on_small_ifu()
                else: self.parameter_fit_on_large_ifu()

            else:
                break

            n_iter += 1
            
            if n_iter > self.maxiter:
                if self.maxiter > 1:
                    logging.warning("[@{0}] Max. number of iterations reached.".format(self.kk))
                break
                 
        
        ## update inventory of last fits
        self.last_fitted_layers.append(self.jj)
        if len(self.last_fitted_layers)>2*self.Ntrack:
            self.last_fitted_layers.remove(self.last_fitted_layers[0])
            

    @property
    def model(self):
        """Create the current best-fit model"""
        return self.Apsf*sparse.dia_matrix((self.sources.data[self.jj,self.valid_components,0]*self.fluxfactors[self.valid_components],[0]),
                                           shape=(self.valid_components.sum(), self.valid_components.sum()))


    @property
    def current(self):
        """Determine the current residuals"""
        ## self.model.sum(axis=1) returns numpy.matrixlib.defmatrix-object
        ## `getA1' returns a flattened numpy.ndarray
        return (self.data/self.sigma)-self.model.sum(axis=1).getA1()

    
    def build_matrix(self, update_unresolved=True, log=True):
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
        update_unresolved : boolean, optional
            If False, the matrix column containing the unresolved stellar
            component (if included) is not recomputed, but taken from a
            previous calculation.
        log : boolean, optional
            If True, the routine method will print a logging.ERROR-message
            every time a source whose PSF is completely outside the FoV is
            encountered.
        
        Returns
        -------
        a scipy.sparse.csc_matrix-instance
            This is the PSF coefficient matrix that is used in the linear
            least-squares fit to determine the source fluxes.
        """
        ##  if unresolved component exists and should not be updated, extract sublist of resolved sources
        if self.sources.includes_unresolved and not update_unresolved:
            psf_instances = [self.psf_instances[i] for i in np.flatnonzero(self.sources.status>0)]
        else: ## otherwise hust use full ist
            psf_instances = self.psf_instances

        ## Initialize parallelization (if requested)
        if self.ncpu > 1:
            pool    = Pool(self.ncpu)
            threads = []
            
            ## determine which sources should be calculated by which CPU
            m = (len(psf_instances)//self.ncpu) + 1
            
            for n in xrange(self.ncpu):
                imin = n*m
                imax = min((n+1)*m, self.sources.nsrc)
                
                threads.append(pool.apply_async(get_psf_data, args=(psf_instances[imin:imax], self.sigma)))
        
            pool.close()
            pool.join()
            
            # TODO: parallelization seems not to work when calling class methods. The different packages are still processed sequentially
            results = list(itertools.chain.from_iterable([threads[n].get() for n in xrange(self.ncpu)]))
        else:
            results = get_psf_data(psf_instances, self.sigma)

        if self.sources.includes_unresolved: ## handle the unresolved component
            if update_unresolved  or self.unresolved_psf==None: ## just store the unresolved sources as a list for later re-use
                self.unresolved_psf = [results[i] for i in np.flatnonzero(self.sources.status==0)]
            else: ## add the existing unresolved PSFs at the right indices
                indices = np.flatnonzero(self.sources.status==0)
                for i in xrange(len(self.unresolved_psf)): results.insert(indices[i], self.unresolved_psf[i])

        ## check if any row does not contribute at all
        self.fluxfactors, specrows = sum_by_group([r[-1] for r in results], self.sources.specrow)
        self.valid_components = (self.fluxfactors>0)
        self.component_mask = (self.valid_components==False)
    
        if self.component_mask.any():
            logging.error("Encountered component(s) that contribute zero flux: {0}".format(specrows[self.component_mask]))

        ## fill the matrix: the indices and intensities of all PSF instances are combined and serve as row indices and values of
        ## the matrix. To obtain the column indices of the matrix, we use the property 'specrow' of the Sources class and correct
        ## for components with zero flux contribution (if they exist)
        i0 = np.hstack((r[0] for r in results)) ## the row indices

        i1 = np.hstack((len(results[i][0])*[self.sources.specrow[i]] for i in xrange(self.sources.nsrc))) ## the column indices
        for bad_index in specrows[self.component_mask][::-1]: i1[i1>bad_index] -= 1 ## correct for components that do not contribute

        f  = np.hstack((r[1] for r in results)) ## the fluxes
        f /= self.fluxfactors[self.valid_components][i1] ## bring all components to an integrated flux of 1

        ## first, a coo-matrix is constructed. Note that it handles the case when the input data contains multiple entries for the
        ## some indices (row, column). After construction, the matrix is converted to a csc matrix to speed up future usage of it.
        self.Apsf = sparse.coo_matrix((f, (i0, i1)), shape=(self.data.size, self.sources.ncmp-(self.component_mask).sum())).tocsc()
        

#   TODO: Enable parallelization for this method
    def get_psf_data(self, imin, imax):
        """
        Calculate the intensity arrays for the PSF instances in the
        provided interval [imin, imax) and return them together with
        the indices of the spaxels for which the PSF profile is defined
        and the integrated PSF flux over the field of view.
        
        Parameters
        ----------
        imin : int
            The index of the first source/PSF instance that is used.
        imax : int
            The index of the first source/PSF instance that is not
            used anymore.
        
        Returns
        -------
        results : list 
            For each PSF instance, the routine returns a list of len=3,
            containing the spaxels on which the PSF profile is defined,
            the relative intensities, normalized by the uncertainties
            of each spaxel, and the integrated profile flux over the whole
            field of view.
            
        Notes
        ----
        This method is currently not used because parallelization does
        not seem to work when a class methood is called. Instead, the
        function get_psf_data as defined above is used.
        """
        results = []

        for psf in self.psf_instances[imin:imax]:
            
            if psf==None: ## sky component
                results.append([np.arange(self.data.size, dtype=np.int32), 1./self.sigma, 1.])

            else: ## stellar component
                spaxels, fluxes = psf.get_flux()
                n = spaxels.size
                if n==0: logging.warning("[@{0}] Source #{1} entirely outside FoV.".format(self.kk, psf.src_id))
                intflux = fluxes.sum()
                results.append([spaxels, fluxes/self.sigma[psf.used], intflux])

        return results


    def parameter_fit_on_large_ifu(self):
        """
        Perform the PSF fit for an IFU with many spaxels. In this case,
        a non-linear least-squares fit is performed for every PSF source
        independently. Depending on the respective fit-flags, either the
        parameters of the PSF and/or the source coordinates are fitted.
        The results of the individual fits are afterwards used to update
        the PSF and/or the coordinate transformation.
        """
        ## Initialize parallelization (if requested)
        # TODO: enable parallelization again and make it work
        # if self.ncpu > 1:
        #     pool    = Pool(self.ncpu)
        #     threads = []

        ## check if (and which) PSF parameters should be fitted
        psf_names  = []
        psf_values = []
        for i in xrange(self.psf_attributes.npars):
            if self.psf_attributes.free[i] and not self.psf_attributes.fitted[i]:
                psf_names.append(self.psf_attributes.names[i])
                psf_values.append(self.psf_attributes.data[self.jj,i,0])

        ## Initialize arrays that will hold results
        results    = []
        posfitted  = [] ## for each source, a boolean flag indicating if the source coordinates are fitted
        psffitted  = [] ## for each source, a boolean flag indicating if the PSF parameters are fitted

        ## Loop over all PSF sources; add source to residuals, fit resulting array and subtract source again.
        for i in self.psf_indices: 

            flux  = self.sources.data[self.jj,self.sources.specrow[i],0]
            names = [] ## the names of the free parameters for this PSF
            x0    = [] ## the initial guesses for the parameter fit for this PSF

            if self.update_coordinates[i]:
                        
                ## get index of source in array of IFS coordinates, save them to place updated coordinates at the right index
                assert self.sources.ifs_coordinates_loaded[i], "PSF source has no IFU coordinates available."
                posindex = self.sources.ifs_coordinates_loaded[:i].sum()

                names.extend(["x_c", "y_c"])
                x0.extend([self.sources.ifs_coordinates[self.jj*self.full_coordinates_fit, posindex, 0, 0],
                           self.sources.ifs_coordinates[self.jj*self.full_coordinates_fit, posindex, 1, 0]])
                posfitted.append(True)
            else:
                posfitted.append(False)

            if i in self.sources.psf_indices:
                names.extend(psf_names)
                x0.extend(psf_values)
                psffitted.append(True)
            else:
                psffitted.append(False)

            # if self.ncpu > 1:
            #     threads.append(pool.apply_async(self.wrapper_optimize_single_psf, args=(x0, names, self.psf_instances[i], flux)))
            # else:
            results.append(self.wrapper_optimize_single_psf(x0=x0, names=names, psf=self.psf_instances[i], flux=flux))

        # if self.ncpu>1:
        #     pool.join()
        #     pool.close()
        #     results = [threads[j].get() for j in len(self.psf_indices)]

        ## collect results from individual PSF fits
        ifs_coords  = []
        ref_coords  = []
        psf_pars    = []

        for n in xrange(len(results)):
            result = results[n]
            i = self.psf_indices[n]

            success = (result[-1]>0)&(result[-1]<=4)
            if not success:
                logging.error("[@{0}] PSF fit failed for source #{1}: {2}".format(self.kk, self.sources.ID[i], result[-2]))
                continue
            
            logstring = "[@{0}] Successfully fitted PSF of source #{1}:".format(self.kk, self.sources.ID[i])

            if posfitted[n]:
                if self.full_coordinates_fit: ## if IFS coordinates should be fitted, update them directly
                    j = self.sources.ifs_coordinates_loaded[:i].sum() ## get index of source in IFS coordinates array
                    self.sources.ifs_coordinates[self.jj,j,:,0] = result[0][:2]
                else: ## save the coordinates to fit the transformation afterwards
                    ifs_coords.append(result[0][:2])
                    ref_coords.append([self.sources.x[i], self.sources.y[i]])
                logstring += " x_c={0:.2f} y_c={1:.2f}".format(float(result[0][0]), float(result[0][1]))
            if psffitted[n]:
                psf_pars.append(result[0][-len(psf_names):])
                for j in xrange(len(psf_names)): logstring += " {0}={1:.2f}".format(psf_names[j], float(psf_pars[-1][j]))
            logging.info(logstring)


        ## update coordinate transformation if available and possible
        if len(ifs_coords)>=2:
                    
            self.sources.transformation.data[self.jj,:,0] = self.sources.transformation._fit_to_data(np.asarray(ref_coords), np.asarray(ifs_coords))
        
            logstring = "[@{0}] Fitted coordinate transformation: ".format(self.kk)
            for i in xrange(self.sources.transformation.npars):
                ## updating the parameters in the PSF-class overwrites the (direct) source positions used for the fit
                getattr(self.PSF, "set_{0}".format(self.sources.transformation.names[i]))(self.sources.transformation.data[self.jj,i,0])
                logstring += "{0}={1:.3f} ".format(self.sources.transformation.names[i], float(self.sources.transformation.data[self.jj,i,0]))
            logging.info(logstring)

        elif not self.full_coordinates_fit:
            logging.error("[@{0}] Cannot update coord. transf. because <2 sources fitted successfully.".format(self.kk))

        ## update PSF parameters
        if self.psf_attributes.contains_free_parameters: ## fit PSF
                
            ## update parameter values in the attributes of the PSF and in the actual PSF profiles; for this, only
            ##  the results of the successful fits are used. If no fit was successful, nothing is updated.
            if len(psf_pars) == 0:
                logging.error("[@{0}] Cannot update PSF parameters because no star yielded a successful fit.".format(self.kk))
            else:
                i = 0
                logstring = "[@{0}] Updated PSF parameters:".format(self.kk)
                for par in psf_names:

                    ii = self.psf_attributes.names.index(par)
                
                    self.psf_attributes.data[self.jj,ii,0] = np.mean([v[i] for v in psf_pars])
                    getattr(self.PSF, "set_{0}".format(par))(self.psf_attributes.data[self.jj,ii,0])
                
                    logstring += "{0}={1:.2f} ".format(par, float(self.psf_attributes.data[self.jj,ii,0]))
                    i += 1
                    
                logging.info(logstring)


    def wrapper_optimize_single_psf(self, x0, names, psf, flux):
        """
        This wrapper around the actual fitting routine 'optimize_single_psf' is
        included to facilitate the parallelizaion of the code and a simple checking
        for the usefulness of the fitted parameters. It works by adding the current
        estimate of the PSF and the flux of the source to the data, calling the fit
        routine 'optimize_single_psf' and providing the results of the fit (after a
        sanity check).

        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted.
        names :  array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        psf : an instance of class pampelmuse.core.psf.profiles.Profile or
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
        _data  = np.copy(self.current)
        _data[psf.used] += flux*psf.f/self.sigma[psf.used]

        ## call non-linear least squares fit
        try:
            return optimize.leastsq(self.optimize_single_psf, x0=x0, Dfun=self.jacobi_single_psf, args=(names, psf, flux, _data), full_output=1, epsfcn=0.0)
        except optimize.minpack.error as error: ## error handler to avoid code that code crashes completely
            return (x0, {}, "Exception of type optimize.minpack.error encountered while executing fit.", 99)


    def optimize_single_psf(self, x0, names, psf, flux, data):
        """
        This is the optimization routine that is passed to the Levenberg-
        Marquard solver scipy.optimize.leastsq. During each iteration, a
        PSF array is created, scaled and subtracted from the data. The
        residuals of the fit are returned.
        
        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted.
        names :  array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        psf : an instance of class pampelmuse.core.psf.profiles.Profile or
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
            try:
                ## the attribute is only set in the 'psf'-instance of the class,
                ## not in all instances! This is different from the behaviour of
                ## getattr(self.PSF, "set_<PAR>")(<VALUE>)
                setattr(psf, names[i], x0[i])
                
            except AssertionError as error: ## if parameter cannot be updated, raise LinAlgError
                logging.error("[@{0}] Encountered AssertionError: {1}".format(self.kk, error))
                raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], names[i]))

        ## return residuals
        return (data[psf.used] - flux*psf.f/self.sigma[psf.used])


    def jacobi_single_psf(self, x0, names, psf, flux, data):
        """
        Calculate the Jacobi matrix for the optimization carried out in
        StarFit.optimize_single_psf. The method is passed to the Levenberg-
        Marquard solver scipy.optimize.leastsq as parameter Dfun.
        
        Parameters
        ----------
        This method takes the same arguments as StarFit.optimize_single_psf.
        
        Returns
        -------
        jacob : nd_array
            The Jacobi matrix with respect to the model parameters included
            in 'x0'.
        """

        jacob = np.zeros((len(x0), psf.used.sum()), dtype=np.float64)
        for i in xrange(len(x0)):
            jacob[i] -= flux*getattr(psf, "diff_{0}".format(names[i]))/self.sigma[psf.used]

        return jacob
        

    def parameter_fit_on_small_ifu(self):
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
        if self.sources.contains_free_coordinates:
            
            ## collect free parameters
            fitPars  = []
            parNames = []
            for i in xrange(self.sources.transformation.npars):
                if self.sources.transformation.free[i] and not self.sources.transformation.fitted[i]:
                    fitPars.append(self.sources.transformation.data[self.jj,i,0])
                    parNames.append(self.sources.transformation.names[i])

            ## call Levenberg-Marquard solver from scipy.optimize (& provide Jacobian)
            bestfit = optimize.leastsq(self.optimize_multi_psf, x0=fitPars, Dfun=self.jacobi_multi_psf, col_deriv=1, args=(parNames,), epsfcn=0.0)[0]
            
            ## update parameters using fit results
            for i in xrange(len(parNames)):
                ii = self.sources.transformation.names.index(parNames[i])
                self.sources.transformation.data[self.jj,ii,0] = bestfit[i]

            ## some logging
            if logging.getLogger().getEffectiveLevel()<=20:
                logging.info("[@{0}] Fitted coordinate transformation:".format(self.kk))
                for i in xrange(self.sources.transformation.npars):
                    logging.info("    {0}={1:.2f}".format(self.sources.transformation.names[i], float(self.sources.transformation.data[self.jj,i,0])))


        ## Fit PSF parameters
        if self.psf_attributes.contains_free_parameters:
    
            ## collect free parameters
            fitPars  = []
            parNames = []
            for i in xrange(self.psf_attributes.npars):
                if self.psf_attributes.free[i] and not self.psf_attributes.fitted[i]:
                    fitPars.append(self.psf_attributes.data[self.jj,i,0])
                    parNames.append(self.psf_attributes.names[i])

            ## call Levenberg-Marquard solver from scipy.optimize (& provide Jacobian)
            try:
                bestfit = optimize.leastsq(self.optimize_multi_psf, x0=fitPars, Dfun=self.jacobi_multi_psf, col_deriv=1, args=(parNames,), epsfcn=0.0)[0]
                success = True
            except optimize.minpack.error: ## error handler to avoid code that code crashes completely
                logging.error("[@{0}] PSF fit failed.".format(self.kk))
                success = False

            ## update parameters using fit results
            if success:
                for i in xrange(len(parNames)):
                    ii = self.psf_attributes.names.index(parNames[i])
                    self.psf_attributes.data[self.jj,ii,0] = bestfit[i]

                ## some logging
                if logging.getLogger().getEffectiveLevel()<=20:
                    logging.info("[@{0}] Fitted PSF model:".format(self.kk))
                    for i in xrange(self.psf_attributes.npars):
                        logging.info("    {0}={1:.2f}".format(self.psf_attributes.names[i], float(self.psf_attributes.data[self.jj,i,0])))

        self.ncpu = NCPU ## recover original number of CPUs to be used


    def optimize_multi_psf(self, x0, names):
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
#        logging.debug("Shape of Jacobi matrix: {0}".format(self.jacobi_multi_psf(x0, names).shape))
#        logging.debug("Shape of data: {0}".format(self.data.shape))
        
        ## update parameters in PSF-class
        for i in xrange(len(x0)):
            try:
                getattr(self.PSF, "set_{0}".format(names[i]))(x0[i])
            except AssertionError: ## if parameter cannot be updated, raise LinAlgError
                raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], names[i]))

        ## calculate new model
        self.build_matrix(update_unresolved=False, log=False)
        model =  self.Apsf*sparse.dia_matrix((self.sources.data[self.jj,self.valid_components,0]*self.fluxfactors[self.valid_components],[0]),
                                             shape=(self.valid_components.sum(), self.valid_components.sum()))

        ## return residuals
        return (self.data/self.sigma) - model.sum(axis=1).getA1()


    def jacobi_multi_psf(self, x0, names):
        """
        Routine that calculates the Jacobi matrix of the residuals. This method
        is parsed to the Levenberg-Marquard solver scipy.optimize.leastsq a
        parameter Dfun when the PSF of multiple sources is fitted simultaneously.

        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted
        names : array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        
        Parameters
        ----------
        The method takes the same arguments as StarFit.optimize_multi_psf
        
        Returns
        -------
        jacob : nd_array
            The Jacobi matrix with respect to the model parameters included
            in 'x0'.
        """
        jacob = np.zeros((len(x0), self.PSF.transform.shape[0]), dtype=np.float64)

        i=-1
        for psf in self.PSF:
            i += 1
            ii = self.sources.specrow[i]
            if ii==self.sources.specrow_sky: continue
            if ii==self.sources.specrow_unresolved: continue
            for j in xrange(len(x0)):
                jacob[j, psf.used] -= self.sources.data[self.jj,ii,0]*getattr(psf, "diff_{0}".format(names[j]))/self.sigma[psf.used]
            
        return jacob


    def update_parameters(self):
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
        if len(self.last_fitted_layers) < 2:
            ## nothing to update yet
            return
        elif len(self.last_fitted_layers) < self.Ntrack:
            ## use all images that have been fitted so far
            LASTFITS = self.last_fitted_layers
        else:
            ## get every second of the recently fitted images
            LASTFITS = [self.last_fitted_layers[i] for i in xrange(len(self.last_fitted_layers)-2, -1, -2)]
        
        logging.debug("[@{0}] Updating initial guesses of parameters using results from adjacent {1} layers.".format(self.kk, len(self.last_fitted_layers)))
        logging.debug("[@{0}] Previous fits in focus: {1}".format(self.kk, LASTFITS))

        if self.sources.contains_free_coordinates:
            if not self.full_coordinates_fit:  ## update coordinate transformation
                for i in np.flatnonzero(self.sources.transformation.free&(self.sources.transformation.fitted==False)):
                    self.sources.transformation.data[self.jj,i,0] = self.sources.transformation.data[LASTFITS,i,0].mean(axis=0)
            else: ## update coordinates
                self.sources.ifs_coordinates[self.jj,self.update_coordinates,:,0] = self.sources.ifs_coordinates[LASTFITS].mean(axis=0)[self.update_coordinates,:,0]

        if self.psf_attributes.contains_free_parameters: ## update PSF
            for i in np.flatnonzero(self.psf_attributes.free&(self.psf_attributes.fitted==False)):
                self.psf_attributes.data[self.jj,i,0] = self.psf_attributes.data[LASTFITS,i,0].mean(axis=0)
