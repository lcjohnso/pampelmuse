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
258, 2015/04/27
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 258

import contextlib
import itertools
import logging
import time
import numpy as np
import scipy
from multiprocessing import cpu_count, Pool
from scipy import optimize, sparse
# from astropy.io import fits

if int(scipy.__version__.split(".")[1]) < 10:
    logging.warning("Version of SciPy is <0.10, using sparse.linalg.lsqr instead of faster scipy.linalg.lsmr")
    from scipy.sparse.linalg import lsqr as lsmr
else:
    from scipy.sparse.linalg import lsmr
# lsmr does the same as lsqr but faster (i.e. the Max Power way), it will only work for scipy versions >=0.10

# required PampelMuse-packages
from pampelmuse.core import BackgroundGrid
from pampelmuse.core.psf import Profile, Gaussian, Moffat


# for a nicer printing of numpy arrays, following suggestion from
# http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
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
        should be summed. It must have the same length as
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
    values.cumsum(out=values, dtype=np.float64)
    # print "CUMSUM:", values
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups


def init_worker(back):
    global background
    background = back


def get_psf_data(psf_instances):
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
        the package pampelmuse.core.psf. May also contain
        integers if the parameter 'background' is set.
    background : instance of pampelmuse.core.background.Background, optional
        For every integer in the psf_instances list, the
        method background.get_flux() is called to obtain
        the flux distribution of the respective background
        component.
        
    Returns
    -------
    results : list 
        For each PSF instance, the routine returns a list of len=3,
        containing the spaxels on which the PSF profile is defined,
        the relative intensities, normalized by the uncertainties
        of each spaxel, and the integrated profile flux over the whole
        field of view.
    """
    global background
    if not hasattr(psf_instances, "__iter__"):
        psf_instances = [psf_instances]
    results = []
    for psf in psf_instances:
        if not isinstance(psf, Profile):  # background component
            # continue
            spaxels, fluxes = background.get_flux(psf)
        else:  # stellar component
            spaxels, fluxes = psf.get_flux()
            if spaxels.size == 0:
                logging.warning("Source #{0} entirely outside FoV.".format(psf.src_id))
        intflux = fluxes.sum()
        results.append([spaxels, fluxes, intflux])
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
    proceeds to the red and blue. This is done because usually the highest
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

        self.maxrad = maxrad

        # supersampling of central pixels
        self.osfactor = osfactor
        self.osradius = osradius

        # initialize background
        if self.sources.is_background.sum() == 1:
            self.background = BackgroundGrid.constant()
        elif self.sources.is_background.sum() > 1:
            self.background = BackgroundGrid(
                np.vstack((self.sources.y[self.sources.is_background], self.sources.x[self.sources.is_background])).T)
        else:
            self.background = None
        if self.background is not None:
            self.background.set_transform(self.ifs_data.instrument.transform)

        # get basic PSF model: Moffat or Gaussian
        if self.psf_attributes.profile == "gauss":
            self.PSF = Gaussian
        elif self.psf_attributes.profile == "moffat":
            self.PSF = Moffat
        else:
            logging.critical("Invalid PSF type: '{0}'".format(self.psf_attributes.psftype))
            raise IOError("PSF type not understood.")

        # initialize PSF-class for every source included
        self.PSF.set_transform(self.ifs_data.instrument.transform)

        logging.info("Initializing PSF instances for {0} sources...".format(self.sources.nsrc))
        self.psf_instances = []
        for i in xrange(self.sources.nsrc):
            if self.sources.status[i] == 4:
                # for each background component, an integer is added to the list giving the index of the corresponding
                # patch in self.background
                self.psf_instances.append(self.sources.is_background[:(i + 1)].cumsum()[-1] - 1)
            else:
                # for each stellar source, an instance of the requested PSF-class is added
                self.psf_instances.append(self.PSF(
                    uc=self.sources.x[i], vc=self.sources.y[i], mag=sources.weights[i],
                    src_id=self.sources.ID[i], maxrad=maxrad, osfactor=osfactor, osradius=osradius))

        # Check if individual source positions should be fitted
        self.full_coordinates_fit = (
            self.sources.ifs_coordinates_free & (~self.sources.ifs_coordinates_fitted)).any()
        if self.full_coordinates_fit:
            logging.warning("Fit of individual source positions requested.")
            self.update_coordinates = (self.sources.status < 4) & (~self.sources.ifs_coordinates_fitted)
        else:
            self.update_coordinates = (self.sources.status == 3) & (~self.sources.ifs_coordinates_fitted)

        # initialize class attributes to be used later
        self.jj = None
        self.data = None
        self._sigma = None
        self.psf_matrix = None
        self._unresolved_matrix = None
        self.valid_components = None

    def __call__(self, layerrange=(0, -1), ntrack=10, assmallifu=False, ncpu=1, maxiter=20, use_sigma=True, nbin=1,
                 **kwargs):
        """
        Execute the fitting process. If a previous analysis of the same data
        was interrupted, it might be resumed. For each layer of the provided
        datacube, the following steps are performed:
        [1) Update the initial guesses or values of the parameters of the
            PSF or the coordinate transformation.]
         2) Initialize the PSF model.
         3) Calculate a PSF profile for every source included
        [4) Fit the parameters of the coordinate transformation or the
            individual source coordinates.]
        [5) Fit the shape parameters of the PSF]
         6) Fit the source fluxes
        Steps in square brackets are optional and only executed if requested.
        
        Parameters
        ----------
        layerrange : tuple
            The indices of the first and last layer of the spectral range to
            be analysed. By default, the whole cube is analysed.
        ntrack :  integer, optional
            In case the PSF and/or the source positions are fitted, the 
            routine will update the initial guesses of the parameters that
            should be fitted prior to fitting a layer. To this aim, it uses
            the results from the fits to the 'ntrack' neighboring layers. If
            you do not want this, set `ntrack' to zero. Default is 10. Note
            that `ntrack' will be divided by `nbin', if layer binning is used.
        assmallifu : boolean, optional
            Flag indicating whether the data will be handled as coming from
            a small or a large IFU. This parameter controls how the PSF and/
            or the positions are fitted. Note that if individual source
            coordinates should be determined, 'asSmallIFU' will automatically
            be set to False. Default is False.
        maxiter : int, optional
            The maximum number of iterations per image. The default is 20.
        use_sigma : boolean, optional
            A flag indicating whether the uncertainties of the IFS data should
            be used in the analysis.
        nbin : int, optional
            The number of layers of the IFS data combined into a single layer
            for the analysis.
        """
        # check which layers should be analysed
        start, stop = layerrange
        if start < 0:
            start = self.sources.ndisp + start
        if stop < 0:
            stop = self.sources.ndisp + (stop + 1)
        self.nbin = nbin
        layers_to_analyse = np.arange(start + self.nbin // 2, stop, self.nbin, dtype=np.int16)
        self.layers_to_analyse = np.zeros(layers_to_analyse.shape, dtype=layers_to_analyse.dtype)
        self.layers_to_analyse[0::2] = layers_to_analyse[layers_to_analyse.size // 2:]
        self.layers_to_analyse[1::2] = layers_to_analyse[:layers_to_analyse.size // 2][::-1]

        omitted_layers = ~np.in1d(np.arange(self.sources.ndisp, dtype=np.int16), self.layers_to_analyse)

        logging.info("Layer range to be analysed: {0}-{1}".format(self.layers_to_analyse.min(),
                                                                  self.layers_to_analyse.max()))
        logging.info("Binning factor per layer: {0}".format(self.nbin))

        # Number of iterations per layer.
        self.maxiter = int(maxiter)
        if self.maxiter > 1:
            logging.info("Maximum iterations per layer: {0}".format(self.maxiter))
        else:
            logging.info("No iteration on individual layers performed.")
            self.maxiter = 1

        # if parameters (of the PSF or the coordinate transformation) are fitted, it must be ensured that the instances
        # have correct dispersion information
        if self.psf_attributes.ndisp != self.sources.ndisp:
            self.psf_attributes.resize(self.sources.ndisp)
            if self.nbin > 1:  # set fitted PSF parameters of omitted layers to NaN
                self.psf_attributes.data[np.ix_(omitted_layers, self.psf_attributes.free, [True, False])] = np.nan

        if self.sources.transformation.valid and self.sources.transformation.ndisp != self.sources.ndisp:
            self.sources.transformation.resize(self.sources.ndisp)
            if self.nbin > 1:  # set fitted transformation parameters of omitted layers to NaN
                self.sources.transformation.data[
                    np.ix_(omitted_layers, self.sources.transformation.free, [True, False])] = np.nan

        # for large IFS data, get indices of sources used to fit PSF/positions
        self.assmallifu = assmallifu
        if not self.assmallifu:
            self.psf_indices = np.union1d(np.flatnonzero(self.update_coordinates), self.sources.psf_indices)

        # whole array with spectra needs to be loaded
        self.sources.load_spectra()

        # if individual source positions should be fitted, we need to load the initial guesses for the positions of all
        # sources on all layers into memory from the start
        # UPDATE 2015/02/09: Always load full coordinate array into memory because doing it per layer is quite slow
        # if self.full_coordinates_fit:
        self.sources.load_ifs_coordinates()

        # Check how many CPUs should be used
        if ncpu is None:
            self.ncpu = cpu_count()
        else:
            self.ncpu = int(ncpu)
        logging.info("Starting analysis using {0} CPU(s).".format(self.ncpu))

        # check if uncertainties should be used
        self.use_sigma = use_sigma

        # The number of previously fitted layers used to obtain initial guesses
        self.ntrack = int(ntrack//self.nbin) + 1 if self.nbin > 1 else ntrack

        # execute fit
        self.timeout()

    def timeout(self):
        """
        The caller for the actual fit routine. The idea of this routine is to
        facilitate the adaption of the fitting process to a graphic-supported
        analysis using PyQt. Check the documentation of the QStarFit-class for
        further details.
        """
        # loop over all layers in a datacube
        for i, self.jj in enumerate(self.layers_to_analyse):
            # A common prefix used for all logging messages related to the analysis of the current layer.
            self.prefix_string = "[{0:0{1}d}/{2}]".format(i, int(np.log10(self.layers_to_analyse.size)) + 1,
                                                          self.layers_to_analyse.size)

            # call fitting routine
            logging.info("{0} Starting fit for layer #{1}.".format(self.prefix_string, self.jj))
            self.iter_fit_layer()

    def iter_fit_layer(self):
        """
        The method is called for every slice of the datacube from the
        'timeout'-method. It performs the following steps.
        (i) Initialization of the PSF;
        (ii) load the data and sigma arrays and apply the mask if one is
             provided;
        (iii) run the iterative scheme used to fit the fluxes, PSF parameters
              and coordinate transformation;
        (iv) estimate the uncertainty of the individual flux of each
             component.
        """
        # update parameter initial guesses (if any):
        self.update_parameters()

        # if spectrum data contains NaN values, they must be replaced with zeros prior to the fit
        if np.isnan(self.sources.data[self.jj, :, 0]).any():
            self.sources.data[self.jj, np.isnan(self.sources.data[self.jj, :, 0]), 0] = 0

        # load IFU coordinates for current layer into memory unless individual positions are fitted
        # not done anymore because it may take ages - instead the whole array is loaded when the fit is started
        # if not self.full_coordinates_fit:
        # self.sources.load_ifs_coordinates(layers=self.jj)

        # update PSF instances
        # -with PSF parameters
        self.update_psf_parameters()

        # -with parameters of coordinate transformation (if available and needed)
        if self.sources.transformation.valid and not np.all((
                self.sources.ifs_coordinates_fitted | self.sources.ifs_coordinates_free)[~self.sources.is_background]):
            self.update_coordinate_transformation()

        # -with direct positions if required
        if self.sources.ifs_coordinates_fitted.any() or self.sources.ifs_coordinates_free.any():
            self.update_ifs_coordinates()

        # WORK ON IFS DATA
        if self.nbin == 1:
            data = self.ifs_data.data.layer(self.jj)
            sigma = self.ifs_data.sigma.layer(self.jj) if self.use_sigma else None
        else:
            jjmin = self.jj - self.nbin // 2
            jjmax = min(self.jj + self.nbin // 2, self.ifs_data.ndisp)
            if self.use_sigma:
                data, sigma = self.ifs_data.integrate(layer_range=(jjmin, jjmax), use_sigma=True)
            else:
                data = self.ifs_data.integrate(layer_range=(jjmin, jjmax), use_sigma=False)
                sigma = None

        # check if mask is available, otherwise only NaN pixels will be masked
        mask = np.isnan(data)
        if self.ifs_data.mask is not None:
            mask |= self.ifs_data.mask.layer(self.jj)
        if self.use_sigma:
            mask |= np.isnan(sigma) | (sigma <= 0)

        self.data = data[~mask]
        if sigma is None:
            self.sigma = np.ones(self.data.shape, dtype=np.float32)
        else:
            if self.ifs_data.is_var:
                self.sigma = np.sqrt(sigma[~mask])
            else:
                self.sigma = sigma[~mask]
        self.weights = np.ones(self.data.shape, dtype=np.float32)

        logging.debug("{0} {1} masked pixels".format(self.prefix_string, mask.sum()))

        # Set spaxel to (x,y) transformation of the PSF-instances, using the transformation of the Instrument instance
        # and the mask
        self.transform = self.ifs_data.instrument.transform[~mask]
        self.PSF.set_transform(self.transform)
        if self.background is not None:
            self.background.set_transform(self.transform)

        # Reset possibly existing unresolved component from last layer
        self._unresolved_matrix = None

        rebuild_matrix = True

        # Start iteration
        n_iter = 1
        while True:

            if self.maxiter > 1:
                logging.info("{0} Iteration {1} (of max. {2})".format(self.prefix_string, n_iter, self.maxiter))

            # create PSF matrix if required, divide by uncertainties
            if rebuild_matrix:
                self.psf_matrix, self.valid_components = self.build_matrix()
                t0 = time.time()
                self.psf_matrix.data /= np.take(self.sigma, self.psf_matrix.indices)
                rebuild_matrix = False
                t1 = time.time()
                logging.debug("{0} Time required to divide matrix by uncertainties: {1:6.3f}s".format(
                    self.prefix_string, t1-t0))

            # Fit fluxes by direct matrix inversion
            # The documentation of scipy states that to efficiently solve the matrix equation, the columns should
            # be scaled to have the same euclidean norm. I am still not convince that this is true, but scale the
            # columns nevertheless.
            intflux = np.sqrt((self.psf_matrix.T*self.psf_matrix).diagonal())
            fit_matrix = self.psf_matrix * sparse.dia_matrix((1. / intflux, [0]),
                                                             shape=(self.psf_matrix.shape[1], self.psf_matrix.shape[1]))
            fit_matrix.data *= np.take(self.weights, fit_matrix.indices)

            t1 = time.time()
            bestfit = lsmr(fit_matrix, self.current * self.weights, atol=1e-5, show=False)
            t2 = time.time()
            logging.debug("{0} Time required to solve matrix equation: {1:6.3f}s".format(self.prefix_string, t2-t1))

            self.sources.data[self.jj, self.valid_components, 0] += bestfit[0] / intflux
            t3 = time.time()
            logging.debug("{0} Time required to update fluxes: {1:6.3f}s".format(self.prefix_string, t3-t2))

            # update noise values
            self.sources.data[self.jj, self.valid_components, 1] = \
                1. / (intflux * np.sqrt(fit_matrix.multiply(fit_matrix).sum(axis=0)).getA1())

            # determine relative change in fluxes
            delta_flux = bestfit[0] / (intflux*self.sources.data[self.jj, self.valid_components, 0])

            # some logging
            logging.info("{0} Chi**2 of current fit: {1:.1f}".format(self.prefix_string, bestfit[3] ** 2))
            with printoptions(precision=4, suppress=True):
                logging.debug("{0} Relative changes in fluxes: {1}".format(self.prefix_string, delta_flux))

            # convergence criterion: median relative change in fluxes <0.005
            if np.median(abs(delta_flux)) < 5e-3 and n_iter > 1:
                break

            # fit PSF/positions if requested
            if self.psf_attributes.contains_free_parameters or self.sources.contains_free_coordinates:
                # PSF/positions fit
                if self.assmallifu:
                    self.parameter_fit_on_small_ifu()
                else:
                    self.parameter_fit_on_large_ifu()
                rebuild_matrix = True
            elif self.use_sigma:
                # apply weighting scheme, similar to the one used in Stetson's DAOphot
                # only possible if uncertainties are being used
                self.weights = 1. / (1. + (abs(self.current) / (2.*self.sigma)) ** 2)
            else:
                break

            n_iter += 1

            if n_iter > self.maxiter:
                if self.maxiter > 1:
                    logging.warning("{0} Max. number of iterations reached.".format(self.prefix_string))
                break

    def update_psf_parameters(self):
        """
        Method to update the PSF parameters in all instances of PSF sources
        prior to fitting a new layer of data.

        :return:
        """
        logstring = "{0} Initial PSF model:".format(self.prefix_string)

        # loop over PSF parameters
        i = 0
        for prm in self.psf_attributes.names:
            if self.psf_attributes.fitted[i]:
                value = self.psf_attributes.data[self.jj, i, 1]
            else:
                if self.psf_attributes.mask[self.jj, i, 0]:
                    logging.warning("Access to NaN value in PSF data, interpolating...")
                    if (~self.psf_attributes.mask[:, i, 0]).sum > 2:
                        value = float(np.interp(self.jj, (~self.psf_attributes.mask[:, i, 0]).nonzero()[0],
                                                self.psf_attributes.data[~self.psf_attributes.mask[:, i, 0], i, 0]))
                    else:
                        value = np.mean(self.psf_attributes.data[~self.psf_attributes.mask[:, i, 0], i, 0])
                    self.psf_attributes.data[self.jj, i, 0] = value
                else:
                    value = self.psf_attributes.data[self.jj, i, 0]
            getattr(self.PSF, "set_{0}".format(prm))(value)
            logstring += " {1}={2:.2f},".format(self.prefix_string, prm, float(value))
            i += 1
        logging.info(logstring[:-1])

    def update_coordinate_transformation(self):
        """
        Method to update the parameters of the coordinate transformation in
        all instances of PSF sources prior to fitting a new layer of data.

        :return:
        """
        logstring = "{0} Initial coord. transformation:".format(self.prefix_string)

        # loop over parameters of coordinate transformation
        i = 0
        for prm in self.sources.transformation.names:
            if self.sources.transformation.fitted[i]:
                value = self.sources.transformation.data[self.jj, i, 1]
            else:
                if self.sources.transformation.mask[self.jj, i, 0]:
                    logging.warning("Access to NaN value in coordinate transformation data, interpolating...")
                    if (~self.sources.transformation.mask[:, i, 0]).sum() > 2:  # interpolate only if >2 valid entries
                        value = float(np.interp(
                            self.jj, (~self.sources.transformation.mask[:, i, 0]).nonzero()[0],
                            self.sources.transformation.data[~self.sources.transformation.mask[:, i, 0], i, 0]))
                    else:  # otherwise take mean of valid entries
                        value = np.mean(
                            self.sources.transformation.data[~self.sources.transformation.mask[:, i, 0], i, 0])
                    self.sources.transformation.data[self.jj, i, 0] = value
                else:
                    value = self.sources.transformation.data[self.jj, i, 0]
            getattr(self.PSF, "set_{0}".format(prm))(value)
            logstring += " {1}={2:.2f},".format(self.prefix_string, prm, float(value))
            i += 1
        logging.info(logstring[:-1])

    def update_ifs_coordinates(self):
        """
        Method to update the IFS coordinates i all instances of PSF sources
        prior to fitting a new layer of data.

        :return:
        """
        k = 0
        logging.info("{0} Updating coordinates of PSF instances...".format(self.prefix_string))
        t0 = time.time()
        for i, psf in enumerate(self.psf_instances):
            if not isinstance(psf, Profile):
                continue
            if self.sources.ifs_coordinates_fitted[i]:
                psf.set_xc(self.sources.ifs_coordinates[self.jj, k, 0, 1])
                psf.set_yc(self.sources.ifs_coordinates[self.jj, k, 1, 1])
            elif self.sources.ifs_coordinates_free[i]:
                psf.set_xc(self.sources.ifs_coordinates[self.jj, k, 0, 0])
                psf.set_yc(self.sources.ifs_coordinates[self.jj, k, 1, 0])
            k += 1

        logging.debug("{0} Time required to update source coordinates: {1:6.4f}s".format(
            self.prefix_string, time.time() - t0))

    @property
    def model(self):
        """Create the current best-fit model"""
        return self.psf_matrix * sparse.dia_matrix((self.sources.data[self.jj, self.valid_components, 0], [0]),
                                                   shape=(self.psf_matrix.shape[1], self.psf_matrix.shape[1]))

    @property
    def current(self):
        """Determine the current residuals"""
        # self.model.sum(axis=1) returns numpy.matrixlib.defmatrix-object; `getA1' returns a flattened numpy.ndarray
        return (self.data / self.sigma) - self.model.sum(axis=1).getA1()

    def build_matrix(self, update_unresolved=True):
        """
        Define the source matrix. The source matrix is a sparse matrix that
        contains the spatial profiles of all resolved components, one per
        column.
        The method initially creates three empty arrays to hold
        (a) the source (PSF) coefficients, divided by the pixel uncertainties
        (b) the (1dim.) coordinates of the coefficients on the data array
            (they are the row indices of the final PSF coefficient matrix)
        (c) the component to which each source belongs
            (they are the column indices of the final PSF coefficient matrix)
        It then loops over all sources in the 'Sources'-instance, obtains
        their PSF and adds it to the respective column of the array. After-
        wards, the arrays are combined into a single sparse matrix.
        
        Parameters
        ----------
        update_unresolved : boolean, optional
            If False, the matrix column containing the unresolved stellar
            component (if included) is not recomputed, but taken from a
            previous calculation.
        
        Returns
        -------
        a scipy.sparse.csc_matrix-instance
            This is the PSF coefficient matrix that is used in the linear
            least-squares fit to determine the source fluxes.
        """
        t0 = time.time()

        # omit possibly unresolved stellar component unless it should be updated
        if not self.sources.includes_unresolved or update_unresolved:
            psf_instances = self.psf_instances
            specrow = self.sources.specrow
        else:
            psf_instances = self.psf_instances[~self.sources.is_unresolved]
            specrow = self.sources.specrow[~self.sources.is_unresolved]

        # Initialize parallelization (if requested)
        if self.ncpu > 1:
            pool = Pool(self.ncpu, initializer=init_worker, initargs=(self.background,))
            threads = []

            # determine which sources should be calculated by which CPU
            m = (len(psf_instances) // self.ncpu) + 1

#             for n in xrange(self.ncpu):
#                 imin = n * m
#                 imax = min((n + 1) * m, self.sources.nsrc)
#                 threads.append(
#                     pool.apply_async(get_psf_data, args=(psf_instances[imin:imax], )))  # , self.background)))
            _results = pool.map_async(get_psf_data, psf_instances)

            pool.close()
            pool.join()
            results = list(itertools.chain.from_iterable(_results.get()))
            # Note: parallelization seems not to work when calling class methods. The different packages are still
            # processed
#             results = list(itertools.chain.from_iterable([threads[n].get() for n in xrange(self.ncpu)]))
        else:
            init_worker(self.background)
            results = get_psf_data(psf_instances)

        # result is a list with length equal to the number of sources, each entry is again a list containing three
        # entries: the spaxels on which the PSF is defined, the corresponding fluxes, and the integrated flux

        t1 = time.time()
        logging.debug("{0} Time required to calculate PSF profiles: {1:.3f}s".format(self.prefix_string, t1 - t0))

        # Calculate combined integrated flux of each component
        intflux, unique_specrow = sum_by_group([r[-1] for r in results], specrow)
        # note that the integrated fluxes are already in the correct order, i.e. sorted by self.sources.specrow, the
        # unresolved component is only included if update_unresolved was set to True

        t2 = time.time()
        logging.debug("{0} Time required to combine components: {1:.3f}s".format(self.prefix_string, t2 - t1))

        # check if any row does not contribute at all
        empty_specrow = unique_specrow[intflux <= 0]
        if len(empty_specrow) > 0:
            logging.warning("Encountered component(s) that contribute zero flux: {0}".format(empty_specrow))

        # Fill the matrix: the indices and intensities of all PSF instances are combined and serve as row indices (i0)
        # and values of the matrix. To obtain the column indices of the matrix (i1), we use the property 'specrow' of
        # the Sources class and correct for components with zero flux contribution (if they exist)
        i0 = np.hstack((r[0] for r in results))  # the row indices

        i1 = np.zeros(i0.shape, dtype=np.int32)
        jstart = 0
        for i in xrange(len(results)):
            if specrow[i] in empty_specrow:
                continue
            jend = jstart + len(results[i][0])
            i1[jstart:jend] = specrow[i]
            jstart = jend

        f = np.hstack((r[1] for r in results))  # the fluxes

        # If sources are outside the field of view, the indices in i1 will not be contiguous but there will be gaps
        # that must be removed.
        if len(empty_specrow) > 0:
            assert np.intersect1d(i1, empty_specrow).size == 0, \
                "i1 contains indices of sources with zero flux contribution."
            for esr in empty_specrow[::-1]:
                i1[i1 > esr] -= 1

        # create matrix: first, a COO matrix is constructed. Note that it handles the case when the input data contains
        # multiple entries for the some indices (row, column). After construction, the matrix is converted to a
        # CSC matrix to speed up future usage of it.
        nrows = self.data.size
        ncols = self.sources.ncmp - len(empty_specrow)

        if self.sources.includes_unresolved:  # handle the unresolved component
            # need to take into account that the row of the unresolved component in the matrix changes if some
            # components do not contribute
            specrow_unresolved = self.sources.specrow_unresolved - (
                empty_specrow < self.sources.specrow_unresolved).sum()
            if update_unresolved or self._unresolved_matrix is None:
                # just store the unresolved sources as a list for later re-use
                r = (i1 != specrow_unresolved)
                self._unresolved_matrix = sparse.coo_matrix((f[~r], (i0[~r], i1[~r])), shape=(nrows, ncols))
                _Apsf = sparse.coo_matrix((f[r], (i0[r], i1[r])), shape=(nrows, ncols)) + self._unresolved_matrix
            else:
                _Apsf = sparse.coo_matrix((f, (i0, i1)), shape=(nrows, ncols)) + self._unresolved_matrix
        else:
            _Apsf = sparse.coo_matrix((f, (i0, i1)), shape=(nrows, ncols))

        t3 = time.time()
        logging.debug("{0} Time required to create matrix: {1:.3f}s".format(self.prefix_string, t3 - t2))

        # return CSC matrix and boolean array indicating which components contribute flux
        zero_component = np.in1d(np.arange(self.sources.ncmp, dtype=np.int32), empty_specrow)
        return _Apsf.tocsc(), ~zero_component

    def parameter_fit_on_large_ifu(self):
        """
        Perform the PSF fit for an IFU with many spaxels. In this case,
        a non-linear least-squares fit is performed for every PSF source
        independently. Depending on the respective fit-flags, either the
        parameters of the PSF and/or the source coordinates are fitted.
        The results of the individual fits are afterwards used to update
        the PSF and/or the coordinate transformation.
        """
        # Initialize parallelization (if requested)
        # TODO: enable parallelization again and make it work
        # if self.ncpu > 1:
        # pool    = Pool(self.ncpu)
        #     threads = []

        # check if (and which) PSF parameters should be fitted
        psf_names = []
        psf_values = []
        for i in xrange(self.psf_attributes.npars):
            if self.psf_attributes.free[i] and not self.psf_attributes.fitted[i]:
                psf_names.append(self.psf_attributes.names[i])
                psf_values.append(self.psf_attributes.data[self.jj, i, 0])

        # Initialize arrays that will hold results
        results = []
        snratios = []
        posfitted = []  # for each source, a boolean flag indicating if the source coordinates are fitted
        psffitted = []  # for each source, a boolean flag indicating if the PSF parameters are fitted

        # Loop over all PSF sources; add source to residuals, fit resulting array and subtract source again.
        for i in self.psf_indices:

            names = ["flux"]  # the names of the free parameters for this PSF
            x0 = [self.sources.data[self.jj, self.sources.specrow[i], 0]]  # the initial guesses for the parameter
            # fit for this PSF
            snratios.append(x0[0] / self.sources.data[self.jj, self.sources.specrow[i], 1])

            if self.update_coordinates[i]:

                # get index of source in array of IFS coordinates, save them to place updated coordinates at the right
                # index
                assert self.sources.ifs_coordinates_loaded[i], "PSF source has no IFU coordinates available."
                posindex = self.sources.ifs_coordinates_loaded[:i].sum()

                names.extend(["xc", "yc"])
                x0.extend([self.sources.ifs_coordinates[self.jj, posindex, 0, 0],
                           self.sources.ifs_coordinates[self.jj, posindex, 1, 0]])
                posfitted.append(True)
            else:
                posfitted.append(False)

            if i in self.sources.psf_indices:
                names.extend(psf_names)
                x0.extend(psf_values)
                psffitted.append(True)
            else:
                psffitted.append(False)

            psf = self.psf_instances[i]

            sigma = self.sigma[psf.used]
            data = self.current[psf.used] + x0[0] * psf.f / sigma

            psf.transform = self.transform[psf.used]
            psf.maxrad = np.inf

            if data.size <= len(x0):
                result = (x0, None, {}, "Number of free parameters exceeds number of datapoints.")
            else:
                try:
                    result = optimize.leastsq(self.optimize_single_psf, x0=x0, Dfun=self.jacobi_single_psf, col_deriv=1,
                                              args=(names, psf, data, sigma), full_output=1, epsfcn=0.0)
                except:  # optimize.minpack.error as error: ## error handler to avoid code that code crashes completely
                    result = (
                        x0, None, {}, "Exception of type optimize.minpack.error encountered while executing fit.", 99)

                    # If uncommented, the following lines save a FITS image of the source for which the fit failed.
                    # used = self.ifs_data.instrument.dist2spax(psf.x, psf.y).argmin(axis=0)
                    # calcdata = np.zeros(self.ifs_data.instrument.nspaxel, dtype=np.float32)
                    # calcdata[used] += data
                    # outdata  = self.ifs_data.instrument.make_grid(calcdata)[int(psf.y.min()):int(psf.y.max()),
                    #                                                         int(psf.x.min()):int(psf.x.max())]
                    # outhdu   = fits.PrimaryHDU(outdata)
                    # outhdu.writeto("{0}_{1}.fits".format(self.jj, psf.src_id), clobber=True)

            # remove individual positions and/or PSF parameters for this profile
            for name in names[1:]:
                delattr(psf, name)

            results.append(result)

            psf.transform = self.transform
            psf.maxrad = self.maxrad
            # if self.ncpu > 1:
            #     threads.append(pool.apply_async(self.wrapper_optimize_single_psf, args=(
            #         x0, names, self.psf_instances[i], flux)))
            # else:
            #     results.append(self.wrapper_optimize_single_psf(
            #         x0=x0, names=names, psf=self.psf_instances[i], flux=flux))

        # if self.ncpu > 1:
        #     pool.join()
        #     pool.close()
        #     results = [threads[j].get() for j in len(self.psf_indices)]

        # collect results from individual PSF fits
        ifs_coords = []
        ref_coords = []
        psf_pars = []
        psf_snrs = []

        for n in xrange(len(results)):
            result = results[n]
            i = self.psf_indices[n]

            success = (result[-1] > 0) & (result[-1] <= 4)
            if not success:
                logging.error("{0} PSF fit failed for source #{1} (S/N={2:.1f}): {3}".format(
                    self.prefix_string, self.sources.ID[i], float(snratios[n]), result[-2]))
                continue

            logstring = "{0} Successfully fitted PSF of source #{1} (S/N={2:.1f}): f={3:.1f}".format(
                self.prefix_string, self.sources.ID[i], float(snratios[n]), result[0][0])

            if posfitted[n]:
                if self.full_coordinates_fit:  # if IFS coordinates should be fitted, update them directly
                    j = self.sources.ifs_coordinates_loaded[:i].sum()  # get index of source in IFS coordinates array
                    self.sources.ifs_coordinates[self.jj, j, :, 0] = result[0][1:3]
                else:  # save the coordinates to fit the transformation afterwards
                    ifs_coords.append(result[0][1:3])
                    ref_coords.append([self.sources.x[i], self.sources.y[i]])
                logstring += " xc={0:.2f} yc={1:.2f}".format(float(result[0][1]), float(result[0][2]))
            if psffitted[n]:
                psf_pars.append(result[0][-len(psf_names):])
                psf_snrs.append(snratios[n])
                for j in xrange(len(psf_names)):
                    logstring += " {0}={1:.2f}".format(psf_names[j], float(psf_pars[-1][j]))
            logging.info(logstring)

        # update coordinate transformation if available and possible
        if len(ifs_coords) >= 2:

            self.sources.transformation.data[self.jj, :, 0] = self.sources.transformation._fit_to_data(
                np.asarray(ref_coords), np.asarray(ifs_coords))

            logstring = "{0} Fitted coordinate transformation: ".format(self.prefix_string)
            for i in xrange(self.sources.transformation.npars):
                # updating the parameters in the PSF-class overwrites the (direct) source positions used for the fit
                getattr(self.PSF, "set_{0}".format(self.sources.transformation.names[i]))(
                    self.sources.transformation.data[self.jj, i, 0])
                logstring += "{0}={1:.3f} ".format(self.sources.transformation.names[i],
                                                   float(self.sources.transformation.data[self.jj, i, 0]))
            logging.info(logstring)

        elif not self.full_coordinates_fit and (~self.sources.ifs_coordinates_fitted).any():
            logging.error("{0} Cannot update coordinate transformation because <2 sources fitted successfully.".format(
                self.prefix_string))

        # update PSF parameters
        if self.psf_attributes.contains_free_parameters:

            # update parameter values in the attributes of the PSF and in the actual PSF profiles; for this, only
            # the results of the successful fits are used. If no fit was successful, nothing is updated.
            if len(psf_pars) == 0:
                logging.error("{0} Cannot update PSF parameters because no star yielded a successful fit.".format(
                    self.prefix_string))
            else:
                i = 0
                logstring = "{0} Updated PSF parameters:".format(self.prefix_string)
                for par in psf_names:
                    ii = self.psf_attributes.names.index(par)

                    # weight individual parameter fits using S/N of sources
                    self.psf_attributes.data[self.jj, ii, 0] = np.average([v[i] for v in psf_pars], weights=psf_snrs)
                    getattr(self.PSF, "set_{0}".format(par))(self.psf_attributes.data[self.jj, ii, 0])

                    logstring += "{0}={1:.2f} ".format(par, float(self.psf_attributes.data[self.jj, ii, 0]))
                    i += 1

                logging.info(logstring)

    def optimize_single_psf(self, x0, names, psf, data, sigma):
        """
        This is the optimization routine that is passed to the Levenberg-
        Marquard solver scipy.optimize.leastsq. During each iteration, a
        PSF array is created, scaled and subtracted from the data. The
        residuals of the fit are returned.
        
        Parameters
        ----------
        x0 : array_like
            The initial guesses of the parameters that are fitted. The first
            parameter must be the flux of the source.
        names :  array_like
            The names of the parameters that are fitted. Must have same length
            as 'x0' of course.
        psf : an instance of class pampelmuse.core.psf.profiles.Profile or
              one of its subclasses
            The PSF model for the source.
        data : nd_array
            The data array used to fit the parameters. Note that the pixel
            values should be normalized by their respective uncertainties.
        sigma : nd_array
            The uncertainties for the 'data' array.

        Returns
        -------
        bestfit : nd_array
            The values of the free parameters obtained in the fit.
        """
        # update PSF using current parameter estimates
        for i in xrange(1, len(x0)):
            try:
                # the attribute is only set in the 'psf'-instance of the class, not in all instances! This is different
                # from the behaviour of getattr(self.PSF, "set_<PAR>")(<VALUE>)
                setattr(psf, names[i], x0[i])
            except AssertionError as error:  # if parameter cannot be updated, raise LinAlgError
                logging.error("{0} Encountered AssertionError: {1}".format(self.prefix_string, error))
                raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], names[i]))
        # return residuals
        return data - x0[0] * (psf.f / sigma if sigma is not None else psf.f)

    def jacobi_single_psf(self, x0, names, psf, data, sigma):
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
        jacob[0] -= (psf.f / sigma if sigma is not None else psf.f)
        for i in xrange(1, len(x0)):
            diff_i = getattr(psf, "diff_{0}".format(names[i]))
            jacob[i] -= x0[0] * (diff_i / sigma if sigma is not None else diff_i)
        return jacob

    def parameter_fit_on_small_ifu(self):
        """
        Perform the positions/PSF fit on the whole array. A non-linear least-
        squares fit that uses Levenberg-Marquardt optimization is performed
        with all resolved sources. This fitting procedure is recommended only
        for IFUs with a small number of spatial pixels. The results of the fit
        are used to update the PSF and/or the parameters of the coordinate
        transformation.
        """

        NCPU = self.ncpu  # no parallelization for this part yet
        self.ncpu = 1  # so set ncpu=1 and save its original value

        # Fit position parameters
        if self.sources.contains_free_coordinates:

            # collect free parameters
            fitPars = []
            parNames = []
            for i in xrange(self.sources.transformation.npars):
                if self.sources.transformation.free[i] and not self.sources.transformation.fitted[i]:
                    fitPars.append(self.sources.transformation.data[self.jj, i, 0])
                    parNames.append(self.sources.transformation.names[i])

            # call Levenberg-Marquard solver from scipy.optimize (& provide Jacobian)
            bestfit = optimize.leastsq(self.optimize_multi_psf, x0=fitPars, Dfun=self.jacobi_multi_psf, col_deriv=1,
                                       args=(parNames,), epsfcn=0.0)[0]

            # update parameters using fit results
            for i in xrange(len(parNames)):
                ii = self.sources.transformation.names.index(parNames[i])
                self.sources.transformation.data[self.jj, ii, 0] = bestfit[i]

            # some logging
            if logging.getLogger().getEffectiveLevel() <= 20:
                logging.info("{0} Fitted coordinate transformation:".format(self.prefix_string))
                for i in xrange(self.sources.transformation.npars):
                    logging.info("    {0}={1:.2f}".format(self.sources.transformation.names[i],
                                                          float(self.sources.transformation.data[self.jj, i, 0])))

        # fit PSF parameters
        if self.psf_attributes.contains_free_parameters:

            # collect free parameters
            fitPars = []
            parNames = []
            for i in xrange(self.psf_attributes.npars):
                if self.psf_attributes.free[i] and not self.psf_attributes.fitted[i]:
                    fitPars.append(self.psf_attributes.data[self.jj, i, 0])
                    parNames.append(self.psf_attributes.names[i])

            # call Levenberg-Marquardt solver from scipy.optimize (& provide Jacobian)
            try:
                bestfit = optimize.leastsq(self.optimize_multi_psf, x0=fitPars, Dfun=self.jacobi_multi_psf, col_deriv=1,
                                           args=(parNames,), epsfcn=0.0)[0]
                success = True
            except:  # optimize.minpack.error: # error handler to avoid code that code crashes completely, seems to
                # depend on numpy version which error is thrown
                logging.error("{0} PSF fit failed.".format(self.prefix_string))
                success = False

            # update parameters using fit results
            if success:
                for i in xrange(len(parNames)):
                    ii = self.psf_attributes.names.index(parNames[i])
                    self.psf_attributes.data[self.jj, ii, 0] = bestfit[i]

                # some logging
                if logging.getLogger().getEffectiveLevel() <= 20:
                    logging.info("{0} Fitted PSF model:".format(self.prefix_string))
                    for i in xrange(self.psf_attributes.npars):
                        logging.info("    {0}={1:.2f}".format(self.psf_attributes.names[i],
                                                              float(self.psf_attributes.data[self.jj, i, 0])))

        self.ncpu = NCPU  # recover original number of CPUs to be used

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
        # logging.debug("Shape of Jacobi matrix: {0}".format(self.jacobi_multi_psf(x0, names).shape))
        # logging.debug("Shape of data: {0}".format(self.data.shape))
        # update parameters in PSF-class
        for i in xrange(len(x0)):
            try:
                getattr(self.PSF, "set_{0}".format(names[i]))(x0[i])
            except AssertionError:  # if parameter cannot be updated, raise LinAlgError
                raise np.linalg.linalg.LinAlgError("Invalid value '{0}' for parameter '{1}'.".format(x0[i], names[i]))

        # calculate new model
        psf_matrix = self.build_matrix(update_unresolved=False)[0]

        model = psf_matrix * sparse.dia_matrix((self.sources.data[self.jj, self.valid_components, 0], [0]),
                                               shape=(self.valid_components.sum(), self.valid_components.sum()))
        # return residuals
        return (self.data / self.sigma) - model.sum(axis=1).getA1()

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
        t0 = time.time()
        jacob = np.zeros((len(x0), self.data.size), dtype=np.float64)

        # The influence of the unresolved sources (if any) is currently ignored when calculating the Jacobi matrix,
        # any sky component has no influence anyway.
        for i in np.flatnonzero((self.sources.status > 0) & (self.sources.status < 4)):
            psf = self.psf_instances[i]
            ii = self.sources.specrow[i]
            for j in xrange(len(x0)):
                diff_i = getattr(psf, "diff_{0}".format(names[j]))
                jacob[j, psf.used] -= self.sources.data[self.jj, ii, 0] * (diff_i / self.sigma[psf.used])

        t1 = time.time()
        logging.debug("Time needed to create Jacobi matrix: {0:.3f}s".format(t1 - t0))

        return jacob

    def update_parameters(self):
        """
        Method is used to update the initial guesses of the parameters of
        the PSF and the coordinate transformation prior to a PSF fit.
        The new initial guesses are determined by calculating the mean
        of the results of the `Ntrack'/2 nearest images that have been
        fitted so far. For the parameters of the PSF and the coordinate
        transformation, a kappa-sigma clipping (kappa=2.5) is performed 
        on the previous results to exclude potential outliers.
        """
        if self.ntrack == 0:
            # no updating requested
            return

        assert(self.jj in self.layers_to_analyse)
        i = np.where(self.layers_to_analyse == self.jj)[0][0]

        if i < 2:
            # nothing to update yet
            return
        elif i <= 2 * self.ntrack:
            # use all images that have been fitted so far
            lastfits = self.layers_to_analyse[:i]
        else:
            lastfits = self.layers_to_analyse[i - 2:max(0, i - 2 - 2 * self.ntrack):-2]

        logging.debug("{0} Updating initial guesses of parameters using results from adjacent {1} layers.".format(
            self.prefix_string, lastfits.size))
        logging.debug("{0} Previous fits in focus: {1}".format(self.prefix_string, lastfits))

        if self.sources.contains_free_coordinates:
            if not self.full_coordinates_fit:  # update coordinate transformation
                for i in np.flatnonzero(self.sources.transformation.free & ~self.sources.transformation.fitted):
                    self.sources.transformation.data[self.jj, i, 0] = np.median(
                        self.sources.transformation.data[lastfits, i, 0])
            else:  # update coordinates
                assert(self.sources.ifs_coordinates_loaded[self.update_coordinates].all())
                update = self.update_coordinates[self.sources.ifs_coordinates_loaded]
                self.sources.ifs_coordinates[self.jj, update, :, 0] = np.median(
                    self.sources.ifs_coordinates[lastfits], axis=0)[update, :, 0]

        if self.psf_attributes.contains_free_parameters:  # update PSF
            for i in np.flatnonzero(self.psf_attributes.free & ~self.psf_attributes.fitted):
                self.psf_attributes.data[self.jj, i, 0] = np.median(self.psf_attributes.data[lastfits, i, 0])
