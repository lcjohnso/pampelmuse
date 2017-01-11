"""
source_selector.py
==================

Copyright 2013-2016 Sebastian Kamann

This file is part of PampelMuse.

PampelMuse is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

PampelMuse is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PampelMuse. If not, see <http://www.gnu.org/licenses/>.


+++
Purpose
-------
This module provides the class SourceSelector that is designed to
facilitate the process of selecting a useful subset of all sources
available in a photometric catalog for the analysis with PampelMuse.

Added
-----
2016/08/09, rev. 353

Latest SVN revision
-------------------
381, 2017/01/11
"""
import logging
import sys
from multiprocessing import cpu_count, Pool
import numpy as np
from scipy import spatial
from .aperture import AperPhoto
from .background import BackgroundGrid
from .sources import Sources
from ..instruments import Instrument, Muse, MusePixtable
from ..filters import ALIASES
from ..psf import *
from ..utils.io import read_asciifile
from ..utils.image import find_offset
from ..utils.statistics import der_snr


__author__ = 'Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)'
__revision__ = 381


logger = logging.getLogger(__name__)


def init_worker(variance, *args):
    """
    This function has the purpose of defining a global variable pointing
    to the variances of the IFS data. This is done in order to reduce the
    overhead of the initialisation of the parallel computation of the S/N
    estimates.

    Parameters
    ----------
    variance : nd_array
        The variances of the individual pixels of the data.
    args
    """
    global g_variance
    g_variance = variance.flatten()


def image_from_psf(sources_data):
    """
    Create a model image from a PSF profile and a set of centroid coordinates
    and magnitudes.

    Parameters
    ----------
    sources_data : dict
        The dictionary must contain the following two keys:
        - psf_instances : an instance of a PSF class
              The PSF profile is required to determine the flux distribution
              across the FoV
        - uc : array_like
            The centroids of the individual stars along the x-direction of the
            reference system
        - vc : array_like
            The centroids of the individual stars along the y-direction of the
            reference system
        - mag : array_like
            The magnitudes of the individual stars in the chosen passband.

    Returns
    -------
    image : nd_array
        The model image created from the input data.
    """
    psf = sources_data['psf']
    image = np.zeros((psf.n_pixel, ), dtype='<f4')

    for i in range(len(sources_data['uc'])):
        psf.uc = sources_data['uc'][i]
        psf.vc = sources_data['vc'][i]
        psf.mag = sources_data['mag'][i]
        psf.reset()

        image[psf.used] += psf.f
        logger.debug('Image creation: Processed star {0}/{1}'.format(i+1, len(sources_data['uc'])))

    return image


def estimate_snr(sources_data):
    """
    Calculate the S/N ratio of all stars in the scene.

    Parameters
    ----------
    sources_data : dict
        The dictionary must contain the following two keys:
        - psf_instances : an instance of a PSF class
              The PSF profile is required to determine the flux distribution
              across the FoV
        - uc : array_like
            The centroids of the individual stars along the x-direction of the
            reference system
        - vc : array_like
            The centroids of the individual stars along the y-direction of the
            reference system
        - mag : array_like
            The magnitudes of the individual stars in the chosen passband.

    Returns
    -------
    snratios : nd_array
        A 1dim. array containing the S/N-ratio for each source
    totfluxes : float
        The integrated flux of all point sources over the individual
        spaxels.
    """
    global g_variance

    snratios = []
    totfluxes = 0.

    psf = sources_data['psf']
    for i in range(len(sources_data['uc'])):
        psf.uc = sources_data['uc'][i]
        psf.vc = sources_data['vc'][i]
        psf.mag = sources_data['mag'][i]
        psf.reset()

        if (g_variance[psf.used] == 0).any():
            logger.error("Sigma image contains invalid pixels with values <=0.")

        # measure S/N
        snratios.append(np.sqrt(((psf.f / np.sqrt(g_variance[psf.used])) ** 2).sum()))

        logger.debug("Estimated S/N for source {0}/{1}: {2}.".format(i+1, len(sources_data['uc']), snratios[-1]))

        # add the flux contribution of this source in the FoV
        totfluxes += psf.f.sum()

    return np.array(snratios, dtype=np.float32), totfluxes


class SourceSelector(object):
    """
    This class is designed to perform the selection process for the sources
    that are used in the PampelMuse extraction process. It defines methods
    that perform the individual steps of the source selection that are called
    by the INITFIT routine of PampelMuse.
    """

    def __init__(self, photometry_file, ifs_data, passband="f814w", zeropoint=None, position_fit=2):
        """
        Initialize a new instance of the SourceSelector class.

        Parameters
        ----------
        photometry_file : str
            The name of the ascii-file containing the photometric reference
            catalog. It should be in a format that is supported by PampelMuse.
            Check the manual for details.
        ifs_data : an instance of pampelmuse.instruments
            This instance should contain all the information about the IFS
            data that is being analysed.
        passband : str, optional
            The name of the photometric filter for which the source selection
            should be performed.
        zeropoint : float, optional
            An initial guess for the photometric zeropoint of the IFS data in
            the choosen passband.
        position_fit : int, optional
            A flag indicating how the source positions on the IFS data should
            be determined. Can be one of:
            0 - from a coordinate transformation with fixed parameters
            1 - from a coordinate transformation with fixed rotation and scale
                parameters but variable offsets.
            2 - from a coordinate transformation with variable rotation and
                scale parameters and variable offsets.
        """
        logger.info('Starting source selection based on {0}-magnitudes.'.format(passband.upper()))
        self.passband = passband
        self.zeropoint = zeropoint

        assert isinstance(ifs_data, Instrument), 'Parameter "ifs_data" muse be instance of an Instrument-class.'
        self.ifs_data = ifs_data
        self.use_wcs = self._check_wcs()

        # get results from crowded-field photometry, initialize sources
        logger.info('Reading photometry results from file "{0}".'.format(photometry_file))

        # Note that the routine will first check if the first (uncommented) line in the file is a header line from
        # which it can infer the column content. Only if that fails, it will use the association in 'usecols' and
        # 'usenames' the 'aliases' provide for each valid passband a list of other possible names the column may have
        self.cfp_results = read_asciifile(photometry_file, usecols=(0, 1, 2, 3, 4, 5),
                                          usenames="id,x,y,{0},ra,dec".format(self.passband),
                                          optional="x,y,ra,dec", aliases=ALIASES)

        # make sure IDs are available
        if "id" not in self.cfp_results.keys():
            logger.critical("Did not find IDs in photometry file.")
            raise IOError("Source IDs must be provided with photometry")

        # make sure magnitudes are available
        if self.passband.lower() not in self.cfp_results.keys():
            logger.critical("Did not find magnitudes in photometry file.")
            raise IOError("Source magnitudes must be provided with photometry.")

        # make sure either RA & Dec or x & y are available; if both are available, RA & Dec are preferred over x & y
        if self.use_wcs:
            if "ra" not in self.cfp_results.keys() or "dec" not in self.cfp_results.keys():
                self.use_wcs = False

        if self.use_wcs:
            logger.info("Using RA & DEC coordinates as input.")

            ra = np.asarray(self.cfp_results["ra"], dtype=np.float64)
            dec = np.asarray(self.cfp_results["dec"], dtype=np.float64)

            # get IFS spaxel coordinates from reference RA and DEC
            if self.ifs_data.wcs.wcs.naxis == 3:
                x, y, _ = self.ifs_data.wcs.wcs_world2pix(ra, dec, np.ones(ra.shape), 1)
            else:
                x, y = self.ifs_data.wcs.wcs_world2pix(ra, dec, 1)
            self.cfp_results["x"] = x.tolist()
            self.cfp_results["y"] = y.tolist()

        else:
            if "x" not in self.cfp_results.keys() or "y" not in self.cfp_results.keys():
                logger.critical("Cannot find input source coordinates.")
                raise IOError("Either x & y or RA & Dec must be provided with photometry.")

        # Perform sanity checks on provided source list (exclude sources with magnitudes NaN or >99):
        self.valid = ~np.isnan(np.asarray(self.cfp_results[passband.lower()], dtype=np.float32))
        self.valid &= (np.asarray(self.cfp_results[passband.lower()], dtype=np.float32) < 99.)
        if not self.valid.all():
            logger.warning("Ignoring {0} source(s) with undefined magnitudes.".format((~self.valid).sum()))
        if len(np.unique(self.cfp_results["id"])) < len(self.cfp_results["id"]):
            logger.error("Catalog contains duplicated IDs.")

        self.position_fit = position_fit
        self.sources = None
        self.psf_class = None
        self.psf_attributes = None
        self.identification_image = None
        self.data_image = None
        self.var_image = None
        self.ccxoff = 0.0
        self.ccyoff = 0.0
        self.signal_to_noise = None

    def _check_wcs(self):
        """
        Internal method that checks if all necessary WCS information is
        available for the IFS data to perform the source selection based on
        RA and Dec instead of pixel coordinates.

        Returns
        -------
        use_wcs : bool
            A flag indicating if enough WCS information is available for the
            IFS data.
        """
        if self.ifs_data.wcs is None:
            return False
        elif self.ifs_data.wcs.naxis < 2:
            return False
        elif (self.ifs_data.wcs.wcs.crval[:2] == 0).any():
            return False
        elif np.linalg.det(self.ifs_data.wcs.wcs.cd[:2, :2]) == 1.0:
            return False
        return True

    def background_setup(self, skysize=-1):
        """
        Method to add one or more background components.

        One background component corresponds to a constant background across
        the field of view. If more than one background component should be
        used, the method will set up a grid. The background in each pixel will
        then be obtained by interpolation of the grid points.

        Parameters
        ----------
        skysize : int, optional
            The size of each cell of the background grid in pixels. Set to -1
            for a constant background across the field of view.

        Returns
        -------
        n_background : int
            The number of background components used.
        """
        if skysize <= 0:
            background = BackgroundGrid.constant()
        else:
            background = BackgroundGrid.from_shape(shape=self.ifs_data.shape, rscale=skysize/2.)
        for i, [y, x] in enumerate(background.centroids):
            # ignore sky components outside FoV
            if self.ifs_data.distance_to_edge([[x, y]])[0] < -skysize and skysize > 0:
                continue
            self.valid = np.r_[self.valid, True]
            self.cfp_results["id"].append(str(-i-1))
            self.cfp_results["x"].append(str(x))
            self.cfp_results["y"].append(str(y))
            self.cfp_results[self.passband.lower()].append("0")
            if "ra" in self.cfp_results.keys():
                self.cfp_results["ra"].append("0")
            if "dec" in self.cfp_results.keys():
                self.cfp_results["dec"].append("0")
        return len(background.centroids)

    def make_sources_instance(self):
        """
        Method to create an instance of pampelmuse.core.Sources containing
        the current list of stellar and background sources.

        The method will run the intialisation method of the Sources-class and
        copy the dispersion information from the IFs data to the new instance.
        Afterwards, the instance can be accessed as SourceSelector().sources.
        """
        self.sources = Sources(ids=np.asarray(self.cfp_results["id"], dtype=np.int32)[self.valid])
        self.sources.x = np.asarray(self.cfp_results["x"], dtype=np.float32)[self.valid]
        self.sources.y = np.asarray(self.cfp_results["y"], dtype=np.float32)[self.valid]
        self.sources.mag = np.asarray(self.cfp_results[self.passband.lower()], dtype=np.float32)[self.valid]
        if "ra" in self.cfp_results.keys():
            self.sources.ra = np.asarray(self.cfp_results["ra"], dtype=np.float64)[self.valid]
        if "dec" in self.cfp_results.keys():
            self.sources.dec = np.asarray(self.cfp_results["dec"], dtype=np.float64)[self.valid]

        # unless a MUSE pixtable is being analysed, copy dispersion information to sources class
        if not isinstance(self.ifs_data, MusePixtable):
            if self.ifs_data.wcs.naxis > 2:
                self.sources.crval = self.ifs_data.wcs.wcs.crval[2]
                self.sources.crpix = self.ifs_data.wcs.wcs.crpix[2]
                self.sources.cdelt = self.ifs_data.wcs.wcs.cd[2, 2]
                self.sources.ctype = self.ifs_data.wcs.wcs.ctype[2]
                self.sources.cunit = self.ifs_data.wcs.wcs.cunit[2].name
            else:
                self.sources.crval = self.ifs_data.wave[0]
                self.sources.crpix = 1
                self.sources.cdelt = np.diff(self.ifs_data.wave).mean()

    def prepare_psf(self, profile='moffat', parameters=None):
        """
        Method to initialize the PSF that should be used in the source
        selection process.

        The method will define two properties, one containing the analytical
        profile that should be used to model the PSF (accessible afterwards as
        SourceSelector().psf_class) and one containing the values of the
        individual parameters that define the profile (accessible afterwards
        as SourceSelector().psf_attributes).

        Parameters
        ----------
        profile : str, optional
            The name of the analytical profile used to model the PSF.
            Currently 'moffat' (default), 'gauss', and 'double_gauss' are
            supported.
        parameters : list, optional
            The values of the parameters of the chosen profile, provided as a
            list that contains one dictionary per parameter. Each dictionary
            should contain three keys:
            - 'name': The name of the parameter
            - 'value': The (initial) value of the parameter
            - 'free': Boolean flag indicating if the parameter value should be
                      optimized in the analysis.
            By default, a round Moffat profile with beta=2.5 and fwhm=3.0pixel
            is used.
        """
        if profile == "moffat":
            self.psf_class = Moffat
        elif profile == "gauss":
            self.psf_class = Gaussian
        elif profile == "double_gauss":
            self.psf_class = DoubleGaussian
        else:
            logger.error('Unknown PSF type: "{0}"'.format(profile))
            return

        if parameters is None:
            parameters = [{'name': 'beta', 'value': 2.5, 'free': False},
                          {'name': 'fwhm', 'value': 3.0, 'free': False}]

        self.psf_attributes = Variables(profile=profile)
        for parameter in parameters:
            # include value and fit request in PSF variables
            j = self.psf_attributes.names.index(parameter['name'])
            self.psf_attributes.data[0, j, 0] = parameter['value']
            self.psf_attributes.free[j] = parameter['free']

        self.psf_attributes.info()

    def create_mock_image(self, n_stars=500, n_cpu=1):
        """
        Method to create a mock image from the catalogue of sources.

        This method creates a two dimensional image using the coordinates and
        magnitudes for the available sources and the current PSF model. The
        dimensions of the image are chosen so that all available sources are
        included. For large catalogs, the parameter 'n_stars' makes sure that
        the computation times stay moderate by only adding the subset of the
        brightest stars to the image. After creation, the image is available
        as SourceSelector().identification_image.

        The image creation is parallelized and it uses the function
        image_from_psf defined in this method.

        Parameters
        ----------
        n_stars : int, optional
            The maximum number of sources that is included in the image. The
            selection of sources is based on magnitudes (bright to faint).
        n_cpu : int, optional
            The number of CPUs used to prepare the image.
        """
        logger.info("Creating mock image from photometry catalogue...")

        if self.sources is None:
            logger.warning('Need to initialize instance of Sources-class first.')
            self.make_sources_instance()

        if self.psf_class is None:
            logger.warning('Need to initialize PSF first.')
            self.prepare_psf()

        shape = (int(self.sources.y.max() - self.sources.y.min()),
                 int(self.sources.x.max() - self.sources.x.min()))

        # initialize single PSF instance per CPU
        logger.info("   adding sources, working on {0} CPUs...".format(n_cpu))
        if n_cpu is None or n_cpu == -1:
            n_cpu = cpu_count()  # use all available CPUs

        psf_parameters = dict(
                (name, self.psf_attributes.data[:, i, 0]) for i, name in enumerate(self.psf_attributes.names))

        # select n_stars brightest sources - note that the selection has no effect on argsort because
        # the background components are at the end of the array
        include = np.argsort(self.sources.mag[:-self.sources.nbackground])[:n_stars]
        to_cpu = np.arange(include.size) % n_cpu

        # to get reasonable count rates in mock image, adjust zeropoint unless known
        zeropoint = self.sources.mag[include[-1]] + 7.5 if self.zeropoint is None else self.zeropoint

        args = []
        for n in range(n_cpu):
            used = include[to_cpu == n]
            psf = self.psf_class.from_shape(shape=shape, uc=0., vc=0., mag=0., maxrad=10., **psf_parameters)
            args.append({'psf': psf,
                         'uc': self.sources.x[used] - self.sources.x.min(),
                         'vc': self.sources.y[used] - self.sources.y.min(),
                         'mag': self.sources.mag[used] - zeropoint})

        if n_cpu > 1:
            pool = Pool(n_cpu)
            _results = pool.map_async(image_from_psf, args)
            pool.close()
            pool.join()
            _image = np.sum(_results.get(), axis=0)
        else:
            _image = image_from_psf(args[0])

        # 3. resample image on regular (x, y) grid
        logger.info("   resampling image on regular 2D grid...")
        self.identification_image = np.zeros(shape, dtype=np.float32)
        self.identification_image[psf.y.astype('<i4'), psf.x.astype('<i4')] = _image

    def create_image_from_ifs(self, n_bin=10):
        """
        This method creates a two-dimensional image from the IFS data.

        The image is created by integrating the IFS data over the passband in
        which the magnitudes were provided. For large data cubes this can be
        quite time-consuming. In those cases, consider using the the parameter
        'n_bin' to reduce the computation time. Afterwards, the image is
        accessible via SourceSelector().data_image. Note that the method will
        also create an image with the average variances, accessible via
        SourceSelector().var_image.

        Parameters
        ----------
        n_bin : int, optional
            The binning factor; if >1, only every n_bin'th layer will be used
            to create the image.
        """

        logger.info("Creating mock {0}-band image from IFS data...".format(self.passband))
        (self.data_image, self.var_image), weights = self.ifs_data.integrate_over_filtercurve(
                self.passband, apply_variances=True, nbin=n_bin)

        # The sigma values are valid for the broadband image and must be increased to be representative for a typical
        # layer. As the image was obtained by integrating over a filter curve, the sigma values are multiplied by
        # sqrt(N) where N is the effective width of the filter curve in units of the spectral sampling.
        self.var_image *= weights[(np.arange(weights.size) % 10) == 0].sum()

        mask = np.isnan(self.data_image)
        if mask.all():
            logger.error("{0}-image created from IFS data contains only NaNs.".format(self.passband))
            logger.error("Check overlap between {0} throughput and IFS wavelength coverage.".format(self.passband))

        # if a MUSE data cube has been provided, try to determine edge of field of view as smallest polygon to includes
        # all valid spaxels.
        if mask.any() and isinstance(self.ifs_data, Muse):
            self.ifs_data.edge = ~mask

        self.var_image[mask] = np.inf
        self.data_image[mask] = 0

        if self.data_image.sum() <= 0:
            logger.error("Flux sum in {0}-image is negative.".format(self.passband))

    def cross_correlate_images(self, ccmaxoff=None, init_guess=None):
        """
        This method tries to measure the offset between the IFS data and the
        coordinates in the source catalogue.

        The method will cross-correlate the image created from the IFS_data
        with the mock image created from the source catalog and then find the
        peak intensity of the cross-correlation signal if a pre-defined area.
        The coordinates of this point are considered as the most-likely offset
        between the two images and are accessible via SourceSelector().ccxoff
        and SourceSelector().ccyoff afterwards.

        Parameters
        ----------
        ccmaxoff : float, optional
            The maximum allowed offset between the two images in pixels.
        init_guess : tuple, optional
            Initial guesses for the offset along the x- and y-axes. The peak
            in the cross-correlation signal is searched in a square with half-
            width 'ccmaxoff' centred on the initial guess.
            IMPORTANT: The offsets are measured relative to the initial guess
                       coordinates.
        """

        if not self.use_wcs:
            logger.error('Cannot perform cross-correlation unless valid WCS instance is available.')
            return

        if self.identification_image is None:
            logger.warning('Need to create identification image for cross-correlation first.')
            self.create_mock_image()

        if self.data_image is None:
            logger.warning('Need to create image from IFS data for cross-correlation first.')
            self.create_image_from_ifs()

        if init_guess is None:  # need to correct for shift between image and catalogue.
            init_guess = (-self.sources.x.min(), -self.sources.y.min())

        if self.data_image.ndim == 1:
            cc_data_image = self.ifs_data.make_image(self.data_image, fill_value=0)
        else:
            cc_data_image = self.data_image

        self.ccxoff, self.ccyoff = find_offset(cc_data_image, self.identification_image, max_off=ccmaxoff,
                                               init_guess=init_guess)
        # Note that the offset are provided with negative signs because the cross-correlation routine determines
        # the offset from the reference image to the IFS data, while we need it vice versa.
        logger.info("   pointing offset from cross-correlation dx={0:.1f}, dy={1:.1f}".format(self.ccxoff, self.ccyoff))

        # initialize/update coordinate transformation
        self.sources.transformation.data = np.array([1., 0., 0., 1., -self.ccxoff, -self.ccyoff], dtype=np.float32)
        if self.position_fit == 2:
            self.sources.transformation.free[:] = True
        elif self.position_fit == 1:
            self.sources.transformation.free[-2:] = True

    def identify_footprint(self, n_stars=100):
        """
        This method allows for a interactive determination of the location of
        the IFS data relative to the source catalog.

        This method will open the IFS data and the mock image created from the
        source catalogue in the PampelMuse GUI. Then the locations of the
        individual sources will be overplotted on the mock image. If the
        cross-correlation via SourceSelector().cross_correlate_images() has
        been run successfully before, the method will also overplot the likely
        location of the IFS data on the mock image and the initial guesses of
        the sources on the IFS data. Note that this method requires PyQt4 to
        be installed on the system.

        Parameters
        ----------
        n_stars : int, optional
            The number of stars that are initially over-plotted on the mock
            image. Note that it will be possible to adjust this number once
            the GUI has been loaded.
        """
        from PyQt4 import QtGui
        from ..graphics.pampelmuse_gui import PampelMuseGui

        logger.info("Preparing GUI for source identification...")

        app = QtGui.QApplication(sys.argv)

        # get magnitude of n_stars'th brightest star
        n_stars = min(n_stars, self.sources.nsrc - self.sources.nbackground)
        mag = self.sources.mag[np.argsort(self.sources.mag[:-self.sources.nbackground])[n_stars]]

        identifier = PampelMuseGui(sources=self.sources, ifsData=self.ifs_data, refImage=self.identification_image)

        if self.use_wcs:  # overplot positions of sources on IFS as determined via cross-correlation
            identifier.addFovToReference(self.ccxoff - self.sources.x.min() + self.ifs_data.shape[1]//2,
                                         self.ccyoff - self.sources.y.min() + self.ifs_data.shape[0]//2)

            sizes = 10.*(10**(-0.2*(self.sources.mag - mag - 1.)))
            sizes[sizes <= 1] = 0
            select = (self.sources.status < 4) & (
                self.sources.mag <= mag) & (
                self.ifs_data.infov(np.vstack((self.sources.x - self.ccxoff, self.sources.y - self.ccyoff)).T))
            identifier.ifsPlot.addData(self.sources.x[select] - self.ccxoff, self.sources.y[select] - self.ccyoff,
                                       s=sizes[select], marker="o", edgecolor="g", facecolor="None")

        # account for offset between reference image and true coordinates
        (dx, dy) = (self.sources.x.min(), self.sources.y.min())
        identifier.refOffsets = np.vstack((self.sources.x - dx, self.sources.y - dy))
        identifier.maxMagSpinBox.setValue(mag)

        identifier.show()
        app.exec_()

        identified = identifier.identifiedSources
        if len(identified) >= 2:
            # determine coordinate transformation using identified sources, when providing the IFS coordinates for the
            # sources that were identified, they need to be provided in the right order (the sources were not
            # necessarily identified in the same order they appear in the catalogue)
            sorted_indices = np.argsort([list(self.sources.ID).index(entry) for entry in identified.keys()])

            self.sources.load_ifs_coordinates(ids=identified.keys())
            self.sources.ifs_coordinates[..., 0] = np.asarray(identified.values())[sorted_indices, :2]
            self.sources.fit_coordinate_transformation()
            if self.position_fit == 2:
                self.sources.transformation.free[:] = True
            elif self.position_fit == 1:
                self.sources.transformation.free[-2:] = True

        elif not self.sources.transformation.valid:
            logger.error("At least 2 sources must be identified.")

    def calibrate_photometry(self, n_cpu=1, update_zeropoint=True):
        """
        This method performs a photometric calibration of the IFS data along
        with an S/N estimation for all the sources in the catalog.

        The method will perform the following steps to estimate the average
        S/N of a source on the IFS data.
        1) Based on a hypothetical zeropoint (i.e. the one provided during the
           initialisation of the instance), the flux profiles of all sources
           are determined and a hypothetical S/N estimate is obtained. This
           step is parallelized and also yields an estimate of the total flux
           in the image.
        2) The integrated flux is compared to the sum of all pixel values in
           the broadband image created from the IFS data before. The ratio of
           these two fluxes can be converted into a magnitude difference
           between the hypothetical and the real zeropoint.
        3) All S/N estimates are corrected for this magnitude difference

        Parameters
        ----------
        n_cpu : int, optional
            the number of CPUs used in the parallelized part of the method.
        update_zeropoint : bool, optional
            Flag indicating if the zeropoint should be updated based on the
            flux comparison with the IFS data. If not, the S/N estimates of
            all sources are based on the zeropoint provided upon
            initialisation of the instance.
        """
        logger.info("Estimating S/N for all sources...")

        if self.var_image is None:
            logger.warning('Need to calculate variances before estimating S/N.')
            self.create_image_from_ifs()

        zeropoint = 25. if self.zeropoint is None else self.zeropoint

        psf_parameters = dict(
                (name, self.psf_attributes.data[:, i, 0]) for i, name in enumerate(self.psf_attributes.names))
        for i, name in enumerate(self.sources.transformation.names):
            psf_parameters[name] = self.sources.transformation.data[:, i, 0]

        # Assign sources to CPUs
        to_cpu = np.arange(self.sources.nsrc) % n_cpu
        to_cpu[self.sources.ID < 0] = n_cpu + 1
        args = []
        for n in range(n_cpu):
            used = np.flatnonzero(to_cpu == n)
            if isinstance(self.ifs_data, MusePixtable):
                psf = self.psf_class.from_shape(shape=self.var_image.shape, uc=0., vc=0., mag=0., maxrad=10.,
                                                **psf_parameters)
            else:
                psf = self.psf_class(transform=self.ifs_data.transform, uc=0., vc=0., mag=0., maxrad=10.,
                                     **psf_parameters)
            args.append({'psf': psf, 'uc': self.sources.x[used], 'vc': self.sources.y[used],
                         'mag': self.sources.mag[used] - zeropoint})

        # estimate S/N and integrated flux from sources in image
        if n_cpu > 1:
            pool = Pool(n_cpu, initializer=init_worker, initargs=(self.var_image,))

            _results = pool.map_async(estimate_snr, args)
            pool.close()
            pool.join()
            results = _results.get()

            self.signal_to_noise = np.zeros(to_cpu.size, dtype='<f4')
            for n in range(n_cpu):
                self.signal_to_noise[to_cpu == n] = results[n][0]
            intflux = np.sum([r[1] for r in results])
        else:
            init_worker(*(self.var_image,))
            self.signal_to_noise, intflux = estimate_snr(args[0])

        logger.info("Estimating the photometric zeropoint...")
        # Calculate ratio between expected flux for current zeropoint and flux in image
        fratio = intflux/self.data_image.sum()
        dmag = -2.5*np.log10(fratio)

        #  update S/N estimates
        self.signal_to_noise /= fratio

        # determine zeropoint if requested
        if update_zeropoint:
            self.zeropoint = zeropoint + dmag
        logger.info("Estimated photometric zeropoint ({0}) is {1:.2f}".format(self.passband, zeropoint + dmag))

    @property
    def fwhm(self):
        if self.psf_class is None:
            return np.nan
        elif isinstance(self.psf_class, DoubleGaussian):
            i = self.psf_attributes.names.index('f12')
            j = self.psf_attributes.names.index('fwhm1')
            k = self.psf_attributes.names.index('fwhm2')
            f12 = np.nanmean(self.psf_attributes.data[:, i, 0])
            fwhm1 = np.nanmean(self.psf_attributes.data[:, j, 0])
            fwhm2 = np.nanmean(self.psf_attributes.data[:, k, 0])
            return f12*fwhm1 + (1. - f12)*fwhm2
        else:
            i = self.psf_attributes.names.index('fwhm')
            return np.nanmean(self.psf_attributes.data[:, i, 0])

    def select(self, reslim=None, sncut=3., confusion_density=0.4, binrad=0.2, use_unres=False, unres_range=2):
        """
        This method performs the source selection process for the PSF fitting,

        Each source is given a status based on criteria such as the estimated
        S/N, the distance brighter sources or the edge of the field of view.
        The status of a source can be one of the following:
            4 - a background/sky spectrum
            2 - a resolved source for which a spectrum will be extracted
            1 - a source that should be resolved according to its brightness,
                but is too close to a brighter source. Its contribution is
                added to the brighter source in a combined PSF.
            0 - an unresolved source that is included in the grainy stellar
                background component
           -1 - an unresolved source that is not considered in the fit.

        Parameters
        ----------
        reslim : float, optional
            If this parameter is set to a value different from None, the
            method will not try to determine the confusion limit itself but
            will  only include sources brighter than the provided value.
        confusion_density : float, optional
            The source density where the confusion limit is reached, provided
            as [no. of sources]/[resolution element]. A value of 0.4 gave good
            results for the simulations performed in Kamann et al. (2013).
        sncut : float, optional
            The minimum S/N a source must have to be still included.
        binrad : float, optional
            The minimum distance for which sources are considered resolved,
            provided relative to the FWHM of the PSF profile. Sources with
            mutual distances below this value will be combined in the
            analysis.
        use_unres : bool, optional
            Flag indicating if an unresolved component should be included in
            the selection.
        unres_range : float, optional
            If an unresolved component should be included, this parameter
            gives the range in magnitudes of the sources that are included in
            the component. Note that the bright end of this range is set to
             the magnitude of the faintest resolved source.
        """
        logger.info("Selecting sources to be used in PSF fitting...")

        if self.signal_to_noise is None:
            logger.warning('Need to estimate S/N for all sources before performing selection.')
            self.calibrate_photometry()

        if np.isnan(self.fwhm):
            logger.error('Need to define PSF before source selection can be performed.')
            return

        # load IFS coordinates for all stars, average over spectral axis (if any), and bring into correct order
        # (including background components)
        self.sources.load_ifs_coordinates()
        ifs_coordinates = np.zeros((self.sources.nsrc, 2), dtype='<f4')
        ifs_coordinates[self.sources.ifs_coordinates_loaded] = np.nanmean(self.sources.ifs_coordinates[..., 0], axis=0)

        # arrays to keep track of the status of each source and the ID of the brightest source in its component
        # (that is identical to the source ID unless a brighter source is so close that the two cannot be separated
        # and are fitted as a single component)
        status = -np.ones((self.sources.nsrc,), dtype=np.int16)
        status[self.sources.ID < 0] = 4  # handle background components
        primary = np.zeros((self.sources.nsrc,), dtype=np.int32)
        primary[self.sources.ID < 0] = self.sources.ID[self.sources.ID < 0]  # handle background components

        # for larger fields of view, such as the MUSE WFM, the stellar density can be very different across its extent.
        # To deal with this, the confusion limit is estimated locally in an area of radius 10xFWHM around each star
        density = np.zeros(self.ifs_data.nspaxel, dtype=np.int16)

        # loop over magnitude-sorted list of indices of the sources the instance of the Sources-class
        # (brightest sources first)
        indices_in_sources_class = np.argsort(self.sources.mag)

        logger.info("       ID   mag    S/N  d_bright  d_edge  status")
        logger.info("                           [spx]   [spx]        ")

        for i in indices_in_sources_class:

            if status[i] == 4:  # ignore background components
                continue

            # stop loop if current source falls below (fixed) confusion limit
            if reslim is not None and self.sources.mag[i] > reslim:
                break

            xi, yi = ifs_coordinates[i]

            # get position of source relative to field of view
            distance_to_pixels = self.ifs_data.distance_to_spaxels(xi, yi)[..., 0]
            distance_to_edge = self.ifs_data.distance_to_edge([[xi, yi]])[0]

            # check if local confusion limit is reached
            slc = distance_to_pixels < (10.*self.fwhm)
            if slc.any():
                # no. of resolution elements is [no. of pixels]/[pixels per resolution element]
                n_resolution = (slc.sum()/(np.pi*(0.5*self.fwhm)**2))

                # confusion limit is given as [no. of sources]/[no. of resolution elements]
                if density[slc].sum()/n_resolution > confusion_density:
                    continue

            # get S/N estimate for the source
            snr = self.signal_to_noise[i]
            if snr < sncut:
                continue

            # penalize S/N if source is outside FoV:
            if distance_to_edge < 0:
                snr *= np.exp(-(distance_to_edge/self.fwhm)**2)
                if snr < sncut:
                    continue
            # otherwise, add source to array for calculation of stellar density
            else:
                density[distance_to_pixels.argmin()] += 1

            # calculate distance to next brighter source
            brighter = np.flatnonzero(status == 2)
            if brighter.any():
                dr = spatial.distance.cdist([[xi, yi]], ifs_coordinates[brighter])[0]
                j_near = dr.argmin()

                # issue warning if two sources have basically the same IFS coordinates
                if dr[j_near] < 1e-5:
                    logger.warning("Source #{0} might be double.".format(self.sources.ID[i]))

                # penalize S/N if a brighter source is located inside the FWHM of the PSF
                if dr[j_near] < self.fwhm:
                    snr *= (1.+0.9*np.log10(dr[j_near]/self.fwhm))

                    if dr[j_near] < (binrad*self.fwhm) or snr < sncut:
                        status[i] = 1
                        primary[i] = self.sources.ID[status == 2][j_near]
                        logger.info("{0:9d}  {1:4.1f}  {2:6.1f}  {3:8.2f}  {4:6.2f}       1".format(
                            self.sources.ID[i], self.sources.mag[i], snr, dr[j_near], distance_to_edge))
                        continue

            status[i] = 2
            primary[i] = self.sources.ID[i]
            logger.info("{0:9d}  {1:4.1f}  {2:6.1f}            {3:6.2f}       2".format(
                self.sources.ID[i], self.sources.mag[i], snr, distance_to_edge))

        logger.info("Results of source selection: {0} resolved, {1} nearby sources.".format(
            (status == 2).sum(), (status == 1).sum()))

        if not (status == 2).any():
            logger.error("Did not find any resolved sources. Aborting...")
            return

        # if requested, update confusion limit. Note that the configuration that is stored in the header of the
        # prm-file may contain two different values for "RESLIM": the initial one provided by the user and the one
        # just determined from the source list.
        if reslim is None:
            reslim = self.sources.mag[status == 2].max()
            logger.info("Estimated confusion limit is m_{0}={1:.1f}".format(self.passband, reslim))

        # if requested, add a component of unresolved stars
        if use_unres:
            unreslim = reslim + unres_range

            logger.info("Including sources to magnitude m_{0}={1:.2f} to unresolved component...".format(
                self.passband, unreslim))

            for i in indices_in_sources_class:

                if status[i] > 0:
                    continue

                if self.sources.mag[i] > unreslim:
                    break

                distance_to_edge = self.ifs_data.distance_to_edge(ifs_coordinates[[i]])[0]
                if distance_to_edge > -1:
                    status[i] = 0
                    primary[i] = -2

            logger.info("Unresolved component contains {0} sources.".format(int((status == 0).sum())))

        self.sources.primary = primary
        self.sources.status = status

    def search_psf_stars(self, saturated=None, max_mag=None, psf_radius=10, aperture_noise=0.1):
        """
        This method searches the photometry for stars suited to constrain the
        the PSF.

        Stars that are considered suitable for constraining the PSF must
        fulfil the following criteria:
        (1) brighter than the provided magnitude limit (parameter max_mag)
        (2) must have been identified as resolved (i.e. status==2) in the
            source catalog
        (3) must not have close companions (sources with status=1)
        # (4) must have an estimated S/N above the provided cut (this is
        #     estimated using aperture photometry)
        # This criterion is currently not used.
        (5) at a distance of at least the PSF FWHM from the edge of the FoV
            (2 x FWHM for MUSE)
        (6) isolation criteria:
            a) relative contribution from neighboring sources inside the PSF
               radius should be smaller than limit given by 'aperture_noise'
               parameter
            b) no brighter source inside 1.5 x PSFRAD distance

        Note that the isolation of a star is measured using a slightly
        modified version of the shoulder drop or topographic prominence (for
        the exact definition of the topographic prominence, see
        http://en.wikipedia.org/wiki/Topographic_prominence). It is a measure
        for how strongly a source dominates its surroundings. The measure used
        here is the flux ratio of the source itself and the brightest source
        within a distance r that we set to the value of the PSF radius.

        Parameters
        ----------
        saturated : list, optional
            A list of source IDs that should be excluded from the PSF star
            search, e.g. because they are saturated in the IFS data.
        max_mag : float, optional
            The maximum allowed magnitude of a PSF star.
        psf_radius : float, optional
            The radius out to which the PSF profiles are calculated. The
            isolation criterion will be checked in a circle with this radius.
        aperture_noise : float, optional
            The maximum allowed contamination of a nearby source inside the
            area where the PSF profile is defined.
        """
        logger.info("Searching for possible PSF stars...")

        if not (self.sources.status == 2).any():
            logger.error('Source selection must be performed before searching for PSF stars.')
            return

        if saturated is not None:
            logger.info("Ignoring following stars flagged as saturated: {0}".format(", ".join(
                [str(star) for star in saturated])))
        else:
            saturated = []

        # load IFS coordinates for all stars, average over spectral axis (if any), and bring into correct order
        # (including background components)
        self.sources.load_ifs_coordinates()
        ifs_coordinates = np.zeros((self.sources.nsrc, 2), dtype='<f4')
        ifs_coordinates[self.sources.ifs_coordinates_loaded] = np.nanmean(self.sources.ifs_coordinates[..., 0], axis=0)

        # loop over magnitude-sorted list of indices of the sources the instance of the Sources-class
        # (brightest sources first)
        indices_in_sources_class = np.argsort(self.sources.mag)

        logger.info('IDs of stars that fulfill criteria for PSF stars:')
        logger.info('      ID|   mag| d_edge [pixel]| shoulder_drop')
        logger.info('--------+------+---------------+--------------')
        for i in indices_in_sources_class:
            if max_mag is not None and self.sources.mag[i] > max_mag:
                break
            if self.sources.status[i] != 2:
                continue
            if self.sources.ID[i] in saturated:
                continue
            if (self.sources.primary == self.sources.ID[i]).sum() != 1:
                continue

            distance_to_edge = self.ifs_data.distance_to_edge(ifs_coordinates[[i]])[0]
            if distance_to_edge < self.fwhm:
                continue
            elif distance_to_edge < 2*self.fwhm and isinstance(self.ifs_data, (Muse, MusePixtable)):
                continue

            # calculate shoulder drop
            dr = spatial.distance.cdist(ifs_coordinates[[i]], ifs_coordinates)[0]
            dr[i] = np.inf  # handle the source itself
            dr[self.sources.status == 4] = np.inf  # handle background components

            if (dr < psf_radius).any():
                assert dr.size == self.sources.mag.size
                dmag = (self.sources.mag[(dr < psf_radius)] - self.sources.mag[i]).min()
                shoulder_drop = 10**(-0.4*dmag)
                if shoulder_drop > aperture_noise:
                    continue
            else:
                shoulder_drop = 0

            # check if brighter star inside 1.5x PSFRAD:
            if (dr < 1.5*psf_radius).any() and (self.sources.mag[dr < 1.5*psf_radius] < self.sources.mag[i]).any():
                continue

            self.sources.status[i] = 3
            logger.info("{0:8d}|{1:6.2f}|{2:15.1f}|{3:14.2f}".format(self.sources.ID[i], self.sources.mag[i],
                                                                     distance_to_edge, shoulder_drop))

        if len(self.sources.psf_indices) == 0:
            logger.critical("No possible PSF sources found!")
            return

    def run_aperture_photometry(self, ids=None, start=0, stop=-1, radius=3, n_cpu=1, n_bin=1):
        """
        Method to perform aperture photometry for a subset of the source
        catalog on the layers of the IFS data.

        Based on the estimates for the centroid coordinates in the IFS data,
        this method will obtain spectra for a given set of sources by
        performing aperture photometry for those sources on all layers of the
        IFs data. At the same time, the method will refine the centroids and
        also obtain estimates of the PSF parameters using the spatial moments
        measured during the analysis. The results will be written to the
        SourceSelector().sources and SourceSelector().psf_attributes
        properties.

        Parameters
        ----------
        ids : list, optional
            The IDs of the stars for which the analysis should be performed.
            If none are provided, the method will use all sources that have
            status=3 in the SourceSelector().sources property.
        start : int, optional
            The first layer of the IFS data for which the analysis should be
            performed.
        stop : int, optional
            The last layer of the IFS data for which the analysis should be
            performed.
        radius : int, optional
           The aperture radius inside which the flux and the moments of each
           source are determined.
        n_cpu : int optional
           The number of CPUs used in the parallized part of the analysis.
        n_bin : int, optional
           The binning factor of the IFS data. If >1, one new layer is created
           by combining the signal from 'n_bin' adjacent layers.
        """
        # carry out aperture spectrophotometry for PSF sources
        logger.info("Initializing aperture spectrophotometry for possible PSF stars...")

        if ids is None:
            ids = self.sources.ID[self.sources.status == 3]
        else:  # make sure IDs are provided in same order as in Sources instance
            ids = self.sources.ID[np.in1d(self.sources.ID, ids, assume_unique=True)]
        if len(ids) == 0:
            logger.error('Did not find any suitable sources.')
            return

        # make sure sources instance has correct dispersion information
        if self.sources.ndisp != self.ifs_data.nlayer:
            logger.warning('Need to update dispersion information in source catalog.')
            self.sources.ndisp = self.ifs_data.nlayer

            # create dummy spectrum data: every spectrum contains only NaNs
            # if a MUSE pixtable is being analysed, a dedicated wavelength array must be created as well.
            if isinstance(self.ifs_data, MusePixtable):
                self.sources.load_spectra(init_wave=True)
                self.sources.data_wave[:] = np.nan
            # otherwise try to add dispersion information to Sources instance
            else:
                self.sources.load_spectra(init_wave=False)
            self.sources.data[:] = np.nan

        if self.sources.transformation.valid and self.sources.transformation.ndisp != self.ifs_data.nlayer:
            self.sources.transformation.resize(self.sources.ndisp)

        # Obtain centroid coordinates for the selected sources
        self.sources.load_ifs_coordinates(ids=ids)
        # Important: the input format of the aperture routine is (y, x) instead of (x, y)
        centroids = np.nanmean(self.sources.ifs_coordinates[..., 0], axis=0)[:, ::-1]

        logger.info('Using following stars in aperture spectrophotometry:')
        logger.info('       ID      x      y')
        for i, (y, x) in enumerate(centroids):
            logger.info('{0:9d}{1:7.1f}{2:7.1f}'.format(ids[i], x, y))

        # initialize aperture routine
        make_aper = AperPhoto(cube=self.ifs_data, centroids=centroids)

        # run it - note that the array that is returned may contain NaN values (if the moment calculation failed or
        # lead to unphysical values). To account for this, masked arrays are used in the following calculations
        # TODO: find out why code crashes for n_cpu != 1
        if n_cpu > 1:
            logger.warning('Parallelization switched off for aperture photometry.')
        results = make_aper(r_aper=radius, layer_range=(start, stop), background="local", n_cpu=1, n_bin=n_bin)

        # for testing purposes:
        # from astropy.io import fits
        # tablehdu = fits.BinTableHDU(results)
        # tablehdu.writeto("/users/skamann/aperture.fits", clobber=True)

        for i, source_id in enumerate(ids):

            j = np.where(self.sources.ID == source_id)[0][0]

            # add flux of source to Sources instance
            self.sources.data[:, self.sources.specrow[j], 0] = results["flux"][i]

            # for pixtables, also need wavelength information for each spectrum
            if isinstance(self.ifs_data, MusePixtable):
                self.sources.data_wave[:, self.sources.specrow[j]] = self.ifs_data.wave

            # NaN pixels are set to zero since DER_SNR skips such pixels
            snr = der_snr(np.ma.array(results["flux"][i], mask=np.isnan(results["flux"][i]), fill_value=0.).filled())
            logger.info("S/N of source #{0} is {1:.1f}".format(source_id, snr))

            # add IFS coordinates to Sources instance
            self.sources.ifs_coordinates[:, i, 0, 0] = results["x"][i]
            self.sources.ifs_coordinates[:, i, 1, 0] = results["y"][i]

        # determine wavelength-dependent coordinate transformation
        if self.sources.transformation.valid:
            logger.info("Determining wavelength-dependent coordinate transformation...")
            self.sources.fit_coordinate_transformation()
            if self.position_fit == 2:
                self.sources.transformation.free[:] = True
            elif self.position_fit == 1:
                self.sources.transformation.free[-2:] = True

        # determine FWHM, E and THETA
        logger.info("Estimating PSF properties using measured moments...")
        x2 = np.ma.array(results['x2'], mask=np.isnan(results['x2']))
        y2 = np.ma.array(results['y2'], mask=np.isnan(results['y2']))
        xy = np.ma.array(results['xy'], mask=np.isnan(results['xy']))

        # get A, B:
        with np.errstate(invalid="ignore"):
            a = np.sqrt((x2 + y2) / 2. + np.sqrt(((x2 - y2) / 2.) ** 2 + xy ** 2))
            a.mask[a <= 0] = True
            b = np.sqrt((x2 + y2) / 2. - np.sqrt(((x2 - y2) / 2.) ** 2 + xy ** 2))
            b.mask[b <= 0] = True

        # FWHM
        fwhm = np.sqrt(8. * np.log(2)) * np.mean(a, axis=0)
        # TODO: should FWHM be updated as well?
        logger.info("Mean FWHM: {0:.2f}pix.".format(float(fwhm.mean())))

        # Ellipticity
        e = 1. - np.mean(b / a, axis=0)
        logger.info("Mean ellipticity: {0:.2f}".format(float(e.mean())))
        ii = self.psf_attributes.names.index("e")
        if self.psf_attributes.free[ii]:
            # need to check if dispersion information already included in PSF variables, otherwise, use
            # numpy.resize that fills enlarged array with repeated copies of available data
            if self.psf_attributes.data.shape[0] == 1:
                self.psf_attributes.data = np.resize(self.psf_attributes.data,
                                                     e.shape + self.psf_attributes.data.shape[1:])
            self.psf_attributes.data[:, ii, 0] = e.data

        # Position angle
        with np.errstate(invalid="ignore"):
            _theta = 0.5 * np.arctan(2 * xy / (x2 - y2))
            _theta -= (-1) ** (_theta < 0) * np.pi / 4.
            _theta += (-1) ** (xy > 0) * np.pi / 4.
            theta = _theta.mean(axis=0)
        logger.info("Mean position angle: {0:.2f}rad.".format(float(theta.mean())))
        ii = self.psf_attributes.names.index("theta")
        if self.psf_attributes.free[ii]:
            if self.psf_attributes.data.shape[0] == 1:
                self.psf_attributes.data = np.resize(self.psf_attributes.data,
                                                     theta.shape + self.psf_attributes.data.shape[1:])
            self.psf_attributes.data[:, ii, 0] = theta.data
