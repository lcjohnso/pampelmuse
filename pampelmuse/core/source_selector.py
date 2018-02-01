"""
source_selector.py
==================

Copyright 2013-2018 Sebastian Kamann

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

Latest Git revision
-------------------
2018/02/01
"""
import logging
import sys
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from .aperture import AperPhoto
from .background import BackgroundGrid
from .coordinates import Transformation
from .sources import Sources
from ..instruments import Instrument, Muse, MusePixtable
from ..psf import *
from ..utils.image import find_offset
from ..utils.statistics import der_snr


__author__ = 'Sebastian Kamann (s.kamann@ljmu.ac.uk)'
__revision__ = 20180201


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
        - data: pandas.DataFrame
              The centroid coordinates and the magnitudes of the individual
              sources, provided as columns 'x', 'y', and 'mag'
        - offset : tuple, optional
              A global offset applied to all sources along x and y. Default is
              (0, 0)
        - zeropoint : float
              The magnitude zeropoint, i.e. the magnitude of a source with an
              integrated flux of 1. Default is 0.

    Returns
    -------
    image : nd_array
        The model image created from the input data.
    """
    psf = sources_data['psf']
    image = np.zeros((psf.n_pixel, ), dtype='<f4')

    offset = sources_data.get('offset', (0., 0.))
    zeropoint = sources_data.get('zeropoint', 0.)

    n_sources = sources_data['data'].shape[0]
    for i, (index, row) in enumerate(sources_data['data'].iterrows(), start=1):
        psf.uc = row['x'] - offset[0]
        psf.vc = row['y'] - offset[1]
        psf.mag = row['mag'] - zeropoint
        psf.reset()

        image[psf.used] += psf.f
        logger.debug('Image creation: Processed star #{0} {1}/{2}'.format(index, i, n_sources))

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
        - data: pandas.DataFrame
              The centroid coordinates and the magnitudes of the individual
              sources, provided as columns 'x', 'y', and 'mag'
        - zeropoint : float
              The magnitude zeropoint, i.e. the magnitude of a source with an
              integrated flux of 1. Default is 0.

    Returns
    -------
    snratios : pandas.Series
        A pandas Series containing the S/N-ratio for each source
    totfluxes : float
        The integrated fluxes of all point sources over the individual
        spaxels.
    """
    global g_variance

    snratios = pd.Series(0., index=sources_data['data'].index)
    totfluxes = 0.

    psf = sources_data['psf']
    zeropoint = sources_data.get('zeropoint', 0.)

    n_sources = sources_data['data'].shape[0]
    for i, (index, row) in enumerate(sources_data['data'].iterrows(), start=1):
        psf.uc = row['x']
        psf.vc = row['y']
        psf.mag = row['mag'] - zeropoint
        psf.reset()

        if (g_variance[psf.used] == 0).any():
            logger.error("Sigma image contains invalid pixels with values <=0.")

        # measure S/N
        snratios[index] = np.sqrt(np.nansum((psf.f / np.sqrt(g_variance[psf.used])) ** 2))

        logger.debug("Estimated S/N for source #{0} [{0}/{1}]: {2}.".format(index, i, n_sources, snratios[index]))

        # add the flux contribution of this source in the FoV
        totfluxes += psf.f.sum()

    return snratios, totfluxes


class SourceSelector(object):
    """
    This class is designed to perform the selection process for the sources
    that are used in the PampelMuse extraction process.

    The defined methods perform the individual steps of the source selection
    process and are used by the INITFIT routine of PampelMuse.
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
            the chosen passband.
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

        # open reference catalog. It should be in plain csv-format. the source IDs are used as index
        try:
            self.cfp_results = pd.read_csv(photometry_file, index_col='id')
        except ValueError:
            logger.critical("Did not find IDs in photometry file.")
            raise IOError("Source IDs must be provided with photometry")

        # make sure magnitudes are available
        if self.passband.lower() not in self.cfp_results.columns:
            logger.critical("Did not find magnitudes in photometry file.")
            raise IOError("Source magnitudes must be provided with photometry.")
        else:
            # rename columns with magnitudes to 'mag'
            self.cfp_results.rename(columns={self.passband.lower(): 'mag'}, inplace=True)

        # make sure either RA & Dec or x & y are available; if both are available, RA & Dec are preferred over x & y
        if "ra" in self.cfp_results.columns and "dec" in self.cfp_results.columns:
            logger.info("Using RA & DEC coordinates as input.")
            ra = np.asarray(self.cfp_results["ra"], dtype=np.float64)
            dec = np.asarray(self.cfp_results["dec"], dtype=np.float64)

            # if we have a WCS instance, use it to convert to pixel coordinates
            if self.use_wcs:
                # get IFS spaxel coordinates from reference RA and DEC
                if self.ifs_data.wcs.wcs.naxis == 3:
                    x, y, _ = self.ifs_data.wcs.wcs_world2pix(ra, dec, np.ones(ra.shape), 1)
                else:
                    x, y = self.ifs_data.wcs.wcs_world2pix(ra, dec, 1)

            # otherwise create mock WCS instance with pixelsize 0.2arcsec
            else:
                wcs = WCS(naxis=2)
                wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
                wcs.wcs.cdelt = [-0.2/3600., 0.2/3600.]
                wcs.wcs.crval = [ra.max(), dec.min()]
                wcs.wcs.crpix = [0, 0]
                x, y = wcs.wcs_world2pix(ra, dec, 1)

            self.cfp_results["x"] = x.tolist()
            self.cfp_results["y"] = y.tolist()

        elif "x" in self.cfp_results.columns or "y" in self.cfp_results.columns:
            self.use_wcs = False
            logger.info("using x- and-coordinates as input.")

        else:
            logger.critical("Cannot find input source coordinates.")
            raise IOError("Either x & y or RA & Dec must be provided with photometry.")

        # Perform sanity checks on provided source list (exclude sources with magnitudes NaN or >99):
        self.cfp_results['valid'] = np.isfinite(self.cfp_results['mag'])
        self.cfp_results.loc[self.cfp_results['mag'] >= 99, 'valid'] = False
        if not self.cfp_results['valid'].all():
            logger.warning("Ignoring {0} source(s) with undefined magnitudes.".format(
                (~self.cfp_results['valid']).sum()))

        # check for duplicate indices (= source IDs)
        if self.cfp_results.index.duplicated().any():
            logger.error('Catalog contains duplicate source IDs.')

        # unless already present, add columns for storing status of a source and whether it is in a combined
        # component with a different primary source
        if 'status' not in self.cfp_results.columns:
            self.cfp_results['status'] = -1
        if 'primary' not in self.cfp_results.columns:
            self.cfp_results['primary'] = 0

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
        else:
            try:
                return np.linalg.det(self.ifs_data.wcs.wcs.cd[:2, :2]) != 1.0
            except AttributeError:
                return (self.ifs_data.wcs.wcs.cdelt[:2] != 0).all()

    def background_setup(self, skysize=-1, slices=False, oversampling=1):
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
        slices : boolean, optional
            Set this flag to create a separate background component for each
            IFU/slice. This is only used for MUSE data.
        oversampling : int, optional
            To improve the sky subtraction for MUSE pixel tables, the spectral
            sampling of the background components can be increased by the
            provided factor.

        Returns
        -------
        n_background : int, optional
            The number of background components used.
        """
        if skysize <= 0:
            background = BackgroundGrid.constant(oversampling=oversampling)
        else:
            background = BackgroundGrid.from_shape(shape=self.ifs_data.shape, rscale=skysize/2.,
                                                   oversampling=oversampling)

        if slices:
            logger.warning('Assigning background components to slices still in testing phase.')
            if isinstance(self.ifs_data, MusePixtable):
                if self.ifs_data.ifuhdu is not None and self.ifs_data.slicehdu is not None:
                    background.add_slices(n_ifu=24, n_slice=48)
                else:
                    logger.error('Need extensions "ifu" and "slice" for assigning background components to slices.')
            else:
                logger.error('Assigning background components to slices only supported for MUSE pixel tables.')

        # create data frame for background data
        n_centroids = len(background.centroids)
        index = pd.Index(np.arange(-1, -n_centroids-1, -1))  # sky components have negative index
        background_data = pd.DataFrame(True, columns=['valid', ], index=index)
        background_data['x'] = background.centroids[:, 0]
        background_data['y'] = background.centroids[:, 1]
        background_data['status'] = 4
        # ignore sky components outside FoV (if any)
        if skysize > 0:
            outside_fov = (self.ifs_data.distance_to_edge(background.centroids) < -skysize)
            background_data.loc[outside_fov, 'status'] = -1
        # change status of slice-only backgrounds (if any) to 5
        if slices:
            last_indices = background_data.tail(background.n_ifu*background.n_slice).index
            background_data.loc[last_indices, 'status'] = 5
        background_data['primary'] = index

        # attach background components to source catalog
        self.cfp_results = self.cfp_results.append(background_data, verify_integrity=True)

        return n_centroids

    def make_sources_instance(self):
        """
        Method to create an instance of pampelmuse.core.Sources containing
        the current list of stellar and background sources.

        The method will run the initialisation method of the Sources-class and
        copy the dispersion information from the IFs data to the new instance.
        Afterwards, the instance can be accessed as SourceSelector().sources.
        """
        self.sources = Sources(catalog=self.cfp_results[self.cfp_results['valid']])

        # unless a MUSE pixtable is being analysed, copy dispersion information to sources class
        if not isinstance(self.ifs_data, MusePixtable):
            if self.ifs_data.wcs.naxis > 2:
                self.sources.crval = self.ifs_data.wcs.wcs.crval[2]
                self.sources.crpix = self.ifs_data.wcs.wcs.crpix[2]
                if self.ifs_data.wcs.wcs.has_cd():
                    self.sources.cdelt = self.ifs_data.wcs.wcs.cd[2, 2]
                else:
                    self.sources.cdelt = self.ifs_data.wcs.wcs.cdelt[2]
                self.sources.ctype = self.ifs_data.wcs.wcs.ctype[2]
                self.sources.cunit = self.ifs_data.wcs.wcs.cunit[2].name
            else:
                self.sources.crval = self.ifs_data.wave[0]
                self.sources.crpix = 1
                self.sources.cdelt = np.diff(self.ifs_data.wave).mean()
        else:
            self.sources.crval = self.ifs_data.wave[0]
            self.sources.crpix = 1
            self.sources.cdelt = np.diff(self.ifs_data.wave).mean()

        # make sure instance has correct dispersion information
        self.sources.n_dispersion = self.ifs_data.nlayer

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
        parameters : iterable, optional
            The values of the parameters of the chosen profile, provided as a
            list that contains one dictionary per parameter. Each dictionary
            should contain three keys:
            - 'name': The name of the parameter
            - 'value': The (initial) value of the parameter
            - 'free': Boolean flag indicating if the parameter value should be
                      optimized in the analysis.
            By default, a round Moffat profile with beta=2.5 and fwhm=2.5pixel
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

        # prepare data and variability flags for default PSF
        n_parameters = len(Variables.PROFILES[profile])
        data = pd.DataFrame(np.zeros((1, n_parameters), dtype=np.float32), columns=Variables.PROFILES[profile])
        free = pd.Series(0, index=Variables.PROFILES[profile])

        # update PSF data and variability flags with user-provided data (if any)
        if parameters is not None:
            for p in parameters:
                data[p['name']] = p['value']
                free[p['name']] = p['free']
        # otherwise make sure fwhm (and beta) are >0
        else:
            data['fwhm'] = 2.5
            if 'beta' in Variables.PROFILES[profile]:
                data['beta'] = 2.5

        # initialise instance with PSF parameters and print basic info about PSF
        self.psf_attributes = Variables(profile=profile, data=data, free=free)
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

        # collect PSF parameters in dictionary for profile initialisation
        psf_parameters = {}
        for name in self.psf_attributes.names:
            psf_parameters[name] = self.psf_attributes.data.at[0, (name, 'value')]

        # select n_stars brightest sources - note that the selection has no effect on argsort because
        # the background components are at the end of the array
        used_ids = self.sources.mag[~self.sources.is_background].sort_values().index[:n_stars]
        to_cpu = np.arange(used_ids.size) % n_cpu

        # to get reasonable count rates in mock image, adjust zeropoint unless known
        zeropoint = self.sources.mag[used_ids[-1]] + 7.5 if self.zeropoint is None else self.zeropoint

        args = []
        psf = None
        for n in range(n_cpu):
            used = used_ids[to_cpu == n]
            psf = self.psf_class.from_shape(shape=shape, uc=0., vc=0., mag=0., maxrad=10., **psf_parameters)
            args.append({'psf': psf, 'data': self.sources.catalog.loc[used, ['x', 'y', 'mag']],
                         'offset': (self.sources.x.min(), self.sources.y.min()), 'zeropoint': zeropoint})

        if n_cpu > 1:
            pool = Pool(n_cpu)
            _results = pool.map_async(image_from_psf, args)
            pool.close()
            pool.join()
            _image = np.sum(_results.get(), axis=0)
        else:
            _image = image_from_psf(args[0])

        # 3. resample image on regular (x, y) grid
        logger.info("   re-sampling image on regular 2D grid ...")
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
                self.passband, apply_variances=True, nbin=n_bin, layer_range=(5, self.ifs_data.nlayer - 5))

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

        # take care of data cubes wih over-subtraction of background
        if self.data_image.sum() <= 0:
            self.data_image -= np.median(self.data_image)
            logger.warning('Flux sum in {0}-image was negative. Subtracted (negative) median flux.'.format(
                self.passband))
            if self.data_image.sum() <= 0:
                fixed_pixels = 0
                while self.data_image.sum() <= 0:
                    self.data_image[np.argmin(self.data_image)] = 0
                    fixed_pixels += 1
                logger.warning(
                    "Flux sum in {0}-image was still negative. Set the {1} most negative pixels to zero.".format(
                        self.passband, fixed_pixels))

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
            width 'ccmaxoff' centred on the initial guess. IMPORTANT: The
            offsets are measured relative to the initial guess coordinates.
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
        self.sources.transformation = Transformation(
            data=pd.DataFrame(np.array([[1., 0., 0., 1., -self.ccxoff, -self.ccyoff]], dtype=np.float32),
                              columns=Transformation.PARAMETERS))
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
        from PyQt5 import QtWidgets
        from ..graphics.pampelmuse_gui import PampelMuseGui

        logger.info("Preparing GUI for source identification...")

        app = QtWidgets.QApplication(sys.argv)

        # get magnitude of n_stars'th brightest star
        n_stars = min(n_stars, self.sources.n_sources - self.sources.n_background - 1)
        mag = self.sources.mag[~self.sources.is_background].sort_values().iloc[n_stars]

        identifier = PampelMuseGui(sources=self.sources, ifsData=self.ifs_data, refImage=self.identification_image)

        # account for offset between reference image and true coordinates
        (dx, dy) = (self.sources.x.min(), self.sources.y.min())
        identifier.refOffsets = {'x': pd.Series(self.sources.x - dx, index=self.sources.ids),
                                 'y': pd.Series(self.sources.y - dy, index=self.sources.ids)}
        identifier.plot_sources_on_ref()

        if self.use_wcs:
            # show approximate location of IFS field of view in reference view
            identifier.addFovToReference(self.ccxoff - self.sources.x.min() + self.ifs_data.shape[1]//2,
                                         self.ccyoff - self.sources.y.min() + self.ifs_data.shape[0]//2)

            # overplot positions of sources on IFS as determined via cross-correlation
            sizes = 10.*(10**(-0.2*(self.sources.mag - mag - 1.)))
            sizes[sizes <= 1] = 0
            select = (self.sources.status < 4) & (
                self.sources.mag <= mag) & (
                self.ifs_data.infov(np.vstack((self.sources.x - self.ccxoff, self.sources.y - self.ccyoff)).T))
            identifier.ifsPlot.addData(self.sources.x[select] - self.ccxoff, self.sources.y[select] - self.ccyoff,
                                       s=sizes[select], marker="o", edgecolor="g", facecolor="None")

        identifier.show()
        app.exec_()

        identified = identifier.identifiedSources

        if len(identified) >= 4:
            # TODO: reduce number of required sources to 2 if only offsets should be determined

            # collect IFS coordinates in common DataFrame and fit coordinate transformation
            ifs_coordinates = pd.DataFrame(
                0., index=[self.sources.wave[0]], columns=pd.MultiIndex.from_product([identified.keys(), ['x', 'y']]))
            for source_id, values in identified.items():
                ifs_coordinates.at[self.sources.wave[0], (source_id, 'x')] = values[0]
                ifs_coordinates.at[self.sources.wave[0], (source_id, 'y')] = values[1]

            self.sources.transformation = Transformation.from_coordinates(self.sources.catalog, ifs_coordinates)

            if self.position_fit == 2:
                self.sources.transformation.free[:] = True
            elif self.position_fit == 1:
                self.sources.transformation.free[-2:] = True

        elif not self.sources.transformation.valid:
            logger.error("At least 4 sources must be identified.")

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

        # collect PSF parameters in dictionary for profile initialisation
        psf_parameters = {}
        for name in self.psf_attributes.names:
            psf_parameters[name] = self.psf_attributes.data.at[0, (name, 'value')]

        # add parameters of coordinate transformation
        for name in self.sources.transformation.names:
            psf_parameters[name] = self.sources.transformation.data[(name, 'value')].iloc[0]

        # Assign sources to CPUs
        if n_cpu is None or n_cpu == -1:
            n_cpu = cpu_count()  # use all available CPUs
        to_cpu = np.arange(self.sources.n_sources) % n_cpu
        to_cpu[self.sources.ids < 0] = n_cpu + 1  # ignore background components
        args = []
        for n in range(n_cpu):
            # in case the data is a MUSE pixtable, make sure the PSF profiles are defined on the same coordinates
            # that are available  in the variance image
            if isinstance(self.ifs_data, MusePixtable):
                psf = self.psf_class.from_shape(shape=self.var_image.shape, uc=0., vc=0., mag=0., maxrad=10.,
                                                **psf_parameters)
            else:
                psf = self.psf_class(transform=self.ifs_data.transform, uc=0., vc=0., mag=0., maxrad=10.,
                                     **psf_parameters)
            args.append({'psf': psf, 'data': self.sources.catalog.loc[to_cpu == n, ['x', 'y', 'mag']],
                         'zeropoint': zeropoint})

        # estimate S/N and integrated flux from sources in image
        if n_cpu > 1:
            pool = Pool(n_cpu, initializer=init_worker, initargs=(self.var_image,))

            _results = pool.map_async(estimate_snr, args)
            pool.close()
            pool.join()
            results = _results.get()

            self.signal_to_noise = pd.concat([r[0] for r in results], verify_integrity=True)
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
            f12 = np.nanmean(self.psf_attributes.data[('f12', 'value')])
            fwhm1 = np.nanmean(self.psf_attributes.data[('fwhm1', 'value')])
            fwhm2 = np.nanmean(self.psf_attributes.data[('fwhm2', 'value')])
            return f12*fwhm1 + (1. - f12)*fwhm2
        else:
            return np.nanmean(self.psf_attributes.data[('fwhm', 'value')])

    def select(self, reslim=None, sncut=3., confusion_density=0.4, binrad=0.2, use_unres=False, unres_range=2):
        """
        This method performs the source selection process for the PSF fitting,

        Each source is given a status based on criteria such as the estimated
        S/N, the distance to brighter sources, and the edge of the field of
        view. The status of a source can be one of the following:
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
            will only include sources brighter than the provided value.
        confusion_density : float, optional
            The source density where the confusion limit is reached, provided
            as [no. of sources]/[resolution element]. A value of 0.4 gave good
            results for the simulations performed in Kamann et al. (2013).
        sncut : float, optional
            The minimum estimated S/N of a source that is still considered as
            resolved.
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

        # for larger fields of view, such as the MUSE WFM, the stellar density can vary significantly across its extent.
        # To deal with this, the confusion limit is estimated locally in an area of radius 10xFWHM around each star
        density = np.zeros(self.ifs_data.nspaxel, dtype=np.int16)

        # load IFS coordinates for all stars (using first layer as reference)
        self.sources.load_ifs_coordinates(layers=0)
        #  get IFS coordinates as DataFrame with source IDs as indices and x & y as columns
        _x = self.sources.ifs_coordinates.loc[:, (slice(None), 'x', 'value')]
        _y = self.sources.ifs_coordinates.loc[:, (slice(None), 'y', 'value')]
        xy = pd.DataFrame(np.stack((_x.values[0], _y.values[0]), axis=1), index=_x.columns.levels[0],
                          columns=['x', 'y'])

        logger.info("       ID   mag    S/N  d_bright  d_edge  status")
        logger.info("                           [spx]   [spx]        ")

        # loop over magnitude-sorted list of indices of the sources the instance of the Sources-class
        # (brightest sources first)
        for source_id, mag in self.sources.mag.sort_values().items():

            if source_id < 0:  # ignore background components
                continue

            # stop loop if current source falls below (fixed) confusion limit
            if reslim is not None and mag > reslim:
                break

            # get x- and y-coordinate
            xi = xy.at[source_id, 'x']
            yi = xy.at[source_id, 'y']

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
            snr = self.signal_to_noise[source_id]
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
            brighter = self.cfp_results.index[self.cfp_results['status'] == 2]
            if brighter.any():
                dr = np.sqrt((xi - xy.loc[brighter, 'x'])**2 + (yi - xy.loc[brighter, 'y'])**2)
                id_near = dr.idxmin()

                # issue warning if two sources have basically the same IFS coordinates
                if dr[id_near] < 1e-5:
                    logger.warning("Source #{0} might be double.".format(source_id))

                # penalize S/N if a brighter source is located inside the FWHM of the PSF
                if dr[id_near] < self.fwhm:
                    snr *= (1.+0.9*np.log10(dr[id_near]/self.fwhm))

                    if dr[id_near] < (binrad*self.fwhm) or snr < sncut:
                        self.cfp_results.at[source_id, 'status'] = 1
                        self.cfp_results.at[source_id, 'primary'] = id_near
                        logger.info("{0:9d}  {1:4.1f}  {2:6.1f}  {3:8.2f}  {4:6.2f}       1".format(
                            source_id, mag, snr, dr[id_near], distance_to_edge))
                        continue

            self.cfp_results.at[source_id, 'status'] = 2
            self.cfp_results.at[source_id, 'primary'] = source_id
            logger.info("{0:9d}  {1:4.1f}  {2:6.1f}            {3:6.2f}       2".format(
                source_id, mag, snr, distance_to_edge))

        logger.info("Results of source selection: {0} resolved, {1} nearby sources.".format(
            np.sum(self.cfp_results['status'] == 2), np.sum(self.cfp_results['status'] == 1)))

        if not (self.cfp_results['status'] == 2).any():
            logger.error("Did not find any resolved sources. Aborting...")
            return

        # if requested, update confusion limit. Note that the configuration that is stored in the header of the
        # prm-file may contain two different values for "RESLIM": the initial one provided by the user and the one
        # just determined from the source list.
        if reslim is None:
            reslim = self.sources.mag[self.cfp_results.index[self.cfp_results['status'] == 2]].max()
            logger.info("Estimated confusion limit is m_{0}={1:.1f}".format(self.passband, reslim))

        # if requested, add a component of unresolved stars
        if use_unres:
            unreslim = reslim + unres_range

            logger.info("Including sources to magnitude m_{0}={1:.2f} to unresolved component...".format(
                self.passband, unreslim))

            for source_id, mag in self.sources.mag.sort_values().items():

                if self.cfp_results.at[source_id, 'status'] > 0:
                    continue

                if mag > unreslim:
                    break

                distance_to_edge = self.ifs_data.distance_to_edge(xy[source_id].values)[0]
                if distance_to_edge > -1:
                    self.cfp_results.at[source_id, 'status'] = 0

            # use brightest unresolved source as "primary"
            idx = self.sources.mag[self.cfp_results.index[self.cfp_results['status'] == 1]].idxmin()
            self.cfp_results.loc[self.cfp_results['status'] == 0, 'primary'] = idx

            logger.info("Unresolved component contains {0} sources.".format((self.cfp_results['status'] == 0).sum()))

        # set properties in sources class
        self.sources.primary = self.cfp_results.loc[self.sources.ids, 'primary']
        self.sources.status = self.cfp_results.loc[self.sources.ids, 'status']
        self.sources.load_spectra(init_wave=isinstance(self.ifs_data, MusePixtable))

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

        # load IFS coordinates for all stars (again using first layer as reference)
        self.sources.load_ifs_coordinates(layers=0)
        #  get IFS coordinates as DataFrame with source IDs as indices and x & y as columns
        _x = self.sources.ifs_coordinates.loc[:, (slice(None), 'x', 'value')]
        _y = self.sources.ifs_coordinates.loc[:, (slice(None), 'y', 'value')]
        xy = pd.DataFrame(np.stack((_x.values[0], _y.values[0]), axis=1), index=_x.columns.levels[0],
                          columns=['x', 'y'])

        logger.info('IDs of stars that fulfill criteria for PSF stars:')
        logger.info('      ID|   mag| d_edge [pixel]| shoulder_drop')
        logger.info('--------+------+---------------+--------------')

        # loop over magnitude-sorted list of indices of the sources the instance of the Sources-class
        # (brightest sources first)
        for source_id, mag in self.sources.mag.sort_values().items():
            if max_mag is not None and mag > max_mag:
                break
            if self.sources.status[source_id] != 2:
                continue
            if source_id in saturated:
                continue
            if (self.sources.primary == source_id).sum() != 1:
                continue

            distance_to_edge = self.ifs_data.distance_to_edge(np.atleast_2d(xy.loc[source_id].values))[0]
            if distance_to_edge < self.fwhm:
                continue
            elif distance_to_edge < 2*self.fwhm and isinstance(self.ifs_data, (Muse, MusePixtable)):
                continue

            # calculate shoulder drop
            dr = np.sqrt((xy.at[source_id, 'x'] - xy['x'])**2 * (xy.at[source_id, 'y'] - xy['y'])**2)
            dr[source_id] = np.inf  # handle the source itself
            dr[self.sources.status >= 4] = np.inf  # handle background components

            if dr.min() < psf_radius:
                dmag = (self.sources.mag[dr[dr < psf_radius].index] - mag).min()
                shoulder_drop = 10**(-0.4*dmag)
                if shoulder_drop > aperture_noise:
                    continue
            else:
                shoulder_drop = 0

            # check if brighter star inside 1.5x PSFRAD:
            if dr.min() < 1.5*psf_radius and self.sources.mag[dr[dr < 1.5*psf_radius].index].min() < mag:
                continue

            self.sources.catalog.at[source_id, 'status'] = 3
            logger.info("{0:8d}|{1:6.2f}|{2:15.1f}|{3:14.2f}".format(source_id, mag, distance_to_edge, shoulder_drop))

        if len(self.sources.psf_indices) == 0:
            logger.error("No possible PSF sources found!")
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
            ids = self.sources.ids[self.sources.status == 3]
        else:  # make sure IDs are available in Sources instance
            ids = self.sources.ids[np.in1d(self.sources.ids, ids, assume_unique=True)]
        if len(ids) == 0:
            logger.error('Did not find any suitable sources for aperture photometry.')
            return

        # make sure sources instance has correct dispersion information
        if self.sources.n_dispersion != self.ifs_data.nlayer:
            logger.warning('Need to update dispersion information in source catalog.')
            self.sources.n_dispersion = self.ifs_data.nlayer

            # create dummy spectrum data: every spectrum contains only NaNs
            # if a MUSE pixtable is being analysed, a dedicated wavelength array must be created as well.
            self.sources.load_spectra(init_wave=isinstance(self.ifs_data, MusePixtable))
            self.sources.data[:] = np.nan

        # Obtain centroid coordinates for the selected sources
        self.sources.load_ifs_coordinates(ids=ids.values)

        #  get IFS coordinates as DataFrame with source IDs as indices and x & y as columns
        _x = self.sources.ifs_coordinates.loc[:, (slice(None), 'x', 'value')]
        _y = self.sources.ifs_coordinates.loc[:, (slice(None), 'y', 'value')]
        xy = pd.DataFrame(np.stack((_x.values[0], _y.values[0]), axis=1), index=_x.columns.levels[0],
                          columns=['x', 'y'])

        logger.info('Using following stars in aperture spectrophotometry:')
        logger.info('       ID      x      y')
        for source_id, row in xy.iterrows():
            logger.info('{0:9d}{1:7.1f}{2:7.1f}'.format(source_id, row['x'], row['y']))

        # initialize aperture routine
        aperture = AperPhoto(cube=self.ifs_data, centroids=xy)

        # run it - note that the array that is returned may contain NaN values (if the moment calculation failed or
        # lead to unphysical values).
        results, background = aperture(r_aper=radius, layer_range=(start, stop), background="local", n_cpu=n_cpu,
                                       n_bin=n_bin)

        # make sure no source returned only NaN
        mean_results = results.mean(axis=0)
        for source_id, values in mean_results.groupby(level=0):
            if np.isnan(values).any():
                logger.warning('Excluding source #{0} as calculation returned only NaNs.'.format(source_id))
                results.drop(source_id, axis=1, inplace=True)
                self.sources.status[source_id] = 2

        # initialize containers for storing profile shape parameters
        fwhm = []
        e = []
        theta = []

        # loop over sources, insert their spectra, calculate FWHM, ellipticity, and pos. angle of profile
        # note that removed source IDs are still in the columns property, see discussion here:
        # https://github.com/pandas-dev/pandas/issues/2770
        for source_id in np.unique(results.columns.get_level_values(0)):

            # get spectrum
            spectrum = results[(source_id, 'flux')]

            # add flux of source to Sources instance
            self.sources.data[(self.sources.specrow[source_id], 'flux')].iloc[results.index] = spectrum.values

            # add coordinates if they are fitted
            if self.sources.ifs_coordinates_free.any():
                self.sources.ifs_coordinates[source_id] = np.nan
                self.sources.ifs_coordinates[
                    (source_id, 'x', 'value')].iloc[results.index] = results[(source_id, 'x')].values
                self.sources.ifs_coordinates[
                    (source_id, 'y', 'value')].iloc[results.index] = results[(source_id, 'y')].values

            # NaN pixels are set to zero since DER_SNR skips such pixels
            snr = der_snr(np.ma.array(spectrum.values, mask=np.isnan(spectrum.values), fill_value=0.).filled())
            logger.info("S/N of source #{0} is {1:.1f}".format(source_id, snr))

            # for PSF stars, determine profile shape, for the origin of the formulae, check the SExtractor manual
            if self.sources.status[source_id] == 3:
                x2 = results[(source_id, 'x2')]
                y2 = results[(source_id, 'y2')]
                xy = results[(source_id, 'xy')]

                a = np.sqrt((x2 + y2) / 2. + np.sqrt(((x2 - y2) / 2.) ** 2 + xy ** 2))
                b = np.sqrt((x2 + y2) / 2. - np.sqrt(((x2 - y2) / 2.) ** 2 + xy ** 2))

                fwhm.append(np.sqrt(8. * np.log(2)) * a)
                e.append(1. - (b / a))
                # according to SExtractor manual, theta should have same size as covariance
                _theta = 0.5 * np.arctan2(2 * xy, (x2 - y2))
                _theta[(_theta < 0) & (xy > 0)] += np.pi/2.
                _theta[(_theta > 0) & (xy < 0)] -= np.pi/2.

                # IMPORTANT: SExtractor measures angle from x-axis in ccw while PampelMuse uses the y-axis as
                # zeropoint. The following lines do the conversion.
                _theta[_theta < 0] += np.pi/2.
                _theta[_theta > 0] -= np.pi/2.

                theta.append(-_theta)

        # calculate (& update) PSF parameters
        logger.info("Estimating PSF properties using measured moments...")
        # check if dispersion information of PSF is correct
        if self.psf_attributes.contains_free_parameters:
            self.psf_attributes.resize(self.sources.wave[results.index.values])

        # FWHM
        # TODO: should FWHM be updated as well?
        mean_fwhm = np.nanmean(fwhm, axis=0)
        logger.info("Mean FWHM: {0:.2f}pix.".format(np.nanmean(mean_fwhm)))

        # Ellipticity
        mean_e = np.nanmean(e, axis=0)
        logger.info("Mean ellipticity: {0:.2f}".format(np.nanmean(mean_e)))
        if self.psf_attributes.free['e']:
            self.psf_attributes.data[('e', 'value')] = mean_e

        # Position angle
        mean_theta = np.nanmean(theta, axis=0)
        logger.info("Mean position angle: {0:.2f}rad.".format(np.nanmean(mean_theta)))
        if self.psf_attributes.free['theta']:
            self.psf_attributes.data[('theta', 'value')] = mean_theta

        # finally bring to same indices as sources instance
        self.psf_attributes.resize(self.sources.wave)

        # determine wavelength-dependent coordinate transformation
        if self.sources.transformation.valid and self.position_fit > 0:
            logger.info("Determining wavelength-dependent coordinate transformation...")
            if self.position_fit == 1:
                fixed = {'A': pd.Series(1., index=self.sources.wave),
                         'B': pd.Series(0., index=self.sources.wave),
                         'C': pd.Series(0., index=self.sources.wave),
                         'D': pd.Series(1., index=self.sources.wave), }
            else:
                fixed = None
            t = Transformation.from_coordinates(self.sources.catalog, results, fixed=fixed)
            # need to update index-information to match that of sources class
            t.data.index = self.sources.wave[results.index.values]
            t.resize(self.sources.wave)
            #  add to sources instance
            self.sources.transformation = t
