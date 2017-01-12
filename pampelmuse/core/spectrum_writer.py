"""
spectrum_writer.py
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
This module implements the class SpectrumWriter that is used to store the
spectra extracted by PampelMuse as individual FITS files.

Added
-----
2016/08/16, rev. 366

Latest SVN revision
-------------------
383, 2017/01/12
"""
import datetime
import logging
import os
import numpy as np
from scipy.stats import mstats, sigmaclip
from astropy.io import fits
from .sources import Sources
from ..instruments import Instrument
from ..psf import Variables
from ..utils.statistics import der_snr


__author__ = 'Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)'
__revision__ = 383


logger = logging.getLogger(__name__)


class SpectrumWriter(object):
    """
    This class is designed to read the spectra extracted by PampelMuse for the
    individual components from the prm-file and write them to individual FITS
    files. If requested, spectrum files will also contain small FITS images
    showing the regions around the extracted sources.
    """

    HEADER_KEYS = ["OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS", "MJD-OBS", "JUL-DATE",
                   "EXPTIME", "RA", "DEC", "EPOCH"]
    AMBI_KEYS = ["AIRM START", "AIRM END", "AMBI FWHM START", "AMBI FWHM END", "AMBI PRES START",
                 "AMBI PRESS END", "AMBI RHUM", "AMBI TAU0", "AMBI TEMP", "AMBI WINDDIR", "AMBI WINDSP"]

    def __init__(self, sources, ifs_data, psf_attributes=None, start=0, stop=-1):
        """
        Initialize a new instance of the SpectrumWriter source.

        Parameters
        ----------
        sources : an instance of pampelmuse.core.Sources
            This parameter must be used to provide all necessary information
            about the the sources that were extracted.
        ifs_data : an instance of pampelmuse.instruments.Instrument
            This parameter must be used to provide all necessary information
            about the IFS data.
        psf_attributes : an instance of pampelmuse.psf.Variables, optional
            This parameter can be used to provide additional information about
            the PSF that was used during the extraction.
        start : int, optional
            The first layer of the IFS data that is included in the spectra.
        stop : int, optional
            The last layer of the IFS data that is included in the spectra.
        """
        assert isinstance(sources, Sources)
        self.sources = sources
        assert isinstance(ifs_data, Instrument)
        self.ifs_data = ifs_data
        if psf_attributes is not None:
            assert isinstance(psf_attributes, Variables)
        self.psf_attributes = psf_attributes

        self.start = start
        self.stop = stop

        # initialize an empty header that will contain the header cards that are the same for all spectra collected
        # from an observation
        self.header = fits.Header()

        # add dispersion information to header
        if self.sources.data_wave_hdu is None:
            self.header["CRPIX1"] = self.sources.crpix - start
            self.header["CRVAL1"] = self.sources.crval
            self.header["CDELT1"] = self.sources.cdelt
            self.header["CTYPE1"] = self.sources.ctype
            self.header["CUNIT1"] = self.sources.cunit
            self.header["WSTART"] = (self.sources.crval - (self.sources.crpix - start - 1)*self.sources.cdelt)
            self.header["WSTEP"] = self.sources.cdelt
        else:
            self.header['WAVE'] = 'WAVE'  # if dedicated HDU with wavelength information exists

        for header_key in self.HEADER_KEYS:
            if self.ifs_data.primary_header is not None and header_key in self.ifs_data.primary_header:
                self.header[header_key] = self.ifs_data.primary_header[header_key]
            elif header_key in self.ifs_data.datahdu.header:
                self.header[header_key] = self.ifs_data.datahdu.header[header_key]
            else:
                logger.warning('Missing header keyword "{0}".'.format(header_key))

        # add JUL-DATE keyword if only MJD-OBS is provided
        if "MJD-OBS" in self.header and "JUL-DATE" not in self.header:
            self.header["JUL-DATE"] = self.header["MJD-OBS"] + 2400000.5
        if "JUL-DATE" in self.header:
            julian_date = self.header["JUL-DATE"]
        else:
            julian_date = 0
        self.date_string = "jd{0:07d}p{1:04d}".format(int(julian_date), int(10000*(julian_date-int(julian_date))))

        # add keywords describing observing conditions
        for ambi_key in self.AMBI_KEYS:
            full_key = "ESO TEL {0}".format(ambi_key)
            if self.ifs_data.primary_header is not None and full_key in self.ifs_data.primary_header:
                self.header["HIERARCH OBSERVATION {0}".format(ambi_key)] = self.ifs_data.primary_header[full_key]
            elif full_key in self.ifs_data.datahdu.header:
                self.header["HIERARCH OBSERVATION {0}".format(ambi_key)] = self.ifs_data.datahdu.header[full_key]
            else:
                pass

        # add estimate for seeing on detector using the fitted FWHM of the PSF profile
        if self.psf_attributes is not None:
            i = list(psf_attributes.names).index("fwhm")
            mean_fwhm = psf_attributes.data[~psf_attributes.mask[:, i, 0], i, 0].mean()
            self.header["hierarch PAMPELMUSE SEEING"] = (mean_fwhm, "Fitted PSF FWHM in spaxels")

        self.xy = None
        self.dmag = None
        self.contaminated = None

    def identify_contaminated(self, passband, kappa=2.5):
        """
        This method identifies stars for which the magnitude derived from the
        spectrum deviates significantly from the input magnitude.

        The method first determines a magnitude in the provided passband from
        each (stellar) spectrum which is then compared to the input magnitude
        of the brightest star contributing to the spectrum. Outliers are
        identified afterwards by performing a kappa-sigma-clipping on the
        scatter between input and recovered magnitudes as a function of input
        magnitude.

        Parameters
        ----------
        passband : str
            The name of the filter in which the magnitudes are derived from
            the spectra. Ideally the same filter used in the source selection
            process.
        kappa : float, optional
            The efficiency of the kappa-sigma-clipping.

        Returns
        -------
        dmag : nd_array
            The deviations between input and recovered magnitudes of all
            sources. Offsets for non-stellar and nearby sources are set to
            NaN. Also stored as property SpectrumWriter.dmag.
        contaminated : nd_array
            Flag indicating if a source has been flagged as contaminated.
            Non-stellar and nearby sources are set to False. Also stored as
            property SpectrumWriter.contaminated.
        """
        # intially set all offsets to NaN
        self.dmag = np.zeros(self.sources.nsrc, dtype=np.float32)
        self.dmag[:] = np.nan

        # load spectra for all sources, determine magnitudes, write offsets to dmag array
        self.sources.load_spectra()
        has_spectrum = np.in1d(self.sources.status, [2, 3])
        components = self.sources.specrow[has_spectrum]
        self.dmag[has_spectrum] = self.sources.get_magnitude(
            component=components, passband=passband) - self.sources.mag[has_spectrum]

        # perform kappa-sigma-clipping in 10 (input) magnitude bins
        bin_edges = mstats.mquantiles(self.sources.mag[np.isfinite(self.dmag)], np.arange(0, 1.0, 0.1))
        bin_numbers = np.digitize(self.sources.mag, bin_edges)

        # loop over bins, find outliers
        self.contaminated = np.zeros(self.sources.nsrc, dtype=np.bool)
        for i in np.unique(bin_numbers):
            slc = np.isfinite(self.dmag) & (bin_numbers == i)
            _, min_clip, max_clip = sigmaclip(self.dmag[slc], kappa, kappa)
            self.contaminated[slc] = (self.dmag[slc] < min_clip) | (self.dmag[slc] > max_clip)

        return self.dmag, self.contaminated

    def add_header_key(self, key, value, comment=None):
        """
        This method is designed to provide additional keywords that should be
        stored in the headers of the output spectra.

        Parameters
        ----------
        key : basestring
            The name of the new key. Note that keywords longer than eight
            digits must use the HIERARCH scheme.
        value : int, float, basestring
            The value of the new parameter.
        comment : basestring, optional
            An optional comment to be stored for the new key.
        """
        if comment is not None:
            self.header[key.upper()] = (value, comment)
        else:
            self.header[key.upper()] = value

    def __call__(self, prefix, folder='spectra', filter_names=None, poststamps=False, radius=10):
        """
        This method performs the storing of the individual spectra to disk.

        Parameters
        ----------
        prefix : basestring
            A common prefix used to create the names of the individual FITS
            files.
        folder : basestring, optional
            The folder in which the spectra should be stored. If it does not
            exist, it will be created.
        filter_names : basestring or list, optional
            One or more filter names for which magnitudes should be estimated
            from the spectra. If the 'poststamp' flag is being set, then the
            images will be created for each filter as well.
            NOTE: PampelMuse only stores a few filter curves internally. See
            the manual for how additional filters can be provided.
        poststamps : bool
            If this flag is set, the routine will store one or more small
            images as additional extensions in each spectrum FITS file. The
            images will show the area around the locations where the spectra
            were extracted. Unless the option 'filter_names' is used, white
            light images will be stored.
        radius : int, optional
            The radius around the sources covered by the poststamp images.
        """
        if not os.path.exists(folder):
            logging.info("Creating folder '{0}' to store spectra...".format(folder))
            os.mkdir(folder)

        if filter_names is not None:
            if isinstance(filter_names, basestring):
                filter_names = [filter_names, ]
        else:
            filter_names = []

        images = {}
        if poststamps:
            if not filter_names:  # create white light image
                logger.info('Creating white light image from IFS data...')
                images['white'] = self.ifs_data.integrate(imin=self.start, imax=self.stop, nbin=10, return_grid=True)[0]
            else:
                for name in filter_names:
                    logger.info('Creating {0}-band from IFS data...'.format(name))
                    images[name] = self.ifs_data.integrate_over_filtercurve(name, nbin=10, return_grid=True)[0][0]

        # load spectra for all components
        self.sources.load_spectra()

        # load IFS coordinates for all sources, get mean coordinates along spectral axis.
        self.sources.load_ifs_coordinates()
        self.xy = np.nanmean(self.sources.ifs_coordinates[..., 0], axis=0)

        # loop over all components, i.e. all spectra
        for i in range(self.sources.ncmp):
            # get index of the primary (brightest) component contributing to the spectrum
            j = self.sources.indexmap[i][0]

            # for storing the poststamps (if requested)
            image_hdus = []

            # sanity check: sources with status 1 (=near brighter source) or -1 (=not used) should have no spectrum
            if self.sources.status[j] in [1, -1]:
                logger.error("Found spectrum for source #{0} with status {1}".format(self.sources.ID[j],
                                                                                     self.sources.status[j]))
                continue

            flag = 0
            # each spectrum is assigned a flag which is the sum of the following numbers:
            #  1 - More than one star contribute to the spectrum
            #  2 - The estimated S/N of the spectrum is <10
            #  4 - The mean flux of the spectrum is <0
            #  8 - The centroid of the star is outside the field of view
            # 16 - The spectrum may be contaminated (recovered magnitude deviates from input magnitude)

            # create primary HDU with spectrum
            primary_hdu = fits.PrimaryHDU(self.sources.data[self.start:self.stop, i, 0], header=self.header)

            # create image HDU with uncertainties, copy dispersion information
            sigma_hdu = fits.ImageHDU(self.sources.data[self.start:self.stop, i, 1], name='SIGMA')
            for key in ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1', 'WSTART', 'WSTEP', 'WAVE']:
                if key in self.header:
                    sigma_hdu.header[key] = self.header[key]

            # check if dedicated wavelength array exists, if so, create new image HDU for it
            if self.sources.data_wave is not None:
                wave_hdu = fits.ImageHDU(self.sources.data_wave[self.start:self.stop, i], name='WAVE')
            else:
                wave_hdu = None

            if self.sources.status[j] in [2, 3]:  # spectrum belongs to a resolved source.

                source_id = self.sources.ID[j]
                logger.info("Processing source #{0} [{1}/{2}]".format(source_id, i+1, self.sources.ncmp))

                primary_hdu.header["HIERARCH STAR ID"] = source_id

                # update RA & Dec values if provided in source catalog
                if not np.isnan(self.sources.ra[j]) and not np.isnan(self.sources.dec[j]):
                    primary_hdu.header["RA"] = self.sources.ra[j]
                    primary_hdu.header["DEC"] = self.sources.dec[j]
                    primary_hdu.header["HIERARCH STAR RA"] = self.sources.ra[j]
                    primary_hdu.header["HIERARCH STAR DEC"] = self.sources.dec[j]

                # add magnitude to header
                if not np.isnan(self.sources.mag[j]):
                    primary_hdu.header["HIERARCH STAR MAG"] = self.sources.mag[j]

                #  Check if more than one source contribute to the spectrum
                cmt = "Whether multiple stars contribute"
                if len(self.sources.indexmap[i]) > 1:
                    flag += 1
                    for l in range(1, len(self.sources.indexmap[i])):
                        # if so, add RA, DEC, and magnitude information for all secondary stars
                        m = self.sources.indexmap[i][l]
                        if not np.isnan(self.sources.ra[m]) and not np.isnan(self.sources.dec[m]):
                            primary_hdu.header["HIERARCH STAR{0} RA".format(l+1)] = self.sources.ra[m]
                            primary_hdu.header["HIERARCH STAR{0} DEC".format(l+1)] = self.sources.dec[m]
                        if not np.isnan(self.sources.mag[m]):
                            primary_hdu.header["HIERARCH STAR{0} MAG".format(l+1)] = self.sources.mag[m]
                    primary_hdu.header["HIERARCH SPECTRUM MULTISTAR"] = (True, cmt)
                    primary_hdu.header["OBJNAME"] = "{0} {1} {2}+".format(prefix.upper(), self.ifs_data.name, source_id)
                else:
                    primary_hdu.header["HIERARCH SPECTRUM MULTISTAR"] = (False, cmt)
                    primary_hdu.header["OBJNAME"] = "{0} {1} {2}".format(prefix.upper(), self.ifs_data.name, source_id)

                # get IFS coordinates for the star and check if it is inside the field of view
                k = np.sum(self.sources.ifs_coordinates_loaded[:j])
                xy_j = self.xy[k]

                # add x- and y-position of PSF centroid in cube
                primary_hdu.header["HIERARCH SPECTRUM XCUBE"] = (xy_j[0], "[spaxel] x-position in cube")
                primary_hdu.header["HIERARCH SPECTRUM YCUBE"] = (xy_j[1], "[spaxel] y-position in cube")

                infield = self.ifs_data.infov([xy_j])[0]
                if not infield:
                    flag += 8
                primary_hdu.header["HIERARCH SPECTRUM INFIELD"] = (infield, "Whether centroid inside FOV")

                edge_distance = self.ifs_data.distance_to_edge([xy_j])[0]
                primary_hdu.header["HIERARCH SPECTRUM EDGEDIST"] = (edge_distance, "[spaxel] Distance to FOV edge")

                # create poststamp image(s) if requested
                if len(images) > 0:
                    image_header = self._header_with_nearby_sources(x_c=xy_j[0], y_c=xy_j[1], radius=radius)
                else:
                    image_header = None
                for filter_name, image in images.items():
                    poststamp_image = SpectrumWriter._extract_poststamp(x_c=xy_j[0], y_c=xy_j[1], image=image,
                                                                        radius=radius)
                    image_hdus.append(fits.ImageHDU(poststamp_image, name=filter_name.upper(), header=image_header))

            elif self.sources.status[j] == 4:
                # for sky component, add x- and y-position of patch centroid in cube
                primary_hdu.header["HIERARCH SPECTRUM XCUBE"] = (self.sources.x[j], "[spaxel] x-centroid in cube")
                primary_hdu.header["HIERARCH SPECTRUM YCUBE"] = (self.sources.y[j], "[spaxel] y-centroid in cube")

            # calculate S/N of the spectrum
            snr = der_snr(self.sources.data[self.start:self.stop, i, 0])
            if snr < 10:
                flag += 2
            if not np.isnan(snr):
                primary_hdu.header["HIERARCH SPECTRUM SNRATIO"] = (snr, "Estimated signal-to-noise")
            else:
                logger.error("S/N estimation returned NaN - Spectrum of source #{0} corrupted.".format(source_id))

            if np.mean(self.sources.data[self.start:self.stop, i, 0]) < 0:
                flag += 4

            # calculate magnitude(s) from spectrum if requested
            for filter_name in filter_names:
                mag = self.sources.get_magnitude(component=i, passband=filter_name)
                if np.isfinite(mag):
                    primary_hdu.header["HIERARCH SPECTRUM MAG {0}".format(filter_name.upper())] = (
                        mag, "AB magnitude from spectrum")
                else:
                    logger.error('{0}-magnitude calculation for source #{1} returned NaN.'.format(
                        filter_name, source_id))
            if self.dmag is not None and np.isfinite(self.dmag[j]):
                primary_hdu.header['HIERARCH SPECTRUM DMAG'] = (self.dmag[j], 'Magnitude difference to input.')
                if self.contaminated is not None and np.isfinite(self.dmag[j]):
                    primary_hdu.header['HIERARCH SPECTRUM CONTAMINATED'] = self.contaminated[j]
                    if self.contaminated[j]:
                        flag += 16

            primary_hdu.header["HIERARCH SPECTRUM QLTFLAG"] = (flag, "Quality flag of source")

            if self.sources.status[j] == 0:
                name = "unresolved"
                primary_hdu.header["HIERARCH SPECTRUM TYPE"] = name
                primary_hdu.header["OBJNAME"] = "{0} {1} UNRES".format(prefix.upper(), self.ifs_data.name)
            elif self.sources.status[j] == 4:
                name = "sky{0:08d}".format(abs(self.sources.ID[j]))
                primary_hdu.header["HIERARCH SPECTRUM TYPE"] = "sky"
                primary_hdu.header["OBJNAME"] = "{0} {1} SKY{2:d}".format(prefix.upper(), self.ifs_data.name,
                                                                          abs(self.sources.ID[j]))
            else:
                name = "id{0:09d}".format(source_id)
                primary_hdu.header["HIERARCH SPECTRUM TYPE"] = "star"
                primary_hdu.header["OBJNAME"] = "{0} {1} {2}".format(prefix.upper(), self.ifs_data.name,
                                                                     abs(self.sources.ID[j]))

            outfile_prefix = "{0}{1}{2}f{3:03d}".format(prefix, name, self.date_string, int(flag))

            utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
            primary_hdu.header["DATE-GEN"] = (utc, "UTC this file was generated.")

            outfile = fits.HDUList(hdus=[primary_hdu, sigma_hdu])
            if len(image_hdus) > 0:
                outfile.extend(image_hdus)
            if wave_hdu is not None:
                outfile.append(wave_hdu)
            outfile.writeto(os.path.join(folder, "{0}.fits".format(outfile_prefix)), clobber=True)

            logger.info("Wrote spectrum '{0}' [{1}/{2}]".format(os.path.join(folder, "{0}.fits".format(outfile_prefix)),
                                                                i+1, self.sources.ncmp))

    @staticmethod
    def _extract_poststamp(image, x_c, y_c, radius=10):
        """
        This internal method is used to extract a subimage of given radius
        around a pixel position in a larger image.

        The subimage that is returned will always have the same size, namely
        2x the radius plus one pixel along each size. If the provided pixel
        position is near the edge of the input image, the area not covered
        by the input image will be filled with zeros in the returned image.

        Parameters
        ----------
        image : nd_array
            The image from which the subimage should be extracted.
        x_c : float
            The centroid of the subimage along the x-axis. Note that the
            x-axis is the second axis in numpy notation.
        y_c : float
            The centroid of the subimage along the y-axis. Note that the
            y-axis is the first axis in numpy notation.
        radius : int
            The radius extracted around the centroid. The image that is
            returned will have a size of (2x radius + 1, 2x radius +1)

        Returns
        -------
        subimage : nd_array
            The extracted image. Note that images not covered by the input
            image (if any) will be filled with zeros.
        """
        x_c = int(round(x_c))
        y_c = int(round(y_c))

        outdata = np.zeros((2*radius+1, 2*radius + 1), dtype=image.dtype)

        dx = (min(radius, x_c), min(radius + 1, image.shape[1] - x_c))
        dy = (min(radius, y_c), min(radius + 1, image.shape[0] - y_c))

        outdata[radius - dy[0]:radius + dy[1], radius - dx[0]:radius + dx[1]] = image[
            y_c - dy[0]:y_c + dy[1], x_c - dx[0]:x_c + dx[1]]

        return outdata

    def _header_with_nearby_sources(self, x_c, y_c, radius):
        """
        Internal image used to create a FITS header containing information
        about the sources found around a pixel position in the IFS data.

        Parameters
        ----------
        x_c : float
            The centroid of the selection along the x-axis.
        y_c : float
            The centroid of the selection along the y-axis.
        radius : int
            The radius of the selection around the centroid.

        Returns
        -------
        header : a astropy.io.fits.Header instance
            The header containing the positions, IDs, and magnitudes (if
            available) of the sources in the footprint.
        """
        if self.xy is None:
            return

        nearby = (abs(self.xy[:, 0] - x_c) < radius) & (abs(self.xy[:, 1] - y_c) < radius)

        image_header = fits.Header()
        n = 1
        for i, j in enumerate(np.flatnonzero(self.sources.ifs_coordinates_loaded)):
            if not nearby[i]:
                continue
            image_header['HIERARCH NEARBY STAR{0:03d} ID'.format(n)] = self.sources.ID[j]
            image_header['HIERARCH NEARBY STAR{0:03d} STATUS'.format(n)] = self.sources.status[j]
            if np.isfinite(self.sources.mag[j]):
                image_header['HIERARCH NEARBY STAR{0:03d} MAG'.format(n)] = self.sources.mag[j]
            image_header['HIERARCH NEARBY STAR{0:03d} X'.format(n)] = self.xy[i, 0] + radius - x_c
            image_header['HIERARCH NEARBY STAR{0:03d} Y'.format(n)] = self.xy[i, 1] + radius - y_c
            n += 1

        return image_header
