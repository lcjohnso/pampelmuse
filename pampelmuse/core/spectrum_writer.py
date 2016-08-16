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


Added
-----
2016/08/16, rev. 366

Latest SVN revision
-------------------
366, 2016/08/16
"""
import datetime
import logging
import os
import numpy as np
from astropy.io import fits
from .sources import Sources
from ..instruments import Instrument
from ..psf import Variables
from ..utils.statistics import der_snr


__author__ = 'Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)'
__revision__ = 366


logger = logging.getLogger(__name__)


class SpectrumWriter(object):

    HEADER_KEYS = ["OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS", "MJD-OBS", "JUL-DATE",
                   "EXPTIME", "RA", "DEC", "EPOCH"]
    AMBI_KEYS = ["AIRM START", "AIRM END", "AMBI FWHM START", "AMBI FWHM END", "AMBI PRES START",
                 "AMBI PRESS END", "AMBI RHUM", "AMBI TAU0", "AMBI TEMP", "AMBI WINDDIR", "AMBI WINDSP"]

    def __init__(self, sources, ifs_data, psf_attributes=None, start=0, stop=-1):
        """

        Parameters
        ----------
        sources
        ifs_data
        psf_attributes
        start
        stop
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
        if self.sources.data_wave is None:
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

    def add_header_key(self, key, value, comment=None):
        """

        Parameters
        ----------
        key
        value
        comment

        Returns
        -------

        """
        if comment is not None:
            self.header[key.upper()] = (value, comment)
        else:
            self.header[key.upper()] = value

    def __call__(self, prefix, folder='spectra', filter_names=None, poststamps=False, radius=10):
        """

        Parameters
        ----------
        prefix
        folder
        filter_names
        poststamps
        radius

        Returns
        -------

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
                    images[name] = self.ifs_data.integrate_over_filtercurve(name, nbin=100, return_grid=True)[0][0]

        # load spectra for all components
        self.sources.load_spectra()

        # loop over all components, i.e. all spectra
        for i in range(self.sources.ncmp):
            # get index of the primary (brightest) component contributing to the spectrum
            j = self.sources.indexmap[i][0]

            # for storing the poststamps (if requested)
            image_hdus = []

            # sanity check: sources with status 1 (=near brighter source) or -1 (=not used) should have no spectrum
            if self.sources.status[j] in [1, -1]:
                logger.error("Found spectrum for source #{0} with status {1}".format(
                    self.sources.ID[j], self.sources.status[j]))
                continue

            flag = 0
            # each spectrum is assigned a flag which is the sum of the following numbers:
            #  1 - More than one star contribute to the spectrum
            #  2 - The estimated S/N of the spectrum is <10
            #  4 - The mean flux of the spectrum is <0
            #  8 - The centroid of the star is outside the field of view

            # load spectrum for current component into memory
            primary_hdu = fits.PrimaryHDU(self.sources.data[self.start:self.stop, i, 0], header=self.header)

            sigma_hdu = fits.ImageHDU(self.sources.data[self.start:self.stop, i, 1])
            sigma_hdu.header['EXTNAME'] = 'SIGMA'

            # copy dispersion information
            for key in ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1', 'WSTART', 'WSTEP', 'WAVE']:
                if key in self.header:
                    sigma_hdu.header[key] = self.header[key]

            if self.sources.data_wave is not None:
                wave_hdu = fits.ImageHDU(self.sources.data_wave[self.start:self.stop, i])
                wave_hdu.header['EXTNAME'] = 'WAVE'
            else:
                wave_hdu = None

            if self.sources.status[j] in [2, 3]:

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

                # load IFS coordinates for the star and check if it is inside the field of view; omit NaN values
                self.sources.load_ifs_coordinates(ids=source_id)
                xy_coord = self.sources.ifs_coordinates[:, 0, :, 0]
                xy_j = np.mean(xy_coord[(~np.isnan(xy_coord)).all(axis=1)], axis=0)

                # add x- and y-position of PSF centroid in cube
                primary_hdu.header["HIERARCH SPECTRUM XCUBE"] = (xy_j[0], "[spaxel] x-position in cube")
                primary_hdu.header["HIERARCH SPECTRUM YCUBE"] = (xy_j[1], "[spaxel] y-position in cube")

                infield = self.ifs_data.infov([xy_j])[0]
                if not infield:
                    flag += 8
                primary_hdu.header["HIERARCH SPECTRUM INFIELD"] = (infield, "Whether centroid inside FOV")

                edge_distance = self.ifs_data.distance_to_edge([xy_j])[0]
                primary_hdu.header["HIERARCH SPECTRUM EDGEDIST"] = (edge_distance, "[spaxel] Distance to FOV edge")

                # create poststamps if requested
                for filter_name, image in images.items():
                    poststamp_image = SpectrumWriter._extract_poststamp(x_c=xy_j[0], y_c=xy_j[1], image=image,
                                                                        radius=radius)
                    image_hdus.append(fits.ImageHDU(poststamp_image, name=filter_name.upper()))

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

            if np.mean(self.sources.data[:, 0, 0]) < 0:
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

        Parameters
        ----------
        image
        x_c
        y_c
        radius

        Returns
        -------

        """
        x_c = int(round(x_c))
        y_c = int(round(y_c))

        outdata = np.zeros((2*radius+1, 2*radius + 1), dtype=image.dtype)

        dx = (min(radius, x_c), min(radius + 1, image.shape[1] - x_c))
        dy = (min(radius, y_c), min(radius + 1, image.shape[0] - y_c))

        outdata[radius - dy[0]:radius + dy[1], radius - dx[0]:radius + dx[1]] = image[
            y_c - dy[0]:y_c + dy[1], x_c - dx[0]:x_c + dx[1]]

        return outdata
