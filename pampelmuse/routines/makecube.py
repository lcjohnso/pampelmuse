"""
makecube.py
===========
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
Provides
--------
function pampelmuse.routines.makecube.makecube

Purpose
-------
This module implements the function 'makecube' that is executed when
PampelMuse is started with the MAKECUBE option. The short function will
transform 2dim. IFS data into a 3dim. datacube. If the IFS data has been
previously processed with the ADDWCS routine, the created datacube will
have valid WCS information in its header.

Latest SVN revision
-------------------
334, 2016/07/26
# IMPORTANT: This module currently is not working, it needs to be bugfixed.
"""
import datetime
import logging
import os


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 334


def add_parser(parser):

    p = parser.add_parser("MAKECUBE", help="""
    The sub-routine creates a 3dim. data cube from 2dim. IFS data and stores it as
    a FITS file under <prefix>.cube.fits. Note that if the IFS data has been
    previously passed to sub-routine ADDWCS, the routine will also set the WCS infor-
    mation in the header of the resulting cube right.""")
    p.set_defaults(func=makecube)


def makecube(pampelmuse):
    """
    Create a data cube from 2dim. IFS data. If available, also add WCS informations
    to the header of the output FITS-file. The resulting cube will be stored under
    <pampelmuse.prefix>.cube.fits, a possibly existing file will be overwritten.
    Note that currently only the data is stored, not the uncertainties or the spectrumMask.
    """

    logging.info("Executing requested routine MAKECUBE [Revision {0:d}].".format(__revision__))

    if pampelmuse.ifs_data.ndim == 3:
        logging.info("IFS data is already 3dim.")
        logging.error("MAKECUBE can only be used with 2dim. data.")

    output_hdu = pampelmuse.ifs_data.make_cube(asPrimary=True)
    
    output_hdu.header.set("ORIGIN", "PampelMuse MAKECUBE - revision {0}".format(__revision__),
                          comment="FITS file originator", after="NAXIS3")
    output_hdu.header.set("DATE", datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S"),
                          comment="UTC timestamp FITS file was created", after="ORIGIN")
    
    if hasattr(pampelmuse, "addwcs"):
        logging.info("Adding WCS information...")
        
        # add WCS keywords if they are available
        keys = ["RADECSYS", "EQUINOX",  "WCSDIM",  "CTYPE2", "CTYPE1", "CRVAL2", "CRVAL1", "CRPIX2", "CRPIX1"]
        insert = ["NAXIS3", "RADECSYS", "EQUINOX", "CTYPE3", "CTYPE2", "CRVAL3", "CRVAL2", "CRPIX3", "CRPIX2"]
        after = [True, True, True, False, False, False, False, False, False]

        for i in xrange(len(keys)):
            if hasattr(pampelmuse.addwcs, keys[i].lower()):
                if after[i]:
                    output_hdu.header.set(keys[i].upper(), getattr(pampelmuse.addwcs, keys[i].lower()),
                                          after=insert[i])
                else:
                    output_hdu.header.set(keys[i].upper(), getattr(pampelmuse.addwcs, keys[i].lower()),
                                          before=insert[i])

        # add WCS CDX_Y matrix if available
        last = "CRPIX3"
        for i in xrange(1, 4):
            for j in xrange(1, 4):
                key = "CD{0}_{1}".format(i, j)
                if hasattr(pampelmuse.addwcs, key.lower()):
                    output_hdu.header.set(key, getattr(pampelmuse.addwcs, key.lower()), after=last)
                    if i == j and "CDELT{0}".format(i) in output_hdu.header:
                        del output_hdu.header["CDELT{0}".format(i)]
                elif key == "CD3_3" and "CDELT3" in output_hdu.header:
                    output_hdu.header.set(key, output_hdu.header["CDELT3"], after=last)
                    del output_hdu.header["CDELT3"]
                else:
                    output_hdu.header.set(key, 0.0, after=last)
                last = key
    
    # store results, existing cube may be overwritten
    outfilename = "{0}.cube.fits".format(pampelmuse.prefix)
    logging.info("Writing data cube to file {0}...".format(outfilename))
    if os.path.exists(outfilename):
        os.remove(outfilename)
        logging.warning("Existing cube has been overwritten.")

    output_hdu.writeto(outfilename)
