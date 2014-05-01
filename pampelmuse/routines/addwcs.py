"""
ADDWCS
======
The routine is designed to add WCS information to the header of the integral-
field data after the pixel positions of some sources (with known RA and Dec)
have been identified (and optionally fitted).
The routine currently requires `pyraf` and `astLib` to work.


Latest SVN revision
-------------------
169, 2014/04/29
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 169

import datetime, logging, os, sys, tempfile
import numpy as np
from astropy.coordinates import Angle
from astropy.io import fits
from pyraf.iraf import images

## required PampelMuse-routines
from pampelmuse.core import Sources
from pampelmuse.core.coordinates import Transformation
from pampelmuse.core.sources import match_on_id
from pampelmuse.functions.io import read_asciifile



def addwcs(pampelmuse):

    logging.info("Executing requested routine ADDWCS [Revision {0:d}].".format(__revision__))

    ## Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    ## Source catalogue
    sources = Sources(data=pampelmuse.hdulist["SOURCES"])

    ## check if additional file with RA- & Dec-coordinates were provided
    if pampelmuse.addwcs.sourcelist != None:
        logging.info("Additional RA- & Dec-file provided: {0}".format(pampelmuse.addwcs.sourcelist))
        cfp_results = read_asciifile(pampelmuse.addwcs.sourcelist, usecols=[0, 1, 2], usenames="id,ra,dec", aliases={"id":["ID",], "ra":["RA"], "dec":["Dec", "DEC"]})
    
        wcs_id = np.asarray(cfp_results["id"], dtype=np.int32)
        wcs_ra = np.asarray(cfp_results["ra"], dtype=np.float32)
        wcs_dec = np.asarray(cfp_results["dec"], dtype=np.float32)
    
        match = match_on_id(sources.ID, wcs_id)
        sources.ra[match] = wcs_ra[np.in1d(wcs_id, sources.ID)]
        sources.dec[match] = wcs_dec[np.in1d(wcs_id, sources.ID)]


    if (sources.ra!=sources.ra).all() or (sources.dec!=sources.dec).all() == None:
        logging.critical("Catalogue of sources does not contain RA- & Dec-information. We stop here.")
        return

    ## Check if PRM-file contains results of successful fit
    if hasattr(pampelmuse, "cubefit") and hasattr(pampelmuse.cubefit, "status"):
        fitStatus = pampelmuse.cubefit.status
    else:
        fitStatus = "Unknown"

    if fitStatus != "SUCCESS":
        logging.error("PRM-file does not contain results of successful fit.")


    ## Collect source positions
    logging.info("Obtaining information about source coordinates...")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        try:
            i = pampelmuse.hdulist.index_of("INPOSITIONS")
            sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        except KeyError:
            pass
    except KeyError:
        pass

    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.transformation = Transformation(hdu=pampelmuse.hdulist[i])
        
    except KeyError:
        logging.warning("No coordinate transformation (HDU 'POSPARS') detected.")

    ## currently the WCS information is fixed for the central layer of the analysed data range
    ## and using all sources with status 2 or 3
    sources.load_ifs_coordinates(ids=sources.ID[(sources.status>1)&(sources.status<4)], layers=sources.ndisp//2)

    ## the routine needs to create a temporary files that contains the pixel and WCS information,
    ## one source per line. To this aim, the tempfile module of python is used that automatically
    ## selects a filename that does not exist and deletes the file afterwards.
    tmp_coord_file = tempfile.NamedTemporaryFile()

    j = 0
    for i in xrange(sources.nsrc):
        if not sources.ifs_coordinates_loaded[i]: continue
        
        if sources.ifs_coordinates_fitted[i]:
            x, y = sources.ifs_coordinates[0,j,:,1]
        else:
            x, y = sources.ifs_coordinates[0,j,:,0]
    
        ra = Angle(sources.ra[i], unit="hour").degrees
        dec = Angle(sources.dec[i], unit="degree").degrees
    
        tmp_coord_file.write("{0}  {1}  {2:7.3f}  {3:7.3f}\n".format(ra, dec, float(x), float(y)))

        j += 1

    tmp_coord_file.flush() ## save input file for calculation
    
    ## a temporary FITS image needs to be created because the only possibility to turn the results from ccmap into useful
    ## header keywords (except for doing the calculations manually) is to update the header of an existing (2dim.!) FITS file
    tmp_fits_root = fits.util._tmp_name("./")
    os.remove(tmp_fits_root)
    tmp_fits_name = "{0}.fits".format(tmp_fits_root)
    tmpHDU = fits.PrimaryHDU(np.zeros(pampelmuse.ifs_data.instrument.shape, dtype=np.int16))
    tmpHDU.writeto(tmp_fits_name)
    
##    tmp_database = tempfile.NamedTemporaryFile() ## open temporary file to hold results

    logging.info("Registering image using IRAF tool 'ccmap'...")
    
    ## When running the pyraf routine ccmap, the standard output is redirected to a temporary file so that the output created
    ## by ccmap can be included in the logging scheme.
    stdout = sys.stdout
    redirect_stdout = tempfile.TemporaryFile()
    sys.stdout = redirect_stdout
    
    images.imcoords.ccmap(tmp_coord_file.name, tempfile.NamedTemporaryFile().name, images=tmp_fits_name, xcol=3, ycol=4, lngcol=1, latcol=2, update=True,
                          results="STDOUT", interactive=False)
    
    sys.stdout = stdout
    
    redirect_stdout.seek(0)
    for line in redirect_stdout: logging.info(line.rstrip("\n"))
    
    ## WCS informations only make sense for data cubes, not for RSS files. Therefore, if the data are 2dim., the WCS related keywords are
    ## stored as hierarch keywords 'hierarch PAMPELMUSE ADDWCS <KEY>' in the PampelMuse PRM-file. Only for 3dim. data, the WCS information
    ## is also directly used to update the data header. The PampelMuse routine MAKECUBE will handle the hierarch keywords correctly.

    tmp_fits_file = fits.open(tmp_fits_name)
    tmp_fits_header = tmp_fits_file[0].header

    for key in ["RADECSYS", "EQUINOX", "WCSDIM", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"]:

        setattr(pampelmuse.addwcs, key.lower(), tmp_fits_header[key])
        if pampelmuse.ifs_data.ndim==3:
            pampelmuse.ifs_data.data_hdu.header.update(key, tmp_fits_header[key])
    
    tmp_fits_file.close()
    os.remove(tmp_fits_name)
    
    pampelmuse.ifs_data.file.flush()
    
    pampelmuse.addwcs.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.addwcs.revision = __revision__
    
    pampelmuse.hdulist[0] = pampelmuse.makeHeader(hdu=pampelmuse.hdulist[0])
    if pampelmuse.addwcs.sourcelist != None:
        pampelmuse.hdulist["SOURCES"] = sources.make_catalogue_hdu()
    pampelmuse.hdulist.flush()
    pampelmuse.hdulist.close()

