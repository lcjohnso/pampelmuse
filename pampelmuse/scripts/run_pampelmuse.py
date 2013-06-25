#! /usr/bin/env python
"""
PAMPELMUSE
==========
The front end. Not much in here except for the basic command-line tool
and some tools to change the configuration of PampelMuse.

CHANGELOG
---------
v0.0.0.1, 2011/09/06, added logo
v0.1.0.0, 2011/10/13, modified how the PampelMuse-scripts are imported
                      to reflect new package-structure.
                      -> first experimental "release"
v0.1.0.1, 2011/11/29, included logging-option
v0.1.0.2, 2011/12/08, added new routine `EXTRACT' to write the pixel
                      positions of the extracted stars to an ASCII-file
v0.1.0.3, 2012/01/13, some minor updates to usage of logging module
v0.1.0.4, 2012/01/17, added posfit module. Still needs some debugging and
                      documentation
v0.2.0.0, 2012/01/29, replaced posfit-module with updateprm-module. Since 
                      the previously applied polynomial fitting of the
                      parameters of the coordinate transformation led to
                      wrong results, the polyfit-module now works directly
                      on the source coordinates instead. To implement this,
                      the structure of the prm-file has been modified. Old
                      prm-files can be updated using the updateprm-module.
v0.2.0.1, 2012/02/12, slightly modified formatting of the logging-messages.
v0.2.1.1, 2012/02/12, fixed bug in format-string of logging-messages.
v0.3.0.0, 2012/03/01, added new script SUBTRES to subtract the contribution
                      of the (resolved) stars from a datacube.
v0.4.0.0, 2012/03/19, modified check of PRM-file structure so that files
                      created using INITFIT version >=0.4.0.0 pass it.
v0.5.0.0, 2012/06/26, added new script ADDWCS that can be used to add WCS
                      information to the header of the IFS datacube.
                      Also updated the help-text.
v0.5.0.1, 2012/07/03, slighly modified logging-initialization because calling
                      `logging.basicConfig` has no effect after a log-message
                      has been triggered.
v0.5.0.2, 2012/07/24, added handling for cases where PyQt4 is not available.
v0.5.1.2, 2012/07/24, some bugfixing wrt availability checks. Now we also
                      check for matplotlib support.
v0.6.0.0, 2012/12/23, added routine SHOTNOI that estimates the shot noise in a
                      datacube.
v0.7.0.0, 2013/03/27, implemented a new way to construct the PSF profiles. In
                      combination with a Levenberg-Marquard algorithm that is
                      now used to perform the non-linear fits, this
                      significantly speeds up the code (by a factor ~10(!)).
                      Also added support for 2dim. input data besides cubes.
v0.8.0.0, 2013/05/15, included the new PampelMuse GUI to facilitate the source
                      identification and parameter inspection/fitting. It
                      provides more functionality than the previous GUIs used
                      for these purposes. Together with the new GUI, the
                      possibility to load a reference image when running
                      INITFIT was provided.
v0.9.0.0, 2013/05/29, revised how the configuration of the code is handled
                      during the analysis. The code now tracks the entire configuration
                      using a dedicated namespace and also writes it to the header of
                      the primary HDU when a PRM-file is saved.
                      The order in which the configuration is used is: 1) header of a
                      PRM-file, 2) command-line input, 3) Pampelmuse xml-file
                      Also the preparation of the IFS data was moved from the subroutines
                      to the main routine (i.e. this code)
v0.9.1.0, 2013/05/31, check whether parameter 'USEQT' is False if dependencies for GUIs
                      are not fulfilled was not updated so far.
v0.9.1.1, 2013/06/25, small changes to logging messages.
v0.9.2.1, 2013/06/25, code crashes when ran without PyQt4 support because methods 'initfit'
                      and 'polyfit' were referenced but not imported.
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.9.2.1"


print """
            ,;:.    PampelMuse, version {0:17}    ,;:.    
           (::;;)   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (::;;)
            `;:'                                             `;:'

        (c) {1}""".format(__version__, __author__)


## IMPORT REQUIRED PYTHON MODULES ##############################################
import argparse, datetime, logging, os, re, sys
import numpy as np, pyfits

## initialize logging
logging.basicConfig(format="%(asctime)s[%(levelname)-8s]: %(message)s",
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.WARNING)

## check if all dependencies are fulfilled: ####################################
use_addwcs=True
try: ## check if astLib is available
    from astLib import astCoords
except ImportError:
    logging.warning("Cannot import module `astLib`.")
    logging.warning("Script ADDWCS cannot be used.")
    use_addwcs = False

try: ## check if pyraf available
    from pyraf.iraf import images
except ImportError:
    logging.warning("Cannot import module `pyraf`.")
    logging.warning("Script ADDWCS cannot be used.")
    use_addwcs = False

## check whether dependencies for graphical backends are fufilled
graphics = True
try:
    from PyQt4 import QtCore, QtGui
except ImportError:
    logging.error("Cannot load `PyQt4`-modules")
    graphics = False
try:
    import matplotlib.pyplot
except ImportError:
    logging.error("Cannot load `matplotlib`-modules")
    graphics = False
if not graphics:
    logging.error("Scripts INITFIT and POLYFIT cannot be used.")



## required PampelMuse-packages ################################################
from pampelmuse.config import xml_interact, PampelMuseConfig, Namespace
from pampelmuse.core.cube import Cube

## the PampelMuse-scripts
if graphics: from pampelmuse.scripts.initfit import initfit
from pampelmuse.scripts.cubefit import cubefit
if graphics: from pampelmuse.scripts.polyfit import polyfit
from pampelmuse.scripts.extract import get_pixpos
from pampelmuse.scripts.updateprm import updateprm
from pampelmuse.scripts.subtres import subtres
from pampelmuse.scripts.shotnoi import shotnoi
if use_addwcs: from pampelmuse.scripts.addwcs import addwcs


CONFIGPATH = os.path.expandvars("$pampelmuse_path/config/")


class GetValuesFromLine(argparse.Action):
    """
    Short routine to write 'name=value'-pairs appended to the '--fromline'
    option into a dictionary.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        outdict = {}
        for _val in values[0].split('-'):
            val = re.sub(r'\s', '', _val) ## remove whitespace chars.
            if len(val.split("=")) != 2:
                raise IOError, "Bad parameter definition in line: {0}"\
                        .format(val)
            name, value = val.split("=")
            outdict[name.upper()] = value
        setattr(namespace, self.dest, outdict)



class GetValuesFromFile(argparse.Action):
    """
    Short routine to extract 'name=value'-pairs from an ASCII-file provided
    using the '--fromfile' option and add them to a dictionary
    """    
    def __call__(self, parser, namespace, values, option_string=None):
        outdict = {}
        for _line in values:
            if not len(_line)==0:
                if not _line[0] == "#":
                    line = re.sub(r'\s', '', _line) ## remove whitespace chars.
                    if not len(line.split("=")) == 2:
                        raise IOError, "Bad parameter definition in file '{0}'"\
                            .format(values.name)+": {0}".format(line)
                    name, value = line.split("=")
                    outdict[name.upper()] = value
        setattr(namespace, self.dest, outdict)



PMparser = argparse.ArgumentParser(prog="PampelMuse", description="""
PampelMuse is a routine to extract stellar spectra from IFU data by usage of
PSF fitting techniques. It is designed for the analysis of IFU observations
of crowded-stellar fields. As the name suggests, it was originally intended
to serve  as an analysis tool for the MUSE-IFU. Yet, PampelMuse is designed
in a generic way and should be able to analyze data coming from similar IFUs
as well.
""")
PMparser.add_argument("prefix", type=str, help="""
The prefix common to all FITS-files required by PampelMuse. The naming of the
files should be according to the following scheme:\n
* PREFIX.dat.fits - the datacube, required by routines INITFIT and CUBEFIT\n
* PREFIX.err.fits - the sigma-cube, required by routines INITFIT and CUBEFIT\n
* PREFIX.msk.fits - a bad-pixel mask (optional), used by routine CUBEFIT if
                    provided\n
* PREFIX.prm.fits - the parameter file of the fit, provided by INITFIT,
                    required by CUBEFIT, POLYFIT and EXTRACT.""")

## Note: 'fromfile' and 'fromline' share the same dest 'parameter'. However,
##       since they are in a 'mutually exclusive group' and their 'default'
##       is set to argparse.SUPPRESS, at most one of them will contibute to
##       the namespace.
group = PMparser.add_mutually_exclusive_group(required=False)
group.add_argument("--fromfile", type=argparse.FileType('r'),
                   action=GetValuesFromFile, dest="parameter",
                   default=argparse.SUPPRESS, help="""
Update configuration using the parameters provided in an ASCII-file: One
parameter per line, name and value separated by '=', lines starting with '#'
are ignored""")
group.add_argument("--fromline", type=str, nargs=1,
                   action=GetValuesFromLine, dest="parameter",
                   default=argparse.SUPPRESS, help="""
Update configuration using the parameters provided in the command line,
following '--fromline'-string. Again, name and value must be separated
by '=' (no whitespace!). Individual parameters  must be comma-separated.""")

PMsubparsers = PMparser.add_subparsers(title='Available routines')


CFparser = PMsubparsers.add_parser("CUBEFIT", help="""
The routine is used for the actual PSF-fitting process. It must be provided
with apropriate input data (created e.g using INITFIT). By default, all the
parameter that have an impact on the fitting process are read from the
PampelMuse configuration-file. However, several ways are provided to change
the configuration on the fly. Check the option-list for details.
""")
CFparser.set_defaults(func=cubefit)

if graphics:
    IFparser = PMsubparsers.add_parser("INITFIT", help="""
The routine is used to prepare the datacube for the PSF fitting process.
It takes as input the datacube and an ASCII-table containing the results
from the crowded-field photometry. The different processing steps are
a) Interactive identification of bright sources in the field.
b) Aperture spectrophotometry of the previously identified sources.
c) Generation of an input catalogue containing intitial guessse of the
   spectra and the positions of the stars.
""")
    IFparser.add_argument("sourcelist", type=str, help="""
ASCII-file containing the results from crowded-field photometry. Lines
starting with '#' are considered comments. The first columns should contain
the names of the columns. The routine will look for columns named `id`, `x`,
`y` and <PASSBAND> as specified in the configuration.""")
    IFparser.set_defaults(func=initfit)

if graphics:
    PFparser = PMsubparsers.add_parser("POLYFIT", help="""
Calling this routine will display a GUI that allows for a visual inspection
of the fit-results (in the <prefix>.prm.fits-file). The GUI can also be used
to fit polynomials to the parameters of the PSF and/or source transformation
with. The results of the fitting process are stored in the <prefix>.prm.fits-
file. If the modified file is fed to the CUBEFIT-routine, PampelMuse will
use the polynomial fits. 
""")
    PFparser.add_argument("reference", type=str, nargs="?", help="""
An optional FITS image to be displayed. The idea here is to load the data
from which the reference coordinates were extracted. In that case, the image
is displayed in the GUI and the locations of the extracted sources are
overplotted.""")
    PFparser.set_defaults(func=polyfit)


EXparser = PMsubparsers.add_parser("EXTRACT", help="""
Routine calculates the pixel positions of the sources that have been extracted
from a datacube. The list of extracted sources is stored as an ASCII-file
under >prefix>.src.dat.""")
EXparser.add_argument("--addcoords", type=str, help="""
A simple ASCII-file that contains the ID and new x and y positions for at
least 4 from the source catalog. If such a file is provided, the routine
will determine the coordinate transformation into the new coordiante system
and also store the list of sources with the additional coordinates in a
separate ASCII-file named <prefix>.add.dat.""")
EXparser.set_defaults(func=get_pixpos)


XFparser = PMsubparsers.add_parser("UPDATEPRM", help="""
The routine can be used to change the structure of a prm-file created by a 
PampelMuse-version < 0.2.0.0 so that the file is compliant with any newer
PampelMuse-version""")
XFparser.set_defaults(func=updateprm)


SRparser = PMsubparsers.add_parser("SUBTRES", help="""
The routine is designed to subtract the contribution of the (resolved) stars
from a datacube. To this aim, it uses a PampelMuse PRM-file along with the
corresponding datacube.""")
SRparser.set_defaults(func=subtres)


SNparser = PMsubparsers.add_parser("SHOTNOI", help="""
Determine shot noise in each spatial pixel of a datacube befor and after the
resolved sources have been deblended. This routine will only work for
successfully analysed cubes.""")
SNparser.add_argument("sourcelist", type=str, nargs="?", help="""
Optional ASCII-file containing the results from crowded-field photometry. Lines
starting with '#' are considered comments. The first columns should contain
the names of the columns. The routine will look for columns named `id`, `x`,
`y` and <PASSBAND> as specified in the configuration.""")
SNparser.set_defaults(func=shotnoi)

if use_addwcs:
    AWparser = PMsubparsers.add_parser("ADDWCS", help="""
The routine can be used to add WCS information to the header of the IFS data
after sources have been identified. To this aim, besides the data a catalogue
must be present that contains the IDs of the sources and columns `RA` and
`Dec`.\n
Note that this routine requires `pyraf` and `astLib` to be available on the
system. Otherwise it won't work.""")
    AWparser.add_argument("sourcelist", type=str, help="""
ASCII-file containing the results from crowded-field photometry. Lines
starting with '#' are considered comments. The first columns should contain
the names of the columns. The routine will look for columns named `id`, `RA`
and `Dec`.""")
    AWparser.set_defaults(func=addwcs)

## A Namespace is created to store the configuration of PampelMuse. The parameters
## included in the namespace will be added as hierarch keywords to the header of
## the primary HDU of the PampelMuse PRM file. As an example, the Namespace parameter
## pampelmuse.initfit.version corresponds to HIERARCH PAMPELMUSE INITFIT VERSION

## parse arguments into preliminary namespace
args = PMparser.parse_args()

pampelmuse = PampelMuseConfig()
pampelmuse.version = __version__
## initialize empty namespace for subroutine that will be called.
setattr(pampelmuse, args.func.__name__, Namespace())

##  make sure PyQt4-dependencies are handled properly:
if not graphics:
    pampelmuse.useqt = False
    
## open configuration file, and read default configuration into  namespace
cfg_interface = xml_interact.ModifyConfig("{0}PMparams.xml".format(CONFIGPATH),
                                          "{0}PMschema.xsd".format(CONFIGPATH))
for key, value in cfg_interface.get_all().iteritems():
    setattr(pampelmuse, key.lower(), value)

## update configuration using command line input
if hasattr(args, "parameter"):
    for key, value in args.parameter.iteritems():
        if hasattr(pampelmuse, key.lower()): ## make sure only valid parameters were provided
            try: setattr(pampelmuse, key.lower(), eval(value))
            except NameError: setattr(pampelmuse, key.lower(), value)
        else:
            logging.error("Unknown configuration parameter provided: '{0}'".format(key))

## access initial PSF information
if pampelmuse.psftype == "moffat": parnames = ["beta", "fwhm", "e", "theta"]
elif pampelmuse.psftype == "gaussian": parnames = ["fwhm", "e", "theta"]
else: logging.critical("Unknown PSF type: '{0}'".format(pampelmuse.psftype))

freepar = np.zeros(len(parnames), dtype=np.bool)
for i in xrange(freepar.size, 0, -1):
    if pampelmuse.psffit >= 2**i:
        freepar[i-1] = True
        pampelmuse.psffit -= 2**i
del pampelmuse.psffit

pampelmuse.psf = Namespace()
pampelmuse.psf.profile = pampelmuse.psftype
del pampelmuse.psftype

for name in parnames:
    i = parnames.index(name)
    setattr(pampelmuse.psf, "parname{0}".format(i), name)
    setattr(pampelmuse.psf, "parvalue{0}".format(i), pampelmuse.psfpars[i])
    setattr(pampelmuse.psf, "freepar{0}".format(i), bool(freepar[i]))
del pampelmuse.psfpars

## access start/stop information
pampelmuse.start = pampelmuse.layerrng[0]
pampelmuse.stop = pampelmuse.layerrng[1]
del pampelmuse.layerrng

    
## add name of PRM file to configuration
pampelmuse.prefix = args.prefix
pampelmuse.filename = "{0}.prm.fits".format(args.prefix)

if not graphics or args.func!=initfit: ## unless INITFIT should be called, open existing PRM file
    
    logging.info("Reading PampelMuse-file: {0}".format(pampelmuse.filename))
    pampelmuse.hdulist  = pyfits.open(pampelmuse.filename, mode="update")

    ## scan header of primary HDU for HIERARCH PAMPELMUSE keywords
    ## Note that these keywords can be either directly connected to
    ## PampelMuse (e.g. HIERARCH PAMPELMUSE IFU) or to one of its
    ## subroutines (e.g. HIERARCH PAMPELMUSE INITFIT VERSION). In the
    ## latter case, the key will be added to the namespace of the
    ## corresponding subroutine.
    pampelmuse.readHeader(pampelmuse.hdulist[0])
    

else: ## add reference catalogue to configuration
    pampelmuse.initfit = Namespace()
    pampelmuse.initfit.sourcelist = args.sourcelist

if graphics and args.func==polyfit:
    if not hasattr(pampelmuse, "polyfit"): pampelmuse.polyfit = Namespace()
    pampelmuse.polyfit.reference = args.reference


## set up log level:
LOGLEVELS   = {0:logging.CRITICAL,
               1:logging.ERROR,
               2:logging.WARNING,
               3:logging.INFO,
               4:logging.DEBUG}
logging.getLogger().setLevel(LOGLEVELS[pampelmuse.verbose])

# Used as a test: an additional logger that gives the file and line where a warning message was triggered
#if pampelmuse.verbose==4:
#    ch = logging.StreamHandler()
#    ch.setLevel(logging.WARNING)
#    ch.setFormatter(logging.Formatter("%(pathname)s [%(lineno)-4s]: %(message)s"))
#    logging.getLogger().addHandler(ch)


## load the IFS data
## The science, sigma and mask data can either be provided in a single file
## (extensions 'DATA', 'ERROR', and 'MASK') or as three separate FITS-files,
## names <root>.dat.fits, <root>.err.fits, and <root>.msk.fits
if os.path.exists("{0}.dat.fits".format(args.prefix)):
    data_ifs = "{0}.dat.fits".format(args.prefix)
elif os.path.exists("{0}.fits".format(args.prefix)):
    data_ifs = "{0}.fits".format(args.prefix)
else:
    logging.critical("Cannot find IFS data: {0}.dat.fits or {0}.fits".format(args.prefix))

if os.path.exists("{0}.err.fits".format(args.prefix)):
    sigma_ifs = "{0}.err.fits".format(args.prefix)
else:
    sigma_ifs = None ## sigma data must be included as an HDU in science data file

if os.path.exists("{0}.msk.fits".format(args.prefix)):
    mask_ifs = "{0}.msk.fits".format(args.prefix)
else:
    mask_ifs = None ## mask data can be included as an HDU in science data file

pampelmuse.ifs_data = Cube(data_ifs, instrument=pampelmuse.ifu, sigma=sigma_ifs, mask=mask_ifs)

## set range of layers to be analyzed properly (in case negative values were provided)
if pampelmuse.start < 0: pampelmuse.start = pampelmuse.ifs_data.ndisp + pampelmuse.start
if pampelmuse.stop < 0: pampelmuse.stop = pampelmuse.ifs_data.ndisp + pampelmuse.stop 

args.func(pampelmuse) ## execute the connected function

