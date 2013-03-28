#! /usr/bin/env python
"""
PAMPELMUSE
==========
The front end. Not much in here except for the basic command-line tool
and some tools to change the configuration of PampelMuse.

CHANGELOG
---------
v0.0.0.1 -2011/09/06 - added logo
v0.1.0.0 -2011/10/13 - modified how the PampelMuse-scripts are imported
                       to reflect new package-structure.
                     - first experimental "release"
v0.1.0.1 -2011/11/29 - included logging-option
v0.1.0.2 -2011/12/08 - added new routine `EXTRACT' to write the pixel
                       positions of the extracted stars to an ASCII-file
v0.1.0.3 -2012/01/13 - some minor updates to usage of logging module
v0.1.0.4 -2012/01/17 - added posfit module. Still needs some debugging and
                       documentation
v0.2.0.0 -2012/01/29 - replaced posfit-module with updateprm-module. Since 
                       the previously applied polynomial fitting of the
                       parameters of the coordinate transformation led to
                       wrong results, the polyfit-module now works directly
                       on the source coordinates instead. To implement this,
                       the structure of the prm-file has been modified. Old
                       prm-files can be updated using the updateprm-module.
v0.2.0.1 -2012/02/12 - slightly modified formatting of the logging-messages.
v0.2.1.1 -2012/02/12 - fixed bug in format-string of logging-messages.
v0.3.0.0 -2012/03/01 - added new script SUBTRES to subtract the contribution
                       of the (resolved) stars from a datacube.
v0.4.0.0 -2012/03/19 - modified check of PRM-file structure so that files
                       created using INITFIT version >=0.4.0.0 pass it.
v0.5.0.0 -2012/06/26 - added new script ADDWCS that can be used to add WCS
                       information to the header of the IFS datacube.
                       Also updated the help-text.
v0.5.0.1 -2012/07/03 - slighly modified logging-initialization because calling
                       `logging.basicConfig` has no effect after a log-message
                       has been triggered.
v0.5.0.2 -2012/07/24 - added handling for cases where PyQt4 is not available.
v0.5.1.2 -2012/07/24 - some bugfixing wrt availability checks. Now we also
                       chek for matplotlib support.
v0.6.0.0 -2012/12/23 - added routine SHOTNOI that estimates the shot noise in a
                       datacube.
v0.7.0.0 -2013/03/27 - implemented a new way to construct the PSF profiles. In
                       combination with a Levenberg-Marquard algorithm that is
                       now used to perform the non-linear fits, this
                       significantly speeds up the code (by a factor ~10(!)).
                       Also added support for 2dim. input data besides cubes.
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.7.0.0"


print """
            ,;:.    PampelMuse, version {0:17}    ,;:.    
           (::;;)   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (::;;)
            `;:'                                             `;:'

        (c) {1}
""".format(__version__, __author__)


## IMPORT REQUIRED PYTHON MODULES ##############################################
import argparse, logging, os, re, sys
import pyfits

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
    logging.error("Cannot import module `pyraf`.")
    logging.warning("Script ADDWCS cannot be used.")
    use_addwcs = False

## check whether dependencies for graphical backends are fufilled
graphics = True
try:
    from PyQt4 import QtCore, QtGui
except ImportError:
    logging.error("Cannot load `PyQt4`-moodules")
    graphics = False
try:
    import matplotlib.pyplot
except ImportError:
    logging.error("Cannot load `matplotlib`-moodules")
    graphics = False
if not graphics:
    logging.warning("Scripts INITFIT and POLYFIT cannot be used.")



## required PampelMuse-packages ################################################
from pampelmuse.config import xml_interact

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


args = PMparser.parse_args()

##  make sure PyQt4-dependencies are handled properly:
if not graphics:
    args.parameter["USEQT"] = "0"

## open configuration file, update if necessary
cfg_interface = xml_interact.ModifyConfig("{0}PMparams.xml".format(CONFIGPATH),
                                          "{0}PMschema.xsd".format(CONFIGPATH))


##  get dtypes of provided parameters (if any) and update their values 
if "parameter" in args:
    names = args.parameter.keys()
    for name in names:
        old_value = cfg_interface.get_parameter(name, return_comment=False)

        dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)",
                           str(type(old_value)))[0][2]

        ## use transform_string-routine provided by xml_interfaces to 
        ## get dtype of parameters right
        success = cfg_interface.update_parameter(\
            name, xml_interact.transform_string(args.parameter[name], dtype))

args.config = cfg_interface.get_all()  ## read configuration


## set up log level:
LOGLEVELS   = {0:logging.CRITICAL,
               1:logging.ERROR,
               2:logging.WARNING,
               3:logging.INFO,
               4:logging.DEBUG}
logging.getLogger().setLevel(LOGLEVELS[args.config["VERBOSE"]])
    

## check whether provided prm-file can be used with current PampelMuse-version.
if args.func.func_name in ["cubefit", "polyfit"]:
    prmFile = pyfits.open("{0}.prm.fits".format(args.prefix))
    try:
        VersionString = prmFile[0].header["PAMPELMUSE INITFIT VERSION"]
        prmStructure = 3
##         if re.match("v([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)",VersionString):
##             groups = re.match("v([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)",
##                               VersionString)
    except KeyError:
        VersionString = "0.0"
        try:
            prmStructure = prmFile["SOURCES"].header["PRMSTRUC"]
        except KeyError:
            prmStructure  = 1

    if prmStructure < 2 and VersionString < "0.4":
        logging.critical("Provided PRM-file not compliant with current")
        logging.critical("PampelMuse-version. Run")
        logging.critical("      'PampelMuse {0} UPDATEPRM'".format(args.prefix))
        logging.critical("to modify PRM-file first.")

        sys.exit()

    prmFile.close()
args.func(args)  ## execute the connected function
