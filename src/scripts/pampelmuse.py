#! /usr/bin/python
"""
PAMPELMUSE
==========
The front end. Not much in here except for the basic command-line tool
and some tools to change the configuration of PampelMuse.

CHANGELOG
---------
v0.0.0.1 -2011/09/06 - added logo
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.0.1"
import argparse, os, re
import xml_interact
from initfit import initfit
from cubefit import cubefit


CONFIGPATH = os.path.expandvars("$pampelmuse_path/config/")


class GetValuesFromLine(argparse.Action):
    """
    Short routine to write 'name=value'-pairs appended to the '--fromline'
    option into a dictionary.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        outdict = {}
        for _val in values[0].split(','):
            val = re.sub(r'\s', '', _val) ## remove whitespace chars.
            if len(val.split("=")) != 2:
                raise IOError, "Bad parameter definition in line: {0}"\
                        .format(val)
            name, value = val.split("=")
            outdict[name] = value
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
                    outdict[name] = value
        setattr(namespace, self.dest, outdict)



PMparser = argparse.ArgumentParser(prog="PampelMuse",
                                   description="""
PampelMuse is a routine to extract stellar spectra from IFU data by usage of
PSF fitting techniques. It is designed for the analysis of IFU observations
of crowded-stellar fields. As the name suggests, it was originally intended
to serve  as an analysis tool for the MUSE-IFU. Yet, PampelMuse is designed
in a generic way and should be able to analyze data coming from similar IFUs
as well.
""")

## Note: 'fromfile' and 'fromline' share the same dest 'parameter'. However,
##       since they are in a 'mutually exclusive group' and their 'default'
##       is set to argparse.SUPPRESS, at most one of them will contibute to
##       the namespace.
group = PMparser.add_mutually_exclusive_group(required=False)
group.add_argument("--fromfile", type=argparse.FileType('r'),
                   action=GetValuesFromFile, dest="parameter",
                   default=argparse.SUPPRESS,
                   help="""Update configuration using the parameters provided
                           in an ASCII-file: One parameter per line, name and
                           value separated by '=', lines starting with '#'
                           are ignored""")
group.add_argument("--fromline", type=str, nargs=1,
                   action=GetValuesFromLine, dest="parameter",
                   default=argparse.SUPPRESS,
                   help="""Update configuration using the parameters provided
                           in the command line, following '--fromline'-string.
                           Again, name and value must be separated by '=' (no
                           whitespace!). Individual parameters  must be comma-
                           separated.""")

PMsubparsers = PMparser.add_subparsers(title='Available routines')


CFparser = PMsubparsers.add_parser("CUBEFIT",
                                   help="""
The routine is used for the actual PSF-fitting process. It must be provided
with apropriate input data (created e.g using INITFIT). By default, all the
parameter that have an impact on the fitting process are read from the
PampelMuse configuration-file. However, several ways are provided to change
the configuration on the fly. Check the option-list for details.
""")

CFparser.add_argument("prefix", type=str,
                    help="""The prefix common to all FITS-files required by the
                            fit-routine. The program will search for files
                            named <prefix>.dat.fits, <prefix>.err.fits and
                            <prefix>.prm.fits. Optionally, a bad-pixel mask
                            can be used (<prefix>.msk.fits).""")
CFparser.set_defaults(func=cubefit)


IFparser = PMsubparsers.add_parser("INITFIT",
                                   help="""
The routine is used to prepare the datacube for the PSF fitting process.
It takes as input the datacube and an ASCII-table containing the results
from the crowded-field photometry. The different processing steps are
a) Interactive identification of bright sources in the field.
b) Aperture spectrophotometry of the previously identified sources.
c) Generation of an input catalogue containing intitial guessse of the
   spectra and the positions of the stars.
""")
IFparser.add_argument("prefix", type=str,
                      help="""
                            The prefix common to all FITS-files required by the
                            routine. The program will search for files
                            named <prefix>.dat.fits, <prefix>.err.fits and
                            optionally a bad-pixel mask (<prefix>.msk.fits).
                            The output will be stored under <prefix>.prm.fits.
                            """)
IFparser.add_argument("sourcelist", type=argparse.FileType('r'),
                      help="""
                            ASCII-file containing the results from crowded-
                            field  photometry. Lines starting with '#' are
                            considered comments. Unless otherwise specified,
                            the routine assumes the following structure:
                            <ID>  <XPOS>  <YPOS>  <MAG> ... (per line)""")
IFparser.set_defaults(func=initfit)

args = PMparser.parse_args()


print """
            ,;:.    PampelMuse, version {0:17}    ,;:.    
           (::;;)   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (::;;)
            `;:'    (c) {1:32}    `;:'
""".format(__version__, __author__)


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
args.func(args)  ## execute the connected function
