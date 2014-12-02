#! /usr/bin/env python
"""
access_configuration.py
=======================

Purpose
-------
A script to access the default configuration of PampelMuse that is stored
in an XML-file. It is possible to add parameters, delete them, update their
properties, or just print their values out on the screen.
Enter 'access_configuration.py --help' for instructions on syntax and possible
parameters.


Latest SVN revision
-------------------
221, 2014/12/01
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 221

import argparse, os, shutil, string

## required PampelMuse modules
from pampelmuse.config import ModifyConfig, transform_string


## get path to XML data containing configuration
__currentdir__ = os.path.dirname(__file__)
__configpath__ = os.path.join(__currentdir__, os.pardir, "config", "data")


class MergeComment(argparse.Action):
    """
    Short routine to merge the individual strings to a single comment
    string. Used as 'action' by the '--comment' option.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, " ".join(values))


class MakeUpper(argparse.Action):
    """
    Short routine to make a provided string uppercase.
    Used as 'action' by the '--ahead' and '--after' options.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, string.upper(values))



parser = argparse.ArgumentParser(description="""
Provides methods to change the PampelMuse configuration. The complete
configuration of PampelMuse is stored in a single XML-file. The methods
can be used to update the XML-file or to read parameters from it.""")

parser.add_argument("method", type=str, choices=['U','A','R'], help="""
Specifies whether a parameter should be [A]dded, [U]pdated or [R]ead.""")
parser.add_argument("name", type=str, help="""
The name of the parameter. Must always be provided, regardless of which
method is used.""")
parser.add_argument("value", type=str, nargs='?', default=None, help="""
The (new) value of the parameter. Required if a new parameter is added,
optional if an existing parameter is updated.""")
parser.add_argument("--type", type=str, default='str', help="""
The type of the parameter. Should always be provided together with a
value. Otherwise the value is assumed to be a string.""")
parser.add_argument("--comment",  type=str, nargs='*', action=MergeComment, help="""
An ADDITIONAL comment to be included in the configuration file. Note
that the comment does NOT replace existing comments (if any).""")
parser.add_argument("--replace_comment", action='store_true', help="""
Replace existing comment(s) (if any) instead of appending the new one.""")
parser.add_argument("--exclude", action='store_false', default=None, help="""
Flag to disable usage of an existing parameter""")
parser.add_argument("--include", action='store_true', default=None, help="""
Flag to enable  usage of an existing  parameter.""")
parser.add_argument("--category", type=str, help="""
Keyword to indicate which part of the configuration is affected by a
parameter""")
parser.add_argument("--ahead", type=str, action=MakeUpper, help="""
If specified, a new parameter will be placed ahead of this one.""")
parser.add_argument("--after", type=str, action=MakeUpper, help="""
If specified, a new parameter will be placed after this one.""")

args = parser.parse_args()

modifier = ModifyConfig(os.path.join(__configpath__, "PMparams.xml"), os.path.join(__configpath__, "PMschema.xsd"))


inuse = max([args.include, args.exclude])


if args.method == "A":
    success = modifier.add_parameter(name=args.name, value=transform_string(args.value, args.type), comment=args.comment, category=args.category, inuse=inuse,
                                     ahead=args.ahead, after=args.after)
elif args.method == "U":

    if args.replace_comment:
        comment_r = args.comment
        args.comment=None
    else:
        comment_r=None

    if args.value != None:
        value=transform_string(args.value, args.type)
    else:
        value=None

    success = modifier.update_parameter(name=args.name, value=value, comment=args.comment, comment_r=comment_r, category=args.category, inuse=inuse)

elif args.method == 'R':
    query_result = modifier.get_parameter(args.name)
    print "{0:10} {1:15}  {2}".format(args.name.upper(), query_result[0], query_result[1])


if args.method in ["A", "U"]:
    shutil.copyfile(os.path.join(__configpath__, "PMparams.xml"),  os.path.join(__configpath__, "PMparams.xml.sav~"))
    modifier.save_changes(style="PMstyle.xsl")
