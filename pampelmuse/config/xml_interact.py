#!/usr/bin/env python
"""
XML_INTERACT
============

PURPOSE:
--------
The routines defined in the `ModifyConfig`-class provide several methods to
update and to read the PampelMuse-configuration that is stored in the 
XML-file '$pampelmuse_path/config/PMparams.xml'. For detailed instructions
on the usage of this class check its documentation.
Furthermore, some auxiliary routines are provided that allow to acces the
PampelMuse-configuration from the command line. 

USAGE:
------
+++Command-line usage:
Type 'PampelMuse CONFIG --help' for instructions on syntax and possible
parameters if you want to change the configuration from the command-line.
(not fully implemented yet)

+++direct usage with class  `ModifyConfig`:
When a new instance of the class is initialized, the user has to provide
the path to the configuration-file and to its XML-Schema (a .xsd-file).
After initialization, the methods 'add_parameter', 'update_parameter' and
'get_parameter' allow interaction with the configuration file. Finally,
the method 'save_changes' can be used to write to possible changes disk.

EXAMPLE FOR USAGE:
------------------
>>> modifier = ModifyConfig('PMparams.xml', 'PMschema.xsd')
>>> modifier.add_parameter(name='PSFTYPE', value='moffat', category='psf',
                           comment='A new PSF')
>>> modifier.update_parameter(name='PSFTYPE', value='gaussian')
>>> print modifier.get_parameter(name='PSFTYPE')
['gaussian', 'A new PSF']
>>> modifier.save_changes()


CHANGELOG:
----------
2011/08/04, v0.0.0.0
2011/08/05, v0.1.0.0, added documentation, modified command-line usage
2011/08/08, v0.2.0.0, added parameter `abort_on_exception` for proper
                      exception handling.
                      added `get_all`-method to read the current configuration
2011/08/09, v0.3.0.0, add options `ahead` and `after` to include new
                      parameters in a specific place of the configuration.
2011/08/10, v0.3.0.1, modified code to allow command-line usage and importing
                      of the defined classes/functions.
2011/08/31, v0.3.1.1, fixed bug in modes 'Update' and 'Read' 
                      (replaced args.parameter with args.name)
2011/09/01, v0.3.2.1, fixed bug in mode 'Update': Code didn't find 'dtype'
                      -parameter, automatically set dtype to string, even if
                      'None' given
2011/09/09, v0.3.3.1, fixed bug in function `transform_string': The routine
                      tried to convert a list/tuple of floats to integers.
2012/07/24, v0.4.0.0, significantly modified the code. It now used `xml`
                      instead of lxml because it is a standard python library.
                      Also included logging module.
2012/07/26, v0.4.0.1, modfied findall command in l384 because including an
                      attribute in the search won't work for earlier
                      xml-versions.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.4.0.0"
import datetime, logging, os, re, string, sys
from xml.etree import ElementTree
from xml.dom.minidom import parseString


def transform_string(thestring, dtype, abort_on_exception=True):
    """
    Short routine to convert a string to one of the following data types:
        float, int, bool, list, tuple

    Parameters
    ----------
    thestring : string
        The string that should be converted.
    dtype : string
        The data type to which the string should be converted
    abort_on_exception : boolean
        Flag to indicate whether an error statement should be raised if
        an exception is encountered (i.e if provided `dtype` is not valid)

    Returns
    -------
        The result of the transformation.
    """
    if dtype == 'str':
        return str(thestring)
    elif dtype == 'float':
        return float(thestring)
    elif dtype == 'int':
        return int(thestring)
    elif dtype == 'bool':
        ## using regular expressions for the transformation to bool
        return bool(int(re.sub("true", "1",
                               re.sub("false", "0", string.lower(thestring))
                               )
                        )
                    )
    elif dtype in ['list', 'tuple']:
        thenumbers = thestring[1:-1].split(',')
        if sum([len(re.findall('\.', s)) for s in thenumbers])==0:
            thelist = [int(x) for x in thenumbers]
        else:
            thelist = [float(x) for x in thenumbers]

        if dtype == 'tuple':
            return tuple(thelist)
        else:
            return thelist
        
    else:
        logging.warning("Unknown datatype encountered:{0}".format(dtype))
        if raise_on_exception:
            raise IOError, "Unknown datatype encountered:{0}".format(dtype)
        else:
            return None



class ModifyConfig:
    """
    Class to interact with the XML-file containing the PampelMuse-
    configuration. The class provides methods to add parameters,
    update existing ones or read their values.
    Note that the configuration-file has to obey the XML-scheme that is
    defined in the PMschema.xsd-file. The routine will check whether this
    is true when the file is opened or modified.
    """
    def __init__(self, filename, schema, abort_on_exception=True):
        """
        Initialize a new instance of the `ModifyConfig`-class

        Parameters
        ----------
        filename : string
            The path to the PampelMuse-configuration-file
        schema : string
            The path to the schema against which the XML-file is validated
        abort_on_exception : boolean
            Flag to indicate whether an error statement should be raised if
            an exception is encountered (i.e if parameter to be updated does
            not exist or definition  of a parameter is not unique)
        """

        ## open XML-schema
##         schema_doc = etree.parse(schema)
##         self.xmlschema = etree.XMLSchema(schema_doc)

        ## specifyXML-parser, include schema
##        self.parser = etree.XMLParser(schema=self.xmlschema)
        self.parser = ElementTree.XMLParser()

        ##  open file, validate against schema
        self.filename = filename
        self.ptree    = ElementTree.parse(filename, self.parser)

        ## get the namespace(s) for XPATH-searches:
        ## WARNING: Hard-coded so far
        self.ns = "{http://www.tobedefined.com/pampelmuse}"

        ## list of all existing parameter-names:
        names = self.ptree.findall("./{0}parameter/{0}name".format(self.ns))
        self.pnames = [name.text for name in names]

        ##  What time is it?
        self.ut        = datetime.datetime.utcnow()
        self.timestamp = self.ut.strftime("%Y-%m-%dT%H:%M:%SZ")

        self.AoE = abort_on_exception
        
        

    def add_parameter(self, name, value, comment=None, category=None,
                      inuse=None, ahead=None, after=None):
        """
        Add a new parameter to the configuration. Note that this method
        should not be used to update existing parameters. For this
        purpose, the `update_parameter`-method should be used.

        Parameters
        ----------
        name : string
            The name of the parameter. It should only contain alpha-
            numerical characters. Note that all characters are auto-
            matically set to uppercase.
        value : a standard python data-type
            The default value of the parameter. Can be either a string,
            a float, an integer, a boolean flag, a list or a tuple.
        comment : string, optional
            A comment to explain the purpose of the parameter.
        category : string,  optional
            A keyword to specify which part of the configuration is
            affected by the new parameter. Currently, it is only used
            for visualization purposes.
        inuse : boolean, optional
            A flag to indicate whether the parameter is used by the
            configuration. By default, `inuse` is set to True. In any case,
            the updated configuration will be validated against the XML-
            schema.
        ahead : string, optional
            If specified, new parameter will be placed ahead of the given
            one in the configuration.
        after : string, optional
            If specified, new parameter will be placed after the given
            one in the configuration.


        Returns
        -------
        valid : an etree.XMLSchema.assertValid return-value
            The new parameter is validated against the schema. If succesful,
            true is returned; otherwise the routine will return an error.
        """

        ## check if parameter conforms to required format
        ## only aplphanumeric characters are allowed 
        new_name = string.upper(name.strip())
        if ''.join(re.findall('[0-9A-Z]', new_name)) != new_name:
            logging.error("Cannot add parameter '{0}' - invalid name!".format(\
                    new_name))
            if self.AoE:
                raise IOError, "Cannot add parameter '{0}' ".format(new_name)\
                    + "since it contains invalid characters!"
            return 0

        ##  check if parameter is already present
        if new_name in self.pnames:

            logging.error("Parameter '{0}' does already exist!".format(\
                    new_name))
            logging.error(" Use `update_parameter`-method!")
            if self.AoE:
                raise IOError, "Parameter '{0}' does already exist!".format(\
                    new_name)
            return 0

        ## get type of parameter using reglular expressions
        dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)",
                           str(type(value)))[0][2]
                
        ##  create new parameter as etree.SubElement of the root
        attrib = {}
        if category != None: attrib["category"] = category
        if inuse != None: attrib["inuse"] = inuse
                
        new_param = ElementTree.SubElement(\
            self.ptree.getroot(), "{0}parameter".format(self.ns), attrib=attrib)

        thename = ElementTree.SubElement(\
            new_param, "{0}name".format(self.ns))
        thename.text = new_name

        thevalue = ElementTree.SubElement(\
            new_param, "{0}value".format(self.ns), attrib={"dtype":str(dtype)})
        thevalue.text = str(value)


        if comment != None:
            thecomment = ElementTree.SubElement(\
                new_param, "{0}comment".format(self.ns))
            thecomment.text = comment

        ## store information about the update in the configuration
        theupdate = ElementTree.SubElement(\
            new_param, "{0}updated".format(self.ns),
            attrib={"timestamp":self.timestamp})

        theupdate.text = "Appended parameter {0} to configuration file."\
            .format(new_name)

        ## check if any information about position of new parameter given
        if ahead!=None and after!=None:
            logging.warning("Cannot use `ahead`- AND `after`-parameter.")

        index=None
        if ahead!=None:
            if not ahead in self.pnames:
                logging.error("Parameter '{0}' does not exist.".format(ahead))
            else:
                index=self.pnames.index(ahead)
        elif after!=None:
            if not after in self.pnames:
                logging.error("Parameter '{0}' does not exist.".format(after))
            else:
                index=self.pnames.index(after)+1

        if index != None:
            self.ptree.getroot().remove(new_param)
            self.ptree.getroot().insert(index, new_param)
            self.pnames.insert(index, new_name)
        else:
            self.pname.append(new_name)

        logging.info("Added parameter '{0} to configuration'".format(new_name))
        
##         return self.xmlschema.assertValid(self.ptree)
        return 1


    def update_parameter(self, name, value=None, comment=None, comment_r=None,
                         category=None, inuse=None, force_update=False):
        """
        Update an existing parameter in the configuration. Note that this
        method should not be used to add new parameters. For this
        purpose, the `add_parameter`-method should be used.
        To update a certain piece of information of an existing parameter,
        set the corresponding method-parameter to a value that is not None.

        Parameters
        ----------
        name : string
            The name of the parameter. Note that all characters are auto-
            matically set to uppercase.
        value : a standard python data-type, optional
            The default value of the parameter. Can be either a string,
            a float, an integer, a boolean flag, a list or a tuple.
        comment : string, optional
            A comment to explain the purpose of the parameter. Note that
            a comment provided here is appended to the existing comment(s)
        comment_r : string, optional
            Same as `comment`, only that exisiting comments (if any) are
            replaced by the new comment.
        category : string,  optional
            A keyword to specify which part of the configuration is
            affected by the new parameter. Currently, it is only used
            for visualization purposes.
        inuse : boolean, optional
            A flag to indicate whether the parameter is used by the
            configuration. By default, `inuse` is set to True.
        force_update : boolean, optional
            If the new parameter `value` has a different type than the old
            one, the default behaviour of the routine is to keep the old
            `value`. To force an update of `value` and `dtype`, set
            `force_update` to  True.

        Returns
        -------
        valid : integer or etree.XMLSchema.assertValid return-value
            In case the parameter is not found or is found multiple times,
            `valid` gives the number of findings. In case on non-matching
            `dtype`, the method returns None. Note that in both cases no
            updates are written to the configuration.
        """
        thename = string.upper(name)
        updates = [] ## a container for all strings describing the
                     ## individual updates

        ## search parameter
        if not thename in self.pnames:
            logging.error("Parameter '{0}' does not exist!".format(thename))
            logging.error(" Use `add_parameter`-method to include it!")

            if self.AoE:
                raise IOError, "Parameter '{0}' does not exist!".format(thename)

            return 0

        ## Found multiple occurences -> abort
        elif self.pnames.count(thename) > 1:
            logging.crtitcal("Meaning of parameter '{0}' ambiguous!".format(\
                thename))
            logging.critical("Please verify XML-file!")
            if self.AoE:
                raise IOError, "Meaning of parameter {0} ambiguous!".format(\
                    thename)
            return len(param_list)


        ##  Found exactly 1 occurence -> go on
        index = self.pnames.index(thename)

        theparam = self.ptree.findall("./{0}parameter".format(self.ns))[index]

        if value != None:  ## check if value should be updated
            dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)",
                               str(type(value)))[0][2]

##             thevalue = theparam.findall("./{0}value[@dtype]".format(self.ns))
##             including attributes only works in latest versions of xml
            thevalue = theparam.findall("./{0}value".format(self.ns))
            assert len(thevalue)==1

            vindex = list(theparam).index(thevalue[0])
            theparam.remove(thevalue[0])

            old_value = thevalue[0].text
            old_dtype = thevalue[0].attrib["dtype"]

            if dtype != old_dtype: ## new data-type does not match old one
                logging.warning("Unmatching `dtype` encountered: {0} vs {1}"\
                                    .format(dtype, old_dtype))
                if not force_update:
                    if self.AoE:
                        raise IOError, "'{0}' requires '{1}', not '{2}'!"\
                            .format(thename, old_dtype, dtype)

                    return 0
                else:
                    logging.warning("Changed dtype of parameter {0}:".format(\
                            thename))
                    logging.warning("   from {0} to {1}".format(\
                            old_dtype, dtype))
   
            thevalue[0].text = str(value)
            theparam.insert(vindex, thevalue[0])

            updates.append(\
                "Changed default value of parameter {0} to {1} (was {2})"\
                                   .format(thename, value, old_value))

        if comment_r != None: ## check if comment should be replaced
            
            ##  delete old comment(s) (if any)
            for comment in theparam.findall("./{0}comment".format(self.ns)):
                theparam.remove(comment)

            comment = comment_r
        
        if comment != None: ##  add new comment
            newcomment = ElementTree.SubElement(\
                theparam, "{0}comment".format(self.ns))
            newcomment.text = comment

            ## schema requires that 'updated'-Elements are placed after
            ## 'comment'-Elements. Following codes assures this.

            oldupdates = theparam.findall("./{0}updated".format(self.ns))
            if len(oldupdates)>0:
                cindex = list(theparam).index(oldupdates[0])
                theparam.remove(newcomment)
                theparam.insert(cindex, newcomment)
            
            
        if category != None:
            if theparam.attrib.has_key("category"):
                old_category = theparam.attrib["category"] 
            else:
                old_category = "None"
            updates.append("Changed category to {0} (was {1})".format(\
                    category, old_category))

            theparam.attrib["category"] = category



        if inuse != None: ## check if usage-flag should be updated
            if theparam.attrib.has_key("inuse"):
                old_inuse = transform_string(\
                    theparam.attrib["inuse"], "bool",
                    abort_on_exception=self.AoE)
            else:
                old_inuse = True

            if old_inuse and not inuse:
                updates.append("Disabled usage of parameter.".format(thename))
            elif inuse and not old_inuse:
                updates.append("Enabled usage of parameter.".format(thename))
            theparam.set("inuse", str(int(inuse)))


            for update in updates: ## write update information to file

                theupdate = ElementTree.SubElement(\
                    theparam, "{0}updated".format(self.ns),
                    attrib={"timestamp":self.timestamp})
                theupdate.text = update

        self.ptree.getroot()[index] = theparam

##         return self.xmlschema.assertValid(self.ptree)
        return 1



    def get_parameter(self, name, return_comment=True):
        """
        Get the default value of a parameter along with the comment
        connected to the parameter (if requested).

        Parameters
        ----------
        name : string
            The name of the parameter. Note that all characters are auto-
            matically set to uppercase.
        return_comment : boolean
            Flag indicating if the return-value includes the comment.

        Returns
        -------
        query_result : None or a standard python data-type
            If the parameter is not found, none is returned. Otherwise,
            the value (if return_comment is False) or a list containing
            value and comment (return_comment is True) is returned.
        """
        thename = string.upper(name)

        ##  search configuration for matching parameter

        if not thename in self.pnames: ## None found -> abort
            logging.error("No such parameter: '{0}'".format(thename))
            if self.AoE:
                raise IOError, " No such parameter: {0}".format(thename)
            return None
        elif self.pnames.count(thename) > 1: ## Multiple found -> abort
            logging.crtitcal("Multiple occurences of parameter '{0}' found!"\
                                 .format(thename))
            logging.critical("Please verify XML-file!")
            if self.AoE:
                raise IOError, "Multiple occurences of parameter '{0}' found!"\
                    .format(thename)
            return len(param_list)


        ## Found one! -> go one & return its value (& comment)
        index = self.pnames.index(thename)

        theparam = self.ptree.findall("./{0}parameter".format(self.ns))[index]

        thevalue = theparam.findall("./{0}value".format(self.ns))[0]
        
        thetype  = thevalue.attrib["dtype"]

        thecomm  = theparam.findall("./{0}comment".format(self.ns))
        comment  = ""
        for comm in list(thecomm):
            comment += "-{0}".format(comm.text)

        if return_comment:
            return [transform_string(thevalue.text, thetype,
                                     abort_on_exception=self.AoE), 
                    comment]
        else:
            return transform_string(thevalue.text, thetype,
                                        abort_on_exception=self.AoE)

            

    def get_all(self):
        """
        Read the current configuration and write it to a dictionary.

        Returns
        -------
        config : dictionary
            A dictionary containing 'name:value'-pairs for all parameters
            specified in the XML-configuration file.
        """
        config = {}

        param_list = self.ptree.findall("./{0}parameter".format(self.ns))

        for parameter in param_list:

            thename  = parameter.findall("./{0}name".format(self.ns))[0]

            thevalue = parameter.findall("./{0}value".format(self.ns))[0]

            thetype  = thevalue.attrib["dtype"]

            config[thename.text] = transform_string(thevalue.text, thetype,
                                                    abort_on_exception=self.AoE)

        return config


    def save_changes(self, filename=None, style=None):

        if filename == None:
            filename = self.filename

        self.ptree.getroot().set("timestamp", self.timestamp)

        txt = ElementTree.tostringlist(self.ptree.getroot(),
                                       encoding='ISO-8859-1')
        if style != None:
            txt.insert(1, '<?xml-stylesheet type="text/xsl" href="{0}"?>'\
                           .format(style))

        outfile = open(filename, "w")
        outfile.write(parseString("".join(txt)).toprettyxml(indent="",newl=""))
        outfile.close()

##         self.ptree.write(filename, pretty_print=pretty_print,
##                          xml_declaration=xml_declaration,
##                          encoding=self.ptree.docinfo.encoding)




### BEGINNING OF COMMAND LINE PART
if __name__ == '__main__':
    import argparse

    CONFIGPATH = os.path.expandvars("$pampelmuse_path/config/")

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





    parser = argparse.ArgumentParser(prog="PampelMuse CONFIG",
                                     description="""
    Provides methods to change the PampelMuse configuration. The complete
    configuration of PampelMuse is stored in a single XML-file. The methods
    can be used to update the XML-file or to read parameters from it.
    """)

    parser.add_argument("method", type=str, choices=['U','A','R'],
                        help="""Specifies whether a parameter should be [A]dded,
                                [U]pdated or [R]ead.""")
    parser.add_argument("name", type=str,
                        help="""The name of the parameter. Must always be
                                provided, regardless of which method is used.
                                """)
    parser.add_argument("value", type=str, nargs='?', default=None,
                        help="""The (new) value of the parameter. Required if a
                                new parameter is added, optional if an existing
                                parameter is updated.""")
    parser.add_argument("--type", type=str, default='str',
                        help="""The type of the parameter. Should always be
                                provided together with a value. Otherwise the
                                value is assumed to be a string.""")
    parser.add_argument("--comment",  type=str, nargs='*', action=MergeComment,
                        help="""An ADDITIONAL comment to be included in the
                                configuration file. Note that the comment does
                                NOT replace existing comments (if any).""")
    parser.add_argument("--replace_comment", action='store_true',
                        help="""Replace existing comment(s) (if any) instead
                                of appending the new one.""")
    parser.add_argument("--exclude", action='store_false', default=None,
                        help="""Flag to disable usage of an existing parameter
                                """)
    parser.add_argument("--include", action='store_true', default=None,
                        help="""Flag to enable  usage of an existing  parameter
                                """)
    parser.add_argument("--category", type=str,
                        help="""Keyword to indicate which part of the
                                configuration is affected by a parameter""")
    parser.add_argument("--ahead", type=str, action=MakeUpper,
                        help="""If specified, a new parameter will be placed
                                ahead of this one.""")
    parser.add_argument("--after", type=str, action=MakeUpper,
                        help="""If specified, a new parameter will be placed
                                after this one.""")

    args = parser.parse_args()

    modifier = ModifyConfig("{0}PMparams.xml".format(CONFIGPATH),
                            "{0}PMschema.xsd".format(CONFIGPATH))

    inuse = max([args.include, args.exclude])


    if args.method == "A":
        success = modifier.add_parameter(name=args.name,
                                         value=transform_string(args.value,
                                                                args.type),
                                         comment=args.comment,
                                         category=args.category,
                                         inuse=inuse,
                                         ahead=args.ahead,
                                         after=args.after
                                         )
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

        success = modifier.update_parameter(name=args.name,
                                            value=value,
                                            comment=args.comment,
                                            comment_r=comment_r,
                                            category=args.category,
                                            inuse=inuse,
                                            )

    elif args.method == 'R':
        query_result = modifier.get_parameter(args.name)
        print "{0:10} {1:15}  {2}".format(string.upper(args.name), 
                                          query_result[0],
                                          query_result[1])

    if args.method in ["A", "U"]:
        os.system('cp {0}PMparams.xml {0}PMparams.xml.sav~'.format(CONFIGPATH))
        modifier.save_changes(style="PMstyle.xsl")


