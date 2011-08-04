#!/usr/bin/python
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
>>> modifier.save:_changes()


CHANGELOG:
----------
2011/08/04, v0.0.0, S.Kamann
"""
__version__ = "0.0.0"
import datetime, os, re, string, sys
from lxml import etree



def transform_string(thestring, dtype):
    """
    Short routine to convert a string to one of the following data types:
        float, int, bool, list, tuple

    Parameters
    ----------
    thestring : string
        The string that should be converted.
    dtype : string
        The data type to which the string should be converted

    Returns:
        The result of the transformation.
    """
    if dtype == 'str':
        return thestring
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
        thelist = [float(x) for x in thestring[1:-1].split(',')]
        if dtype == 'tuple':
            return tuple(thelist)
        else:
            return thelist
        
    else:
        print "WARNING: Unknown datatype encountered:{0}".format(dtype)
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
    def __init__(self, filename, schema):
        """
        Initialize a new instance of the `ModifyConfig`-class

        Parameters:
        -----------
        filename : string
            The path to the PampelMuse-configuration-file
        schema : string
            The path to the schema against which the XML-file is validated
        """

        ## open XML-schema
        schema_doc = etree.parse(schema)
        self.xmlschema = etree.XMLSchema(schema_doc)

        ## specifyXML-parser, include schema
        self.parser = etree.XMLParser(schema=self.xmlschema)

        ##  open file, validate against schema
        self.filename = filename
        self.ptree  = etree.parse(filename, self.parser)
        self.config = self.ptree.getroot()

        ## get the namespace(s) for XPATH-searches:
        self.nsmap  = self.ptree.getroot().nsmap
        self.loc_ns = {"pm":self.nsmap[None]}
        self.prefix = "{{{0}}}".format(self.loc_ns["pm"])

        ##  What time is it?
        self.ut        = datetime.datetime.utcnow()
        self.timestamp = self.ut.strftime("%Y-%m-%dT%H:%M:%SZ")


    def add_parameter(self, name, value, comment=None, category=None,
                      inuse=None):
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
            configuration. By default, `inuse` is set to True.

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
            print "Cannot add parameter '{0}' ".format(new_name)\
                + "since it contains invalid characters!"
            return 0

        ##  check if parameter is already present
        param_list = self.ptree.xpath("//pm:parameter[pm:name='{0}']"\
                                          .format(new_name),
                                      namespaces=self.loc_ns)
        if len(param_list) > 0:
            print "Parameter '{0}' does already exist!".format(new_name)
            print "Use `update_parameter`-method!"
            return 0

        ## get type of parameter using reglular expressions
        dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)",
                           str(type(value)))[0][2]
                
        ##  create new parameter as etree.SubElement of the root
        attrib = {}
        if category != None: attrib["category"] = category
        if inuse != None: attrib["inuse"] = inuse


        new_param = etree.SubElement(self.config, self.prefix+'parameter',
                                     attrib=attrib)

        thename = etree.SubElement(new_param, self.prefix+'name')
        thename.text = new_name

        thevalue = etree.SubElement(new_param, self.prefix+"value",
                                    attrib={"dtype":str(dtype)})
        thevalue.text = str(value)


        if comment != None:
            thecomment = etree.SubElement(new_param, self.prefix+"comment")
            thecomment.text = comment

        ## store information about the update in the configuration
        theupdate = etree.SubElement(new_param, self.prefix+"updated",
                                     attrib={"timestamp":self.timestamp})
        theupdate.text = "Appended parameter {0} to configuration file."\
            .format(new_name)

        return self.xmlschema.assertValid(self.ptree)



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
            The updated configuration is validated against the schema.
            If succesful, true is returned; otherwise the routine will return
            an error.
        """
        thename = string.upper(name)
        updates = [] ## a container for all strings describing the
                     ## individual updates

        ## search parameter
        param_list = self.ptree.xpath("//pm:parameter[pm:name='{0}']"\
                                          .format(thename),
                                      namespaces=self.loc_ns) 
        if len(param_list) == 0: ## Found no occurence -> abort
            print "Parameter '{0}' does not exist!".format(thename)
            print "Use `add_parameter`-method to include it!"
            return 0
        elif len(param_list) > 1: ## Found multiple occurences -> abort
            print "WARNING: Meaning of parameter {0} ambiguous!".format(thename)
            print "Please verify XML-file!"
            return len(param_list)
        else:
            theparam = param_list[0] ##  Found exactly 1 occurence -> go on
            if value != None:  ## check if value should be updated
                dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)",
                           str(type(value)))[0][2]
                old_dtype =  theparam.xpath("./pm:value/@dtype",
                                            namespaces=self.loc_ns)[0]

                if dtype != old_dtype: ## new data-type does not match old one
                    if not force_update:
                        print "Provided value is of type '{0}',".format(dtype),
                        print " but '{0}' requires '{1}'!"\
                            .format(thename, theparam.attrib["dtype"])
                        return
                    else:
                        updates.append("Changed dtype of parameter {0}:"\
                                           .format(thename)+" {0} to {1}"\
                                           .format(old_dtype, dtype))
                        theparam.xpath("./pm:value/@dtype",
                                       namespaces=self.loc_ns)[0] = dtype
                
                old_value = theparam.xpath("./pm:value",
                                           namespaces=self.loc_ns)[0].text
                theparam.xpath("./pm:value", namespaces=self.loc_ns)[0].text\
                    = str(value)
                updates.append("Changed default value of parameter {0}:"\
                                   .format(thename)+" {0} to {1}"\
                                   .format(old_value, value))

            if comment_r != None: ## check if comment should be replaced
                _old = theparam.findall("./pm:comment", namespaces=self.loc_ns)
                for comment in _old: ##  delete old comment(s) (if any)
                    theparam.remove(comment)
                    comment = comment_r
            if comment != None: ##  add new comment
                newcomment = etree.SubElement(theparam, self.prefix+"comment")
                newcomment.text = comment

                ## schema requires that 'updated'-Elements are placed after
                ## 'comment'-Elements. Following codes assures this.
                _oldupdates = theparam.findall("./pm:updated",
                                               namespaces=self.loc_ns)
                if len(_oldupdates)>0:
                    _oldupdates[0].addprevious(newcomment)
            
            if category != None:
                old_category = theparam.xpath("./@category",
                                namespaces=self.loc_ns)[0]
                if old_category != category:
                    updates.append("Changed category of parameter {0}:"\
                                       .format(thename)+" {0} to {1}"\
                                       .format(old_category, category))
                theparam.set("category", category)

            if inuse != None: ## check if usage-flag should be updated
                old_inuse = transform_string\
                    (theparam.xpath("./@inuse", namespaces=self.loc_ns)[0], 
                     "bool")
                if old_inuse and not inuse:
                    updates.append("Disabled usage of parameter {0}."\
                                       .format(thename))
                elif inuse and not old_inuse:
                    updates.append("Enabled usage of parameter {0}."\
                                       .format(thename))
                theparam.set("inuse", str(int(inuse)))


            for update in updates: ## write update information to file
                theupdate =etree.SubElement(theparam, self.prefix+"updated",
                                            attrib={"timestamp":self.timestamp})
                theupdate.text = update

            return self.xmlschema.assertValid(self.ptree)



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
        param_list = self.ptree.xpath("//pm:parameter[pm:name='{0}']"\
                                          .format(thename),
                                      namespaces=self.loc_ns) 
        if len(param_list) == 0: ## None found -> abort
            print "WARNING: No such parameter: {0}".format(thename)
            return
        elif len(param_list) > 1: ## Multiple found -> abort
            print "WARNING: Meaning of parameter {0} ambiguous!".format(thename)
            print "Please verify XML-file!"
            return len(param_list)
        else: ## Found one! -> go one & return its value (& comment)
            theparam = param_list[0]
            thetype  = theparam.xpath("string(@dtype)")
            thevalue = theparam.xpath("string(pm:value)",
                                      namespaces=self.loc_ns)
            thecomm  = theparam.xpath("string(pm:comment)",
                                      namespaces=self.loc_ns)

            if return_comment:
                return [transform_string(thevalue, thetype), thecomm]
            else:
                return transform_string(thevalue, thetype)
       


    def save_changes(self, filename=None, xml_declaration=True,
                     pretty_print=True):

        if filename == None:
            filename = self.filename

        self.config.set("timestamp", self.timestamp)

        self.ptree.write(filename, pretty_print=pretty_print,
                         xml_declaration=xml_declaration,
                         encoding=self.ptree.docinfo.encoding)




import argparse


class MergeComment(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, " ".join(values))



parser = argparse.ArgumentParser(description="Provides methods to change the PampelMuse configuration.")
parser.add_argument("method", type=str, choices=['U','A','G'])
parser.add_argument("parameter",  type=str)
parser.add_argument("value", type=str, nargs='?', default=None)
parser.add_argument("--type", type=str, default='str')
parser.add_argument("--comment",  type=str, nargs='*', action=MergeComment)
parser.add_argument("--exclude", action='store_false', default=None)
parser.add_argument("--include", action='store_true', default=None)
parser.add_argument("--category", type=str)
args = parser.parse_args()

CONFIGPATH = os.path.expandvars("$pampelmuse_path/config/")

modifier = ModifyConfig("{0}PMparams.xml".format(CONFIGPATH),
                        "{0}PMschema.xsd".format(CONFIGPATH))

inuse = max([args.include, args.exclude])

if args.method == "A":
    success = modifier.add_parameter(name=args.parameter,
                                     value=transform_string(args.value,
                                                            args.type),
                                     comment=args.comment,
                                     category=args.category,
                                     inuse=inuse
                                     )
elif args.method == "U":
    success = modifier.update_parameter(name=args.parameter,
                                        value=transform_string(args.value,
                                                               args.type),
                                        comment=args.comment,
                                        category=args.category,
                                        inuse=inuse
                                        )
elif args.method == 'G':
    query_result = modifier.get_parameter(args.parameter)
    print "{0:10}{1:25}{2}".format(string.upper(args.parameter), 
                                   query_result[0],
                                   query_result[1])

if args.method in ["A", "U"]:
    os.system('cp {0}PMparams.xml {0}PMparams.xml.sav~'.format(CONFIGPATH))
    modifier.save_changes()


