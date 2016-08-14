"""
xml_interact.py
===============
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
function pampelmuse.config.xml_interact.transform_string
class pampelmuse.config.xml_interact.ModifyConfig

Purpose
-------
The module implements the class 'ModifyConfig' which provides several methods
to update and to read the PampelMuse-configuration that is stored in the 
XML-file 'PMparams.xml' in the config/ folder of the PampelMuse distribution.
When a new instance of the class is initialized, the user has to provide
the path to the configuration-file and to its XML-Schema (a .xsd-file).
After initialization, the methods 'add_parameter', 'update_parameter' and
'get_parameter' allow interaction with the configuration file. Finally,
the method 'save_changes' can be used to write to possible changes disk.

Example for usage:
>>> modifier = ModifyConfig('PMparams.xml', 'PMschema.xsd')
>>> modifier.add_parameter(name='PSFTYPE', value='moffat', category='psf', comment='A new PSF')
>>> modifier.update_parameter(name='PSFTYPE', value='gaussian')
>>> print modifier.get_parameter(name='PSFTYPE')
['gaussian', 'A new PSF']
>>> modifier.save_changes()

The auxiliary function 'Transform_string' is designed to transform a string
into another type, such as a float or a list.

Latest SVN revision
-------------------
340, 2016/07/29
"""
import datetime
import logging
import re
import string
from xml.etree import ElementTree
from xml.dom.minidom import parseString


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 340


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
        # using regular expressions for the transformation to bool
        return bool(int(re.sub("true", "1", re.sub("false", "0", string.lower(thestring)))))
    elif dtype in ['list', 'tuple']:
        thenumbers = thestring[1:-1].split(',')
        if thenumbers == [""]:
            thelist = []
        elif sum([len(re.findall('\.', s)) for s in thenumbers]) == 0:
            thelist = [int(x) for x in thenumbers]
        else:
            thelist = [float(x) for x in thenumbers]

        if dtype == 'tuple':
            return tuple(thelist)
        else:
            return thelist
        
    else:
        logging.warning("Unknown datatype encountered:{0}".format(dtype))
        if abort_on_exception:
            raise IOError("Unknown datatype encountered:{0}".format(dtype))
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
        # # open XML-schema
        # schema_doc = etree.parse(schema)
        # self.xmlschema = etree.XMLSchema(schema_doc)

        # # specifyXML-parser, include schema
        # self.parser = etree.XMLParser(schema=self.xmlschema)
        self.parser = ElementTree.XMLParser()

        # open file, validate against schema
        self.filename = filename
        self.ptree = ElementTree.parse(filename, self.parser)

        # get the namespace(s) for XPATH-searches:
        # WARNING: Hard-coded so far
        self.ns = "{http://www.tobedefined.com/pampelmuse}"

        # list of all existing parameter-names:
        names = self.ptree.findall("./{0}parameter/{0}name".format(self.ns))
        self.pnames = [name.text for name in names]

        #  What time is it?
        self.ut = datetime.datetime.utcnow()
        self.timestamp = self.ut.strftime("%Y-%m-%dT%H:%M:%SZ")

        self.AoE = abort_on_exception

    def add_parameter(self, name, value, comment=None, category=None, inuse=True, ahead=None, after=None):
        """
        Add a new parameter to the configuration. Note that this method should
        not be used to update existing parameters. For this purpose, the
        `update_parameter`-method should be used.

        Parameters
        ----------
        name : string
            The name of the parameter. It should only contain alphanumerical
            characters. Note that all characters are automatically set to
            uppercase.
        value : a standard python data-type
            The default value of the parameter. Can be either a string, a
            float, an integer, a boolean flag, a list or a tuple.
        comment : string, optional
            A comment to explain the purpose of the parameter.
        category : string,  optional
            A keyword to specify which part of the configuration is affected
             by the new parameter. Currently, it is only used for
             visualization purposes.
        inuse : boolean, optional
            A flag to indicate whether the parameter is used by the
            configuration. By default, `inuse` is set to True. In any case,
            the updated configuration will be validated against the XML-
            schema.
        ahead : string, optional
            If specified, new parameter will be placed ahead of the given one
            in the configuration.
        after : string, optional
            If specified, new parameter will be placed after the given one in
            the configuration.


        Returns
        -------
        valid : an etree.XMLSchema.assertValid return-value
            The new parameter is validated against the schema. If successful,
            true is returned; otherwise the routine will return an error.
        """
        # check if parameter conforms to required format - only alphanumeric characters are allowed
        new_name = string.upper(name.strip())
        if ''.join(re.findall('[0-9A-Z]', new_name)) != new_name:
            logging.error("Cannot add parameter '{0}' - invalid name!".format(new_name))
            if self.AoE:
                raise IOError("Cannot add parameter '{0}' since it contains invalid characters!".format(new_name))
            return 0

        #  check if parameter is already present
        if new_name in self.pnames:

            logging.error("Parameter '{0}' does already exist!".format(new_name))
            logging.error(" Use `update_parameter`-method!")
            if self.AoE:
                raise IOError("Parameter '{0}' does already exist!".format(new_name))
            return 0

        # get type of parameter using regular expressions
        dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)", str(type(value)))[0][2]
        
        #  create new parameter as etree.SubElement of the root
        attrib = {}
        if category is not None:
            attrib["category"] = category
        if inuse is not None:
            attrib["inuse"] = inuse
        
        new_param = ElementTree.SubElement(self.ptree.getroot(), "{0}parameter".format(self.ns), attrib=attrib)

        thename = ElementTree.SubElement(new_param, "{0}name".format(self.ns))
        thename.text = new_name

        thevalue = ElementTree.SubElement(new_param, "{0}value".format(self.ns), attrib={"dtype": str(dtype)})
        thevalue.text = str(value)

        if comment is not None:
            thecomment = ElementTree.SubElement(new_param, "{0}comment".format(self.ns))
            thecomment.text = comment

        # store information about the update in the configuration
        theupdate = ElementTree.SubElement(new_param, "{0}updated".format(self.ns),
                                           attrib={"timestamp": self.timestamp})

        theupdate.text = "Appended parameter {0} to configuration file.".format(new_name)

        # check if any information about position of new parameter given
        if ahead is not None and after is not None:
            logging.warning("Cannot use `ahead`- AND `after`-parameter.")

        index = None
        if ahead is not None:
            if ahead not in self.pnames:
                logging.error("Parameter '{0}' does not exist.".format(ahead))
            else:
                index = self.pnames.index(ahead)
        elif after is not None:
            if after not in self.pnames:
                logging.error("Parameter '{0}' does not exist.".format(after))
            else:
                index = self.pnames.index(after) + 1

        if index is not None:
            self.ptree.getroot().remove(new_param)
            self.ptree.getroot().insert(index, new_param)
            self.pnames.insert(index, new_name)
        else:
            self.pnames.append(new_name)

        logging.info("Added parameter '{0} to configuration'".format(new_name))
        
        # return self.xmlschema.assertValid(self.ptree)
        return 1

    def update_parameter(self, name, value=None, comment=None, comment_r=None, category=None, inuse=None,
                         force_update=False):
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
            Same as `comment`, only that existing comments (if any) are
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
        updates = []  # a container for all strings describing the individual updates

        # search parameter
        if thename not in self.pnames:
            logging.error("Parameter '{0}' does not exist!".format(thename))
            logging.error(" Use `add_parameter`-method to include it!")

            if self.AoE:
                raise IOError("Parameter '{0}' does not exist!".format(thename))
            return 0

        # Found multiple occurrences -> abort
        elif self.pnames.count(thename) > 1:
            logging.critical("Meaning of parameter '{0}' ambiguous!".format(thename))
            logging.critical("Please verify XML-file!")
            if self.AoE:
                raise IOError("Meaning of parameter {0} ambiguous!".format(thename))
            return self.pnames.count(thename)

        #  Found exactly 1 occurrence -> go on
        index = self.pnames.index(thename)

        theparam = self.ptree.findall("./{0}parameter".format(self.ns))[index]

        if value is not None:  # check if value should be updated
            dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)", str(type(value)))[0][2]

            # thevalue = theparam.findall("./{0}value[@dtype]".format(self.ns))
            # including attributes only works in latest versions of xml
            thevalue = theparam.findall("./{0}value".format(self.ns))
            assert len(thevalue) == 1

            vindex = list(theparam).index(thevalue[0])
            theparam.remove(thevalue[0])

            old_value = thevalue[0].text
            old_dtype = thevalue[0].attrib["dtype"]

            if dtype != old_dtype:  # new data-type does not match old one
                logging.warning("Unmatching `dtype` encountered: {0} vs {1}".format(dtype, old_dtype))
                if not force_update:
                    if self.AoE:
                        raise IOError("'{0}' requires '{1}', not '{2}'!".format(thename, old_dtype, dtype))

                    return 0
                else:
                    logging.warning("Changed dtype of parameter {0}:".format(thename))
                    logging.warning("   from {0} to {1}".format(old_dtype, dtype))
   
            thevalue[0].text = str(value)
            theparam.insert(vindex, thevalue[0])

            updates.append("Changed default value of parameter {0} to {1} (was {2})".format(thename, value, old_value))

        if comment_r is not None:  # check if comment should be replaced
            
            # delete old comment(s) (if any)
            for comment in theparam.findall("./{0}comment".format(self.ns)):
                theparam.remove(comment)

            comment = comment_r
        
        if comment is not None:  # add new comment
            newcomment = ElementTree.SubElement(theparam, "{0}comment".format(self.ns))
            newcomment.text = comment

            # schema requires that 'updated'-Elements are placed after
            # 'comment'-Elements. Following codes assures this.

            oldupdates = theparam.findall("./{0}updated".format(self.ns))
            if len(oldupdates) > 0:
                cindex = list(theparam).index(oldupdates[0])
                theparam.remove(newcomment)
                theparam.insert(cindex, newcomment)
            
        if category is not None:
            if "category" in theparam.attrib:
                old_category = theparam.attrib["category"]
            else:
                old_category = "None"
            updates.append("Changed category to {0} (was {1})".format(category, old_category))

            theparam.attrib["category"] = category

        if inuse is not None:  # check if usage-flag should be updated
            if "inuse" in theparam.attrib:
                old_inuse = transform_string(theparam.attrib["inuse"], "bool", abort_on_exception=self.AoE)
            else:
                old_inuse = True

            if old_inuse and not inuse:
                updates.append("Disabled usage of parameter.".format(thename))
            elif inuse and not old_inuse:
                updates.append("Enabled usage of parameter.".format(thename))
            theparam.set("inuse", str(int(inuse)))

            for update in updates:  # write update information to file

                theupdate = ElementTree.SubElement(theparam, "{0}updated".format(self.ns),
                                                   attrib={"timestamp": self.timestamp})
                theupdate.text = update

        self.ptree.getroot()[index] = theparam

        # return self.xmlschema.assertValid(self.ptree)
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

        # search configuration for matching parameter

        if thename not in self.pnames:  # None found -> abort
            logging.error("No such parameter: '{0}'".format(thename))
            if self.AoE:
                raise IOError(" No such parameter: {0}".format(thename))
            return None
        elif self.pnames.count(thename) > 1:  # Multiple found -> abort
            logging.critical("Multiple occurrences of parameter '{0}' found!".format(thename))
            logging.critical("Please verify XML-file!")
            if self.AoE:
                raise IOError("Multiple occurrences of parameter '{0}' found!".format(thename))
            return self.pnames.count(thename)

        # Found one! -> go one & return its value (& comment)
        index = self.pnames.index(thename)

        theparam = self.ptree.findall("./{0}parameter".format(self.ns))[index]

        thevalue = theparam.findall("./{0}value".format(self.ns))[0]
        
        thetype = thevalue.attrib["dtype"]

        thecomm = theparam.findall("./{0}comment".format(self.ns))
        comment = ""
        for comm in list(thecomm):
            comment += "-{0}".format(comm.text)

        if return_comment:
            return [transform_string(thevalue.text, thetype, abort_on_exception=self.AoE), comment]
        else:
            return transform_string(thevalue.text, thetype, abort_on_exception=self.AoE)

    def get_all(self, return_comment=False):
        """
        Read the current configuration and write it to a dictionary.

        Parameters
        ----------
        return_comment : boolean
            Flag indicating if the returned values include the comment.

        Returns
        -------
        config : dictionary
            A dictionary containing 'name:value'-pairs for all parameters
            specified in the XML-configuration file.
        """
        config = {}

        param_list = self.ptree.findall("./{0}parameter".format(self.ns))

        for parameter in param_list:

            thename = parameter.findall("./{0}name".format(self.ns))[0]

            thevalue = parameter.findall("./{0}value".format(self.ns))[0]

            thetype = thevalue.attrib["dtype"]

            if return_comment:
                thecomm = parameter.findall("./{0}comment".format(self.ns))
                comment = ""
                for comm in list(thecomm):
                    comment += "-{0}".format(comm.text)
                config[thename.text] = [transform_string(thevalue.text, thetype, abort_on_exception=self.AoE), comment]
            else:
                config[thename.text] = transform_string(thevalue.text, thetype, abort_on_exception=self.AoE)

        return config

    def save_changes(self, filename=None, style=None):

        if filename is None:
            filename = self.filename

        self.ptree.getroot().set("timestamp", self.timestamp)

        txt = ElementTree.tostringlist(self.ptree.getroot(), encoding='ISO-8859-1')
        if style is not None:
            txt.insert(1, '<?xml-stylesheet type="text/xsl" href="{0}"?>'.format(style))

        outfile = open(filename, "w")
        outfile.write(parseString("".join(txt)).toprettyxml(indent="", newl=""))
        outfile.close()

        # self.ptree.write(filename, pretty_print=pretty_print, xml_declaration=xml_declaration,
        #                  encoding=self.ptree.docinfo.encoding)
