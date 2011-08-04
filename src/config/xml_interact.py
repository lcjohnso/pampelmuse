#!/usr/bin/python
import datetime, os, re, string, sys
from lxml import etree



def transform_string(thestring, dtype):
    if dtype == 'str':
        return thestring
    elif dtype == 'float':
        return float(thestring)
    elif dtype == 'int':
        return int(thestring)
    elif dtype == 'bool':
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

    def __init__(self, filename, schema):

        schema_doc = etree.parse(schema)
        self.xmlschema = etree.XMLSchema(schema_doc)

        self.parser = etree.XMLParser(schema=self.xmlschema)

        self.filename = filename

        self.ptree  = etree.parse(filename, self.parser)
##         self.validator = self.ptree.docinfo.externalDTD

        self.nsmap  = self.ptree.getroot().nsmap

        ## local namespace for XPATH-searches:
        self.loc_ns = {"pm":self.nsmap[None]}
        self.prefix = "{{{0}}}".format(self.loc_ns["pm"])

        self.config = self.ptree.getroot()

        self.ut        = datetime.datetime.utcnow()
        self.timestamp = self.ut.strftime("%Y-%m-%dT%H:%M:%SZ")


    def add_parameter(self, name, value, comment=None, category=None,
                      inuse=None):

        ## check if parameter conforms to required format
        ## only aplphanumeric characters are allowed 
        new_name = string.upper(name.strip())
        if ''.join(re.findall('[0-9A-Z]', new_name)) != new_name:
            print "Cannot add parameter '{0}' ".format(new_name)\
                + "since it contains invalid characters!"
            return

        for child in self.config.getchildren():
            if child[0].text == new_name:
                print "Parameter '{0}' does already exist!".format(new_name)
                print "Use `update_parameter`-method!"
                return

        dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)",
                           str(type(value)))[0][2]
                
        attrib = {}
        if category != None: attrib["category"] = category
        if inuse != None: attrib["inuse"] = inuse


        new_param = etree.SubElement(self.ptree.getroot(),
                                     self.prefix+'parameter',
                                     attrib=attrib)

        thename = etree.SubElement(new_param, self.prefix+'name')
        thename.text = new_name

        thevalue = etree.SubElement(new_param, self.prefix+"value",
                                    attrib={"dtype":str(dtype)})
        thevalue.text = str(value)

##         if dtype == "bool":
##             thevalue = etree.SubElement(new_param, self.prefix+"flag")
##             thevalue.text = value
##         elif dtype == "int":
##             thevalue = etree.SubElement(new_param, self.prefix+"integer")
##             thevalue.text = value
##         elif dtype == "float":
##             thevalue = etree.SubElement(new_param, self.prefix+"float")
##             thevalue.text = value
##         else:
##             thevalue = etree.SubElement(new_param, self.prefix+"string")
##             thevalue.text = str(value)

        if comment != None:
            thecomment = etree.SubElement(new_param, self.prefix+"comment")
            thecomment.text = comment
            
        theupdate = etree.SubElement(new_param, self.prefix+"updated",
                                     attrib={"timestamp":self.timestamp})
        theupdate.text = "Appended parameter {0} to configuration file."\
            .format(new_name)

        return self.xmlschema.assertValid(self.ptree)



    def update_parameter(self, name, value=None, comment=None, comment_r=None,
                         category=None, inuse=None, force_update=False):

        thename = string.upper(name)
        updates = []

        param_list = self.ptree.xpath("//pm:parameter[pm:name='{0}']"\
                                          .format(thename),
                                      namespaces=self.loc_ns) 
        if len(param_list) == 0:
            print "Parameter '{0}' does not exist!".format(thename)
            print "Use `add_parameter`-method to include it!"
            return 0
        elif len(param_list) > 1:
            print "WARNING: Meaning of parameter {0} ambiguous!".format(thename)
            print "Please verify XML-file!"
            return len(param_list)
        else:
            theparam = param_list[0]
            if value != None:
                dtype = re.findall(r"(^<type)( ')([a-z0-9]*.[a-z0-9]*)('>)",
                           str(type(value)))[0][2]
                old_dtype =  theparam.xpath("./pm:value/@dtype",
                                            namespaces=self.loc_ns)[0]

                if dtype != old_dtype:
                    if not force_update:
                        print "Provided value is of type '{0}',".format(dtype),
                        print " but '{0}' requires '{1}'!"\
                            .format(thename, theparam.attrib["dtype"])
                        return 0
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

            if comment_r != None:
                _old = theparam.findall("./pm:comment", namespaces=self.loc_ns)
                for comment in _old:
                    theparam.remove(comment)
                    comment = comment_r
            if comment != None:
                newcomment = etree.SubElement(theparam, self.prefix+"comment")
                newcomment.text = comment

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

            if inuse != None:
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


            for update in updates:
                theupdate =etree.SubElement(theparam, self.prefix+"updated",
                                            attrib={"timestamp":self.timestamp})
                theupdate.text = update

            return self.xmlschema.assertValid(self.ptree)



    def get_parameter(self, name, return_comment=True):
        
        thename = string.upper(name)

        param_list = self.ptree.xpath("//pm:parameter[pm:name='{0}']"\
                                          .format(thename),
                                      namespaces=self.loc_ns) 
        if len(param_list) == 0:
            print "WARNING: No such parameter: {0}".format(thename)
            return 0
        elif len(param_list) > 1:
            print "WARNING: Meaning of parameter {0} ambiguous!".format(thename)
            print "Please verify XML-file!"
            return len(param_list)
        else:
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


