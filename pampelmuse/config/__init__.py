"""
__init__.py
============
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
class pampelmuse.config.Namespace
class pampelmuse.config.PampelMuseConfig

Purpose
-------
The module implements two classes that are used to carry the configuration
parameters of PampelMuse through the analysis.
The class 'Namespace' is, as the name suggests, a pure namespace, suited
to store, update and read attributes. For each subroutine of PampelMuse, an
individual instance of Namespace will be created.
The class 'PampelMuseConfig' is also designed as a pure namspace. However,
it defines two additional methods PampelMuseConfig.readHeader() and
PampelMuseConfig.makeHeader() that can be used to read a configuration from
a FITS HDU and to store the current configuration of PampelMuse again in
the header of a FITS HDU.

Latest SVN revision
-------------------
340, 2016/07/29
"""
import ast
import datetime
import logging
import os
import re
from astropy.io import fits
from .xml_interact import ModifyConfig, transform_string


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 340


class Namespace():
    pass


class PampelMuseConfig(Namespace):

    def __init__(self):
        """
        Initialize an instance of class 'PampelMuseConfig'
        """
        
        self.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
        
        # the following list contains all the keywords that can currently be changed during the analysis, i.e. when
        # calling a routine different from INITFIT or SINGLESRC:
        self.changeable = ["verbose", "interact", "cpucount", "maxiter", "polytype", "polydeg", "subunres", "subsky",
                           "usesigma", "layerrng", "layerbin", "psfrad"]

        # read the default configuration from he XML file that is part of PampelMuse and stored in the data directory
        # of this package.
        __currentdir__ = os.path.abspath(os.path.dirname(__file__))
        self.config_reader = ModifyConfig(os.path.join(__currentdir__, "data", "PMparams.xml"),
                                          os.path.join(__currentdir__, "data", "PMschema.xsd"))

        self.start = 0
        self.stop = -1
        for key, value in self.config_reader.get_all().iteritems():
            if key.lower() == "layerrng":
                self.start = value[0]
                self.stop = value[1]
            setattr(self, key.lower(), value)

    def update_parameter(self, key, value, only_changeable=False):
        """
        This method is used to change a configuration parameter for the
        current analysis. It performs a number of consistency checks before
        updating the parameter in order to not mess up the configuration.

        Parameters
        ----------
        key : string
            The name of the parameter that should be updated.
        value : string or appropriate type
            The updated value of the parameter. If a string is provided, the
            method will check whether it can be evaluated, i.e. transformed
            into another type.
        only_changeable : bool, optional
            If this flag is set to True, only the parameters specified in the
            'changeable' attribute can be specified

        Returns
        -------
        success : int
            0 is update successful, 1 if failed with warning, 2 if failed with
            error.
        """
        if not hasattr(self, key.lower()):  # make sure only valid parameters are included
            logging.error("Unknown configuration parameter provided: '{0}'".format(key))
            return 2

        elif not only_changeable or key.lower() in self.changeable:  # make sure parameter can still be modified

            # get current value of the parameter
            current_value = getattr(self, key.lower())

            if isinstance(current_value, bool):
                try:
                    value = ast.literal_eval(value)  # bool(<string>) will always return True
                except ValueError:
                    logging.error("New value for parameter {0} cannot be casted to boolean.".format(key.upper()))
                    return 1
            elif isinstance(current_value, int):
                try:
                    value = int(value)
                except ValueError:
                    logging.error("New value for parameter {0} cannot be casted to integer.".format(key.upper()))
                    return 1
            elif isinstance(current_value, float):
                try:
                    value = float(value)
                except ValueError:
                    logging.error("New value for parameter {0} cannot be casted to float.".format(key.upper()))
                    return 1
            elif isinstance(current_value, str):
                try:
                    value = str(value)
                except ValueError:
                    logging.error("New value for parameter {0} cannot be casted to string.".format(key.upper()))
                    return 1
            elif isinstance(current_value, tuple):
                try:
                    value = ast.literal_eval(value)
                    if key.lower() == "layerrng":  # the layer range must be transformed into start and stop values
                        self.start = value[0]
                        self.stop = value[1]
                except ValueError:
                    logging.error("New value for parameter {0} cannot be casted to tuple.".format(key.upper()))
                    return 1
            elif isinstance(current_value, list):
                try:
                    value = ast.literal_eval(value)
                except ValueError:
                    logging.error("ew value for parameter {0} cannot be casted to list.".format(key.upper()))
                    return 1
            else:  # The code should never come here.
                logging.error("Something went terribly wrong.")
                return 2

            logging.debug("Updating parameter {0} to new value {1}.".format(key.upper(), value))
            setattr(self, key.lower(), value)
            return 0
        else:
            logging.warning("Configuration parameter '{0}' cannot be changed during the analysis.".format(key))
            return 1

    def readHeader(self, header):
        """
        Read a configuration from the header of a FITS HDU.
        
        Parameters
        ----------
        header : an astropy.io.fits.Header instance
            The HDU from which the configuration should be read. The
            routine will scan the header for HIERARCH keywords starting
            with PAMPELMUSE.
        """
        config_finder = re.compile(r"PAMPELMUSE\s([A-Z]+)\s?([A-Z]\w+)?$")
        
        for entry in header.cards:
            match = config_finder.match(entry.keyword)
            if match is not None:
                if match.group(2) is None:
                    setattr(self, match.group(1).lower(), entry.value)
                else:
                    if not hasattr(self, match.group(1).lower()):  # create sub-namespace if it does not exist
                        setattr(self, match.group(1).lower(), Namespace())
                    if match.group(1).lower() == "saturated":
                        self.saturated.append(entry.value)
                    else:
                        setattr(self.__dict__[match.group(1).lower()],  match.group(2).lower(), entry.value)
                        # add parameter to sub-namespace

    def makeHeader(self, hdu=None):
        """
        Save the current configuration in a astropy.io.fits.Header instance.
        
        Parameters
        ----------
        hdu : an astropy.io.fits HDU instance, optional
            The HDU used to store the configuration, if none is provided,
            an empty HDU will be created.
        """
        if hdu is not None:
            header = hdu.header
        else:
            header = fits.Header()

        for entry in ["revision", "utc", "verbose", "ifu", "interact", "useqt", "cpucount", "smallifu", "passband",
                      "ccmaxoff", "aperrad", "apernois", "reslim", "updaterl", "confdens", "snrcut", "binrad",
                      "useunres", "unresbin", "globsky", "start", "stop", "layerbin", "psfrad", "jumpstart",
                      "usesigma", "maxiter", "polytype", "polydeg", "subunres", "subsky"
                      ]:
           
            if hasattr(self, entry):
                header["hierarch PAMPELMUSE {0}".format(entry.upper())] = getattr(self, entry)
         
        if hasattr(self, "ifs_data"):
            if hasattr(self.ifs_data, "instrument"):
                header["hierarch PAMPELMUSE IFU"] = getattr(self.ifs_data.instrument, "name")
         
        if hasattr(self, "psf"):
            if hasattr(self.psf, "profile"):
                header["hierarch PAMPELMUSE PSF PROFILE"] = getattr(self.psf, "profile")
         
            i = 0
            while hasattr(self.psf, "parname{0}".format(i)):
                header["hierarch PAMPELMUSE PSF PARNAME{0}".format(i)] = getattr(self.psf, "parname{0}".format(i))
                header["hierarch PAMPELMUSE PSF PARVALUE{0}".format(i)] = getattr(self.psf, "parvalue{0}".format(i))
                header["hierarch PAMPELMUSE PSF FREEPAR{0}".format(i)] = getattr(self.psf, "freepar{0}".format(i))
                i += 1

        if hasattr(self, "saturated"):
            for i, star_id in enumerate(self.saturated):
                header["hierarch PAMPELMUSE SATURATED ID{0:03d}".format(i)] = star_id

        if hasattr(self, "initfit"):
            for entry in ["revision", "utc", "ccxoff", "ccyoff", "reslim", "photzero"]:
                if hasattr(self.initfit, entry):
                    header["hierarch PAMPELMUSE INITFIT {0}".format(entry.upper())] = getattr(self.initfit, entry)

        if hasattr(self, "cubefit"):
            for entry in ["revision", "utc", "status", "lastlayer"]:
                if hasattr(self.cubefit, entry):
                    header["hierarch PAMPELMUSE CUBEFIT {0}".format(entry.upper())] = getattr(self.cubefit, entry)
                    
        if hasattr(self, "polyfit"):
            for entry in ["revision", "utc"]:
                if hasattr(self.polyfit, entry):
                    header["hierarch PAMPELMUSE POLYFIT {0}".format(entry.upper())] = getattr(self.polyfit, entry)

        if hasattr(self, "subtres"):
            for entry in ["revision", "utc"]:
                if hasattr(self.subtres, entry):
                    header["hierarch PAMPELMUSE SUBTRES {0}".format(entry.upper())] = getattr(self.subtres, entry)

        if hasattr(self, "shotnoi"):
            for entry in ["revision", "utc"]:
                if hasattr(self.shotnoi, entry):
                    header["hierarch PAMPELMUSE SUBTRES {0}".format(entry.upper())] = getattr(self.shotnoi, entry)

        if hasattr(self, "addwcs"):
            for entry in ["revision", "utc", "RADECSYS", "EQUINOX", "WCSDIM", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
                          "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"]:
                if hasattr(self.addwcs, entry.lower()):
                    header["hierarch PAMPELMUSE ADDWCS {0}".format(entry.upper())] = getattr(
                        self.addwcs, entry.lower())
        if hdu is None:
            return header
        else:
            return None
