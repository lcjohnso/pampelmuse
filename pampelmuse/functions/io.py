"""
io.py
=====

Provides
--------
pampelmuse.functions.io.read_asciifile

Purpose
-------
This module implements functions that handle I/O actions of files. The
current list of functions defined in this module contains:

1) read_asciifile
Read data from an ascii file and return it as a dictionary that contains
a list of strings for each column.

To import a function from this module (e.g. read_asciifile):
from pampelmuse.core.io import read_asciifile

Latest SVN revision
-------------------
279, 2015/08/21
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 279


import logging
import re


def read_asciifile(filename, usenames=None, usecols=None, optional="", aliases={}):
    """
    Reads an ascii file and stores its data in a dictionary. The input
    must be a plain ascii file containing the data in well separated
    columns. Empty lines, or lines that are commented out (starting 
    with '#') are ignored. The routine will automatically check whether
    the first (non-ignored) line of the file contains the columns names.
    The output can be restricted to a subset of the available columns.

    Parameters
    ----------
    filename : string
        The name of the ascii file to be read.
    usenames : string or iterable, optional
        The names of the columns to be returned, provided either as a
        single string of (comma-separated) names or a list-like object
        containing the names.
        Note that if no header line is found containing the column
        names, the routine will use the provided names as keys in the
        dictionary that is returned. To this aim, the number of names
        must match the number of (requested or available) columns.
    usecols : iterable
        A list-like object of integers indicating the indices of the
        columns to be returned. Note that this parameter has no effect
        if a header line is detected and the parameter 'usenames' is
        provided.
    optional : stringor iterable, optional
        The names of the columns that are optional. No log messages will
        be produced if these columns are missing.
    aliases : dictionary, optional
        If provided, it must be a dictionary mapping names provided via
        'usenames' to a list of alternative names (aliases) under which
        the requested columns may be found.

    Returns
    -------
    output : dictionary
        A mapping from the column names to their data. Note that for each
        column, the data will be a list of strings. No transformation is
        performed to integers or floats.
    """
    datafile = open(filename)

    indices = {}
    output = {}

    re_header = re.compile(r"([a-zA-Z_]\w*)")  # regular expression matching the header line (if any); currently each
    # name must start with a letter and may only include alpha-numeric characters
    re_data = re.compile(r"([0-9\.\-]+|[a-zA-Z]+)")  # regular expression matching a data line; currently an entry in
    # a data line must be either a number (float or integer) or a sequence of letters

    # get optional column names (if any)
    if isinstance(optional, str):
        optional = [name.lower() for name in optional.split(",")]
    else:
        optional = [name.lower() for name in optional]

    i = 0
    for line in datafile.readlines():  # loop over lines in file
    
        if re.match(r"\s*#.*", line) is not None:
            continue  # omit commented lines

        if re.match(r"^\s*$", line) is not None:
            continue  # omit empty lines (or those containing only whitespace)

        if i == 0:  # first (i.e. index=0) valid line may be header line
            line0 = line  # save line in case it is a data line and must be added later
            headerline = re_header.findall(line0)
            headerline = [name.lower() for name in headerline]  # make all column names lowercase
            # check if some columns have the same name, if so, names of successive occurrences get trailing '_'s
            for j in xrange(len(headerline)):
                while headerline[j] in headerline[:j]:
                    headerline[j] += "_"

        elif i == 1:  # for second (i.e. index=1) valid line, some checks need to be performed: If the number of
            # entries detected in the first and second line is the same, the first line is considered a header line;
            # otherwise, the first line is considered a data line
            
            dataline1 = re_data.findall(line)

            if len(dataline1) == len(headerline):  # case 1: same number of entries in first two lines - first line is
                # header

                if usenames is not None:
                    # column names provided - ignore possibly provided column indices (parameter 'usecols')
                    # Note that lowercase letters will be used for all names 
                    if isinstance(usenames, str):
                        usenames = [name.lower() for name in usenames.split(",")]  # in case single string was provided
                    else:
                        usenames = [name.lower() for name in usenames]
                    usecols = []

                    assert hasattr(usenames, "__iter__"), "not iterable: {0}".format(usenames)
                    for name in usenames:
                        # loop over provided names and check whether they (or their aliases) are available.
                        if name in headerline:  # check for name itself
                            indices[name] = headerline.index(name)
                            logging.info("Detected column named '{0}'.".format(name))
                        elif name in aliases.keys():  # check for aliases
                            for alias in aliases[name]:
                                if alias in headerline:
                                    indices[name] = headerline.index(alias)
                                    logging.info("Using column '{0}' as an alias for column '{1}'.".format(alias, name))
                        if name not in indices.keys() and name not in optional:  # check if column required or optional
                            logging.error("Cannot find column corresponding to '{0}'.".format(name))

                elif usecols is not None:  # columns indices provided
                    assert hasattr(usecols, "__iter__"), "not iterable : {0}".format(usecols)
                    usenames = []
                    for j in usecols:
                        if j < len(headerline):
                            indices[headerline[j]] = j
                        else:
                            logging.error("Provided column index ({0}) out of valid range (<{1}).".format(
                                j, len(headerline)))
                else:  # neither column names nor indices were provided
                    for j in xrange(headerline):
                        indices[headerline[j]] = j
                    
                # initialize output using data from second line
                for name, index in indices.iteritems():
                    output[name] = [dataline1[index], ]

            else:
                # case 2: different number of entries in first two lines - first line is data
                if usecols is not None:
                    assert hasattr(usecols, "__iter__"), "not iterable : {0}".format(usecols)
                else:
                    usecols = range(len(dataline1))

                if usenames is not None:  # if column names have been provided, they are used to name the output columns
                    # and the number of provided names must match the number of (requested or available) columns
                    if isinstance(usenames, str):
                        usenames = usenames.split(",")
                    assert hasattr(usenames, "__iter__"), "not iterable: {0}".format(usenames)
                    
                    if len(usenames) != len(usecols):
                        logging.error("No. of provided column names not compatible with no. of columns.")
                        usenames = ["col{0}".format(j) for j in usecols]
                else:
                    # otherwise the columns are numerated like col0, col1, col2, ...
                    usenames = ["col{0}".format(j) for j in usecols]

                # make sure number of data entries in first two lines is the same (otherwise the file has an
                # unsupported structure)
                dataline0 = re_data.findall(line0) 
                assert len(dataline0) == len(dataline1), "Cannot determine structure of ascii-file."

                logging.warning(
                    "Did not detect header with column names. Make sure the following column assignment is correct!")

                for j in xrange(len(usenames)):  # initialize output using data from first two columns
                    if usecols[j] < len(dataline0):
                        logging.warning("   '{0}': column #{1}".format(usenames[j], usecols[j]))
                        indices[usenames[j]] = usecols[j]
                        output[usenames[j]] = [dataline0[usecols[j]], dataline1[usecols[j]]]
                    else:
                        logging.error("Index of column '{0}' ({1}) out of valid range (<{2}).".format(
                            usenames[j], usecols[j], len(dataline0)))
             
            ncols = len(dataline1)
   
        else:  # for each remaining line, just try too add it to existing output data
            datalinei = re_data.findall(line)
            if not len(datalinei) == ncols:
                logging.error("Structure of line #{0} incompatible with previous lines.".format(i))
            else:
                for name, index in indices.iteritems():
                    output[name].append(datalinei[index])
                    
        i += 1

    return output
