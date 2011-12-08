"""
EXTRACT
=======
Short routine to extract the pixel positions of the resolved sources from
the PRM-file and store them in a separate ASCII-file. The structure of the
ASCII-file will be: one source per line, 4 columns with ID, x pos., y pos.
and a color code. The color code depends on the status of a source. The
first line contains the column names.
In case an ASCII-file is provided containing the positions of at least 4
resolved source in an arbitrary reference frame. The routine will also
determine the coordinate transformation into that reference frame and create
another ASCII-file containing the pixel positions of all resolved sources in
the additional reference frame. This option can be used to create a source
list that can be overplotted onto an arbitrary FITS-image of the target.

CHANGELOG:
----------
2011/12/08, v.0.1.0.0
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.0.0"

import argparse, logging
import numpy as np, pyfits

## required PampelMuse routines
from pampelmuse.functions.trans_coords import trans_coords

## the color-code attached to a source depends on its `status'
COLORS = {1:"#33CCCC", 2:"#0000FF", 3:"#FF3300"}


def get_pixpos(arguments):

    ## arguments is a namespace that MUST contain the following entries:
    ## 1) prefix - the prefix common to all FITS-files to be used
    ## [2) addcoords - if the transformation into an  additional reference
    ##                 frame should be performed]  
    ## [3) config - a dictionary containing the current PampelMuse
    ##              configuration]

    print ""
    print 21* " "+"***  Extract, version {0}  ***".format(__version__)+21*" "
    print ""
  
  
    ## List of required definitions: #################
    prefix    = arguments.prefix                    ##
    addcoords = arguments.addcoords                 ##
    ##################################################
    
    ##  open PRM-file, extract information
    inFile = pyfits.open("{0}.prm.fits".format(prefix))

    A = np.mean(inFile['POSPARS'].data[0]) 
    B = np.mean(inFile['POSPARS'].data[1])
    C = np.mean(inFile['POSPARS'].data[2])
    D = np.mean(inFile['POSPARS'].data[3])

    ID     = inFile['SOURCES'].data.field('id')
    xInput = inFile['SOURCES'].data.field('x')
    yInput = inFile['SOURCES'].data.field('y')
    status = inFile['SOURCES'].data.field('status')


    ## store list of extracted sources
    outFile = open("{0}.src.dat".format(prefix), "w")
    outFile.write("    id      x      y    color\n")
    for i in xrange(inFile["SOURCES"].data.size):
        if status[i] in [1,2,3]:
            xPix = A*xInput[i]+B*yInput[i]+C
            yPix = A*yInput[i]-B*xInput[i]+D

            outFile.write("{0:6d}  {1:5.2f}  {2:5.2f}  {3}\n".format(\
                    int(ID[i]), xPix, yPix, COLORS[status[i]]))
    outFile.close()


    ##  check if additional file should be stored
    if addcoords != None:
        srcList = np.array(inFile['SOURCES'].data,
                       dtype=inFile['SOURCES'].data.dtype)

        addCoords = np.genfromtxt(addcoords, usecols=(1,2,3),
                                  dtype=[('id', '>i4'),
                                         ('x', '>f4'),
                                         ('y', '>f4')])

        addParams = trans_coords(srcList, addCoords)
        A = addParams['A'][0]
        B = addParams['B'][0]
        C = addParams['C'][0]
        D = addParams['D'][0]

        addOutFile = open("{0}.add.dat".format(prefix), "w")
        addOutFile.write("    id       x       y    color\n")

        for i in xrange(inFile["SOURCES"].data.size):
            if status[i] in [1,2,3]:
                xPix = A*xInput[i]+B*yInput[i]+C
                yPix = A*yInput[i]-B*xInput[i]+D

                addOutFile.write("{0:6d}  {1:6.1f}  {2:6.1f}  {3}\n".format(\
                        int(ID[i]), xPix, yPix, COLORS[status[i]]))

        addOutFile.close()

    inFile.close()
