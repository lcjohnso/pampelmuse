import numpy as np, pyfits
from pampelmuse.functions.step_poly_fit import StepPolyFit

## Still needs debugging!!!
## Code will  crash if x_ifu and y_ifu have been calculated before.

## import matplotlib.pyplot as plt

## def calcSigma(inData, size=200):
## 
##     outData = np.zeros(inData.shape, dtype="float32")
##     for i in xrange(inData.shape[0]):
##         selPix = np.copy(inData[max(0, i-size):min(inData.shape[0], i+size)])
##         selPix.sort()
##         outData[i] = selPix[size//20:-size//20].std()
## 
##     return outData


def posfit(arguments):

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    nBins    = 3                                          ##
    pOrder   = 3                                          ##
    overlap  = 400                                        ##

    prmFile = pyfits.open("{0}.prm.fits".format(prefix), mode="update")

    Nsources = prmFile["SOURCES"].data.size

    ## read transformation parameters
    valueShape = (prmFile['POSPARS'].header['NAXIS1'],)
    
    posPars    = np.zeros(4,
                          dtype=[('name', 'a1'),
                                 ('value', 'f8', valueShape)])
    posPars['name'] = ['A','B','C','D']

    for i in xrange(4):
        posPars['value'][i] = prmFile['POSPARS'].data[i]
    
    outData = np.zeros(Nsources, dtype=[("x_ifu", np.float32, valueShape),
                                        ("y_ifu", np.float32, valueShape)])

##     fig = plt.figure()
##     ax1 = fig.add_axes([0.12,0.12,0.83,0.83])

    for j in xrange(Nsources):
        if prmFile["SOURCES"].data.field("status")[j] < 4:
            x = prmFile["SOURCES"].data.field("x")[j]
            y = prmFile["SOURCES"].data.field("y")[j]

            _x_init = posPars["value"][0]*x + posPars["value"][1]*y \
                + posPars["value"][2]
            _y_init = posPars["value"][0]*y - posPars["value"][1]*x \
                + posPars["value"][3]

            outData["x_ifu"][j] \
                = StepPolyFit(_x_init,order=pOrder,bins=nBins,overlap=overlap)
            outData["y_ifu"][j] \
                = StepPolyFit(_y_init,order=pOrder,bins=nBins,overlap=overlap)

##             if j == 2:
##                 ax1.plot(_y_init, "g-")
##                 ax1.plot(outData["y_ifu"][j], "r-")
## 
##     plt.show()

    coordTable = pyfits.BinTableHDU(outData)
    fullTable  = pyfits.new_table(prmFile["SOURCES"].columns+coordTable.columns)
    fullTable.header.update("EXTNAME", "SOURCES")

    prmFile["SOURCES"] = fullTable
    
    prmFile.flush()


##     newTbl = pyfits.new_table(prmFile["SOURCES"].columns + newCols.columns)
##     print newCols.data["x_ifu"]


## meanSlope = np.average(smoothPos['x']-smoothPos['x'].mean(axis=1)[:,np.newaxis], axis=0, weights=smoothPos['x_err'])
