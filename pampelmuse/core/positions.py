"""
POSITIONS
=========
The routine is designed as an interface for the loading, modification and
storage of the coordinate transformation that is used by PampelMuse. The main
class is `CoordinateTransformation' and is designed to provide a convenient
way to transform from input to IFU coordinates. It is used as a super-class by
the `Sources'-class of PampelMuse.
Versions <0.4.0.0 also contained a second class, `Positions' for direct access
to IFU coordinates. The methods of this class have been transferred to the
`Sources'-class and the `Positions'-class deleted. 

CHANGELOG
---------
2012/02/28, v0.1.0.0
2012/03/01, v0.1.0.1, modified matchOnID-function so that it returns the
                      indices of the `parent'-array in the same order the
                      provided IDs are given.
2012/03/07, v0.2.0.0, major updates to the CoordinateTransformation-class so
                      that it makes use of the new ValueArray-structure.
                      Usage of `Positions'-class is deprecated, required methods
                      have been transfered to Sources-class in pampelmuse.core
2012/03/16, v0.4.0.0, some major updates to the `CoordinateTransformation'-
                      class; the` Positions'-class was deleted; some
                      documentation has been added.
2012/03/28, v0.4.1.0, some debugging in `calcTrans`-method.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.4.1.0"
import logging
import numpy as np, pyfits
from PyQt4 import QtCore

## required PampelMuse-packages:
from pampelmuse.functions.step_poly_fit import StepPolyFit
from pampelmuse.core.value_array import ValueArray


def myLstsqSigma(A, b, sigma=None):
    """
    The routine is a wrapper around the the numpy.linalg.lstsq-routine. It is
    designed as an extension that allows the user to also provide uncertainties
    together with the data when minimizing |Ax-b|.

    Parameters
    ----------
    A : array_like, shape (M, N)
        "Coefficient" matrix.
    b : array_like, shape (M,) or (M, K)
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    sigma : array_like, same shape as b, optional
        Uncertainties associated to the data provided in `b`.

    Returns
    -------
    x : ndarray, shape (N,) or (N, K)
        Least-squares solution.  The shape of `x` depends on the shape of
        `b`.
    residues : ndarray, shape (), (1,), or (K,)
        Sums of residues; squared Euclidean 2-norm for each column in
        ``b - a*x``.
        If the rank of `a` is < N or > M, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : ndarray, shape (min(M,N),)
        Singular values of `a`.
    """
    if not sigma: ## normal function call if no uncertainties provided
        return np.linalg.lstsq(A, b)
    else:         ## otherwise divide A and b by uncertainties
        b_ = b/sigma
        if b_.ndim > 1:
            results = [np.zeros((4,b_.shape[1]), dtype=np.float32)]
            for i in xrange(b_.shape[1]):
                results[0][i] = np.linalg.lstsq(A/sigma[:,i][:,np.newaxis], b_)
            return results
        else:
            return np.linalg.lstsq(A/sigma[:,np.newaxis], b_)




def matchOnID(parent, child):
    """
    Compare to 1dim. lists for common entries. For each element in `children`
    that is also part of the `parent`-list, the index of in the `parent'-list
    is returned.

    Parameters
    ----------
    parent : array_like, 1dim.
        The list that that is scanned
    child : array_like, 1dim.
        The list containing the entries that `parent' is scanned for.

    Returns
    -------
    indices : nd_array
        The index of every `child'-entry in the `parent' list (if found).
    """
    return np.nonzero((child[:,np.newaxis]-parent)==0)[1]



class CoordinateTransformation(object):
    """
    The class that stores information about the coordinate transformation that
    is used by PampelMuse. So far, the coordinates transformation is 
        x = A*u + B*v + C
        y = A*v - B*u + D
    With (x,y) being the coordinates in IFU pixels and (u,v) being the input
    (or reference) coordinates. A-D are the parameters of the transformation.
    """

    parameters = ["A", "B", "C", "D"]


    def __init__(self):
        """
        Intialize an instance of the `CoordinateTransformation`-class. The
        __init__-method does not take any input arguments, it just defines
        some properties that are needed later on.
        """
        self.ctData   = None  ## will contain the parameters of the transf.
        self.ctFree   = None  ## will contain flags about whether a polynomial
                              ## is a free parameter in the fit.
        self.ctPoly   = None  ## will contain flags about whether a polynomial
                              ## fit exists for a parameter.
        self.ctValues = None  ## will later be used as a ValueArray for
                              ## convenient access to ctData.
        self.npars    = len(self.parameters) ## no. of parameters
 


    def makeNewTrans(self, **kwargs):
        """
        Create a new coordinate transformation. One or more parameters of the
        transformation might be provided as keyword arguments:
        
        Parameters
        ----------
        kwargs : float or array_like
            Any parameter of the transformation, either as a constant value
            or including dispersion information. Parameters initially not
            provided might later be added using the `addParameter`-method.
        """
        logging.info("Initializing coordinate transformation.")

        self.backup = False

##         self.unsavedTrans = True

        self.initStructure() ## Initialize framework of the coord. transf.

        for name, value in kwargs.iteritems():
            self.addParameter(name, value)


        
    def calcTrans(self, inCoords, outCoords):
        """
        Calculate a new coordinate transformation using the provided input
        (reference) and output (IFU) coordinates. Uncertainties associated
        to the output coordinates might also be provided.

        Parameters
        ----------
        inCoords : structured array
            The array must contain at least 3 fields, named `id`, `x` and `y`,
            containing the IDs of the sources and their reference x- and y-
            coordinates
        outCoords :  structured array
            Again, the coordinate must contain the three fields `id`, `x` and
            `y`. Sources are matched to their counterparts in the `inCoords`-
            array based on the ID. If the `outCoords`-array contains fields
            named `x_err` and `y_err`, those are taken as the uncertainties
            associated with `x` and `y`.
        """
        logging.info("Determining coordinate transformation.")

        self.backup = False

        ## check if provided data is valid
        for name in ['id', 'x', 'y']:
            if not name in inCoords.dtype.names:
                logging.error(\
                    "Cannot find field '{0}' in input coords.".format(name))
                return
            if not name in outCoords.dtype.names:
                logging.error(\
                    "Cannot find field '{0}' in output coords.".format(name))
                return

        if outCoords["x"].shape != outCoords["y"].shape:
            logging.error("Output x- & y-coords. have different dimensions.")
            return
        else:
            disp = (outCoords["x"].ndim>1)
        if disp:
            ndisp = outCoords["x"].shape[1]
        else:
            ndisp = 1

        ## check if errors for out-coordinates are provided
        if "x_err" in outCoords.dtype.names \
                and "y_err" in outCoords.dtype.names:
            err = True
            logging.info("Found uncertainties of output coords.")
        else:
            err = False
            logging.info("No uncertainties of output coords. found.")


        inSel  = []
        outSel = []

        ## find matching coordinate pairs
        for i in xrange(inCoords.size):
            inID = inCoords['id'][i]
            if not inID in outCoords['id']:
                continue
            inSel.append(i)
            outSel.append(np.where(outCoords['id']==inID)[0][0])
    
        m = len(inSel) ## No of matching sources provided
        if m<2:
            logging.critical("Number of matching sources <2!")
            return
        elif m<4:
            logging.warning("Number of matching sources <4!")
        else:
            logging.info("Number of matching sources: {0}".format(m))


        ## create ndarrays for fitting
        inCoordArr  = np.zeros((2, m), dtype=np.float32)
        inCoordArr[0] = inCoords["x"][inSel]
        inCoordArr[1] = inCoords["y"][inSel]

        outCoordArr = np.zeros((2, m, ndisp), dtype=np.float32)
        outCoordArr[0] = outCoords["x"][outSel][:,np.newaxis]
        outCoordArr[1] = outCoords["y"][outSel][:,np.newaxis]

        if err:
            outSigmaArr = np.zeros(outCoordArr.shape, dtype=np.float32)
            outSigmaArr[0] = outCoords["x_err"][outSel]
            outSigmaArr[1] = outCoords["y_err"][outSel]
        else:
            outSigmaArr = None

        coordTrans = self.quickFit(inCoordArr, outCoordArr,
                                   outSigma=outSigmaArr)[0]

        self.makeNewTrans()

        for i in xrange(self.npars):
            self.addParameter(name=self.parameters[i], value=coordTrans[i])


        ## Only logging from this line
        logging.info(\
            "Determined coord. transf. for {0} image(s).".format(ndisp))

        logging.info("Best-fit parameters:")
        for i in xrange(len(self.parameters)):
            name = self.parameters[i]
            logging.info(\
                "   {0} = {1:8.3f}".format(name, self.ctData[0,i].mean()))

        logging.info("Offsets of individual sources:")
        logging.info("       ID     dx     dy")
        [x, y] = self.__call__(x=inCoordArr[0], y=inCoordArr[1])
        for i in xrange(m):
            J = inSel[i]
            K = outSel[i]

            logging.info("   {0:6d}  {1:+5.2f}  {2:+5.2f}".format(\
                    int(inCoords['id'][J]),
                    float((x[i]-outCoordArr[0,i]).mean(axis=-1)),
                    float((y[i]-outCoordArr[1,i]).mean(axis=-1))))



    def quickFit(self, inCoords, outCoords, outSigma=None):
        """
        A short-cut for calculating a new coordinate transformation, without
        any checking of the input.

        Note that the routine returns the output of the least-squares fit.
        This behavior is different from the `calcTrans`-method that
        initializes a new coordinate transformation.
        
        Parameters
        ----------
        inCoords : array_like, shape (2, N)
           The input coordinates (u and v) for a sample of N>2 sources.
        outCoords : array_like, shape (2, N) or (2, N, K)
           The output coordinates (x and y) for the same sample of sources
           as in `inCoords`. If `outCoords` is 3dimensional, the transf. is
           determined for the K coordinates.
        outSigma : array_like, shape (2, N) or (2, N, K), optional
           The uncertainties (if any) associated with the output coordrinates.
           Must be of same size as `outCoords` if provided.

        Returns
        -------
        The results of the least squares fit. Consult the documentation of
        myLstsqSigma for more information on the output.

        """
        m = inCoords[0].shape[0]
        M = 2*m

        A = np.ones((M,4), dtype=np.float32)
        A[:m,0] =  inCoords[0].flatten()
        A[m:,0] =  inCoords[1].flatten()
        A[:m,1] =  inCoords[1].flatten()
        A[m:,1] = -inCoords[0].flatten()
        A[m:,2] = 0.
        A[:m,3] = 0.

        sigma = None
        if outCoords[0].ndim>1:
            b = np.vstack(outCoords)
            ## to check: might be better to use np.hstack(np.squueze(outCoords))
            if outSigma != None: sigma = np.vstack(outSigma)
        else:
            b = np.hstack(outCoords)
            if outSigma != None: sigma = np.hstack(outSigma)

        return myLstsqSigma(A, b, sigma=sigma)



    def readTransHDU(self, HDU, backup=False):
        """
        Load an existing coordinate transfomation from a FITS BinTable-HDU.

        Parameters
        ----------
        HDU : a pyfits BinTableHDU-instance
            The HDU to load. It should at least contain the fields `name` and
            `value` for each parameter. Fields `freepar`, `isfitted` and
            `polyfit` are optional.
        backup : bool, optional
            Flag indicating whether existing data should be backed up if it
            is overwritten with new data.
        """
        if "EXTNAME" in HDU.header:
            logging.info("Initializing coord. transf. from HDU '{0}'.".format(\
                    HDU.header["EXTNAME"]))
        else:
            logging.info("Initializing coord. transf. from unnamed HDU.")


        ## check if required fields 'name' and 'value exist':
        for field in ["name", "value"]:
            if not field in HDU.data.names:
                logging.critical(\
                    "Cannot find field '{0}' in transf. HDU.".format(field))
                return

        self.backup = backup

        self.initStructure()

        ## read data from selected HDU
        freepar  = False
        isfitted = False
        polyfit  = None

        ## add the parameters to the coord. tranf. definition
        for i in xrange(HDU.data.size):
            if "freepar" in HDU.data.names:
                freepar=HDU.data.field("freepar")[i]
            if "isfitted" in HDU.data.names:
                isfitted=HDU.data.field("isfitted")[i]
            if "polyfit" in HDU.data.names:
                polyfit=HDU.data.field("polyfit")[i]
                
            self.addParameter(name=HDU.data.field("name")[i],
                              value=HDU.data.field("value")[i],
                              freepar=freepar,
                              isfitted=isfitted,
                              polyfit=polyfit)

##         self.unsavedTrans = False



    def initStructure(self):
        """
        The method is used to prepare the required arrays containing the
        properties of the PSF parameters. It should be called immediately
        after a PSF has been initialized and before any parameters are
        added. It is called automatically by methods `makeNewTrans`,
        `calcTrans` and `loadTransHDU`.
        """
        self.disp       = False
        self.lockSize   = False ## Cannot resize after call to `makeValues`
                                ## unless `delValues` is called
        self.missing    = [name for name in self.parameters]
        self.ctData = np.zeros((2,self.npars,1),  dtype=np.float32)
        self.ctFree = np.zeros((self.npars,),  dtype=np.bool)
        self.ctPoly = np.zeros((self.npars,),  dtype=np.bool)
 

        
    def addParameter(self, name, value, freepar=False, isfitted=False,
                     polyfit=None):
        """
        Add a parameter to an existing coord. transf. definition.

        Note that this routine cannot be used to update the values and size
        of an existing parameter.

        Parameters
        ----------
        name : string
            The name of the parameter to add, either `A`, `B`, C` or `D`.
        value : float or array_like
           The (initial) value of the parameter, either as a single float or 
           a 1dim. array.
        freepar : bool,  optional
           A flag indicating whether the parameter is free if the coordinate
           transformation is fitted.
        isfitted : bool, optional
           A flag indicating if a polynomial fit to the `value` exists. If
           True, `polyfit` must be specified.
        polyfit : array_like, optional
           The polynomial  fit to `value`. Only used if flag `isfitted` is
           set.
        """
        if not name in self.missing:
            logging.error("Cannot add parameter: '{0}' not missing.".format(\
                    name))
            return
        
        i = self.parameters.index(name)

        ## check if value has correct data shape, reshape self.ctData if needed 
        value = np.asarray(value)
        if value.size > self.ctData.shape[-1]:
            if self.disp:
                logging.warning("Differing data size of parameter '{0}'.")
                logging.warning("Transformation structure will be resized.")
            self.resizeStructure(size=value.size)
        elif value.size < self.ctData.shape[-1] and value.size > 1:
            logging.error("Cannot add parameter: '{0}' has invalid data size."\
                              .format(name))
            return

        ##  check if polyfit information is valid
        if isfitted and polyfit==None:
            logging.error("Cannot add parameter: '{0}'".format(name))
            logging.error(" No `polyfit' available, but `isfitted'-flag set.")
            return
        
        if polyfit != None:
            polyfit = np.asarray(polyfit)
            if polyfit.size != value.size:
                logging.error("Cannot add parameter: '{0}'".format(name))
                logging.error("Incompatible sizes of `value' and `polyfit'.")
                return
        
        self.ctData[0,i] = value
        self.ctFree[i]   = freepar
        self.ctPoly[i]   = isfitted
        if polyfit    !=  None: self.ctData[1,i]  = polyfit
##         if self.backup: self.value_init[name] = value

        self.missing.remove(name)

##         self.unsavedTrans = True

        logging.info("Added parameter '{0}' to transformation.".format(name))



    @property
    def verifyTrans(self):
        """
        Property to check whether the coord. transf.  definition is valie,
        i.e. whether all of its parameters have been provided.
        """
        if not "missing" in dir(self):
            return False

        if self.missing != []:
##             logging.warning("Some coord. transf. parameters not specified:")
##             for prm in self.missing:
##                 logging.warning("   Missing parameter '{0}'".format(prm))
            return False
        else:
            return True



    def info(self):
        """
        The method pretty-prints some information about the status of the
        coord. transf. definition out on the scree.
        """
        logging.info("Summary of coordinate transformation instance:")
##         logging.info(" Valid: {0}".format(self.verifyTrans))
        logging.info(" Parameter  defined?  free?  polyfit?")

        for i in xrange(self.npars):
            name = self.parameters[i]

            logString = " *{0:8s}".format(name)
            if not name in self.missing:
                logString += "      [x]"
            else:
                logstring += "      [ ]     [ ]       [ ]"
                logging.info(logString)
                continue

            if self.ctFree[i]: logString += "     [x]"
            else: logString += "     [ ]"

            if self.ctPoly[i]: logString += "       [x]"
            else: logString += "       [ ]"

            logging.info(logString)
                


    def provideParNames(self):
        """
        Short routine whose sole purpose is to provide a list of the names of
        all the parameters to the `InspectFitData`-routine of PampelMuse.

        Returns
        -------
        names :QtCore.QStringList
            The names of the parameters of the coord. transf.
        """
        return QtCore.QStringList(self.parameters)



    def resizeStructure(self, size):
        """
        The method can be used to resize the arrays containing the data of
        the individual parameters (the `ctData`-property). This is for example
        useful to add wavelength information to the coord. transf. definition.

        Note that this routine can only be called if the `ctValues`-property
        is not currently filled with data.

        Parameters
        ----------
        size : integer, >0
            The new size of the structure. It it exceeds the current size, the
            additional entries are appended to the `ctData`-property and filled
            with the values from the last existing slice.
        """
        if self.lockSize:
            logging.error("Cannot resize after calling `makeValues'.")
            logging.error("To resize, call `delValues' first.")
            return

        if self.ctData.shape[-1] == size:
            return

        if self.ctData.shape[-1] == 1 and size > 1:
            self.disp = True

        kMax = min(size, self.ctData.shape[-1])-1
        newData = np.zeros((self.ctData.shape)[:2]+(size,),  dtype=np.float32)
        newData[...,:kMax] = self.ctData[...,:kMax]
        if size > self.ctData.shape[-1]:
            newData[...,kMax:] = self.ctData[...,-1][:,:,np.newaxis]
        self.ctData = newData


##         if self.backup:
##             Needs to be implemented
        
##         self.unsavedTrans = True



    def makeValues(self):
        """
        The method views the `ctData`-property as a ValueArray, using the
        property `ctValues`. The idea behind this the ValueArray-structure is
        to combine the all the information about the parameters in a 2dim.
        array that
        i) redirects `__getitem__`-calls automatically to the right values,
           i.e the polynomial fit if available, the data  itself otherwise. 
        ii) handles `__setitem__`-calls such that only parameters whose
            `ctFree`-flag is True and whose `ctPoly`-flag is False are
            updated.
        """
        if not self.verifyTrans:
            logging.error(" PSF definition is not valid.")
            return

        self.ctValues = ValueArray(\
            self.ctData, free=self.ctFree, poly=self.ctPoly)

        self.lockSize = True ## notresizing possible if `ctValues` contains
                             ## data



    def delValues(self):
        """        
        Empty the `ctValues`-property (if it contains data). This might be
        necessary in case the data structure needs to  be resized after
        method `makeValues` has been called.
        """
        if not self.lockSize:
            return

        self.ctValues = None
        self.lockSize = False

        logging.warning("Deleted attribute `values', unlocked data size.")

    

    @property
    def Fit(self):
        """
        Property to check whether any of the parameters of the coordinate
        transformation are free parameters in a fit.
        Criterion for a free parameter: `ctFree`-flag is True, `ctPoly`-flag
                                        is False.
        """

        if not self.verifyTrans:
            return False
        
        return (self.ctFree&(self.ctPoly==False)).any()

            

    def __call__(self, x, y, index=Ellipsis):
        """
        Call the coordinate transformation with a set of input (reference)
        coordinates and return the IFU coordinates.

        Parameters
        ----------
        x :  float or array_like, with shape (N,)
           The input x-coordinate(s) of the sources.
        y :  float or array_like, with shape (N,)
           The input y-coordinate(s) of the sources. Must have same shape as
           `x`.
        index : integer, slice- or Ellipsis-object
           The indices of the coord. transf. that should be used to calculate
           the IFU-coordinates. If `index` is an integer, the returned coords.
           will be 1D. Otherwise, they will be 2D with shape (N, len(slice)).

        Returns
        -------   
        x_out : array_like, with shape (N,) or (N, len(slice))
            The IFU x-coordinates. The shape of the array will depend on the
            provided input.
        y_out : array_like, with shape (N,) or (N, len(slice))
            The IFU y-coordinates. The shape of the array will depend on the
            provided input.
        """

        ## index can be any valid slice definition:  an integer, a slice()-
        ## object or an Ellipsis-object
        
        if not self.verifyTrans:
            logging.error("Cannot determine pixel positions of sources:")
            logging.error(" No valid coord. transf. avalable.")
            return None

        if not self.lockSize:
            self.makeValues()

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.ndim != 1:
            logging.error("Provided x- & y-coords have invalid dimensions.")
        if x.shape != y.shape:
            logging.error("Length of provided x- & y-coords differ.")
            return None
        
        A = self.ctValues[0, index] 
        B = self.ctValues[1, index]
        C = self.ctValues[2, index]
        D = self.ctValues[3, index]

        xOut = A*x[:,np.newaxis] + B*y[:,np.newaxis] + C
        yOut = A*y[:,np.newaxis] - B*x[:,np.newaxis] + D

        return [xOut, yOut]



    def makeTransHDU(self,  includePolyFit=True):
        """
        Write the current coord. transf. definition to a FITS BinTableHDU.
        The table will contain fields named `name`, `value`, `freepar`,
        [`isfitted`, `polyfit`]

        Parameters
        ----------
        includePolyFit : bool, optional
            If the `includePolyFit`-flag is set, the returned table will
            contain the fields named `isfitted` and `polyfit` containing
            information about  the polynomial fit (if any).

        Returns
        -------
        outHDU : pyfits BinTableHDU-instance
            A  FITS binary table named `POSPARS` and containing all the
            information about the coord. transf.
        """
##         if self.unsavedTrans:
##             logging.info("Saving updated coord-trans. to FITS BinTable-HDU.")
##         else:
##             logging.warning(\
##                 "Saving unchanged coord-trans. to FITS BinTable-HDU.")


        if not self.disp: 
            dataFmt = "E"
        else:
            dataFmt = "{0}E".format(self.ctData.shape[-1])

        outColumns = [
            pyfits.Column(name="name", format="10A", array=self.parameters),
            pyfits.Column(name="value", format=dataFmt,
                          array=np.squeeze(self.ctData[0])),
            pyfits.Column(name="freepar", format="L", array=self.ctFree)
            ]

        if includePolyFit:
            outColumns.extend([\
                    pyfits.Column(\
                        name="isfitted", format="L", array=self.ctPoly),
                    pyfits.Column(\
                        name="polyfit", format=dataFmt,
                        array=np.squeeze(self.ctData[1]))])
            
        outHDU = pyfits.new_table(outColumns, header=pyfits.Header(\
                cards=[pyfits.Card(key="EXTNAME", value="POSPARS")]))

        outHDU.add_checksum()

##         self.unsavedTrans = False

        return outHDU
