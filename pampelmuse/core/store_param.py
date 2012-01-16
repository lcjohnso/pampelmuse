"""
STORE_PARAM
===========
Routine is designed to write the results of the INITFIT-routine to a
file. The output will be a multi-extension FITS-file containing the
complete list of considered sources, the independent spectra and the
parameters of the coordinate transformation.  

CHANGELOG
---------
2011/09/06, v0.0.0.0
2011/09/09, v0.0.0.1, added a proper naming scheme for the individual
                      extensions of the created FITS-file.
                      changed column names in `SOURCES'-table to lowercase.
2011/09/09, v0.1.0.0, spectra of sky and unressolved component are now
                      written into separate HDUs.
2011/09/19, v0.1.0.1, modifed how the individual HDUs are created.
2011/09/20, v0.1.1.1, fixed bug that containing spectra was generated with
                      wrong shape
2011/09/26, v0.2.0.0, instead of storing sky spectrum and `unresolved'
                      spectrum in dedicated HDUs, they are stored in the
                      same HDUs as the `resolved' spectra now. The sky must
                      be included in the `sourceList'-parameter of __init__
2011/10/27, v0.2.1.0, fixed bug that code crashed if no unresolved component
                      was included.
2011/11/29, v0.2.1.1, removed option to perform photometric calibration. The
                      magnitude zeropoint must now be provided. Also, the 
                      spectra of sources without aperture spectrophotometry
                      are no longer infered using other spectra but are set
                      to a constant value.
                      Logging is included.
2011/12/05, v0.2.1.2, added a `method to store an image HDU containing the
                      PSF parameters as a function of wavelength.
2012/01/16, v0.2.2.2, fixed bug that the code crashed if a secondary component
                      hat no primary counterpart. Instead now the status of the
                      secondary is increased and a warning message is logged.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.2.2.2"
import datetime, logging, pyfits, os
import numpy as np



class StoreParam:
    """
    The class is used to prepare and store the prm.fits-file that contains
    the input for the PampelMuse-FITCUBE routine. It is usually called by
    the INITFIT-routine.
    """
    def __init__(self, sourceList, wlStart, wlStep, specPix):
        """
        Initialize an instance of the `StoreParam'-class.

        Parameters
        ----------
        sourceList : numpy_record_array
            The catalog containing the sources in the field. The structured
            array must contain fields named 'id', 'x', 'y' and 'mag'. The sky
            (if any) should have ID -1.
        wlStart : float
            The wavelength of the first image of the datacube.
        wlStep : float
            The dispersion of the datacube.
        """
        self.sourceList = sourceList
        self.wlStart    = wlStart
        self.wlStep     = wlStep
        self.specPix    = specPix

        self.ut        = datetime.datetime.utcnow()
        self.timestamp = self.ut.strftime("%Y-%m-%dT%H:%M:%SZ")

        logging.info("Creating primary HDU.")
        self.priHDU = pyfits.PrimaryHDU()
        self.priHDU.header.update("CREATIME", self.timestamp,
                                  "Date and time this file was created")
        self.HDUList = [self.priHDU]
        

        
    def storeSourceCat(self, selectedSources):
        """
        Create a FITS-table containing the all sources that will be
        considered during the fit.
        The FITS-table will contain the columns 'ID', 'X', 'Y', 'MAG'
        'PRIMARY', 'SPECROW'. 'SPECROW' gives the row of the source
        spectrum in the arrray containing all spectra.

        Parameters
        ----------
        selectedSources : numpy_record_array
            The catalog of all sources that should be written to the FITS-
            file. It must contain fields named 'id', 'status' and 'primary'.
            'status' should be an integer in the range [-1,3] and defines
            how the source is used:  4 - sky spectrum
                                     3 - resolved, own spectrum
                                     2 - resolved, inferred spectrum
                                     1 - resolved, but next to brighter
                                         source
                                     0 - unresolved
                                    -1 - excluded
            'primary' should contain the ID of the brightest source of a
            fitted component. If 'status'=1, 'primary' is the ID of the
            brighter source, if 'status'=0, 'primary'=-1, else 'primary'
            is identical with the source ID.
        """
        self.selectedSources = selectedSources

        logging.info("Creating catalog of all sources to be included in fit.")
        ## make sure no source with status=1 has dependent primary component
        for i in xrange(self.selectedSources.size):
            if self.selectedSources['status'][i] == 1:
                secondary = self.selectedSources['id'][i]
                primary   = self.selectedSources['primary'][i]

                prim = np.where(self.selectedSources['id']==primary)[0]
                if prim.size > 0:
                    j = prim[0]
                    self.selectedSources['primary'][i]\
                        = self.selectedSources['primary'][j]
                else:
                    logging.warn(\
                        "Source #{0} is secondary,".format(secondary)+\
                            "but primary [#{0}] is missing".format(primary))
                    self.selectedSources['status'][i] = 2

        ## match sources in the two lists
        self.selector = [np.where(self.sourceList['id']==i)[0][0] \
                             for i in selectedSources['id']]

        ## prepare column 'SPECROW':
        ## get ID of 'primary' component for each source with 'status'=1
        priCompID  = selectedSources['primary'][selectedSources['status']==1]
        ## get the positions of the 'primary' components 
        priCompPos = [np.where(selectedSources['id']==i)[0][0] \
                        for i in priCompID]
        ## assign a new row to each independent source
        self.specrow = np.cumsum(selectedSources['status']>1) - 1
        ## assign an existing row to each dependent source
        self.specrow[selectedSources['status']==1] = self.specrow[priCompPos]
        ## assign common row to all unresolved sources
        if (selectedSources['status']==0).any():
            self.specrow[selectedSources['status']==0] = self.specrow.max()+1

        self.sourceCat = pyfits.ColDefs(\
            [pyfits.Column(name='id', format='J',
                           array=selectedSources['id']),
             pyfits.Column(name='x', format='E',
                           array=self.sourceList['x'][self.selector]),
             pyfits.Column(name='y', format='E',
                           array=self.sourceList['y'][self.selector]),
             pyfits.Column(name='mag', format='E',
                           array=self.sourceList['mag'][self.selector]),
             pyfits.Column(name='status', format='I',
                           array=selectedSources['status']),
             pyfits.Column(name='primary', format='J',
                           array=selectedSources['primary']),
             pyfits.Column(name='specrow', format='J',
                           array=self.specrow)
             ]
            )
        
        self.sourceCatHDU = pyfits.new_table(self.sourceCat)
        self.sourceCatHDU.header.update('EXTNAME', 'SOURCES')
        self.HDUList.append(self.sourceCatHDU)

        self.includeUnres = (selectedSources['status']==0).any()


    def  storeSourceSpec(self, inputSpectra, magZ, spatPix=256):
        """
        create a FITS image to hold all the spectra that are used in
        the analysis.

        Parameters
        ----------
        inputSpectra : numpy_record_array
            A catalog of sources whose spectra have been measured, e.g via
            aperture spectrophotometry. The structured array must contain
            fields named 'id' and 'spectrum'.
        magZ : float, optional
            The magnitude zeropoint of the observation, defined here as the
            magnitude of a source whose spectrum has a mean flux of 1.
##             If  none is given, it is calculated from the provided spectra.
##        includeSky : boolean, optional
##            Whether or not a sky spectrum should be included.  If True, the
##            routine searches the catalog `InputSpectra' for a source with ID
##            -1.
        spatPix : integer, optional
            The number of spatial pixels in the datacube. This number is
            used  to scale the spectrum of the unresolved component (if
            it exists)
        """
        
        logging.info("Creating Image HDU that will contain spectra.")

        ## create array to hold spectra
        Nspec=self.specrow.max()+1
        specData = np.zeros((Nspec, self.specPix), dtype='float32')

##         ## determine magnitude zeropoint (unless given)
##         ownSpecID = inputSpectra['id'][inputSpectra['id']>-1] ## exclude sky
##         
##         ii = [np.where(inputSpectra['id']==i)[0][0] for i in ownSpecID]
## 
##         if not magZ:
## 
##             jj =[np.where(self.sourceList['id']==i)[0][0] for i in ownSpecID]
## 
##             meanFlux = np.mean(inputSpectra['spectrum'][ii,:], axis=1)
##             selMags  = self.sourceList['mag'][jj]
## 
##             magZ = selMags + 2.5*np.log10(meanFlux)
##             magZmean = magZ.mean()
##         else:
##             magZmean = magZ

        self.priHDU.header.update("MAGZ", magZ,
                                  "Estimated magnitude zeropoint")

##         ## calculate mean of provided source spectra
##         meanSpectrum = inputSpectra['spectrum'][ii].mean(axis=0)
##         meanSpectrum /= meanSpectrum.mean()

        if self.includeUnres:
            unresFlux = 0.

        ## loop over all sources and fill array 
        for i in xrange(self.selectedSources.size):
            
            ## source has own spectrum
            if self.selectedSources['status'][i] >= 3:

                if not self.selectedSources['id'][i] in inputSpectra['id']:
                    continue

                logging.debug("   Source #{0} has spectrum.".format(\
                        self.selectedSources['id'][i]))

                jj = np.where(inputSpectra['id']\
                                  ==self.selectedSources['id'][i])[0][0]
                specData[self.specrow[i]] = inputSpectra['spectrum'][jj]
            
            ## spectrum must be inferred
            elif self.selectedSources['status'][i] >= 0:

                mag_i = self.sourceList['mag'][self.selector][i]
                meanFlux = 10**(-0.4*(mag_i-magZ))

                if self.selectedSources['status'][i] in [1,2]:
                    logging.debug("   Source #{0} has no spectrum.".format(\
                            self.selectedSources['id'][i]))

                    specData[self.specrow[i]] += meanFlux ## *meanSpectrum
                else:
                    logging.debug("   Source #{0} in unresolved comp.".format(\
                            self.selectedSources['id'][i]))

                    unresFlux += 10**(-0.4*(mag_i-magZ))


        if self.includeUnres:
            unresFlux /= spatPix
            specData[-1] = unresFlux ## *meanSpectrum

        ##  create HDUs to hold the spectra
        self.addImaHDU(data=specData, extname='SPECTRA', crpix1=1,
                       crval1=self.wlStart, cdelt1=self.wlStep)
            


    def storePosPars(self, pospars):
        """
        Create a FITS-image that contains the parameters of the coordinate
        transformation (as a function of wavelength).

        Parameters
        ----------
        pospars :  numpy_record_array
            The structured array must contain 4 fields (named 'A', 'B', 'C',
            'D') with the parameters of the transformation.
        """

        logging.info("Creating Image HDU to store paramaters of coord. trans.")

        outData = np.zeros((4,pospars.size), dtype='float32')
        outData[0] = pospars['A']
        outData[1] = pospars['B']
        outData[2] = pospars['C']
        outData[3] = pospars['D']

        self.addImaHDU(data=outData, extname='POSPARS', crpix1=1,
                       crval1=self.wlStart, cdelt1=self.wlStep)



    def storePSF(self, psfType, psfPars):
        """
        Create a FITS-image that contains the parameters of the point-spread
        function(as a function of wavelength).

        Parameters
        ----------
        psfType : string
            The analytical model used to model the PSF.
        psfPars :  numpy_record_array
            For each PSF parameter, the structured array must contain one
            field with the parameters name. Note that no validation of the
            input is performed at this point. It is up to the user to provide
            a valid set of input parameters for a certain PSF type.
        """

        logging.info("Creating Image HDU to store paramaters of PSF.")
        
        nPars = len(psfPars.dtype.names)

        psfArgs = {"psfType":psfType}

        outData = np.zeros((nPars, psfPars.size), dtype='float32')
        for i in xrange(nPars):
            outData[i] = psfPars[psfPars.dtype.names[i]]
            psfArgs["psfPar{0}".format(i+1)] = psfPars.dtype.names[i]

        self.addImaHDU(data=outData, extname='PSFPARS', crpix1=1,
                       crval1=self.wlStart, cdelt1=self.wlStep, **psfArgs)


        
    def addImaHDU(self, data, **kwargs):

        newHDU = pyfits.ImageHDU(data)
        for key, value in kwargs.iteritems():
            newHDU.header.update(str.upper(key), value)
        self.HDUList.append(newHDU)



    def writeto(self, filename, clobber=False):
        """ 
        Collect the created HDUs and write them to a common FITS-file.
        
        Parameters
        ----------
        filename : string
            The name under which the file will be stored.
        clobber : boolean, optional
            Flag indicating whether an existing file will be overwritten.
        """
        
        logging.info("Saving results in `{0}'".format(filename))

        outHDU = pyfits.HDUList(self.HDUList)
        
        if clobber:
            logging.info("Existing file has been overwritten.")
            if os.path.exists(filename):  os.remove(filename)
        outHDU.writeto(filename)
        


