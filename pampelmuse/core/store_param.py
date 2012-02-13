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
2012/01/30, v0.3.0.0, updated the code to produce PRM-files in agreement with
                      the new structure.
2012/02/03, v0.3.0.1, changed the documentation a bit.
2012/02/12, v0.3.1.2, few bugfixes in method `storeSourceCat': instead of
                      self.selectedSources, selectedSources was used in
                      several lines; fixed that code crashed occasionally since
                      some sources had non-existing primary components.
                      Made usage of method `storePSF' more convenient in case
                      that PSF is not wavelength-dependent.
2012/02/13, v0.3.1.3, added default values for parameters `wlStart' and
                      `wlStep' to make them optional rather than mandatory.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.3.1.3"
import datetime, logging, pyfits, os
import numpy as np



class StoreParam:
    """
    The class is used to prepare and store the prm.fits-file that contains
    the input for the PampelMuse-FITCUBE routine. It is usually called by
    the INITFIT-routine.
    """
    def __init__(self, sourceList, specPix, wlStart=1, wlStep=1):
        """
        Initialize an instance of the `StoreParam'-class.

        Parameters
        ----------
        sourceList : numpy_record_array
            The catalog containing the sources in the field. The structured
            array must contain fields named 'id', 'x', 'y' and 'mag'. The sky
            (if any) should have ID -1.
        specpix : int
            The no. of pixels along the dispersion axis.
        wlStart : float, optional
            The wavelength of the first image of the datacube.
        wlStep : float, optional
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

                    newPrim = self.selectedSources['primary'][j]
                    if np.where(self.selectedSources['id']==newPrim)[0].size>0:
                        self.selectedSources['primary'][i]\
                            = self.selectedSources['primary'][j]
                else:
                    logging.warn(\
                        "Source #{0} is secondary,".format(secondary)+\
                            "but primary [#{0}] is missing".format(primary))
                    self.selectedSources['status'][i] = 2
                    self.selectedSources['primary'][i] = secondary


        ## match sources in the two lists
        self.selector = [np.where(self.sourceList['id']==i)[0][0] \
                             for i in self.selectedSources['id']]

        ## prepare column 'SPECROW':
        ## get ID of 'primary' component for each source with 'status'=1
        priCompID = self.selectedSources['primary'][\
            self.selectedSources['status']==1]
        ## get the positions of the 'primary' components
        priCompPos = [np.where(self.selectedSources['id']==i)[0][0] \
                         for i in priCompID]
        ## assign a new row to each independent source
        self.specrow = np.cumsum(self.selectedSources['status']>1) - 1
        ## assign an existing row to each dependent source
        self.specrow[self.selectedSources['status']==1] \
            = self.specrow[priCompPos]
        ## assign common row to all unresolved sources
        if (self.selectedSources['status']==0).any():
            self.specrow[self.selectedSources['status']==0] \
                = self.specrow.max()+1

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

        ## to indicate prm-file is compliant with new structure:
        self.sourceCatHDU.header.update('PRMSTRUC', 2)
        self.HDUList.append(self.sourceCatHDU)

        self.includeUnres = (selectedSources['status']==0).any()


    def  storeSourceSpec(self, inputSpectra, magZ=None, spatPix=None):
        """
        create a FITS image to hold all the spectra that are used in
        the analysis.

        Note: Older versions used the set of provided spectra to infer the
              spectra of sources without own spectrum. Since version 0.2.1.1
              this behavior was changed.

        Parameters
        ----------
        inputSpectra : numpy_record_array
            A catalog of sources whose spectra have been measured, e.g via
            aperture spectrophotometry. It can also be an empty array in which
            case all spectra are assigned a continuum based on their magnitude.
            If not empty, it must contain fields named 'id' and 'spectrum'.
        magZ : float ## , optional
            The magnitude zeropoint of the observation, defined here as the
            magnitude of a source whose spectrum has a mean flux of 1.
##             If  none is given, it is calculated from the provided spectra.
##        includeSky : boolean, optional
##            Whether or not a sky spectrum should be included.  If True, the
##            routine searches the catalog `InputSpectra' for a source with ID
##            -1. If none is found, the sky spectrum will be zero. DEPRECATED.
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

        if magZ == None:
            logging.warning("No magnitude zeropoint provided.")
            logging.warning("Source spectra will be empty unless provided.")
            self.priHDU.header.update("MAGZ", "None",
                                      "Estimated magnitude zeropoint")
        else:
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

                if magZ == None:
                    continue

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
            if spatPix != None and magZ != None:
                unresFlux /= spatPix
                specData[-1] = unresFlux ## *meanSpectrum
            else:
                logging.warning("Cannot estimate spectrum of unresolved stars.")
                logging.warning("Either `magZ' or `spatPix' is undefined.")

        ##  create HDUs to hold the spectra
        self.addImaHDU(data=specData, extname='SPECTRA', crpix1=1,
                       crval1=self.wlStart, cdelt1=self.wlStep)
            


    def storePosPars(self, posPars, posFit=2):
        """
        Create a FITS-table that contains the parameters of the coordinate
        transformation (as a function of wavelength).

        Parameters
        ----------
        posPars :  numpy_record_array
            The structured array must contain fields (named e.g. 'A', 'B',
            'C', 'D') with the parameters of the transformation.
        posFit : integer
            `posFit' indicates how the source positions are fitted. Check
            the PampelMuse-README for details.
        """

        logging.info("Creating Table HDU to store paramaters of coord. trans.")

        nPars = len(posPars.dtype)
        nDim  = int(posPars.size)

        outData = pyfits.new_table([\
                pyfits.Column(name="name", format="1A",
                              array=np.zeros((nPars,))),
                pyfits.Column(name="value", format="{0}E".format(nDim),
                              array=np.zeros((nPars, nDim))),
                pyfits.Column(name="freepar", format="L",
                              array=np.zeros((nPars,))),
                pyfits.Column(name="isfitted", format="L",
                              array=np.zeros((nPars,))),
                pyfits.Column(name="polyfit", format="{0}E".format(nDim),
                              array=np.zeros((nPars, nDim)))
                ])

        i = 0
        for name in posPars.dtype.names:
            outData.data.field('name')[i] = name
            outData.data.field('value')[i] = posPars[name]
            if name in ["C", "D"] and posFit > 0:
                outData.data.field('freepar')[i] = True
            if name in ['A', 'B'] and posFit > 1:
                outData.data.field('freepar')[i] = True
            i += 1

        outData.header.update("EXTNAME", "POSPARS")

        self.HDUList.append(outData)



    def storePSF(self, psfType, psfPars, psfFit):
        """
        Create a FITS-table that contains the parameters of the point-spread
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
        psfFit : array_like
            A flag indicating whether a parameter is fitted. Must be of same
            length as `psfPars'.
        """

        logging.info("Creating Table HDU to store paramaters of PSF.")
        
        nPars = len(psfPars.dtype.names)
        nDim  = int(psfPars.size)

        if nDim == 1:
            vFormat = "E"
            vData   = np.zeros((nPars,))
        else:
            vFormat = "{0}E".format(nDim)
            vData   = np.zeros((nPars, nDim))

        outData = pyfits.new_table([\
                pyfits.Column(name="name", format="10A",
                              array=np.zeros((nPars,))),
                pyfits.Column(name="value", format=vFormat, array=vData),
                pyfits.Column(name="freepar", format="L",
                              array=np.zeros((nPars,))),
                pyfits.Column(name="isfitted", format="L",
                              array=np.zeros((nPars,))),
                pyfits.Column(name="polyfit", format=vFormat, array=vData)
                ])

        i = 0
        for name in psfPars.dtype.names:
            outData.data.field('name')[i] = name
            outData.data.field('value')[i] = psfPars[name]
            outData.data.field('freepar')[i] = psfFit[i]
            i += 1

        outData.header.update("EXTNAME", "PSFPARS")
        outData.header.update("PSFTYPE", psfType)

        self.HDUList.append(outData)


        
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
            if os.path.exists(filename):
                os.remove(filename)
                logging.info("Existing PRM-file has been overwritten.")
        outHDU.writeto(filename)
        


