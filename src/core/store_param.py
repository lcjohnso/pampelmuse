"""
STORE_PARAM
===========
Routine is designed to write the results of the INITFIT-routine to a
file. The output will be a multi-extension FITS-file containing the
complete list of considered sources, the independent spectra and the
parameters of the coordinate transformation.  

CHANGELOG
---------
2011/09/06, v0.0.0.0.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.0.0"
import datetime, pyfits, os
import numpy as np



class StoreParam:
    """
    The class is used to prepare and store the prm.fits-file that contains
    the input for the PampelMuse-FITCUBE routine. It is usually called by
    the INITFIT-routine.
    """
    def __init__(self, sourceList, wlStart, wlStep):
        """
        Initialize an instance of the `StoreParam'-class.

        Parameters
        ----------
        sourceList : numpy_record_array
            The catalog containing the sources in the field. The structured
            array must contain fields named 'id', 'x', 'y' and 'mag'.
        wlStart : float
            The wavelength of the first image of the datacube.
        wlStep : float
            The dispersion of the datacube.
        """
        self.sourceList = sourceList
        self.wlStart    = wlStart
        self.wlStep     = wlStep

        self.ut        = datetime.datetime.utcnow()
        self.timestamp = self.ut.strftime("%Y-%m-%dT%H:%M:%SZ")

        self.priHDU = pyfits.PrimaryHDU()
        self.priHDU.header.update("CREATIME", self.timestamp,
                                  "Date and time this file was created")

        
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
            how the source is used:  3 - resolved, own spectrum
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
        self.specrow[selectedSources['status']==0] = self.specrow.max()+1

        self.sourceCat = pyfits.ColDefs(\
            [pyfits.Column(name='ID', format='J',
                           array=selectedSources['id']),
             pyfits.Column(name='X', format='E',
                           array=self.sourceList['x'][self.selector]),
             pyfits.Column(name='Y', format='E',
                           array=self.sourceList['y'][self.selector]),
             pyfits.Column(name='MAG', format='E',
                           array=self.sourceList['mag'][self.selector]),
             pyfits.Column(name='STATUS', format='I',
                           array=selectedSources['status']),
             pyfits.Column(name='PRIMARY', format='J',
                           array=selectedSources['primary']),
             pyfits.Column(name='SPECROW', format='J',
                           array=self.specrow)
             ]
            )
        
        self.sourceCatHDU = pyfits.new_table(self.sourceCat)

        self.includeUnres = (selectedSources['status']==0).any()


    def  storeSourceSpec(self, inputSpectra, magZ=None, includeSky=True,
                         spatPix=256):
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
            magnitude of a source  whose spectrum has a mean flux of 1. If
            none is given, it is calculated from the provided spectra.
        includeSky : boolean, optional
            Whether or not a sky spectrum should be included.  If True, the
            routine searches the catalog `InputSpectra' for a source with ID
            -1.
        spatPix : integer, optional
            The number of spatial pixels in the datacube. This number is
            used  to scale the spectrum of the unresolved component (if
            it exists)
        """

        ## create array to hold spectra
        Nspec=self.specrow.max()+includeSky+self.includeUnres
        specData = np.zeros((Nspec, inputSpectra['spectrum'].shape[1]),
                            dtype='float32')

        ## determine magnitude zeropoint (unless given)
        ownSpecID = inputSpectra['id'][inputSpectra['id']>-1]
        
        ii = [np.where(inputSpectra['id']==i)[0][0] for i in ownSpecID]

        if not magZ:

            jj = [np.where(self.sourceList['id']==i)[0][0] for i in ownSpecID]

            meanFlux = np.mean(inputSpectra['spectrum'][ii,:], axis=1)
            selMags  = self.sourceList['mag'][jj]

            magZ = selMags + 2.5*np.log10(meanFlux)
            magZmean = magZ.mean()
        else:
            magZmean = magZ

        self.priHDU.header.update("MAGZ", magZmean,
                                  "Estimated magnitude zeropoint")

        ##  calculate mean of provided source spectra
        meanSpectrum = inputSpectra['spectrum'][ii].mean(axis=0)
        meanSpectrum /= meanSpectrum.mean()

        if self.includeUnres:
            unresFlux = 0.

        ## loop over all sources and fill array 
        for i in xrange(self.selectedSources.size):

            ## source has own spectrum
            if self.selectedSources['status'][i] == 3:

                jj = np.where(inputSpectra['id']\
                                  ==self.selectedSources['id'][i])[0][0]
                specData[self.specrow[i]] = inputSpectra['spectrum'][jj]
            
            ## spectrum must be inferred
            elif self.selectedSources['status'][i] >= 0:
                mag_i = self.sourceList['mag'][self.selector][i]
                meanFlux = 10**(-0.4*(mag_i-magZmean))

                if self.selectedSources['status'][i] in [1,2]:
                    specData[self.specrow[i]] += meanFlux*meanSpectrum
                else:
                    unresFlux += 10**(-0.4*(mag_i-magZmean))

        
        if self.includeUnres:
            unresFlux /= spatPix
            specData[-(1+includeSky)] = unresFlux*meanSpectrum

        if includeSky:
            skySpecPos = np.where(inputSpectra['id']==-1)[0][0]
            specData[-1] = inputSpectra['spectrum'][skySpecPos]
            
        self.sourceSpecHDU = pyfits.ImageHDU(specData)



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
        outData = np.zeros((4,pospars.size), dtype='float32')
        outData[0] = pospars['A']
        outData[1] = pospars['B']
        outData[2] = pospars['C']
        outData[3] = pospars['D']

        self.posParsHDU = pyfits.ImageHDU(outData)
        self.posParsHDU.header.update('CRPIX1', 1)
        self.posParsHDU.header.update('CRVAL1', self.wlStart)
        self.posParsHDU.header.update('CDELT1', self.wlStep)



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
        outHDU = pyfits.HDUList([self.priHDU, self.sourceCatHDU, 
                                 self.sourceSpecHDU, self.posParsHDU])
        
        if clobber:
            if os.path.exists(filename):  os.remove(filename)
        outHDU.writeto(filename)
        


