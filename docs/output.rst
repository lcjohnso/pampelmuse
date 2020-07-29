

Output files
============

.. _output-prm:

The PRM-file
------------


.. _output-psf:

The PSF-file
------------

For each PSF star in the analysis (i.e. each star that has status 3 in the :ref:`PRM-file <output-prm>`), the FITS
file will contain four HDUs with the following content.

* The values of the fitted PSF parameters in each layer that was analysed. The information is stored as a FITS-table,
  containing one column per fitted parameter and the wavelength information in the different rows. The names of these
  extensions are `PM_<source ID>`, where `<source ID>` is the ID of the star in extension `SOURCES` of the PRM-file.
* The radial profiles of the PSF stars in the IFS data as a function of wavelength. This information is stored as a
  two-dimensional array, containing the profile for one layer per column. The (approximate) wavelengths of the layers
  can be reconstructed from the `CRVAL1` and `CDELT1` keys in the header. The names of these extensions are
  `PR_<source ID>`, where again `<source ID>` is the ID of the star in extension `SOURCES` of the PRM-file. Note that
  the common radii of the individual points of all profiles (in spaxels) are provided as the data of the primary
  extension of the FITS-file.
* The two-dimensional profile of each PSF star extracted from the IFS data as a function of wavelength is stored as a
  mini data cube. Again, (approximate) dispersion information is encoded in the `CRVAL1` and `CDELT1` keys in the
  header. The names of these extensions are `DAT_<source ID>`.
* The two-dimensional residual maps of the PSF fits to every star as a function of wavelength are also stored as mini
  data cubes. The format is just the same as for the two dimensional profiles and the extensions are named
  `RES_<source ID>`.
