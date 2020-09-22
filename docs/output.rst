

Output files
============

.. _output-prm:

The PRM-file
------------

.. table:: The different extensions of the PRM-file and their respective contents.
    :name: table-prm-info
    :width: 100%

    +-----------+--------------+---------------------------------------------------+
    | EXTNAME   | HDU Type     | Purpose                                           |
    +===========+==============+===================================================+
    | PRIMARY   | Primary      | Header contains the configuration of PampelMuse.  |
    +-----------+--------------+---------------------------------------------------+
    | SOURCES   | Binary Table | The source catalogue, contains basic information  |
    |           |              | about every source used in the extraction.        |
    +-----------+--------------+---------------------------------------------------+
    | SPECTRA   | Image        | Contains the extracted spectra and an estimate    |
    |           |              | of the uncertainties.                             |
    +-----------+--------------+---------------------------------------------------+
    | POSPARS   | Binary Table | Contains the coordinate transformation from       |
    |           |              | reference to IFS coordinates.                     |
    +-----------+--------------+---------------------------------------------------+
    | POSITIONS | Binary Table | Contains the parameters of the polynomial fits    |
    |           |              | to the IFS coordinates.                           |
    +-----------+--------------+---------------------------------------------------+
    | PSFPARS   | Binary Table | Contains the parameters of the fitted PSF.        |
    +-----------+--------------+---------------------------------------------------+



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


.. _output-spectra:

The extracted spectra
---------------------

When using :ref:`GETSPECTRA <getspectra>` to write the set of extracted spectra to disk, the code will create one FITS
file per spectrum, using the following naming convention::

    "{prefix:s}id{source_id:09d}jd{jd_int:d}p{jd_frac:04d}f{flag:03d}.fits".format(
        prefix="ngc1234",  # The value of the `getspectra|prefix` configuration key
        source_id=2345,  # The ID of the source in the reference catalogue
        jd_int=int(2459110.1234),  # The integer part of the Julian Date of the observation
        jd_frac=int(1e4*np.mod(2459110.1234, 1)),  # The fractional part of the Julian Date of the observation
        flag=0  # The quality flag of the spectrum
        )

Hence each filename is composed of a common prefix that can be specified in the user configuration file, the ID of the
source in the reference catalogue, the Julian Date of the observation, and a quality flag. The quality flag is
generated via the following bitmask:

.. table:: Spectrum properties considered in quality flags.
    :name: table-quality-flags
    :widths: auto

    =====  ====
    Value  Meaning
    =====  ====
     1     More than one source contribute to the spectrum.
     2     The average S/N of the spectrum is :math:`<10`.
     4     The mean flux of the spectrum is negative.
     8     The centroid of the source is outside the field of view of the data cube.
    =====  ====

.. note::

    In case the configuration key `getspectra|skipflag` is set to :code:`True`, the part :code:`"f{flag:03d}"` is
    omitted from the filenames.

.. note::

    Spectra of sky components are stored as well. Their naming convention is the same as for the stellar spectra, only
    that the part :code:`"id{source_id:09d}"` is replaced with :code:`"sky{sky_id:09d}"`, where :code:`sky_id` is the
    (internally assigned) index of the sky component.

FITS header keywords
^^^^^^^^^^^^^^^^^^^^

Each spectrum written to disk will contain a number of keywords in the primary FITS header, providing some information
about its properties and quality. An overview is given below.

OBJECT :code:`str`
    The object name provided in the FITS header of the data cube.

TELESCOP :code:`str`
    The telescope name provided in  the FITS header of the data cube.

INSTRUME :code:`str`
    The instrument name provided in the FITS header of the data cube.

DATE-OBS :code:`str`
    The date and time of the observation.

JUL-DATE :code:`float`
    The Julian Date of the observation.

MJD-OBS :code:`float`
    The Modified Julian Date of the observation.

EXPTIME :code:`float`
    The exposure time in seconds.

RA :code:`float`
    The right ascension of the source.

DEC :code:`float`
    The declination of the source.

OBJNAME :code:`str`
    An internally generated object name of the source.

HIERARCH PAMPELMUSE SEEING :code:`float`
    The seeing estimated from the PSF parameters, in spaxels.

HIERARCH STAR ID :code:`int`
    The source ID in reference catalogue.

HIERARCH STAR RA :code:`float`
    The right ascension of the source.

HIERARCH STAR DEC :code:`float`
    The declination of the source.

HIERARCH STAR MAG :code:`float`
    The input magnitude of the source as provided in the reference catalogue.

HIERARCH SPECTRUM MULTISTAR :code:`bool`
    A flag indicating if multiple sources contribute to the spectrum.

HIERARCH SPECTRUM XCUBE :code:`float`
    The source position along the x-axis of the cube in spaxels.

HIERARCH SPECTRUM YCUBE :code:`float`
    The source position along the y-axis of the cube in spaxels.

HIERARCH SPECTRUM INFIELD :code:`bool`
    A flag indicating if the source centroid is inside the cube field of view.

HIERARCH SPECTRUM EDGEDIST :code:`float`
    The distance of the source centroid to the edge of cube field of view in spaxels. Negative values imply that the
    source centroid is outside the field of view.

HIERARCH SPECTRUM SNRATIO :code:`float`
    The average signal-to-noise of the spectrum, measured per pixel via the :code:`DER_SNR` algorithm presented in
    `ST-ECF Newsletter 42 <https://www.spacetelescope.org/about/further_information/stecfnewsletters/hst_stecf_0042/>`_.

HIERARCH SPECTRUM MAG <PASSBAND>  :code:`float`
    The AB magnitude recovered from the spectrum in the passband used for the source selection, as specified by the
    `catalog|passband` parameter in the user configuration file. Note that the magnitude is determined under the
    assumption that the flux unit of the cube is :math:`\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,{\rm Å}^{-1}`.

HIERARCH SPECTRUM MAG DELTA :code:`float`
    The difference :math:`\delta` between input and recovered magnitude of the source.

HIERARCH SPECTRUM MAG ACCURACY :code:`float`
    The relative accuracy of the recovered magnitude. This value is computed by comparing the difference :math:`\delta`
    between input and recovered magnitude to the standard deviation :math:`\sigma` of the differences obtained for
    sources of similar brightness, using the formula

    .. math::
        a = \frac{1}{1 + \left(\frac{\delta}{2\cdot\sigma}\right)^2}

HIERARCH SPECTRUM TYPE :code:`str`
    The type of source.

HIERARCH SPECTRUM QLTFLAG :code:`int`
    The quality flag of the spectrum.
