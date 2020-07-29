
.. _data:

The input data
==============

Supported instruments
---------------------

In general, PampelMuse is designed such  that it can correctly handle integral-field data regardless of which
instrument was used to observe them. However, because of the huge variety of data formats that are produced by the
individual instruments, I cannot guarantee that the code will work out of the box for each instrument. So far, I
successfully tested PampelMuse using data from

    * PMAS [Roth2005]_
    * ARGUS 2002Msngr.110....1P
    * MUSE 2010SPIE.7735E..08B
    * KMOS \2004SPIE.5492.1179S
    * WiFeS

The routine will try to figure out which instrument the data is coming from by searching for the standard header
keyword `INSTRUME`. For some instruments (like PMAS) this keyword is not set. In such cases, if you want to make use
of the instrument-specific features of PampelMuse (see :ref:`Integral field data <integral-field-data>`), you should
update the FITS header of your data accordingly.

If you have data from another instrument that is not properly handled by PampelMuse, please
:ref:`let me know <contact>`. On the other hand, if your data is handled correctly and the instrument does not appear
in the list above, that would be very helpful for me to know as well.

.. _integral-field-data:

Integral field data
-------------------

.. important::
    Keep in mind that PampelMuse is *not* a data reduction pipeline. All analysis steps assume that the integral field
    data have been properly reduced and that all the instrumental artifacts have been removed. While PampelMuse does
    perform checks to confirm the integrity of the provided data, they are not exhaustive. In case your data contain
    artifacts that interfere with the analysis (such as voxels with zero variance that are not masked), the analysis
    will most likely fail.

The integral field data can be provided in different formats, depending on which instrument was used in the
observations:

    * As a three-dimensional data cube with two spatial axes (`NAXIS1` and `NAXIS2`) and a spectral axis (`NAXIS3`).
      This format can always be used, even if the data were obtained with an instrument that is unknown to PampelMuse.

    * As a two-dimensional image with one spatial and one spectral axis. The advantage of this format is that it can
      also be used with instruments that have an irregular (i.e. non-rectangular) spatial sampling. However, for this
      format to work PampelMuse must know how to arrange the different spaxels in the field of view. Currently this
      format is only supported for PMAS and ARGUS data.

    * MUSE data may also be provided in the form of a pixtable as it is produced by the pipeline recipe
      `muse_scipost`, when the option `--save-individual` is being used. [#f1]_ The analysis of MUSE pixtables with
      PampelMuse requires the data to be sorted by increasing wavelength. This can be achieved by running the
      dedicated :ref:`PTSORT routine <ptsort>` before starting the analysis.

Regardless of the input format, the data file to be analysed must always be provided in :ref:`user configuration file
<user-configuration>` using the parameter

global|prefix : string
    The prefix (i.e. usually the filename without the .fits ending) of the integral field data file to be analysed.

Besides the actual data, PampelMuse also requires an estimate of their variances for the analysis. Optionally, a bad
pixel mask can be used in addition. Data, variances, and mask can either be provided as different extensions of a
single FITS file or as three individual FITS files. In the former case, the routine will use the extension names to
identify what is stored in which extension, as listed in the following table.

==============  =========  ========================
Content         Required?  `EXTNAME`
==============  =========  ========================
Data            y          `DATA`
Variances       y          `STAT`, `ERROR`, `SIGMA`
Bad-pixel mask  n          `MASK`
==============  =========  ========================

If instead individual FITS files are provided, their filenames should have the same prefix and be suffixed `.dat.fits`
(the data), `.err.fits` (the variances), and `.msk.fits` (the bad pixel mask).

If provided, the bad pixel mask should be zero for valid pixels and one for pixels that should be excluded. Instead of
providing a dedicated mask, one can also set the invalid pixels to NaN and PampelMuse will ignore them in the analysis.

Although not required, it is highly advantageous if the header of the FITS file containing the IFS data contains some
basic WCS information, i.e. at least the keywords `RA`, `DEC`, and the matrix `CDi_j`. PampelMuse will use this
information to determine the spaxel scale and the approximate location of the field of view on the sky.

PampelMuse also offers the possibility to analyse only a part of the data cube or to bin the data in spectral
direction in order to speed up parts of the analysis. To this aim, the section `layers` has been introduced in the
configuration. It contains the following keys.

layers|first : integer
    The index of the first bin to be analysed.

layers|last : integer
    The index of the last layer to be analysed.

layers|binning : integer
    The binning applied in spectral direction.


.. _reference-catalogue:

The reference catalogue
-----------------------

As PampelmMse does not include a dedicated source detection algorithm, the analysis relies on the presence of a
reference catalogue, containing coordinates and magnitudes of the stars in and around the observed field. Ideal are
source catalogues generated from high-resolution imaging data (e.g. obtained with the Hubble Space Telescope or via
ground-based adaptive optics imaging), using passbands that overlap at least partially with the wavelength range of the
integral field data.

.. important::

    The quality of the reference catalogue (especially its completeness and astrometric accuracy) will strongly
    influence the results of the analysis. For example, a common shortcoming of catalogues generated from Hubble data
    is that at least some of the brightest stars are missing because they were saturated in the exposures. If this is
    the case, try to add them manually.

For PampelMuse to recognize the reference catalogue, its file name *must* be provided in the :ref:`user configuration
file <user-configuration>`, using the parameter

catalog|name : string
    The filename of the astrometric reference catalogue used for the analysis.

The reference catalogue should be present as a plain csv-file with the first (non-commented) line containing the column
names. PampelMuse will recognize column names such as ``ra``, ``dec``, or ``id`` and process the column content
accordingly. The reference catalogue must contain the following columns:

    * A unique identifier for every source, named ``id``. This must be a non-negative integer of not more than 10
      digits.
    * Two columns containing the source coordinates in a reference system. Both pixel coordinates (named ``x``, ``y``)
      and world coordinates (named ``ra``, ``dec``) are accepted, and priority is given to the latter if both are
      present. If provided, the world coordinates should be in units of decimal degrees.
    * A magnitude for every source in a passband that overlaps at least partially with the spectral range of the
      integral field data. The filter name must also be used as column name and needs to be specified in the
      :ref:`user configuration file <user-configuration>`, using the parameter

    catalog|passband : string
        The name of the filter in which the magnitudes provided in the reference catalogue are given.

Here is an example of how a valid reference catalogue file may look like::

    id,ra,dec,f606w
    1,12.764311,-26.986152,13.29
    2,12.751325,-26.986431,14.87
    4,12.760123,-26.976514,12.25
    ...

.. note::
    A fifth column labelled ``status`` is required when using the :ref:`SINGLESRC routine <singlesrc>` instead of the
    :ref:`INITFIT routine <initfit>`.

Saturated stars
^^^^^^^^^^^^^^^

In some occasions, it may happen that the brightest stars from the reference catalogue are saturated in the integral
field data. In such cases, it is recommended to flag these stars, using the parameter

catalog|saturated : tuple of integers
    The list of stars from the reference catalogue that appear saturated in the IFS data. The ID(s) of the saturated
    star(s) as provided in the column ``id`` of the reference catalogue must be provided.

PampelMuse will ignore stars flagged as saturated when determining the PSF of the observation.

A note about filters
^^^^^^^^^^^^^^^^^^^^

PampelMuse comes with a limited number of filter curves. Currently, the transmission curves for `F555W`, `F606W`,
`F775W`, `F814W`, the Johnson bands and the infrared `J`, `H`, and `K` are provided. If the input photometry uses
another filter, it is possible to provide PampelMuse with the required information. To this aim, set the environment
variable `PMFILTERPATH`. It must point to the folder in which the transmission curve of the filter is stored, e.g.::

    export PMFILTERPATH=$HOME/data/filters

The transmission curve of the filter must be available as a plain ascii file, containing at least two columns. The
first column must contain the wavelength in units of Angstrom, the second column must contain the transmission in
absolute values, i.e. ranging from 0 to 1. The name of the ascii file must be the name of the filter in capital
letters. For example, if you want to use the SDSS g filter, the file should be named G or SDSS_G, but not g.txt or
sdss_g. In the reference catalogue, you should use the same label for the photometry as you use for the filename.


.. rubric:: Footnotes

.. [#f1] For details, check the MUSE pipeline manual available at the
         `ESO homepage <https://www.eso.org/sci/software/pipelines/muse/>`_ or the pipeline paper [Weilbacher2020]_.