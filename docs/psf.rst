

.. _psf:

Point Spread Function
=====================

Overview on available profiles
------------------------------

PampelMuse uses analytical PSF profiles. Most of the available profiles are based on either a Gaussian or a Moffat
profile. An extensive discussion of the profiles and all associated formulae can be found in
:ref:`here <appendix_profiles>`. Here, only the main formulae and most important properties are presented.

The analytical profile of the Gaussian profile as a function of the radius :math:`r` is given by

.. math::
    G(r) = \Sigma_\text{0}\exp\left\{\frac{-r^2}{2\sigma^2}\right\}

whereas the Moffat profile is described via

.. math::
 M(r) = \Sigma_\text{0}\left[1+\left(\frac{r}{r_{\text d}}\right)^2\right]^{-\beta}

For either profile, :math:`\Sigma_\mathrm{0}` is the central intensity. The width of the Gaussian is controlled by the
standard deviation :math:`\sigma`, while the effective radius :math:`r_{\text d}` defines the width of the Moffat.
However, instead of :math:`\sigma` or :math:`r_{\text d}`, PampelMuse uses the *full width at half maximum (FWHM)* as
a measure for the width of either profile. The conversion formulae from :math:`\sigma` and :math:`r_{\text d}` to
:math:`{\text FWHM}` can be found :ref:`here <appendix_profiles>`).

.. note::
    The :math:`{\text FWHM}` is always measured in units of spaxels.

Besides the :math:`{\text FWHM}`, the Moffat profile has a second shape parameter, :math:`\beta`. It measures the
kurtosis of the profile, i.e. the strengths of the wings of the PSF. Higher values of :math:`\beta` correspond to less
pronounced wings and vice versa.

.. important::
    From my experience, :math:`\beta\sim2.5` is a typical value for a seeing-limited PSF. For this value, a Moffat
    profile has much more pronounced wings than a Gaussian profile. Therefore I strongly recommend using the Moffat
    profile because the Gaussian profile will almost certainly underestimate the wings of the PSF.

Besides the plain Gaussian and Moffat profiles, PampelMuse also supports combinations of the two:

    * A **Moffat-Gauss profile**, consisting of an outer Moffat profile (with parameters :math:`beta` and
      :math:`{\text FWHM}_1`) and a central Gaussian profile (with parameter :math:`{\text FWHM}_1`). An additional
      parameter :math:`f_{12}` controls the fractional flux in the Moffat component.

    * A **double-Gauss profile**, consisting of an outer and a central Gaussian component (with parameters
      :math:`{\text FWHM}_1` and :math:`{\text FWHM}_2`, respectively). Again, an additional parameter :math:`f_{12}`
      is used, controlling the fractional flux in the outer component.

    * A **double-Moffat profile**, consisting of an outer and a central Moffat component (with parameters
      :math:`{\text FWHM}_1`, :math:`\beta_1` and :math:`{\text FWHM}_2`, :math:`\beta_2`), respectively. As for the
      double-Gauss profile, the parameter :math:`f_{12}` controls the fractional flux of the outer component.

PampelMuse supports elliptical PSF profiles. The radius :math:`r` incorporates the ellipticity :math:`e` and the
position angle :math:`\theta` of a PSF profile. Thus, besides the actual centroid coordinates, the Moffat profile has
4 free parameters, the Gaussian profile has 3. For combined profiles, only a global ellipticity and position angle can
be used.

Adaptive optics data
^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 1.0
    To facilitate the handling of adaptive optics (AO) data, in particular MUSE narrow field mode (NFM) observations,
    the MAOPPY PSF model developed by [Fetick2019]_ is made available in PampelMuse. Besides the parameters :math:`e`
    and :math:`\theta` that it shares with the other PSF models, the MAOPPY PSF has five parameters that can be
    optimized during the analysis. They are the Fried parameter :math:`r_0`, the kurtosis :math:`\beta` of the Moffat
    contribution, the effective radius :math:`\alpha` of the Moffat contribution, the AO-corrected phase power spectral
    density (PSD) background :math:`bck`, and the residual variance :math:`amp`. Note that some parameters have been
    renamed compared to the names provided in Table 2 of [Fetick2019]_ in order to avoid confusion with other fitting
    parameters.

.. important::
    In case you use this PSF model, please cite [Fetick2019]_.

Specifying the PSF to use
-------------------------

To specify the properties of the PSF during the analysis, one needs to modify the `psf` section of the PampelMuse
:ref:`configuration <user-configuration>`.

Setting the PSF type and parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, one needs to specify which type of profile should be used. This can be using the following configuration
parameter.

psf|profile : string
    The analytical PSF profile used during the analysis. Currently supported are `gauss`, `moffat`, `double_gauss`,
    `double_moffat`, `moffat_gauss`, and `maoppy`.

Next, initial guesses for the PSF parameters must be defined. As already mentioned, the available profiles are
described by different sets parameters. This is summarized in the following table.

.. table:: Overview of PSF parameters
    :name: table-psf-parameters

    +-----------+-------------+-------------+----------------+-----------------+----------------+----------+
    | Parameter | `gauss`     | `moffat`    | `double_gauss` | `double_moffat` | `moffat_gauss` | `maoppy` |
    | index     |             |             |                |                 |                |          |
    | :math:`n` |             |             |                |                 |                |          |
    +===========+=============+=============+================+=================+================+==========+
    | 1         | `fwhm`      | `beta`      | `f12`          | `f12`           | `f12`          | `r0`     |
    +-----------+-------------+-------------+----------------+-----------------+----------------+----------+
    | 2         | `e`         | `fwhm`      | `fwhm1`        | `beta1`         | `beta`         | `bck`    |
    +-----------+-------------+-------------+----------------+-----------------+----------------+----------+
    | 3         | `theta`     | `e`         | `fwhm2`        | `fwhm1`         | `fwhm1`        | `amp`    |
    +-----------+-------------+-------------+----------------+-----------------+----------------+----------+
    | 4         |             | `theta`     | `e`            | `beta2`         | `fwhm2`        | `alpha`  |
    +-----------+-------------+-------------+----------------+-----------------+----------------+----------+
    | 5         |             |             | `theta`        | `fwhm2`         | `e`            | `beta`   |
    +-----------+-------------+-------------+----------------+-----------------+----------------+----------+
    | 6         |             |             |                | `e`             | `theta`        | `e`      |
    +-----------+-------------+-------------+----------------+-----------------+----------------+----------+
    | 7         |             |             |                | `theta`         |                | `theta`  |
    +-----------+-------------+-------------+----------------+-----------------+----------------+----------+

For each parameter of the chosen PSF profile, an initial guess can be defined in the `psf` section of the
configuration, e.g.:

psf|fwhm : float
    The full width at half maximum of the selected PSF profile. Must be :math:`>0`.

Finally, one needs to define which of the PSF parameters should be optimized during the analysis and which remain fixed
at their initial guesses. To this aim, PampelMuse uses a single configuration parameter:

psf|fit : integer
    Defines which PSF parameters are optimized during the analysis. The value should set to
    :math:`\Sigma_{n} 2^n`, with the sum including the index :math:`n` of every PSF parameter to be optimized (see
    :ref:`table <table-psf-parameters>`).

Assuming one wants to model the PSF as a Moffat profile, with a fixed value of :math:`\beta=2.5`, and free parameters
:math:`{\text FWHM}`, :math:`e`, and :math:`\theta`, the `psf` section of the configuration could look as follows.

.. code-block:: json

    {
        "psf": {
            "profile": "moffat",
            "beta": 2.0,
            "fwhm": 2.5,
            "e": 0.2,
            "theta": 0.5,
            "fit": 28
        }
    }

The values provided for :math:`{\text FWHM}`, :math:`e`, and :math:`\theta` serve as initial guesses for the fit.

Setting the PSF radii
^^^^^^^^^^^^^^^^^^^^^

Each PSF profile used during the analysis will be defined out to a maximum radius, beyond which the surface intensity
of the profile is believed to be negligible. This radius can be changed using the following parameter.

psf|radius : float
    The radius out to which a PSF profile is calculated, given in units of spaxels.

As a rule of thumb, the radius parameter should be large enough so that no bright halos remain around the bright stars
when creating the residuals of the analysis with the :ref:`SUBTRES routine <subtres>`.

.. versionadded:: 1.0
    PampelMuse also allows one to scale the PSF radius according to the flux of a star, such that a larger radius is
    used for brighter sources. This is useful for example for the analysis of MUSE narrow field mode data, where the
    PSF halos around the brightest stars can extend out to :math:`\sim50-100` spaxels. As using such a large value for
    all sources will significantly slow down the analysis, one can set the following configuration parameter.

    psf|scaling : float
        Scaling factor for the PSF radius, must be :math:`>0` and :math:`<1`. It is defined such that the PSF radius
        will shrink by a factor :math:`2` for every drop in flux by the provided value. A flux reference, the brightest
        star in the field is used, for which the PSF profile will be defined out to `psf|radius`.


In-depth analysis of fitted PSF
-------------------------------

PampelMuse offers the option to store detailed information about the the PSF fits it performed when running the
:ref:`CUBEFIT routine <cubefit>`. To this aim, the following parameter should be enabled.

psf|inspect : boolean
    If enabled, a FITS-file will be created that contains the details of the individual PSF fits. For each source used
    to fit the PSF, four extensions will be created, containing the input data of the fits (named `DAT_<ID>`), radial
    profiles of the input data (named `PR_<ID>`), the fit residuals (named `RES_<ID>`), and the fitted parameters
    (named `PM_<ID>`). The IDs used to construct the extension names are the same as in the :ref:`reference catalogue
    <reference_catalogue>`. All data will be stored as a function of wavelength. The file is saved under
    `<prefix>.psf.fits`, where the same prefix is used that was provided in the user configuration file to identify the
    integral field data.

.. note::
    This setting only has an effect only if PampelMuse is run with the `global|smallifu` key set to False or if the
    analysis has been started using the :ref:`SINGLESRC routine <singlesrc>` routine.

Further information about the content of the FITS file that is created can be found :ref:`here <output-psf>`.