
.. _troubleshooting:

Frequently asked questions
==========================

How to increase the number of PSF stars?
----------------------------------------

PampelMuse requires a certain number of PSF stars in order to determine the coordinate transformation from reference to
spaxel coordinates. In some cases, the number of PSF stars found by the :ref:`INITFIT routine <initfit>` is too low,
resulting in the error message `OSError: Need at least 4 sources to fit coordinate transformation.` The following
parameters can be tweaked to increase the number of PSF stars found by :ref:`INITFIT <initfit>`.

* `initfit|apernois`: The default setting of $0.1$ will result in only the most isolated stars to be considered as
  PSF stars. Increasing this value to something like $0.4$ often strongly increases the number of PSF stars found
  without significantly affecting the accuracy of the fitted PSF

* `initfit|psfmag`: By default, only stars that are 3 magnitudes brighter than the faintest star that is extracted from
  the integral field data are considered as PSF stars. This can lead to problems in cases where the field of view is
  only moderately crowded, but most visible stars have similar magnitudes. Consider setting `initfit|psfmag` explicitly,
  to the magnitude of the faintest star you consider useful for a PSF determination.
