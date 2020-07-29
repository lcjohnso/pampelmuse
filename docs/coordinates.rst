
.. _coordinates:

Source coordinates
==================

In the common usage case, the coordinates :math:`(u^n,\,v^n)` of each known source :math:`n` in a reference system are
provided. In order to obtain the spaxel coordinates :math:`(x^n,\,y^n)` of the source in the integral field data, it is
assumed that a global affine transformation exists that can be described by

* a scaling by a factor :math:`\xi` along the :math:`u`-axis
* a scaling by a factor :math:`\nu` along the :math:`v`-axis
* a rotation around an angle :math:`\theta`
* shifts :math:`(x_{\text 0},\,y_{\text 0})` along the :math:`x`- and :math:`y`-axis of the data cube

Thus, the coordinate transformation can be summarized by the following formulae:

.. math::

    x^n &= \xi\,\cos\theta\,u^n &- \nu\,\sin\theta\,v^n &+ x_{\text 0} \\
    y^n &= \xi\,\sin\theta\,u^n &+ \nu\,\cos\theta\,v^n &+ y_{\text 0}

Substituting :math:`A=\xi\,\cos\theta`, :math:`B=-\nu\,\cos\theta`, :math:`C=-\nu\,\sin\theta`, and
:math:`D=\xi\,\sin\theta`, one obtains

.. math::

 x^n &= A\,u^n &+ C\,v^n &+ x_{\text 0} \\
 y^n &= D\,v^n &- B\,u^n &+ y_{\text 0}

The parameters :math:`A` to :math:`y_{\text 0}` of the coordinate transformation can be optimized in the analysis. Their
behaviour during the extraction is controlled via the parameter

initfit|posfit : integer
    It defines how the coordinate transformation is handled during the analysis:
    :math:`0` - no optimization;
    :math:`1` - only optimize the shifts, i.e. parameters :math:`x_{\text 0}` and :math:`y_{\text 0}`;
    :math:`2` - optimize all :math:`6` parameters of the transformation.

.. note::
    Older versions of PampelMuse only used :math:`4` parameters to determine the coordinate transformation, which is
    the special case of an affine transformation where :math:`\xi=\nu`.
