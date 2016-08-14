"""
__init__.py
===========
Copyright 2013-2016 Sebastian Kamann

This file is part of PampelMuse.

PampelMuse is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

PampelMuse is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PampelMuse. If not, see <http://www.gnu.org/licenses/>.


+++
Last SVN revision
-----------------
334, 2016/07/26
"""
from .cubefit import cubefit
from .initfit import initfit
from .getspectra import getspectra
from .makecube import makecube
from .polyfit import polyfit
from .ptsort import ptsort
from .subtres import subtres
from .shotnoi import shotnoi
from .singlesrc import singlesrc


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 334


__all__ = ["cubefit", "getspectra", "initfit", "makecube", "polyfit", "ptsort", "shotnoi", "subtres", "singlesrc"]

