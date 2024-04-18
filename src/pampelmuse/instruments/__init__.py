"""
PampelMuse
==========
Copyright 2013-2016 Sebastian Kamann

This file is part of PampelMuse.

PampelMuse is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

PampelMuse is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PampelMuse. If not, see <http://www.gnu.org/licenses/>.
"""
from .instrument import Instrument
from .generic_ifs import GenericIFS
from .muse import Muse, MusePixtable
from .pmas import Pmas
from .argus import Argus, SimIfu

__all__ = ["GenericIFS", "Muse", "MusePixtable", "Pmas", "Argus", "SimIfu"]

available_instruments = {"PMAS": Pmas, "GIRAFFE": Argus, "ARGUS": Argus, "MUSE": Muse, "SIMIFU": SimIfu}
