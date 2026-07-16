"""
PampelMuse
==========
Copyright 2013-2026 Sebastian Kamann

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
import logging
from importlib import import_module
from functools import lru_cache
from .instrument import Instrument
from .generic_ifs import GenericIFS


logger = logging.getLogger(__name__)


_INSTRUMENTS = {
    "PMAS": (".pmas", "Pmas"),
    "ARGUS": (".argus", "Argus"),
    "GIRAFFE": (".argus", "Argus"),
    "MUSE": (".muse", "Muse"),
    "PIXTABLE": (".muse", "MusePixtable"),
    "SIMIFU": (".argus", "SimIfu"),
    "MEGARA": (".megara", "Megara"),
}

def known_instruments():
    return tuple(_INSTRUMENTS.keys())

@lru_cache
def available_instruments():
    available = {}
    for name, (module_path, class_name) in _INSTRUMENTS.items():
        try:
            module = import_module(module_path, __name__)
        except ImportError as e:
            logger.debug(f"Instrument '{name}' is not available: {e}")
            continue
        available[name] = getattr(module, class_name)
    return available
