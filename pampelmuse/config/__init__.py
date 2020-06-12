"""
__init__.py
============
Copyright 2013-2018 Sebastian Kamann

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


+++
Provides
--------
list changeable: all parameters that can be modified during the analysis

Purpose
-------
The config package handles the default configuration of PampelMuse.

Last changed
------------
2018/07/04
"""
__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"


changeable = ["verbose", "interact", "cpucount", "iterations", "type", "degree", "subtract",
              "usesigma", "first", "last", "binning", "radius", "inspect", "mask", "include",
              "exclude", "offsets"]
