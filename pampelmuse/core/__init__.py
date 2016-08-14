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
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PampelMuse. If not, see <http://www.gnu.org/licenses/>.


+++
Latest SVN revision
-------------------
333, 2016/07/25
"""
# import copy_reg
# from types import *
from .sources import Sources  # order counts here! aperture loads Sources, so it must be loaded first
from .aperture import AperPhoto
from .background import BackgroundGrid


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 333


# # for multi-threading
# def _pickle_method(method):
#     func_name = method.im_func.__name__
#     obj = method.im_self
#     cls = method.im_class
#     if func_name.startswith('__') and not func_name.endswith('__'):
#         cls_name = cls.__name__.lstrip('_')
#         if cls_name:
#             func_name = '_' + cls_name + func_name
#     return _unpickle_method, (func_name, obj, cls)
#
#
# def _unpickle_method(func_name, obj, cls):
#     for cls in cls.mro():
#         try:
#             func = cls.__dict__[func_name]
#         except KeyError:
#             pass
#         else:
#             break
#     return func.__get__(obj, cls)
#
#
# copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)
