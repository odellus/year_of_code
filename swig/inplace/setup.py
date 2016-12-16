#! /usr/bin/env python

from distutils.core import Extension, setup
from distutils import sysconfig

import numpy

try:
    numpy_include = numpy.get_include()
except:
    numpy_include = numpy.get_numpy_include()

# inplace extension module
_inplace = Extension("_inplace",
    ["inplace.i", "inplace.c"],
    include_dirs=[numpy_include],
    )

setup( name="inplace function",
    description="inplace takes a double array and doubles each of its elements in-place.",
    author="Thomas Wood",
    version="1.0",
    ext_modules=[_inplace])
