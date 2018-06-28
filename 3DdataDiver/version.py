from __future__ import absolute_import, division, print_function
from os.path import join as pjoin


_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases
# _version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "temann: package for predicting Seebeck coefficients"
# Long description will go up on the pypi page
long_description = """
3DdataDiver
======
3ddatadiver is a package for the processing and visualization of 3D AFM data. Users interact with the package primarily
through a GUI that is locally run. Upon uploading .mat or .HDF5 files hidden data cleaning functions are called that
correct sample slope, linearize Zsensor data, concat Zsensor with phase/amp data, and generate a three numpy arrays: 
full, approach, and retract dataset. Users can interact with the data by viewing a full 3D rending, slices in 3D or 2D
cartesian coordinate systems, and animations of these slices. At any point the user can save a .csv file of the data
they are currently viewing to the folder the GUI is being run.
License
=======
``3ddatadiver`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2018--, Xueqiao Zhang, Renlong Zheng &
Ellen Murphy, The University of Washington.
"""

NAME = "3ddatadover"
MAINTAINER = "Ellen Murphy"
MAINTAINER_EMAIL = "emurphy@uw.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/EdwardZheng0312/3ddatadiver"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Xueqiao Zhang <xueqiaozhang@hotmail.com>" +\
         "Renlong Zheng <renloz@uw.edu>, " +\
         "Ellen Murphy <murphy89@uw.edu>"
AUTHOR_EMAIL = ""
description = "AFM 3D data diver"


NAME = "3ddatadiver"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/EdwardZheng0312/VisualanalysisAFM"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Renlong Zheng; Ellen Murphy; Xueqiao Zhang"
AUTHOR_EMAIL = "renloz@uw.edu; emurphy2028@gmail.com; xueqiaozhang@hotmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'3DdataDiver': [pjoin('data', '_data', '*')],
                '': ['*.csv']}
REQUIRES = ["numpy", "h5py", "tkinter", "matplotlib", "os"]

PACKAGE_DATA = {}
REQUIRES = []
