#import os
from setuptools import setup, find_packages

# Get version and release info, which is all stored in 3DdataDiver/version.py
#ver_file = os.path.join('3DdataDriver', 'version.py')
#with open(ver_file) as f:
#    exec(f.read())

NAME = "3DdataDiver"
DESCRIPTION = "AFM 3D_data_Diver"
AUTHOR = "Renlong Zheng; Ellen Murphy; Xueqiao Zhang"
AUTHOR_EMAIL = "renloz@uw.edu; emurphy2028@gmail.com; xueqiaozhang@hotmail.com"
URL = "https://github.com/EdwardZheng0312/VisualanalysisAFM"
#VERSION = __import__(PACKAGE).__version__
VERSION="0.10"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    #long_description=read("README.md"),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="BSD",
    url=URL,
    packages=find_packages(exclude=["tests.*", "tests"]),
    #package_data=find_package_data(
	#		PACKAGE,
	#		only_in_packages=False),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Framework :: Django",
    ],
    zip_safe=False,
    )
