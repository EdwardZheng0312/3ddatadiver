from setuptools import setup, find_packages
import os
# Get version and release info, which is all stored in 3ddatadiver/version.py
#ver_file = os.path.join('3ddatadiver', 'version.py')
#with open(ver_file) as f:
#    exec(f.read())

NAME = "3ddatadiver"
DESCRIPTION = "AFM 3d_data_diver"
AUTHOR = "Renlong Zheng; Ellen Murphy; Xueqiao Zhang"
AUTHOR_EMAIL = "renloz@uw.edu; emurphy2028@gmail.com; xueqiaozhang@hotmail.com"
URL = "https://github.com/EdwardZheng0312/VisualanalysisAFM"
VERSION = "0.10"
PACKAGES = "3ddatadiver"
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="BSD",
    url=URL,
    packages=PACKAGES,
   
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
if __name__ == '__main__':
    setup(**opts)