from setuptools import setup, find_packages


NAME = "3ddatadiver"
DESCRIPTION = "AFM 3d_data_diver"
AUTHOR = "Renlong Zheng; Ellen Murphy; Xueqiao Zhang"
AUTHOR_EMAIL = "renloz@uw.edu; emurphy2028@gmail.com; xueqiaozhang@hotmail.com"
URL = "https://github.com/EdwardZheng0312/VisualanalysisAFM"
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
    py_modules=['3DdataDiver']
   
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
