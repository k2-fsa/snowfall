# coding=utf-8
import os
from pathlib import Path

from setuptools import find_packages, setup

setup(
    name='snowfall',
    version='0.1',
    python_requires='>=3.6.0',
    description='Speech processing recipes using Lhotse and K2',
    author='The K2 and Lhotse Development Team',
    license='Apache-2.0 License',
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed"
    ],
)
