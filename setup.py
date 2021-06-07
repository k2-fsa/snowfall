#!/usr/bin/env python3

from setuptools import find_packages, setup
from pathlib import Path

snowfall_dir = Path(__file__).parent
install_requires = (snowfall_dir / 'requirements.txt').read_text().splitlines()
extras_require = {'test': ['pytest']}
extras_require['dev'] = extras_require['test']

setup(
    name='snowfall',
    version='0.1',
    python_requires='>=3.6.0',
    description='Speech processing recipes using k2 and Lhotse.',
    author='The k2 and Lhotse Development Team',
    license='Apache-2.0 License',
    scripts=['snowfall/bin/snowfall'],
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
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
