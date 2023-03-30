#!/usr/bin/env python3

from setuptools import setup, find_packages
# import os
# import datetime

# Load the WorldCereal version info.
#
# Note that we cannot simply import the module, since dependencies listed
# in setup() will very likely not be installed yet when setup.py run.
#
# See:
#   https://packaging.python.org/guides/single-sourcing-package-version

__version__ = None

with open('src/worldcereal/_version.py') as fp:
    exec(fp.read())

    version = __version__

# Configure setuptools

setup(
    name='worldcereal',
    version=version,
    author='Kristof Van Tricht',
    author_email='kristof.vantricht@vito.be',
    description='WorldCereal Classification',
    url='https://github.com/WorldCereal/worldcereal-classification',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data=True,
    package_data={
        '': ['resources/*', 'resources/**/*', 'resources/**/*',
             'resources/*/*', 'resources/*/*/*', 'resources/*/*/*/*'],
    },
    zip_safe=True,
    install_requires=[
        'catboost==1.0.6',
        'cloudpickle==2.2.0',
        'fire==0.4.0',
        'geopandas==0.9.0',
        'h5py==2.10.0',
        'hdmedians==0.14.2',
        'joblib==1.2.0',
        'loguru<0.7',
        'matplotlib==3.5.3',
        'numba<0.57',
        'numexpr==2.8.3',
        'numpy==1.18.5',
        'openeo==0.12.1',
        'pandas==1.4.4',
        'pyarrow==9.0.0',
        'pyod==1.0.0',
        'python-dateutil<2.9',
        'protobuf==3.20.3',
        'rasterio==1.2.10',
        'requests<3',
        'satio==1.1.15',
        'scikit-image==0.19.3',
        'scikit-learn==1.1.2',
        'scipy==1.4.1',
        'Shapely==1.8.4',
        'tensorflow==2.3.0',
        'tqdm==4.64.1',
        'utm<0.8',
        'xarray==2022.3.0',
        'zarr==2.12.0',
    ],
    test_suite='tests',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    entry_points={
        'console_scripts': [
            'worldcereal=worldcereal.__main__:main'
        ]
    }
)
