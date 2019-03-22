# coding: utf-8

"""Setup for the installation of the 'maxcut' package."""

import setuptools
from setuptools.command.install import install

setuptools.setup(
    name='maxcut',
    version='0.1',
    packages=setuptools.find_packages(),
    include_package_data=True,
    author='Paul Andrey',
    description='max-cut problem solvers using a variety of approaches',
    license='GPLv3',
    install_requires=[
        'cvxpy >= 1.0',
        'networkx >= 2.0',
        'numpy >= 1.12'
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6"
    ]
)
