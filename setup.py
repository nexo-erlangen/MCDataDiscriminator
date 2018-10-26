#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='cnns',
    version='1.0',
    description='Runs CNNs',
    url='https://github.com/nexo-erlangen/UVWireRecon',
    author='Tobias Ziegler and Johannes Link',
    author_email='tobias.ziegler@fau.de',
    packages=find_packages(),
    include_package_data=True,
)