#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='keras-video-generators',
    version='1.0.4',
    description='Keras sequence generators for video data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Patrice Ferlet',
    author_email='metal3d@gmail.com',
    url='https://github.com/metal3d/keras-video-generators',
    packages=find_packages(),
    install_requires=[
        'keras>=2',
        'numpy',
        'opencv-python',
    ]
)
