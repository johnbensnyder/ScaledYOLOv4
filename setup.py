#!/usr/bin/env python
from setuptools import setup, find_packages

install_requires = ["tensorboard",
                    "mish_cuda@git+https://github.com/thomasbrandon/mish-cuda/"]

setup(
    name="scaled_yolo_v4",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/johnbensnyder/ScaledYOLOv4",
    description="ScaledYOLOv4",
    packages=find_packages(),
    install_requires=install_requires,
)