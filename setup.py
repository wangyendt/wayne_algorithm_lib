#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: setup
@time: 2020/3/10 14:38
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywayne",
    version="1.0.0.0.1",
    author="Wayne",
    author_email="wang121ye@hotmail.com",
    description="Some useful tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangyendt/wangye_algorithm_lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
