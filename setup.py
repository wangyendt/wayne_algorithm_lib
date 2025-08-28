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
from pywayne.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#") and not line.startswith("~")]


# 解析所有依赖
all_reqs = parse_requirements('requirements.txt')

# 定义核心依赖
core_reqs = [
    "ipdb",
    "natsort",
    "sortedcontainers",
    "tqdm",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "configparser",
    "setuptools",
    "filelock",
    "pyyaml",
    "requests",
    "websockets",
    "python-dotenv"
]

# 定义可选依赖
optional_reqs = [req for req in all_reqs if req not in core_reqs]

setuptools.setup(
    name="pywayne",
    version=__version__,
    author="Wayne",
    author_email="wang121ye@hotmail.com",
    description="Some useful tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangyendt/wangye_algorithm_lib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    dependency_links=[
        'https://pypi.tuna.tsinghua.edu.cn/simple/',
        'http://mirrors.aliyun.com/pypi/simple/',
        'http://pypi.douban.com/simple/',
        'https://pypi.python.org/simple',
        'https://pypi.mirrors.ustc.edu.cn/simple/',
    ],
    install_requires=core_reqs,
    extras_require={
        'full': optional_reqs,
        'gui': ["easygui", "pynput"],
        'image': ["pillow"],
        'aws': ["boto3", "botocore", "oss2"],
        'data': ["h5py", "seaborn", "pyperclip", "statsmodels"],
        'geo': ["concave_hull", "alphashape", "shapely"],
        'bot': ["lark-oapi", 'gtts'],
        'crypto': ["cryptography"]
    },
    packages=setuptools.find_packages(),
    python_requires='>=3',
    scripts=[
        'bin/gettool',  # shell script
        'bin/gettool.py',  # python script
        'bin/gitstats',    # shell script
        'bin/gitstats.py',  # python script
        'bin/cmdlogger',   # shell script
        'bin/cmdlogger.py'  # python script
    ]
)
