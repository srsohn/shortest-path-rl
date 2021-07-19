# Copyright 2021 Shortest Path RL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for installing shortest-path-rl as a pip module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import setuptools

VERSION = '1.0.0'

install_requires = [
    #'DeepMind-Lab',      # See installation instructions in README.md
    'absl-py>=0.7.0',
    'dill>=0.2.9',
    #'enum>=0.4.7',
    # Won't be needed anymore when moving to python3.
    #'futures>=3.2.0',
    'gin-config>=0.1.2',
    'gym>=0.10.9',
    'numpy>=1.16.0',
    'opencv-python>=4.0.0.21',
    'pypng>=0.0.19',
    'pytype>=2019.1.18',
    'scikit-image>=0.14.2',
    'six>=1.12.0',
    'tensorflow-gpu>=1.12.0,<2.0',
]

description = ('Shortest-Path RL. This is the code that allows reproducing '
               'the results in the scientific paper '
               'http://arxiv.org/abs/2107.06405')

setuptools.setup(
    name='sprl',
    version=VERSION,
    packages=setuptools.find_packages(),
    description=description,
    long_description=description,
    url='https://github.com/srsohn/shortest-path-rl',
    author='Sungryull Sohn',
    author_email='srsohn@umich.edu',
    install_requires=install_requires,
    extras_require={
        'video': ['sk-video'],
        'mujoco': [
            # For installation of dm_control see:
            # https://github.com/deepmind/dm_control#requirements-and-installation.
            'dm_control',
            'functools32',
            'scikit-image',
        ],
    },
    license='Apache 2.0',
    keywords='reinforcement-learning deepmind-lab shortest-path',
)
