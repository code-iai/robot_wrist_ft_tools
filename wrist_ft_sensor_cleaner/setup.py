#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
   packages=['wrist_ft_sensor_cleaner'],
   package_dir={'': 'src'}
)

setup(**d)