from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

import yaml
import os
import subprocess
from bsi_utils.basic_utils import create_config_file


setup(
    name="authenticaudioguard",
    version="0.0.1",
    description="This package contains the code for the authenticaudioguard",
    author="Patrick Kühn",
    packages=find_packages(),
    include_package_data=True,
)