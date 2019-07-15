#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Generic dependencies
from setuptools import setup, find_packages
import unittest

# Local dependencies
from src._version import VERSION

try:
    from src._pers_data import NAME, EMAIL, URL
except (FileExistsError, ModuleNotFoundError):
    NAME        = "Dummy RTD"
    EMAIL       = 'dummy.rtd@rtd.io'
    URL         = 'https://readthedocs.io'


# Get unittest functions
def test_suite():
    test_loader = unittest.TestLoader()
    return test_loader.discover('tests', pattern='test_*.py')


# Long description from the README.md file
with open("README.md", "r") as file_handle:
    long_description = file_handle.read()


# Generic requirements
tests_require = ['numpy', 'pyodbc']
_rem_requires_for_rtd = ['pyodbc']
install_requires = [p_req for p_req in tests_require
                    if p_req not in _rem_requires_for_rtd]


_setup_data = {
    'name':                 'ablvcobas',
    'version':              VERSION,
    'description':          'Compare ABL and Cobas values from blood tests',
    'long_description':     long_description,

    'author':               NAME,
    'author_email':         EMAIL,
    'url':                  URL,

    'packages':             find_packages(),
    'test_suite':           'setup.test_suite',
    'tests_require':        tests_require,
    'install_requires':     install_requires,
}

if __name__ == '__main__':
    setup(**_setup_data)
