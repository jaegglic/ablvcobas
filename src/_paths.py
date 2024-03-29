#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Turn the raw data into features for modeling.
"""

# Standard library
import os
import platform

# Third party requirements

# Local imports

# Define file separator
os_name = platform.system()
if os_name == 'Windows':
    file_sep = '\\'
elif os_name == 'Linux':
    file_sep = '/'
elif os_name == 'Darwin':
    file_sep = ':'
else:
    msg = f'Unknown file separator for operating system "{os_name}"'
    raise ValueError(msg)

# Path for the project's root directory
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\
            + file_sep

# Path to the data folders
PATH_DATA           = f'{PATH_ROOT}data{file_sep}'
PATH_DATA_RAW       = f'{PATH_DATA}raw{file_sep}'
PATH_DATA_PROCESSED = f'{PATH_DATA}processed{file_sep}'

# Path to the model folder
PATH_MODELS         = f'{PATH_ROOT}models{file_sep}'

# Path to the figures
PATH_REP            = f'{PATH_ROOT}reports{file_sep}'
PATH_FIG            = f'{PATH_REP}figures{file_sep}'

if __name__ == '__main__':
    _indent = '  '
    print()

    print("Project's root path:")
    print(_indent, PATH_ROOT)

    print("Projects data paths:")
    print(_indent, PATH_DATA)
    print(_indent, PATH_DATA_RAW)
    print(_indent, PATH_DATA_PROCESSED)

    print("Projects model path:")
    print(_indent, PATH_MODELS)
