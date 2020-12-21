# TODO
# 

__author__ = 'lingson'

from setuptools import setup
from setuptools import find_packages

# Package meta-data.
NAME = 'unique_name'  # Unique name for publishing to PyPI
DESCRIPTION = 'short description'  # Short description for the project
# Bitbucket
URL = 'https://bitbucket.org/lingson/project_name_lowercase'
# Github
# URL = 'https://github.com/lingson/project_name_lowercase'
EMAIL = 'dirk.lingson@gmail.com'
AUTHOR = 'Dirk Wangsadirdja'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'
LICENSE=''

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=['project_name_lowercase'],
    license=LICENSE,
    url=URL,
)