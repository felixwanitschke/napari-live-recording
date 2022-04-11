#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'napari_live_recording'
DESCRIPTION = 'A napari plugin for live video recording with a generic camera device.'
URL = 'https://github.com/jethro33/napari-live-recording'
EMAIL = 'jacopo.abramo@gmail.com'
AUTHOR = 'Jacopo Abramo'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.2.0'

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

about = {}
about['__version__'] = "0.2.0"

requirements = []
with open('requirements.txt') as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

# https://github.com/pypa/setuptools_scm
use_scm = {"write_to": "napari_live_recording/_version.py"}

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

setup(
    name='napari_live_recording',
    author="Jacopo Abramo",
    author_email='jacopo.abramo@gmail.com',
    license='MIT',
    url='https://github.com/jethro33/napari-live-recording',
    description='A napari plugin for live video recording with a generic camera device.',
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=requirements,
    include_package_data=True,
    version='0.2.0',
    setup_requires=['setuptools_scm'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Framework :: napari',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License'
    ],
    entry_points={
        'napari.plugin': [
            'napari-live-recording = napari_live_recording',
        ],
    },
    cmdclass={
        'upload': UploadCommand,
    },
)