#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from setuptools import setup, find_packages

__version__ = '0.0.a1'
def readme():
    """
    Longer description from readme.
    """
    with open('ReadMe.md', 'r') as readmefile:
        return readmefile.read()


def requirements():
    """
    Get requirements to install.
    """
    with open('requirements.txt', 'r') as requirefile:
        return [line.strip() for line in requirefile.readlines() if line]


setup(
    name='utilities',
    version=__version__,
    description='Some utilities.',
    long_description=readme(),
    classifiers=[
        'License :: GPL v3.0 License',
        'Intended Audience :: Developers',
        'Intended Audience :: Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    platforms=[
        'Environment :: Console',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Windows'
    ],
    scripts=['utilities.py'],
    keywords='',
    url='https://github.com/HoudaSebti/pneumonia-detector.git',
    author='Houda Sebti [Leakmited]',
    author_email='houda.sebti@leakmited.com',
    license='License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    packages=find_packages(exclude=[]),
    install_requires=requirements(),
    include_package_data=True,
    zip_safe=False
)
