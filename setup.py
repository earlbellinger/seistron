#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

#with open('HISTORY.rst') as history_file:
#    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Earl Patrick Bellinger",
    author_email='earl.bellinger@yale.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    description="An open-science package for stellar phenomenology",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,# + '\n\n' + history,
    include_package_data=True,
    keywords='seistron',
    name='seistron',
    packages=find_packages(include=['seistron', 'seistron.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/earlbellinger/seistron',
    version='0.0.2', # bump version here, in requirements_dev.txt, and in pyproject.toml
    zip_safe=False,
)
