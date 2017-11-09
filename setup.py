import os
from setuptools import find_packages
from setuptools import setup

version = '0.1.dev1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

install_requires = [
    'numpy',
    'scikit-learn',
    'six',
]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    'pytest-pep8',
]

setup(
    name="binary-kmodes",
    version=version,
    description="Python library for binary K-Modes clustering",
    long_description="\n\n".join([README]),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Machine Learning",
        ],
    keywords="",
    author="Geoffrey French",
    # author_email="brittix1023 at gmail dot com",
    url="https://github.com/Britefury/binary-kmodes",
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        },
    )
