# setup.py
from setuptools import setup, find_packages

setup(
    name="classifier_selection_module",
    version=open('./classifier_selector/version.py').read().split('=')[1].strip().strip("'"),
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
)