# -*- coding: utf-8 -*-
# Ocean Perception, University of Southampton
# Author: Jose Cappelletto - j.cappelletto@soton.ac.uk - cappelletto@gmail.com

# setup.py to manage installation of the package and its dependencies
# package name: bayesian_predictor
# package source folder: src/
# dependencies: numpy, scipy, torch, pandas, scikit-learn, blitz

from importlib_metadata import entry_points
from setuptools import setup, find_packages

# TODO: Complete git hook to retrieve version from git tags
# get the version number from the version file
# with open(os.path.join('src', 'version.py')) as f:
#     exec(f.read())

def run_setup():

    # get the long description from the README file
    # TODO: Merge with short/specific decription provided during setup() call
    with open('README.md') as f:
        long_description = f.read()
    if long_description is None:
        long_description = 'No description available'

    # the requirements can be retrieved from the conda environment file
    # with open('environment.yml') as f:
    #     requirements = f.read().splitlines()

    setup(
        name='bayesian_predictor',
        version='0.1.6',
        description='Bayesian NN training/inference engine to learn mappings between latent representations of low resolution maps and high resolution maps',
        author='Jose Cappelletto',
        author_email='j.cappelletto@soton.ac.uk',
        url='https://github.com/cappelletto/bayesian_inference',
        license='GPLv3', # check if oplab requires MIT for all packages

        packages=['bnn_inference'],

        entry_points={
            'console_scripts': [
                'bnn_train = bnn_inference.bnn_train:main',
                'bnn_predict = bnn_inference.bnn_predict:main',
            ],
        },

        # TODO: need to recreate dep matrix.
        # Pytorch does not support CUDA 11.5, using older 10.2 for the conda env
        install_requires = [
            "blitz-bayesian-pytorch==0.2.7",
            "numpy==1.19.0", # check for update of pandas
            "pandas==0.25.3",
            "torch==1.7.0",  # check for a newer version of torch, supporting multi-GPU and improved queries on allocated memory
            "torchvision==0.8.1",
            "scikit-learn==0.23.1", # scikit was bumped to 1.XX series. Maybe worth doing the update with the rest of the packages
            "scipy==1.5.0",
        ]
)

if __name__ == "__main__":
    run_setup()
