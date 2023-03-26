[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# bnn-geotech-predictor
A ML framework to infer high resolution terrain properties (e.g. slope, rugosity, vehicle landability) from remotely
sensed low resolution maps. A Bayesian neural network [https://github.com/piEsposito/blitz-bayesian-deep-learning/] is used to model the relationship between a compact representation of the terrain and the target output. Unsupervised terrain feature extraction is done via  Location Guided Autoencoder [https://github.com/ocean-perception/location_guided_autoencoder] or by using contrastive learning (GeoCLR).

# Requirements
System requeriments for a typ. small network (5 layers, 256 nodes per layer max)
* CUDA 10+ capable GPU
* 8 GB of RAM
* 2C/4T CPU

For replicability purposes, conda is used to manage the environment setup with all the dependencies.

# Installation
TODO: Complete detailed description:
* Point to env.yaml
* Refer to conda env creation from file
* Install blitz from conda channel
* Done

# Usage
bnn_train.py --help
bnn_predict.py --help
