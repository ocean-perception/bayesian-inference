# bnn-geotech-predictor
Predictive engine that can infer terrain properties (e.g. slope, vehicle landability) from remotely
sensed topo-bathymetry maps. A Bayesian neural network [https://github.com/piEsposito/blitz-bayesian-deep-learning/] is used to model the relationship between a 
compact representation of the terrain and the target output. Terrain features (descriptors) can be obtained from a pre-trained Location Guided Autoencoder [https://github.com/ocean-perception/location_guided_autoencoder]

# Requirements
These are the system requeriments for a typ. small network (5 layers, 256 nodes per layer max)
* CUDA 10+ capable GPU
* 8 GB of RAM
* 2C/4T CPU

Separate section for dependencies (sw)

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
