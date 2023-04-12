[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# bnn-geotech-predictor
A ML framework to infer high resolution terrain properties (e.g. slope, rugosity, vehicle landability) from remotely
sensed low resolution maps. A Bayesian neural network [https://github.com/piEsposito/blitz-bayesian-deep-learning/] is used to model the relationship between a compact representation of the terrain and the target output. Unsupervised terrain feature extraction is done via  Location Guided Autoencoder [https://github.com/ocean-perception/location_guided_autoencoder] or by using contrastive learning (GeoCLR).

# Requirements
Current implementation uses a small fully connected neural network (5 layers, 256 nodes per layer max). The GPU memory footprint is ~500MB so multiple train/predict instances can be dispatched. The minimum required system is

* GPU card with >1 GB
* CUDA 10+
* 8 GB of RAM
* 2C/4T CPU

# Installation
Start by cloning the repository (it will include the blitz submodule):

```
$ git clone https://github.com/cappelletto/bayesian-inference.git
```

## Conda environment
For improved stability and compatibility, using virtual environment (venv) or conda is recommended. You can create a new conda environment with:

```
$ conda create --name bnninference python=3.10
```

Once created, activate it using:

```
$ conda activate bnninference
```

## Dependencies

Now we can proceed to install the general dependencies listed in the requirements file:

```
$ pip install -r environment/requirements.txt
```

And then install the blitz submodule
```
$ cd bnn_inference/submodules/blitz/
$ python setup.py install
```

## Bayesian predictor
Finally, we can install the *bayesian-predictor* package via pip as [^1]:
```
$ cd ../..
$ python setupy. build
$ pip install .
```

# Usage
The current implementation is separated into two modules, one for training and another one for predictions. Both use the same sintaxis to define inputs, outputs, training/inference parameters, etc. For a complete list of all the available features and flags please run any of the modules with the ` --help ` flag

For the training module:
```
bnn_train.py --help
```
It will return something similar to:

```
INFO > Bayesian NN training module: learning hi-res terrain observations from feature representation of low resolution priors
usage: bnn_train [-h] [-i INPUT] [-l LATENT] [-t TARGET] [-k KEY] [-o OUTPUT] [-u UUID] [-n NETWORK] [-g LOGFILE]
                 [-c CONFIG] [-e EPOCHS] [-s SAMPLES] [-x XRATIO] [--scale SCALE] [--lr LR]
                 [--lambda_recon LAMBDA_RECON] [--lambda_elbo LAMBDA_ELBO] [--gpu GPU] [--uncertainty]

Bayesian Neural Network training module

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to CSV containing the latent representation vector for each input entry (image). The 'UUID'
                        is used to match against the target file entries
  -l LATENT, --latent LATENT
                        Name of the key used for the columns containing the latent vector. For example, a h=8 vector
                        should be read as 'latent_0,latent_1,...,latent_7'
  -t TARGET, --target TARGET
                        Path to CSV containing the target entries to be used for training/validation. The 'UUID' is used
                        to match against the input file entries
  -k KEY, --key KEY     Keyword that defines the field to be learnt/predicted. It must match the column name in the
                        target file
  -o OUTPUT, --output OUTPUT
                        File containing the expected and inferred value for each input entry. It preserves the input
                        file columns and appends the corresponding prediction
  -u UUID, --uuid UUID  Unique identifier string used as key for input/target example matching. The UUID string must
                        match for both the input (latent) file and the target file column identifier
  -n NETWORK, --network NETWORK
                        Output path to write the trained Bayesian Neural Network in PyTorch compatible format.
  -g LOGFILE, --logfile LOGFILE
                        Output path to the logfile with the training / validation error for each epoch. Used to inspect
                        the training performance
  -c CONFIG, --config CONFIG
                        Path to YAML configuration file (optional)
  -e EPOCHS, --epochs EPOCHS
                        Define the number of training epochs
  -s SAMPLES, --samples SAMPLES
                        Define the number of samples for sample_elbo based posterior estimation
  -x XRATIO, --xratio XRATIO
                        Define the training (T) ratio as the proportion of the complete dataset used for training. T + V
                        = 1.0
  --scale SCALE         Define the output target scaling factor. Default: 1.0 (no scaling))
  --lr LR               Define the learning rate for the optimizer. Default: 0.001
  --lambda_recon LAMBDA_RECON
                        Define the lambda value for the reconstruction loss. Default: 10.0
  --lambda_elbo LAMBDA_ELBO
                        Define the lambda value for the ELBO KL divergence cost. Default: 1.0
  --gpu GPU             Index of CUDA device to be used. Default: 0
  --uncertainty         Add flag to export uncertainty in the output file
```


```
bnn_predict.py --help
```


[^1]: Verify you are back in the root folder of this repository