[![bnn_inference](https://github.com/ocean-perception/bnn_inference/actions/workflows/bnn_inference.yml/badge.svg)](https://github.com/ocean-perception/bnn_inference/actions/workflows/bnn_inference.yml)
[![Docker](https://github.com/ocean-perception/bnn_inference/actions/workflows/docker.yml/badge.svg)](https://github.com/ocean-perception/bnn_inference/actions/workflows/docker.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# bnn_inference
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
$ git clone https://github.com/cappelletto/bnn_inference.git
```

## Docker image
For improved stability and compatibility, using docker is recommended. You can pull the latest docker image with:

```bash
docker pull ocean-perception/bnn_inference:latest
```

and run it using our alias:

```bash
alias bnn_inference='docker run --rm -it --ipc=private -e USER=$(whoami) -h $HOSTNAME --user $(id -u):$(id -g) --volume $(pwd):/data -v /etc/passwd:/etc/passwd:ro --name=bnn_$(whoami)_$(date +%Y%m%d_%H%M%S) ghcr.io/ocean-perception/bnn_inference:latest'
```

## As a python package
You can also install the package natively in your computer using python. You can install it bare bones or in a virtual environment running the following command from the root of bnn_inference.

```bash
pip install -r requirements.txt
pip install -U -e .
```

# Usage
The current implementation is separated into tree commands: train, predict and join_predictions. They use the same syntax to define inputs, outputs, training/inference parameters, etc. For a complete list of all the available features and flags please run any of the modules with the ` --help ` flag

## Configuration file
To supply the program arguments, you can use a YAML configuration file like the one below. Alternatively, you can also provide some or all argument via command-line.

```yaml
# Which GPU to use
gpu_index: 0

# Input files
## Latent csv file
latent_csv: /home/username/latent.csv
latent_key: latent_

## Target csv file
target_csv: /home/username/target.csv
target_key: prediction

## Key to match the latent and target csv files
uuid_key: relative_path

# Training parameters
num_epochs: 100
num_samples: 10
xratio: 0.9
scale_factor: 1.0
learning_rate: 1e-3
lambda_recon: 10.0
lambda_elbo: 1.0
```

## Main program:

The main program is called `bnn_inference` and it is used to dispatch the other three commands. To see the list of available commands, run:

```bash
$ bnn_inference -h

     ● ●  Ocean Perception
     ● ▲  University of Southampton

 Copyright (C) 2022-2023 University of Southampton
 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to
 redistribute it.

 INFO ▸ Running bnn_inference version 0.1.0

 Usage: bnn_inference [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion            Install completion for the current shell.                        │
│ --show-completion               Show completion for the current shell, to copy it or customize   │
│                                 the installation.                                                │
│ --help                -h        Show this message and exit.                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ join_predictions                                                                                 │
│ predict                                                                                          │
│ train                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Training:

To train a new model, run the following command:

```bash

```bash
$ bnn_inference train -h

     ● ●  Ocean Perception
     ● ▲  University of Southampton

 Copyright (C) 2022-2023 University of Southampton
 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to
 redistribute it.

 INFO ▸ Running bnn_inference version 0.1.0

 Usage: bnn_inference train [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│    --config                                        TEXT     Path to a YAML configuration file.   │
│                                                             You can use the file exclusively or  │
│                                                             overwrite any arguments via CLI.     │
│ *  --latent-csv                                    TEXT     Path to CSV containing the latent    │
│                                                             representation vector for each input │
│                                                             entry (image). The 'UUID' is used to │
│                                                             match against the target file        │
│                                                             entries                              │
│                                                             [default: None]                      │
│                                                             [required]                           │
│    --latent-key                                    TEXT     Name of the key used for the columns │
│                                                             containing the latent vector. For    │
│                                                             example, a h=8 vector should be read │
│                                                             as 'latent_0,latent_1,...,latent_7'  │
│                                                             [default: latent_]                   │
│ *  --target-csv                                    TEXT     Path to CSV containing the target    │
│                                                             entries to be used for               │
│                                                             training/validation. The 'UUID' is   │
│                                                             used to match against the input file │
│                                                             entries                              │
│                                                             [default: None]                      │
│                                                             [required]                           │
│ *  --target-key                                    TEXT     Keyword that defines the field to be │
│                                                             learnt/predicted. It must match the  │
│                                                             column name in the target file       │
│                                                             [default: None]                      │
│                                                             [required]                           │
│    --uuid-key                                      TEXT     Unique identifier string used as key │
│                                                             for input/target example matching.   │
│                                                             The UUID string must match for both  │
│                                                             the input (latent) file and the      │
│                                                             target file column identifier        │
│                                                             [default: relative_path]             │
│    --output-csv                                    TEXT     File containing the expected and     │
│                                                             inferred value for each input entry. │
│                                                             It preserves the input file columns  │
│                                                             and appends the corresponding        │
│                                                             prediction                           │
│    --output-network-filename                       TEXT     Output path to write the trained     │
│                                                             Bayesian Neural Network in PyTorch   │
│                                                             compatible format.                   │
│    --logfile-name                                  TEXT     Output path to the logfile with the  │
│                                                             training / validation error for each │
│                                                             epoch. Used to inspect the training  │
│                                                             performance                          │
│    --num-epochs                                    INTEGER  Defines the number of training       │
│                                                             epochs                               │
│                                                             [default: 100]                       │
│    --num-samples                                   INTEGER  Defines the number of samples for    │
│                                                             sample_elbo based posterior          │
│                                                             estimation                           │
│                                                             [default: 10]                        │
│    --xratio                                        FLOAT    Defines the training (T) ratio as    │
│                                                             the proportion of the complete       │
│                                                             dataset used for training. T + V =   │
│                                                             1.0                                  │
│                                                             [default: 0.9]                       │
│    --scale-factor                                  FLOAT    Defines the output target scaling    │
│                                                             factor. Default: 1.0 (no scaling))   │
│                                                             [default: 1.0]                       │
│    --learning-rate                                 FLOAT    Defines the learning rate for the    │
│                                                             optimizer                            │
│                                                             [default: 0.001]                     │
│    --lambda-recon                                  FLOAT    Defines the lambda value for the     │
│                                                             reconstruction loss.                 │
│                                                             [default: 10.0]                      │
│    --lambda-elbo                                   FLOAT    Defines the lambda value for the     │
│                                                             ELBO KL divergence cost              │
│                                                             [default: 1.0]                       │
│    --gpu-index                                     INTEGER  Index of CUDA device to be used.     │
│                                                             [default: 0]                         │
│    --cpu-only                     --no-cpu-only             If set, the training will be         │
│                                                             performed on the CPU. This is useful │
│                                                             for debugging purposes.              │
│                                                             [default: no-cpu-only]               │
│    --help                     -h                            Show this message and exit.          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

```
## Predict
To predict the target value for a given latent representation, run the following command:

```bash
$ bnn_inference predict -h

     ● ●  Ocean Perception
     ● ▲  University of Southampton

 Copyright (C) 2022-2023 University of Southampton
 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to
 redistribute it.

 INFO ▸ Running bnn_inference version 0.1.0

 Usage: bnn_inference predict [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│    --config                                        TEXT     Path to a YAML configuration file.   │
│                                                             You can use the file exclusively or  │
│                                                             overwrite any arguments via CLI.     │
│ *  --latent-csv                                    TEXT     Path to CSV containing the latent    │
│                                                             representation vector for each input │
│                                                             entry (image). The 'UUID' is used to │
│                                                             match against the target file        │
│                                                             entries                              │
│                                                             [default: None]                      │
│                                                             [required]                           │
│    --latent-key                                    TEXT     Name of the key used for the columns │
│                                                             containing the latent vector. For    │
│                                                             example, a h=8 vector should be read │
│                                                             as 'latent_0,latent_1,...,latent_7'  │
│                                                             [default: latent_]                   │
│ *  --target-key                                    TEXT     Keyword that defines the field to be │
│                                                             learnt/predicted. It must match the  │
│                                                             column name in the target file       │
│                                                             [default: None]                      │
│                                                             [required]                           │
│    --output-csv                                    TEXT     File containing the expected and     │
│                                                             inferred value for each input entry. │
│                                                             It preserves the input file columns  │
│                                                             and appends the corresponding        │
│                                                             prediction                           │
│ *  --output-network-filename                       TEXT     Trained Bayesian Neural Network in   │
│                                                             PyTorch compatible format.           │
│                                                             [default: None]                      │
│                                                             [required]                           │
│    --num-samples                                   INTEGER  Defines the number of samples for    │
│                                                             sample_elbo based posterior          │
│                                                             estimation                           │
│                                                             [default: 20]                        │
│    --scale-factor                                  FLOAT    Defines the output target scaling    │
│                                                             factor. Default: 1.0 (no scaling))   │
│                                                             [default: 1.0]                       │
│    --gpu-index                                     INTEGER  Index of CUDA device to be used.     │
│                                                             [default: 0]                         │
│    --cpu-only                     --no-cpu-only             If set, the training will be         │
│                                                             performed on the CPU. This is useful │
│                                                             for debugging purposes.              │
│                                                             [default: no-cpu-only]               │
│    --help                     -h                            Show this message and exit.          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Join predictions
To join the predictions with the input file, run the following command:

```bash
bnn_inference join_predictions -h

     ● ●  Ocean Perception
     ● ▲  University of Southampton

 Copyright (C) 2022-2023 University of Southampton
 This program comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to
 redistribute it.

 INFO ▸ Running bnn_inference version 0.1.0

 Usage: bnn_inference join_predictions [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│    --config              TEXT  Path to a YAML configuration file. You can use the file           │
│                                exclusively or overwrite any arguments via CLI.                   │
│ *  --latent-csv          TEXT  Path to CSV containing the latent representation vector for each  │
│                                input entry (image). The 'UUID' is used to match against the      │
│                                target file entries                                               │
│                                [default: None]                                                   │
│                                [required]                                                        │
│ *  --target-csv          TEXT  Path to CSV containing the target entries to be used for          │
│                                training/validation. The 'UUID' is used to match against the      │
│                                input file entries                                                │
│                                [default: None]                                                   │
│                                [required]                                                        │
│ *  --target-key          TEXT  Keyword that defines the field to be learnt/predicted. It must    │
│                                match the column name in the target file                          │
│                                [default: None]                                                   │
│                                [required]                                                        │
│    --output-csv          TEXT  File containing the expected and inferred value for each input    │
│                                entry. It preserves the input file columns and appends the        │
│                                corresponding prediction                                          │
│    --help        -h            Show this message and exit.                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

[^1]: Verify you are back in the root folder of this repository
