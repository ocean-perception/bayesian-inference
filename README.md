[![bnn_inference](https://github.com/ocean-perception/bnn_inference/actions/workflows/bnn_inference.yml/badge.svg)](https://github.com/ocean-perception/bnn_inference/actions/workflows/bnn_inference.yml)
[![Docker](https://github.com/ocean-perception/bnn_inference/actions/workflows/docker.yml/badge.svg)](https://github.com/ocean-perception/bnn_inference/actions/workflows/docker.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Description
A small ML framework capable of inferring high-resolution properties of the terrain from low-resolution priors. The framework is based on Bayesian Neural Networks (BNN) that learns the relation between latent representation of the priors and the target properties. The framework is designed to be used in conjunction with georef_semantics [https://github.com/ocean-perception/georef_semantics], but it can be used with any other latent representation.
Some of the currently tested applications are:
- Inferring high-resolution terrain slope, rugosity and roughness from low-resolution bathymetry
- Predicting vehicle landability (e.g. underwater, aerial and planetary landers) from low-resolution bathymetry
- Inferring seafloor habitat classes from low-resolution acoustics (MBES, SSS)
- Predicting high-resolution optical classes distribution from acoustic priors

In case you use this framework in your research, please cite the following paper [https://ieeexplore.ieee.org/document/9705711]:

```bibtex
@inproceedings{cappelletto_autonomous_2021,
	title = {Autonomous {Identification} of {Suitable} {Geotechnical} {Measurement} {Locations} using {Underwater} {Vehicles}},
	doi = {10.23919/OCEANS44145.2021.9705711},
	booktitle = {{OCEANS} 2021: {San} {Diego} – {Porto}},
	author = {Cappelletto, Jose and Thornton, Blair and Bodenmann, Adrian and Yamada, Takaki and Massot-Campos, Miquel and Sangekar, Mehul and White, David and Dix, Justin and Newborough, Darryl},
	year = {2021},
	note = {ISSN: 0197-7385},
}
```

Please see the [Credits](#Credits) section at the end of this document.

# Requirements
Current implementation uses a small fully connected neural network (5 layers, 256 nodes per layer max). The GPU memory footprint is ~500MB so multiple train/predict instances can be dispatched at the same time. The minimum required system is:

* GPU card with >1 GB
* CUDA 10+
* 8 GB of RAM
* 2C/4T CPU

You can use CPU-only mode in case you do not have a GPU card. Additionally, by reducing the number of Monte Carlo samples, you can reduce the training and inference time. However, the quality of your predictions in terms of uncertainty will be affected.

# Installation

Start by cloning the repository (it will include the blitz submodule):

```
$ git clone https://github.com/ocean-perception/bnn_inference.git
```

## Docker image
For improved stability and compatibility, using docker is recommended. You can pull the latest docker image with:

```bash
docker pull ghcr.io/ocean-perception/bnn_inference:latest
```

and run it using this alias:

```bash
alias bnn_inference='docker run --rm -it --ipc=private -e USER=$(whoami) -h $HOSTNAME --user $(id -u):$(id -g) --volume $(pwd):/data -v /etc/passwd:/etc/passwd:ro --name=bnn_$(whoami)_$(date +%Y%m%d_%H%M%S) ghcr.io/ocean-perception/bnn_inference:latest'
```

You can append the alias to your `~/.bashrc` file to make it permanent.

## As a python package
You can also install the package natively in your computer using python. It is recommended to install it within a virtual environment (e.g. conda, venv) running the following command from within the root folder of this repository:

```bash
pip install -r requirements.txt
pip install -U -e .
```

# Usage
(TODO: Improve this section) The current implementation is separated into three commands: train, predict and join_predictions. They use the same syntax to define inputs, outputs, training/inference parameters, etc. For a complete list of all the available features and flags please run any of the modules with the ` --help ` flag

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
lambda_loss: 10.0
lambda_elbo: 1.0
```

## Main program:

The main program `bnn_inference` is used to dispatch the other three commands. To see the list of available commands, run:

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
│    --log-filename                                  TEXT     Output path to the logfile with the  │
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

[^1]: Verify you are back in the root folder of this repository



# Credits
- The Bayesian BNN uses as backbone the *blitz* library available at [https://github.com/piEsposito/blitz-bayesian-deep-learning/] 
- Prior and target image features are using Location Guided Autoencoders and GeoCLR. The code is available at [https://github.com/ocean-perception/georef_semantics]