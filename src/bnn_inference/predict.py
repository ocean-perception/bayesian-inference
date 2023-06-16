# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, Ocean Perception Lab, Univ. of Southampton
All rights reserved.
Licensed under GNU General Public License v3.0
See LICENSE file in the project root for full license information.
"""
# Author: Jose Cappelletto (j.cappelletto@soton.ac.uk)

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from bnn_inference.tools.bnn_model import BayesianRegressor
from bnn_inference.tools.console import Console
from bnn_inference.tools.predictor import PredictiveEngine
from bnn_inference.train import get_torch_device


def predict_impl(
    latent_csv,
    latent_key,
    target_key,
    output_csv,
    output_network_filename,
    num_samples,
    scale_factor,
    gpu_index,
    cpu_only,
):
    Console.info(
        "Bayesian NN inference module. Predicting hi-res terrain maps from lo-res features"
    )

    # we are in prediction (inference) mode
    Console.info(
        "Prediction mode enabled. Looking for pretained network and input latent vectors"
    )
    # Looking for CSV with latent vectors (input)
    if os.path.isfile(latent_csv):
        Console.info("Latent input file: ", latent_csv)
    else:
        Console.error(
            "Latent input file ["
            + latent_csv
            + "] not found. Please check the provided input path (-l, --latent)"
        )

    # check for pre-trained network
    if os.path.isfile(output_network_filename):
        Console.info("Pre-trained network file [", output_network_filename, "] found")
    else:
        Console.quit("No pre-trained network found at: ", output_network_filename)

    if output_csv == "":
        date_str = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
        output_csv = date_str + "_bnn_predictions.csv"

    # if output file exists, warn user
    if os.path.isfile(output_csv):
        Console.warn(
            "Output file [",
            output_csv,
            "] already exists. It will be overwritten (default action)",
        )
    else:
        Console.info("Output file: ", output_csv)
    # ELBO k-sampling for posterior estimation. The larger the better the MLE, but slower. Good range: 5~25
    # WARNING: current MLE relies on the assumption that the predicted output follows a symmetric distribution, spec. Gaussian.
    if num_samples:
        k_samples = num_samples
    else:
        k_samples = 20
    # this is the 'key' that is used to identity the target output (single) or the column name for the predictions
    if target_key:
        output_key = target_key
    else:
        output_key = "predicted"
    # user defined keyword (affix) employed to detect the columns containing our input values (latent space representation of the bathymetry images)
    if latent_key:
        input_key = latent_key
    else:
        input_key = "latent_"  # default expected from LGA based pipeline
    # if necessary, the user can provide a scaling factor for the input values
    if scale_factor:
        scaling_factor = scale_factor
    else:
        scaling_factor = 1.0

    Console.info("Loading latent input [", latent_csv, "]")
    np_latent, n_latents, df = PredictiveEngine.loadData(
        latent_csv, input_key_prefix="latent_"
    )

    Console.info("Loading pretrained network [", output_network_filename, "]")
    device = get_torch_device(gpu_index, cpu_only)

    if torch.cuda.is_available():
        Console.info("Using CUDA")
        trained_network = torch.load(
            output_network_filename
        )  # load pretrained model (dictionary)
        # we need to determine the number of outputs by looking at the linear_output layer
        output_size = len(
            trained_network["model_state_dict"]["linear_output.weight"]
            .data.cpu()
            .numpy()
        )
        regressor = BayesianRegressor(n_latents, output_size).to(
            device
        )  # Single output being predicted
    else:
        Console.warn("Using CPU")
        trained_network = torch.load(
            output_network_filename, map_location=torch.device("cpu")
        )  # load pretrained model (dictionary)
        output_size = len(
            trained_network["model_state_dict"]["linear_output.weight"]
            .data.cpu()
            .numpy()
        )
        regressor = BayesianRegressor(n_latents, output_size).to(
            device
        )  # Single output being predicted

    regressor.load_state_dict(
        trained_network["model_state_dict"]
    )  # load state from deserialized object
    regressor.eval()  # switch to inference mode (set dropout layers)

    # Show information about the model dictionary
    # Model dictionary contains:
    # model_dict = {'epochs': num_epochs,
    #               'batch_size': data_batch_size,
    #               'learning_rate': learning_rate,
    #               'lambda_fit_loss': lambda_fit_loss,
    #               'elbo_kld': elbo_kld,
    #               'model_state_dict': regressor.state_dict()}

    print(
        "Model dictionary loaded network ||"
    )  # For each key in the dictionary, we can check if defined and show warning if not
    print("\tEpochs: ", trained_network["epochs"])
    print("\tBatch size: ", trained_network["batch_size"])
    print("\tLearning rate: ", trained_network["learning_rate"])
    print("\tLambda fit loss: ", trained_network["lambda_fit_loss"])
    print("\tELBO k-samples: ", trained_network["elbo_kld"])

    # Apply any pre-existing scaling factor to the input
    X_norm = np_latent  # for large latents, input to the network
    print("X_norm [min,max]", np.amin(X_norm), "/", np.amax(X_norm))

    # Then, check the dataframe which should contain the same ordered rows from the latent space (see final step of training/validation)
    idx = 0
    Xp_ = torch.tensor(X_norm).float()  # convert normalized intput vector into tensor

    ########################################################################

    # Network is pretrained so we start inferring
    uncertainty = []
    predicted = []  # == y

    # For every input (row) we draw a K samples from the posterior
    for x in Xp_:
        predictions = []
        for n in range(k_samples):
            y_ = regressor(x.to(device)).detach().cpu().numpy()
            # p = regressor(x.to(device)).item()
            predictions.append(y_)  # 1D output, retieve single item

        p_mean = np.mean(predictions, axis=0) * scaling_factor
        p_stdv = np.std(predictions, axis=0) * scaling_factor
        predicted.append(p_mean)
        uncertainty.append(p_stdv)

        idx = idx + 1
        Console.progress(idx, len(Xp_))

    ########################################################################
    ########################################################################
    print("Total predicted rows: ", len(predicted))

    # we repeat this for predicted and uncertainty
    column_names = []
    for i in range(
        output_size
    ):  # for each entry 'i' we create a column with the name 'y_i'
        # the column names is created by prepending 'p_' to the column names of the y_df
        column_names.append(
            "pred_" + output_key + "_" + str(i)
        )  # TODO: use the same naming convention as in the training dataframe (retrieved from NN model dictionary maybe?)
        # column_names.append('predicted_' + str(i))
    _pdf = pd.DataFrame(predicted, columns=column_names)

    # we repeat this for predicted and uncertainty
    column_names = []
    for i in range(
        output_size
    ):  # for each entry 'i' we create a column with the name 'y_i'
        # the column names is created by prepending 'p_' to the column names of the y_df
        column_names.append(
            "std_" + output_key + "_" + str(i)
        )  # TODO: use the same naming convention as in the training dataframe (retrieved from NN model dictionary maybe?)
        # column_names.append('predicted_' + str(i))
    _udf = pd.DataFrame(uncertainty, columns=column_names)

    pred_df = df.copy()  # make a copy, then we append the results
    pred_df = pd.concat([pred_df, _pdf], axis=1)
    pred_df = pd.concat([pred_df, _udf], axis=1)
    new_cols = pred_df.columns.values
    new_cols[
        0
    ] = "uuid"  # this should be the first non-index column, expected to be the uuid

    # Let's clean the dataframe before exporting it
    # 1- Drop the latent vector (as it can be massive and the is no need for most of our maps and pred calculations)
    pred_df.drop(
        list(
            pred_df.filter(regex=input_key)
        ),  # the regex string could be updated to match any user-defined latent vector name
        axis=1,  # search in columns
        inplace=True,
    )  # replace the current df, no need to reassign to a new variable

    print(pred_df.head())
    output_name = output_csv
    Console.info("Exporting predictions to:", output_name)
    pred_df.index.names = ["index"]
    pred_df.to_csv(output_name)
    Console.info("Done!")
    return 0
