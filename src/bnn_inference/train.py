# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, Ocean Perception Lab, Univ. of Southampton
All rights reserved.
Licensed under GNU General Public License v3.0
See LICENSE file in the project root for full license information.
"""
# Author: Jose Cappelletto (j.cappelletto@soton.ac.uk)

import os
import statistics

# Import general libraries
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import sklearn dataset parsers and samples
from sklearn.model_selection import train_test_split

from bnn_inference.tools.bnn_model import BayesianRegressor

# Toolkit specific imports
from bnn_inference.tools.console import Console
from bnn_inference.tools.dataloader import CustomDataloader

################################################################
# TODO: Automate invocation of this script from the command line
# bnn_inference train --latent-csv ${SS}/hf_sampled_all_10m_latents.csv --target-csv ${SS}/hf_sampled_closest_10m_labels.csv --target-key vector_class_ --uuid-key relative_path --num-epochs 200 --output-csv hf_CE_${EXP}.csv --output-network-filename net_CE_${EXP}.pth --log-filename log_CE_${EXP}.csv --latent-key input_latent_ --num-samples 7 --lambda-elbo 10 --lambda-recon 100 --num-epochs 300
# export EXP="elbo10_ce100
################################################################

def set_filenames(output, logfile, network, n_latents, num_epochs, n_samples):
    # for each output file, we check if user defined name is provided. If not, use default naming convention
    filename_suffix = (
        "H" + str(n_latents) + "_E" + str(num_epochs) + "_S" + str(n_samples)
    )
    # Console.warn("Suffix:", filename_suffix)
    if output is None:
        predictions_filename = "bnn_predictions_" + filename_suffix + ".csv"
    else:
        predictions_filename = output
    if os.path.isfile(predictions_filename):
        Console.warn(
            "Output file [",
            predictions_filename,
            "] already exists. It will be overwritten (default action)",
        )
    else:
        Console.info("Output file:   \t", predictions_filename)
    if logfile is None:
        log_filename = "bnn_logfile_" + filename_suffix + ".csv"
    else:
        log_filename = logfile
    if os.path.isfile(log_filename):
        Console.warn(
            "Log file [",
            log_filename,
            "] already exists. It will be overwritten (default action)",
        )
    else:
        Console.info("Log file:      \t", log_filename)
    if network is None:
        network_filename = (
            "bnn_" + filename_suffix + ".pth"
        )  # PyTorch compatible network definition file
    else:
        network_filename = network
    if os.path.isfile(network_filename):
        Console.warn(
            "Trained output [",
            network_filename,
            "] already exists. It will be overwritten (default action)",
        )
    else:
        Console.info("Trained output:\t", network_filename)

    return predictions_filename, log_filename, network_filename


def get_torch_device(gpu_index, cpu_only=False):
    if torch.cuda.is_available() and not cpu_only:
        if torch.cuda.device_count() > 1:
            if gpu_index is None or gpu_index == 0:
                device = torch.device("cuda:0")
                torch.cuda.set_device("cuda:0")
            else:
                device = torch.device("cuda:1")
                torch.cuda.set_device("cuda:1")
        else:
            device = torch.device("cuda:0")
        Console.info("CUDA detected, using device: ", device)
    else:
        Console.info("Using CPU")
        device = torch.device("cpu")
    return device


def train_impl(
    latent_csv,
    latent_key,
    target_csv,
    target_key,
    uuid_key,
    output_csv,
    output_network_filename,
    log_filename,
    num_epochs,
    num_samples,
    xratio,
    scale_factor,
    learning_rate,
    lambda_recon,
    lambda_elbo,
    loss_method,
    gpu_index,
    cpu_only,
):
    Console.info(
        "Bayesian NN training module: learning hi-res terrain observations from feature representation of low resolution priors"
    )

    Console.info("Loading dataset: " + latent_csv)
    X_df, y_df, index_df = CustomDataloader.load_dataset(
        input_filename=latent_csv,  # dataset containing the input. e.g. the latent vector
        target_filename=target_csv,  # target dataset containing the key to be predicted, e.g. mean_slope
        matching_key=uuid_key,
        target_key_prefix=target_key,
        input_key_prefix=latent_key,
    )  # relative_path is the common key in both tables

    X = X_df.to_numpy(
        dtype=np.float64
    )  # Explicit numeric data conversion to avoid silent bugs with implicit string conversion
    y = y_df.to_numpy(dtype=np.float64)  # Apply to both target and latent data
    # We need to peek the number of latent variables to configure the network and set up the filenames
    n_latents = X.shape[
        1
    ]  # this is the only way to retrieve the size of input latent vectors
    n_pairs = X.shape[0]  # number of pairs in the dataset
    # If the number of loaded pairs is zero, print error and exit
    if n_pairs == 0:
        Console.error("No data loaded. Check input and target files")
        sys.exit(1)
    # If the number of loaded pairs is less than the batch size, print warning and exit
    data_batch_size = 1
    if n_pairs < data_batch_size:
        # print the error message with the number of pairs and the batch size
        Console.error(
            "The number of loaded pairs [",
            n_latents,
            "] is less than the batch size ",
            data_batch_size,
            "]. Check input and target files",
        )
        sys.exit(1)

    Console.info("Data loaded...")

    # set the filenames for the model and the training log
    predictions_filename, log_filename, network_filename = set_filenames(
        output_csv,
        log_filename,
        output_network_filename,
        n_latents,
        num_epochs,
        num_samples,
    )

    # To maintain equivalent metrics for normal, log-normal data, we need to normalize the data
    # However the normalization must be reversible at prediction and interpretration time.
    # We can used fixed normalization IF we know the data range (e.g. 0-90 degrees for slope)

    # Let's use existing MinMaxScaler, but we need to manually set the parameters
    # X = StandardScaler().fit_transform(X)
    # y = StandardScaler().fit_transform(np.expand_dims(y, -1)) # this is resizing the array so it can match Size (D,1) expected by pytorch
    # norm = MinMaxScaler().fit(y)
    # y_norm = norm.transform(y)      # min max normalization of our output data
    # y_norm = y
    # norm = MinMaxScaler().fit(X)
    # X_norm = norm.transform(X)      # min max normalization of our input data
    X_norm = X
    Console.warn("Xnorm_Shape", X_norm.shape)
    Console.warn("y_shape", y.shape)

    # We impose fixed normalization for the input data, as we know the expected data range.
    # Warning: we do not use the data to fit the scaler as there is no guarantee that the data sample covers all the expected range
    y_norm = y / scale_factor

    n_latents = X_norm.shape[1]  # retrieve the size of input latent vectors
    n_targets = y_norm.shape[1]  # retrieve the size of output targets
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    print(
        "X_orig [min,max]: ",
        "{:.4}".format(np.amin(X)),
        "/",
        "{:.4}".format(np.amax(X)),
    )
    print(
        "X_norm [min,max]: ",
        "{:.4}".format(np.amin(X_norm)),
        "/",
        "{:.4}".format(np.amax(X_norm)),
    )
    print(
        "Y_orig [min,max]: ",
        "{:.4}".format(np.amin(y)),
        "/",
        "{:.4}".format(np.amax(y)),
    )
    print(
        "Y_norm [min,max]: ",
        "{:.4}".format(np.amin(y_norm)),
        "/",
        "{:.4}".format(np.amax(y_norm)),
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_norm, y_norm, test_size=xratio, shuffle=True  # 8:2 ratio
    )
    # Convert train and test vectors to tensors
    X_train, y_train = torch.Tensor(X_train).float(), torch.Tensor(y_train).float()
    X_valid, y_valid = torch.Tensor(X_valid).float(), torch.Tensor(y_valid).float()
    y_train = torch.unsqueeze(
        y_train, -1
    )  # PyTorch will complain if we feed the (N).Tensor rather than a (NX1).Tensor
    y_valid = torch.unsqueeze(y_valid, -1)  # we add an additional dummy dimension

    device = get_torch_device(gpu_index, cpu_only)

    # set the device
    Console.warn("Using device:", torch.cuda.current_device())
    regressor = BayesianRegressor(n_latents, n_targets).to(device)
    # regressor.init
    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)  # learning rate
    # criterion = (
    #     torch.nn.MSELoss()
    # )  # mean squared error loss (squared L2 norm). Used to compute the regression fitting error
    # # criterion = torch.nn.CosineEmbeddingLoss()  # cosine similarity loss

    criterion = torch.nn.CrossEntropyLoss()
        
    # criterion = torch.nn.MSELoss()

    # print("Model's state_dict:")
    # for param.Tensor in regressor.state_dict():
    #     print(param.Tensor, "\t", regressor .state_dict()[param.Tensor].size())
    data_batch_size = 18

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=data_batch_size, shuffle=True
    )

    ds_valid = torch.utils.data.TensorDataset(X_valid, y_valid)
    dataloader_valid = torch.utils.data.DataLoader(
        ds_valid, batch_size=data_batch_size, shuffle=True
    )

    # Log of training and validation losses
    train_loss_history = []
    train_fit_loss_history = []
    train_kld_loss_history = []
    valid_loss_history = []
    valid_fit_loss_history = []
    valid_kld_loss_history = []

    lambda_fit_loss = lambda_recon  # regularization parameter for the fit loss (cost function is the sum of the scaled fit loss and the KL divergence loss)
    elbo_kld = lambda_elbo  # regularization parameter for the KL divergence loss

    # Print the regularisation parameters (lambda)
    print("MSE-Loss lambda: ", lambda_fit_loss)
    print("ELBO KLD lambda: ", elbo_kld)
    # Print the asked number of samples
    print("Number of samples: ", num_samples)
    # Configure the model for training    
    regressor.train()  # set to training mode, just in case
    # regressor.freeze_() # while frozen, the network will behave as a normal network (non-Bayesian)
    regressor.unfreeze_()  # we no longer start with "warming-up" phase of non-Bayesian training

    # Create customized criterion function
    # Add output layer normalization option: L1 or L2 norm
    # Add option to configure cosine or MSELoss
    # Improve constant torch.ones for CosineEmbeddingLoss, or juts use own cosine distance loss (torch compatible)

    if loss_method == "mse":
        regressor_sample_elbow_weighed = regressor.sample_elbo_weighted_mse
    elif loss_method == "cosine_similarity":
        regressor_sample_elbow_weighed = regressor.sample_elbo_weighted_cos_sim
    else:
        Console.error("Unknown loss_regularisation_method:", loss_regularisation_method)
        Console.error("Valid options are: mse, cosine_similarity")
        Console.quit("Loss regularisation method not supported")

    # print ("Forcing to use sample_elbo ========================")
    # regressor_sample_elbow_weighed = regressor.sample_elbo

    try:
        for epoch in range(num_epochs):
            # if (epoch == 2):          # we train in non-Bayesian way during a first phase of P-epochs (P:50) as 'warm-up'
            #     regressor.unfreeze_()
            #     Console.info("Unfreezing the network")

            # We store a list of losses for each epoch (multiple samples per epoch)
            train_loss = []
            valid_loss = []
            # Loss (cost) values are separated into fit_loss and kld_loss
            train_fit_loss = []
            train_kld_loss = []
            valid_fit_loss = []
            valid_kld_loss = []

            for i, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                # labels.shape = (h,1,1) is adding an extra dimension to the tensor, so we need to remove it
                labels = labels.squeeze(2)
                _loss, _fit_loss, _kld_loss = regressor_sample_elbow_weighed(
                    inputs=datapoints.to(device),
                    labels=labels.to(device),
                    criterion=criterion,  # MSELoss
                    sample_nbr=num_samples,
                    criterion_loss_weight=lambda_fit_loss,
                    complexity_cost_weight=elbo_kld / X_train.shape[0],
                )  # normalize the complexity cost by the number of input points
                _loss.backward()  # the returned loss is the combination of fit loss (MSELoss) and complexity cost (KL_div against a nominal Normal distribution )
                optimizer.step()
                train_loss.append(_loss.item())  # keep track of training loss
                train_fit_loss.append(_fit_loss.item())
                # When the network is frozen the complexity cost is not computed and the kld_loss is 0
                # The problem is that the return type changes from a Tensor to a scalar
                if type(_kld_loss) is torch.Tensor:
                    train_kld_loss.append(_kld_loss.item())
                else:
                    train_kld_loss.append(0.0)

            for k, (valid_datapoints, valid_labels) in enumerate(dataloader_valid):
                # calculate the fit loss and the KL-divergence cost for the test points set
                valid_labels = valid_labels.squeeze(2)
                _loss, _fit_loss, _kld_loss = regressor_sample_elbow_weighed(
                    inputs=valid_datapoints.to(device),
                    labels=valid_labels.to(device),
                    criterion=criterion,
                    sample_nbr=num_samples,
                    criterion_loss_weight=lambda_fit_loss,  # regularization parameter to balance multiobjective cost function (fit loss vs KL div)
                    complexity_cost_weight=elbo_kld / X_valid.shape[0],
                )
                valid_loss.append(_loss.item())  # keep track of training loss
                valid_fit_loss.append(_fit_loss.item())
                # When the network is frozen the complexity cost is not computed and the kld_loss is 0
                # The problem is that the return type changes from a Tensor to a scalar
                if type(_kld_loss) is torch.Tensor:
                    valid_kld_loss.append(_kld_loss.item())
                else:
                    valid_kld_loss.append(0.0)

            mean_train_loss = statistics.mean(train_loss)
            mean_valid_loss = statistics.mean(valid_loss)
            mean_train_fit_loss = statistics.mean(train_fit_loss)
            mean_valid_fit_loss = statistics.mean(valid_fit_loss)
            mean_train_kld_loss = statistics.mean(train_kld_loss)
            mean_valid_kld_loss = statistics.mean(valid_kld_loss)

            # Log of training and validation losses
            train_loss_history.append(mean_train_loss)
            train_fit_loss_history.append(mean_train_fit_loss)
            train_kld_loss_history.append(mean_train_kld_loss)

            valid_loss_history.append(mean_valid_loss)
            valid_fit_loss_history.append(mean_valid_fit_loss)
            valid_kld_loss_history.append(mean_valid_kld_loss)

            Console.info(
                "Epoch ["
                + str(epoch)
                + "] Train (MSE + KLD): {:.3f}".format(mean_train_loss)
                + " = {:.3f}".format(mean_train_fit_loss)
                + " + {:.3f}".format(mean_train_kld_loss)
                + "    | Valid (MSE + KLD): {:.3f}".format(mean_valid_loss)
                + " = {:.3f}".format(mean_valid_fit_loss)
                + " + {:.3f}".format(mean_valid_kld_loss)
            )
            Console.progress(epoch, num_epochs)

    except KeyboardInterrupt:
        Console.warn("Training interrupted...")
        # sys.exit()

    Console.info("Training completed. Saving the model...")
    # create dictionary with the trained model and some training parameters
    model_dict = {
        "epochs": num_epochs,
        "batch_size": data_batch_size,
        "learning_rate": learning_rate,
        "lambda_fit_loss": lambda_fit_loss,
        "elbo_kld": elbo_kld,
        "optimizer": optimizer.state_dict(),
        "model_state_dict": regressor.state_dict(),
    }

    print("Network name:", output_network_filename)
    torch.save(model_dict, output_network_filename)

    export_df = pd.DataFrame(
        [
            train_loss_history,
            train_fit_loss_history,
            train_kld_loss_history,
            valid_loss_history,
            valid_fit_loss_history,
            valid_kld_loss_history,
        ]
    ).transpose()
    export_df.columns = [
        "train_loss",
        "train_fit_loss",
        "train_kld_loss",
        "valid_loss",
        "valid_fit_loss",
        "valid_kld_loss",
    ]
    export_df.index.names = ["index"]
    export_df.to_csv(log_filename, index=False)

    idx = 0
    # for x in X_valid:
    regressor.eval()  # we need to set eval mode before running inference
    # this will set dropout and batch normalization (if any) to evaluation mode

    Console.info("Testing predictions [train dataset]...")
    # Xp_ = torch.Tensor(X_norm).float()
    Xt_ = torch.Tensor(X_train).float().to(device)
    Xv_ = torch.Tensor(X_valid).float().to(device)

    Xp_ = Xt_
    y_list = (
        y_train.squeeze().tolist()
    )  # when converted to list, the shape is (N,) and will be stored in the same "cell" of the dataframe

    uncertainty = []
    predicted = []  # == y
    for x in Xp_:
        predictions = []
        for n in range(num_samples):
            # TODO: verify mismatch when training using batch_size > 1
            # Add a dimension to the input data to match the input shape of the network
            x_ = x.unsqueeze(0)
            y_ = regressor(x_.to(device)).detach().cpu().numpy()
            predictions.append(
                y_
            )  # N-dimensional output, stack/append as "single item"
            # print (x_)
            # print (y_)
            # print (str(n), " ------------------------------")

        p_mean = np.mean(predictions, axis=0)
        p_stdv = np.std(predictions, axis=0)
        predicted.append(p_mean)
        uncertainty.append(p_stdv)

        idx = idx + 1
        Console.progress(idx, len(Xp_))

    # predicted might contain a dimension with size 1, we need to squeeze it
    predicted = np.squeeze(predicted)

    # y_list, predicted and uncertainty lists need to be converted into sub-dataframes with as many columns as n_targets
    column_names = []
    for i in range(
        n_targets
    ):  # for each entry 'i' we create a column with the name 'y_i'
        column_names.append("target_" + y_df.columns[i])

    _ydf = pd.DataFrame(y_list, columns=column_names)

    # we repeat this for predicted and uncertainty
    column_names = []
    for i in range(
        n_targets
    ):  # for each entry 'i' we create a column with the name 'y_i'
        # the column names is created by prepending 'p_' to the column names of the y_df
        column_names.append("pred_" + y_df.columns[i])
        # column_names.append('predicted_' + str(i))
    _pdf = pd.DataFrame(predicted, columns=column_names)

    column_names = []
    for i in range(
        n_targets
    ):  # for each entry 'i' we create a column with the name 'y_i'
        column_names.append("uncertainty_" + y_df.columns[i])
        # column_names.append('uncertainty_' + str(i))

    # predicted might contain a dimension with size 1, we need to squeeze it
    uncertainty = np.squeeze(uncertainty)

    _udf = pd.DataFrame(uncertainty, columns=column_names)

    # Append _ydf dataframe to pred_df
    pred_df = _ydf
    pred_df = pd.concat([pred_df, _pdf], axis=1)
    # Check if --uncertainty flag is set
    # if args.uncertainty:
    pred_df = pd.concat([pred_df, _udf], axis=1)

    Console.warn(
        "Exported [train dataset] predictions to: ", "train_" + predictions_filename
    )
    pred_df.to_csv("train_" + predictions_filename, index=False)

    ######################################################################################################################
    # We repeat the same procedure for the validation dataset
    ######################################################################################################################

    Console.info("Testing predictions [validation dataset]...")
    Xp_ = Xv_
    y_list = (
        y_valid.squeeze().tolist()
    )  # when converted to list, the shape is (N,) and will be stored in the same "cell" of the dataframe

    # Once trained, we start inferring
    # expected = []
    uncertainty = []
    predicted = []  # == y
    idx = 0
    for x in Xp_:
        predictions = []
        for n in range(num_samples):
            x_ = x.unsqueeze(0)
            y_ = regressor(x_.to(device)).detach().cpu().numpy()
            predictions.append(
                y_
            )  # N-dimensional output, stack/append as "single item"
        p_mean = np.mean(predictions, axis=0)
        p_stdv = np.std(predictions, axis=0)
        predicted.append(p_mean)
        uncertainty.append(p_stdv)

        idx = idx + 1
        Console.progress(idx, len(Xp_))

    # y_list, predicted and uncertainty lists need to be converted into sub-dataframes with as many columns as n_targets
    column_names = []
    for i in range(
        n_targets
    ):  # for each entry 'i' we create a column with the name 'y_i'
        column_names.append("target_" + y_df.columns[i])

    _ydf = pd.DataFrame(y_list, columns=column_names)

    # we repeat this for predicted and uncertainty
    column_names = []
    for i in range(
        n_targets
    ):  # for each entry 'i' we create a column with the name 'y_i'
        # the column names is created by prepending 'p_' to the column names of the y_df
        column_names.append("pred_" + y_df.columns[i])
        # column_names.append('predicted_' + str(i))
    # predicted might contain a dimension with size 1, we need to squeeze it
    predicted = np.squeeze(predicted)
    _pdf = pd.DataFrame(predicted, columns=column_names)

    column_names = []
    for i in range(
        n_targets
    ):  # for each entry 'i' we create a column with the name 'y_i'
        column_names.append("uncertainty_" + y_df.columns[i])
    # predicted might contain a dimension with size 1, we need to squeeze it
    uncertainty = np.squeeze(uncertainty)
    _udf = pd.DataFrame(uncertainty, columns=column_names)

    # Finally, let's append _ydf dataframe to pred_df
    # pred_df = pd.concat([pred_df, _ydf], axis=1)
    pred_df = _ydf
    pred_df = pd.concat([pred_df, _pdf], axis=1)
    # Check if --uncertainty flag is set
    # if args.uncertainty:
    pred_df = pd.concat([pred_df, _udf], axis=1)

    Console.warn(
        "Exported [validation dataset] predictions to: ", "valid_" + predictions_filename
    )
    pred_df.to_csv("valid_" + predictions_filename, index=False)
