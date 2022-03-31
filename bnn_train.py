# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, Ocean Perception Lab, Univ. of Southampton
All rights reserved.
Licensed under GNU General Public License v3.0
See LICENSE file in the project root for full license information.
"""
# Author: Jose Cappelletto (j.cappelletto@soton.ac.uk) 

# Import general libraries
import re # Regular expressions (eventually to be deprecated)
import sys
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
# Import sklearn dataset parsers and samples
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Helper libaries (viz)
import matplotlib.pyplot as plt
# Toolkit
from tools.console import Console
from tools.dataloader import CustomDataloader
from tools.predictor import PredictiveEngine
from tools.bnn_model import BayesianRegressor
import tools.parser as par
import statistics
import math

def handler(signum, frame):
    Console.warn ("CTRL + C pressed. Stopping...")
    exit(1)

def main(args=None):
    description_str = "Bayesian Neural Network training module"
    formatter = lambda prog: argparse.HelpFormatter(prog, width=120)
    parser = argparse.ArgumentParser(description=description_str, formatter_class=formatter)
    # argparse.HelpFormatter(parser,'width=120')
    par.add_arguments(parser)

    Console.info("Bayesian Neural Network for hi-res inference from low resolution acoustic priors (LGA-Bathymetry)")
    if len(sys.argv) == 1 and args is None: # no argument passed? show help as some parameters were expected
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(args)  # retrieve parsed arguments

    # Start verifying mandatory arguments
    # [mandatory] Check if input file (latents) exists
    if os.path.isfile(args.input):
        Console.info("Latent input file:\t", args.input)
    else:
        Console.error("Latent input file [" + args.input + "] not found. Please check the provided input path (-i, --input)")
    # [mandatory] target file
    if os.path.isfile(args.target):
        Console.info("Target input file:\t", args.target)
    else:
        Console.error("Target input file [" + args.target + "] not found. Please check the provided input path (-t, --target)")

    # this is the key that is used to identity the target output (single) or the column name for the predictions
    if (args.key):
        output_key = args.key
        Console.info("Using user-defined target key:\t[", output_key, "]")
    else:
        output_key = 'measurability'
        Console.warn("Using default target key:     \t[", output_key, "]")
    # user defined keyword (prefix) employed to detect the columns containing our input values (latent space representation of the bathymetry images)
    if (args.latent):
        latent_key = args.latent
        Console.info("Using user-defined latent key:\t[", latent_key, "]")
    else:
        latent_key = 'latent_'
        Console.info("Using default latent key:     \t[", latent_key, "]")
    # user defined keyword (prefix) employed to detect the columns containing our input values (latent space representation of the bathymetry images)
    if (args.uuid):
        UUID = args.uuid
        Console.info("Using user-defined UUID:\t[", UUID, "]")
    else:
        UUID = 'relative_path'
        Console.info("Using default UUID:     \t[", UUID, "]")
    # these parameters are only used in training mode
    if (args.epochs):
        num_epochs = args.epochs
    else:
        num_epochs = 100    # default
    # number of random samples used by sample_elbo to estimate the mean/std for each inference epoch
    if (args.samples):
        # Verify the number of samples is larger than 2, otherwise Monte Carlo sampling is not possible (won't make sense)
        if (args.samples > 2):
            n_samples = args.samples
        else:
            # If the number of samples is not larger than 2, show an error and exit
            Console.error("The number of MC samples must be larger than 2. Please provide a number larger than 2 (-s, --samples)")
            exit(1)
    else:
        n_samples = 10      # default

    # Check if user specified a learning rate
    if (args.lr):
        learning_rate = args.lr
        Console.info("Using user-defined learning rate:\t[", learning_rate, "]")
    else:
        learning_rate = 0.001 # Default value

    # Check if user specified a lambda for reconstruction loss
    if (args.lambda_recon):
        lambda_recon = args.lambda_recon
        Console.info("Using user-defined lambda for reconstruction loss:\t[", lambda_recon, "]")
    else:
        lambda_recon = 10.0

    # Check if user specified a lambda for ELBO KL loss
    if (args.lambda_elbo):
        lambda_elbo = args.lambda_elbo
        Console.info("Using user-defined lambda for ELBO KL loss:\t[", lambda_elbo, "]")
    else:
        lambda_elbo = 1.0


    dataset_filename = args.input   # dataset containing the input. e.g. the latent vector
    target_filename  = args.target  # target dataset containing the key to be predicted, e.g. mean_slope
    Console.info("Loading dataset: " + dataset_filename)
    X_df, y_df, index_df = CustomDataloader.load_dataset( input_filename  = dataset_filename, 
                                                    target_filename = target_filename,
                                                    matching_key    = UUID,
                                                    target_key_prefix      = output_key,
                                                    input_key_prefix = latent_key)    # relative_path is the common key in both tables
    
    X = X_df.to_numpy(dtype='float')   # Explicit numeric data conversion to avoid silent bugs with implicit string conversion
    y = y_df.to_numpy(dtype='float')   # Apply to both target and latent data
    
    n_latents = X.shape[1]      # this is the only way to retrieve the size of input latent vectors
    Console.info("Data loaded...")

    # for each output file, we check if user defined name is provided. If not, use default naming convention
    filename_suffix = "H" + str(n_latents) + "_E" + str(num_epochs) + "_S" + str(n_samples)
    # Console.warn("Suffix:", filename_suffix)

    if (args.output is None):
        predictions_name = "bnn_predictions_" + filename_suffix +  ".csv"
    else:
        predictions_name = args.output
    if os.path.isfile(predictions_name):
        Console.warn("Output file [", predictions_name, "] already exists. It will be overwritten (default action)")
    else:
        Console.info("Output file:   \t", predictions_name)

    if (args.logfile is None):
        logfile_name = "bnn_logfile_" + filename_suffix +  ".csv"
    else:
        logfile_name = args.logfile
    if os.path.isfile(logfile_name):
        Console.warn("Log file [", logfile_name, "] already exists. It will be overwritten (default action)")
    else:
        Console.info("Log file:      \t", logfile_name)

    if (args.network is None):
        network_name = "bnn_" + filename_suffix +  ".pth"   # PyTorch compatible netwrok definition file
    else:
        network_name = args.network
    if os.path.isfile(network_name):
        Console.warn("Trained output [", network_name, "] already exists. It will be overwritten (default action)")
    else:
        Console.info("Trained output:\t", network_name)

    if (args.config):
        Console.warn("Configuration file provided:\t", args.config, " will be ignored (usage not implemented yet)")

    # For testing purposes, you can use toydataset loader
    # X, y, index_df = CustomDataloader.load_toydataset(dataset_filename, target_key_prefix = output_key, input_prefix= latent_key, matching_key='uuid')    # relative_path is the common key in both tables

    # To maintain equivalent metrics for normal, log-normal data, we need to normalize the data
    # However the normalization must be reversible at prediction and interpretration time.
    # We can used fixed normalization IF we know the data range (e.g. 0-90 degrees for slope)

    # Let's use existing MinMaxScaler, but we need to manually set the parameters
    # (min, max) = (0, 90)
    # X = StandardScaler().fit_transform(X)
    # y = StandardScaler().fit_transform(np.expand_dims(y, -1)) # this is resizing the array so it can match Size (D,1) expected by pytorch
    # norm = MinMaxScaler().fit(y)
    # y_norm = norm.transform(y)      # min max normalization of our output data
    # y_norm = y
    # norm = MinMaxScaler().fit(X)
    # X_norm = norm.transform(X)      # min max normalization of our input data
    X_norm = X

    Console.warn ("Xnorm_Shape", X_norm.shape)
    Console.warn ("y_shape", y.shape)

    # We impose fixed normalization for the input data, as we know the expected data range.
    # Warning: we do not use the data to fit the scaler as there is no guarantee that the ata sample covers all the expected range
    _d      = np.array([       0.0,         1.0])
    # _log_d  = np.array([np.log(0.01), np.log(90.0)])   # this scaler can be used to transform the data from log-normal range
    scaler = MinMaxScaler(feature_range=(0, 1.0))
    scaler.fit_transform(_d.reshape(-1, 1)) # by using _d, we are constructing a scaler that maps slope from 0-90 degrees to 0-1
#    y = np.expand_dims(y, -1)
    y_norm = scaler.transform(y)

    n_latents = X_norm.shape[1]      # retrieve the size of input latent vectors
    n_targets = y_norm.shape[1]      # retrieve the size of output targets

    print ("X [min,max]\t",      np.amin(X),"/",      np.amax(X))
    print ("X_norm [min,max]\t", np.amin(X_norm),"/", np.amax(X_norm))
    print ("Y [min,max]\t",      np.amin(y),"/",      np.amax(y))
    print ("Y_norm [min,max]\t", np.amin(y_norm),"/", np.amax(y_norm))

    X_train, X_valid, y_train, y_valid = train_test_split(X_norm,
                                                        y_norm,
                                                        test_size=.2, # 8:2 ratio
                                                        shuffle = True) 
    # Convert train and test vectors to tensors
    X_train, y_train = torch.Tensor(X_train).float(), torch.Tensor(y_train).float()
    X_valid, y_valid = torch.Tensor(X_valid).float(), torch.Tensor(y_valid).float()
    y_train = torch.unsqueeze(y_train, -1)  # PyTorch will complain if we feed the (N).Tensor rather than a (NX1).Tensor
    y_valid = torch.unsqueeze(y_valid, -1)   # we add an additional dummy dimension

    if torch.cuda.is_available():
        Console.info("Using CUDA")
        device = torch.device("cuda")
    else:
        Console.warn("Using CPU")
        device = torch.device("cpu")

    regressor = BayesianRegressor(n_latents, n_targets).to(device)  # Single output being predicted
    # regressor.init
    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate) # learning rate
    # criterion = torch.nn.MSELoss()  # mean squared error loss (squared L2 norm). Used to compute the regression fitting error
    criterion = torch.nn.CosineEmbeddingLoss()  # cosine similarity loss 


    # print("Model's state_dict:")
    # for param.Tensor in regressor.state_dict():
    #     print(param.Tensor, "\t", regressor .state_dict()[param.Tensor].size())
    data_batch_size = 18

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=data_batch_size, shuffle=True)

    ds_valid = torch.utils.data.TensorDataset(X_valid, y_valid)
    dataloader_valid = torch.utils.data.DataLoader(ds_valid, batch_size=data_batch_size, shuffle=True)

    iteration = 0
    # Log of training and validation losses
    train_loss_history = []
    train_fit_loss_history = []
    train_kld_loss_history = []
    valid_loss_history = []
    valid_fit_loss_history = []
    valid_kld_loss_history = []


    lambda_fit_loss = lambda_recon  # regularization parameter for the fit loss (cost function is the sum of the scaled fit loss and the KL divergence loss)
    elbo_kld        = lambda_elbo   # regularization parameter for the KL divergence loss 
    print (regressor)       # show network architecture (this can be retrieved later, but we show it for debug purposes)

    print ("MSE-Loss lambda: ", lambda_fit_loss);  # Print the regularisation parameter for regression loss
    print ("ELBO KLD lambda: ", elbo_kld);  # Print regularisation parameter for KL divergence loss
    regressor.train()   # set to training mode, just in case
    # regressor.freeze_() # while frozen, the network will behave as a normal network (non-Bayesian)
    regressor.unfreeze_()   # we no longer start with "warming-up" phase of non-Bayesian training


    # Create customized criterion function
    # Add output layer normalization option: L1 or L2 norm
    # Add option to configure cosine or MSELoss
    # Improve constant torch.ones for CosineEmbeddingLoss, or juts use own cosine distance loss (torch compatible)

    try:
        for epoch in range(num_epochs):
            # if (epoch == 2):          # we train in non-Bayesian way during a first phase of P-epochs (P:50) as 'warm-up'
            #     regressor.unfreeze_()
            #     Console.info("Unfreezing the network")

            # We store a list of losses for each epoch (multiple samples per epoch)
            train_loss = []
            valid_loss = []
            #Loss (cost) values are separated into fit_loss and kld_loss
            train_fit_loss = []
            train_kld_loss = []
            valid_fit_loss = []
            valid_kld_loss = []

            for i, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                # labels.shape = (h,1,1) is adding an extra dimension to the tensor, so we need to remove it
                labels = labels.squeeze(2)
                # print ("labels.shape", labels.shape)
                _loss, _fit_loss, _kld_loss = regressor.sample_elbo_weighted(inputs=datapoints.to(device),
                                labels=labels.to(device),
                                criterion=criterion,    # MSELoss
                                sample_nbr=n_samples,
                                criterion_loss_weight = lambda_fit_loss,
                                complexity_cost_weight=elbo_kld/X_train.shape[0])  # normalize the complexity cost by the number of input points
                _loss.backward() # the returned loss is the combination of fit loss (MSELoss) and complexity cost (KL_div against a nominal Normal distribution )
                optimizer.step()
                train_loss.append    (_loss.item())  # keep track of training loss
                train_fit_loss.append(_fit_loss.item())
                # When the network is frozen the complexity cost is not computed and the kld_loss is 0
                # The problem is that the return type changes from a Tensor to a scalar
                if (type(_kld_loss) == torch.Tensor):
                    train_kld_loss.append(_kld_loss.item())
                else:
                    train_kld_loss.append(0.0)

                # print ("_loss.item()", _loss.item(), "\t_fit_loss.item()", _fit_loss.item(), "\t_kld_loss", _kld_loss)
            
            for k, (valid_datapoints, valid_labels) in enumerate(dataloader_valid):
                # calculate the fit loss and the KL-divergence cost for the test points set
                valid_labels = valid_labels.squeeze(2)
                _loss, _fit_loss, _kld_loss = regressor.sample_elbo_weighted(inputs=valid_datapoints.to(device),
                                    labels=valid_labels.to(device),
                                    criterion=criterion,
                                    sample_nbr=n_samples,
                                    criterion_loss_weight = lambda_fit_loss,   # regularization parameter to balance multiobjective cost function (fit loss vs KL div)
                                    complexity_cost_weight=elbo_kld/X_valid.shape[0])

                valid_loss.append    (_loss.item())  # keep track of training loss
                valid_fit_loss.append(_fit_loss.item())
                # When the network is frozen the complexity cost is not computed and the kld_loss is 0
                # The problem is that the return type changes from a Tensor to a scalar
                if (type(_kld_loss) == torch.Tensor):
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

            Console.info("Epoch [" + str(epoch) + "] Train (MSE + KLD): {:.3f}".format(mean_train_loss) + " = {:.3f}".format(mean_train_fit_loss) + " + {:.3f}".format(mean_train_kld_loss) + 
                                              "    | Valid (MSE + KLD): {:.3f}".format(mean_valid_loss) + " = {:.3f}".format(mean_valid_fit_loss) + " + {:.3f}".format(mean_valid_kld_loss) )
            Console.progress(epoch, num_epochs)

    except KeyboardInterrupt:
        Console.warn("Training interrupted...")
        # sys.exit()

    Console.info("Training completed. Saving the model...")
    # create dictionary with the trained model and some training parameters
    model_dict = {'epochs': num_epochs,
                  'batch_size': data_batch_size,
                  'learning_rate': learning_rate,
                  'lambda_fit_loss': lambda_fit_loss,
                  'elbo_kld': elbo_kld,
                  'optimizer': optimizer.state_dict(),
                  'model_state_dict': regressor.state_dict()}
    torch.save(model_dict, network_name)

    export_df = pd.DataFrame([train_loss_history, train_fit_loss_history, train_kld_loss_history, valid_loss_history, valid_fit_loss_history, valid_kld_loss_history]).transpose()
    export_df.columns = ['train_loss', 'train_fit_loss', 'train_kld_loss', 'valid_loss', 'valid_fit_loss', 'valid_kld_loss']

    export_df.index.names=['index']
    export_df.to_csv(logfile_name, index = False)

    # Once trained, we start inferring
    expected = []
    uncertainty = []
    predicted = [] # == y

    Console.info("Testing predictions...")
    idx = 0 
    # for x in X_valid:
    Xp_ = torch.Tensor(X_norm).float()

    regressor.eval() # we need to set eval mode before running inference
                     # this will set dropout and batch normalization to evaluation mode
    for x in Xp_:
        predictions = []
        for n in range(n_samples):
            y_ = regressor(x.to(device)).detach().cpu().numpy()
            predictions.append(y_) #N-dimensional output, stack/append as "single item"

        p_mean = np.mean(predictions, axis=0)
        p_stdv = np.std(predictions, axis=0)
        predicted.append(p_mean)
        uncertainty.append(p_stdv)

        idx = idx + 1
        Console.progress(idx, len(Xp_))

    y_list = y_norm.squeeze().tolist()  # when converted to list, the shape is (N,) and will be stored in the same "cell" of the dataframe

    # y_list, predicted and uncertainty lists need to be converted into sub-dataframes with as many columns as n_targets
    column_names = []
    for i in range(n_targets): # for each entry 'i' we create a column with the name 'y_i'
        column_names.append('y_' + str(i))    
    
    _ydf = pd.DataFrame(y_list, columns=column_names)

    # we repeat this for predicted and uncertainty
    column_names = []
    for i in range(n_targets): # for each entry 'i' we create a column with the name 'y_i'
        # the column names is created by prepending 'p_' to the column names of the y_df
        column_names.append('pred_' + y_df.columns[i])
        # column_names.append('predicted_' + str(i))    
    _pdf = pd.DataFrame(predicted, columns=column_names)

    column_names = []
    for i in range(n_targets): # for each entry 'i' we create a column with the name 'y_i'
        column_names.append('std_' + y_df.columns[i])
        # column_names.append('uncertainty_' + str(i))    
    _udf = pd.DataFrame(uncertainty, columns=column_names)

    pred_df  = pd.DataFrame ([index_df]).transpose()
    pred_df.columns = ['uuid']

    # Finally, let's append _ydf dataframe to pred_df
    pred_df = pd.concat([pred_df, _ydf], axis=1)
    pred_df = pd.concat([pred_df, _pdf], axis=1)
    # pred_df = pd.concat([pred_df, _udf], axis=1) # temporarily disabled, we do not use uncertainty yet

    Console.warn("Exported predictions to: ", predictions_name)
    pred_df.to_csv(predictions_name, index = False)

if __name__ == '__main__':
    main()
