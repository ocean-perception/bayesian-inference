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
import matplotlib
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

    if len(sys.argv) == 1 and args is None: # no argument passed? error, some parameters were expected
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(args)  # retrieve parsed arguments
    Console.info("Bayesian Neural Network for hi-res inference from low res acoustic priors (LGA-Bathymetry)")

    # Start verifying mandatory arguments
    # let's check if input files exist
    # [mandatory] input file (latents)
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
        n_samples = args.samples
    else:
        n_samples = 10      # default

    Console.info("Geotech landability/measurability predictor from low-res acoustics. Uses Bayesian Neural Networks as predictive engine")
    dataset_filename = args.input # dataset containing the predictive input. e.g. the latent vector
    target_filename  = args.target  # output variable to be predicted, e.g. mean_slope
    # dataset_filename = "data/output-201811-merged-h14.xls"     # dataset containing the predictive input
    # target_filename = "data/target/koyo20181121-stat-r002-slo.csv"  # output variable to be predicted
    Console.info("Loading dataset: " + dataset_filename)
    X, y, index_df = CustomDataloader.load_dataset(dataset_filename, target_filename, matching_key=UUID, target_key = output_key, latent_name_prefix=latent_key)    # relative_path is the common key in both tables
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


    # Check optional output filenames, if missing generate the default output names based on the training/data parameters
    pretrained_network = ""

    # X, y, index_df = CustomDataloader.load_toydataset(dataset_filename, target_key = output_key, input_prefix= latent_key, matching_key='uuid')    # relative_path is the common key in both tables

    # n_sample = X.shape[0]
    # X = StandardScaler().fit_transform(X)
    # y = StandardScaler().fit_transform(np.expand_dims(y, -1)) # this is resizing the array so it can match Size (D,1) expected by pytorch
    # norm = MinMaxScaler().fit(y)
    # y_norm = norm.transform(y)      # min max normalization of our output data
    y_norm = 10*y
    # norm = MinMaxScaler().fit(X)
    # X_norm = norm.transform(X)      # min max normalization of our input data
    X_norm = X

    # X_norm = X_ext
    n_latents = X_norm.shape[1]      # this is the only way to retrieve the size of input latent vectors

    Console.warn ("Xnorm_Shape", X_norm.shape)
    # print ("E", X_ext.size)

    print ("X [min,max]", np.amin(X),"/", np.amax(X))
    print ("X_norm [min,max]", np.amin(X_norm),"/", np.amax(X_norm))
    print ("Y [min,max]", np.amin(y),"/", np.amax(y))
    print ("Y_norm [min,max]", np.amin(y_norm),"/", np.amax(y_norm))

    X_train, X_test, y_train, y_test = train_test_split(X_norm,
                                                        y_norm,
                                                        test_size=.15, # 3:1 ratio
                                                        shuffle = True) 

    X_train, y_train = torch.Tensor(X_train).float(), torch.Tensor(y_train).float()
    X_test, y_test   = torch.Tensor(X_test).float(),  torch.Tensor(y_test).float()

    y_train = torch.unsqueeze(y_train, -1)  # PyTorch will complain if we feed the (N).Tensor rather than a (NX1).Tensor
    y_test = torch.unsqueeze(y_test, -1)    # we add an additional dummy dimension
    # sys.exit(1)
    if torch.cuda.is_available():
        Console.info("Using CUDA")
    else:
        Console.warn("Using CPU")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regressor = BayesianRegressor(n_latents, 1).to(device)  # Single output being predicted
    # regressor.init
    optimizer = optim.Adam(regressor.parameters(), lr=0.0015) # learning rate
    criterion = torch.nn.MSELoss()

    # print("Model's state_dict:")
    # for param.Tensor in regressor.state_dict():
    #     print(param.Tensor, "\t", regressor .state_dict()[param.Tensor].size())

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

    ds_test = torch.utils.data.TensorDataset(X_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=True)

    iteration = 0
    # Training time
    test_hist = []
    uncert_hist = []
    train_hist = []
    fit_hist = []
    ufit_hist = []

    elbo_kld = 0.2

    print (regressor)   # show network architecture

    print ("ELBO KLD factor: ", elbo_kld/X_train.shape[0]);
    regressor.train()   # set to training mode, just in case
    regressor.freeze_()

    try:
        for epoch in range(num_epochs):
            if (epoch == 50):          # we train in non-bayesian way during a first phase
                regressor.unfreeze_()

            train_loss = []
            # kl_loss = []
            fl_loss = []
            for i, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                
                loss, fit_loss, kl_loss = regressor.sample_elbo(inputs=datapoints.to(device),
                                labels=labels.to(device),
                                criterion=criterion,    # MSELoss
                                sample_nbr=n_samples,
                                criterion_loss_weight = 2000,
                                complexity_cost_weight=elbo_kld/X_train.shape[0])  # normalize the complexity cost by the number of input points
                loss.backward() # the returned loss is the combination of fit loss (MSELoss) and complexity cost (KL_div against the )
                optimizer.step()
                train_loss.append(loss.item())  # keep track of training loss
                fl_loss.append(fit_loss.item())

            test_loss = []  # complete loss for test dataset
            fit_loss = []   # regression (fitting) only loss for test dataset
            trfit_hist = []
            
            for k, (test_datapoints, test_labels) in enumerate(dataloader_test):
                sample_loss, fit_sample_loss, kl_loss = regressor.sample_elbo(inputs=test_datapoints.to(device),
                                    labels=test_labels.to(device),
                                    criterion=criterion,
                                    sample_nbr=n_samples,
                                    criterion_loss_weight = 2000,
                                    complexity_cost_weight=elbo_kld/X_test.shape[0])

                # fit_sample_loss = regressor.sample_elbo(inputs=test_datapoints.to(device),
                #                     labels=test_labels.to(device),
                #                     criterion=criterion,
                #                     sample_nbr=n_samples,
                #                     complexity_cost_weight=0)   # we are interested in the reconstruction/prediction loss only (no KL cost)

                test_loss.append(sample_loss.item())
                fit_loss.append(fit_sample_loss.item())

            mean_test_loss = statistics.mean(test_loss)
            stdv_test_loss = statistics.stdev(test_loss)

            mean_train_loss = statistics.mean(train_loss)
            mean_trfit_loss = statistics.mean(fl_loss)


            mean_fit_loss = statistics.mean(fit_loss)
            stdv_fit_loss = statistics.stdev(fit_loss)

            Console.info("Epoch [" + str(epoch) + "] Train: {:.2f}".format(mean_train_loss) + " TFit: {:.3f}".format(mean_trfit_loss) + " // Valid.loss: {:.2f}".format(mean_test_loss) + " VFit.loss: {:.3f}  ***".format(mean_fit_loss) )
            Console.progress(epoch, num_epochs)

            test_hist.append(mean_test_loss)
            uncert_hist.append(stdv_test_loss)
            train_hist.append(mean_train_loss)
            trfit_hist.append(mean_trfit_loss)

            fit_hist.append(mean_fit_loss)
            ufit_hist.append(stdv_fit_loss)


    except KeyboardInterrupt:
        Console.warn("Training interrupted...")
        # sys.exit()

    Console.info("Training completed!")
    torch.save(regressor.state_dict(), network_name)

    export_df = pd.DataFrame([train_hist, trfit_hist, test_hist, uncert_hist, fit_hist, ufit_hist]).transpose()
    export_df.columns = ['train_error', 'train_fit_loss', 'test_error', 'test_error_stdev', 'test_loss', 'test_loss_stdev']

    # export_df.index.names=['index']
    export_df.to_csv(logfile_name, index = False)
    # export_df.to_csv("bnn_train_report.csv")
    # df = pd.read_csv(input_filename, index_col=0) # use 1t column as ID, the 2nd (relative_path) can be used as part of UUID

    # Once trained, we start inferring
    expected = []
    uncertainty = []
    predicted = [] # == y

    Console.info("Testing predictions...")
    idx = 0 
    # for x in X_test:
    Xp_ = torch.Tensor(X_norm).float()

    regressor.eval() # we need to set eval mode before running inference
                     # this will set dropout and batch normalization to evaluation mode
    for x in Xp_:
        predictions = []
        for n in range(n_samples):
            p = regressor(x.to(device)).item()
            # print ("p.type", type(p)) ----> float
            # print ("p.len", len(p))
            predictions.append(p) #1D output, retieve single item

        # print ("pred.type", type(predictions))
        # print ("pred.len", len(predictions))    ---> 10 (n_samples)

        p_mean = statistics.mean(predictions)
        p_stdv = statistics.stdev(predictions)
        idx = idx + 1

        # print ("p_mean", type(p_mean))  --> float

        predicted.append(p_mean)
        uncertainty.append(p_stdv)

        Console.progress(idx, len(Xp_))

    y_list = y_norm.squeeze().tolist()

    xl = np.squeeze(X_norm).tolist()

    pred_df  = pd.DataFrame ([y_list, predicted, uncertainty, index_df]).transpose()
    pred_df.columns = ['y', 'predicted', 'uncertainty', 'uuid']
    Console.warn("Exported predictions to: ", predictions_name)
    pred_df.to_csv(predictions_name, index = False)
    # print (pred_df.head())

if __name__ == '__main__':
    main()
