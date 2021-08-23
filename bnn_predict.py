# Importe general libraries
import re
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
# Import blitz (BNN) modules
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
# Import sklearn dataset parsers and samples
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Helper lirbaries (viz)
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

def main(args=None):
    description_str = "Bayesian Neural Network inference module"
    formatter = lambda prog: argparse.HelpFormatter(prog, width=120)
    parser = argparse.ArgumentParser(description=description_str, formatter_class=formatter)
    # argparse.HelpFormatter(parser,'width=120')
    par.add_arguments(parser)

    if len(sys.argv) == 1 and args is None: # no arggument passed? error, some parameters were expected
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(args)  # retrieve parsed arguments
    Console.info("Bayesian Neural Network for hi-res inference from low res acoustic priors (LGA-Bathymetry)")

    # we are in training mode
    Console.info("Prediction mode enabled. Looking for pretained network and input latent vectors")
    # Looking for CSV with latent vectors (input)
    if os.path.isfile(args.input):
        Console.info("Latent input file: ", args.input)
    else:
        Console.error("Latent input file [" + args.input + "] not found. Please check the provided input path (-l, --latent)")

    # check for pre-trained network
    # if output file exists, warn user

    if os.path.isfile(args.network):
        Console.info("Pre-trained network file [", args.network, "] found")
    else:
        Console.error("No pre-trained network found at: ", args.network)
        Console.info ("Terminating...")
        return -1

    if os.path.isfile(args.output):
        Console.warn("Output file [", args.output, "] already exists. It will be overwritten (default action)")
    else:
        Console.info("Output file: ", args.output)
    # ELBO k-sampling for posterior estimation. The larger the better the MLE, but slower. Good range: 5~25
    if (args.samples):
        n_samples = args.samples
    else:
        n_samples = 20
    # this is the key that is used to identity the target output (single) or the column name for the predictions
    if (args.key):
        output_key = args.key
    else:
        output_key = 'measurability'
    # user defined keyword (affix) employed to detect the columns containing our input values (latent space representation of the bathymetry images)
    if (args.latent):
        input_key = args.latent
    else:
        input_key = 'latent_'   # default expected from LGA based pipeline 

    Console.info("Loading latent input [", args.input ,"]")
    np_latent, n_latents, df = PredictiveEngine.loadData(args.input, latent_name_prefix= 'latent_')

    Console.info("Loading pretrained network [", args.network ,"]")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regressor = BayesianRegressor(n_latents, 1).to(device)  # Single output being predicted
    regressor.load_state_dict(torch.load(args.network)) # load state from deserialized object
    regressor.eval()    # switch to inference mode (set dropout layers)
   
    # print("Model's state_dict:")
    # for param_tensor in regressor.state_dict():
    #     print(param_tensor, "\t", regressor.state_dict()[param_tensor].size())

    # Apply any pre-existing scaling factor to the input
    X_norm = np_latent  # for large latents, input to the network
    # X_norm = np_latent/10.0  # for large latents, input to the network
    print ("X_norm [min,max]", np.amin(X_norm),"/", np.amax(X_norm))

    # Then, check the dataframe which should contain the same ordered rows from the latent space (see final step of training/validation)
    # Console.info("testing predictions...")
    idx = 0 
    Xp_ = torch.tensor(X_norm).float()  # convert normalized intput vector into tensor

########################################################################
########################################################################
########################################################################
########################################################################

    # iteration = 0
    # # Training time
    # test_hist = []
    # uncert_hist = []
    # train_hist = []
    # fit_hist = []
    # ufit_hist = []
    # elbo_kld = 1.0

    # # Once trained, we start inferring
    expected = []
    uncertainty = []
    predicted = [] # == y

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


########################################################################
########################################################################
########################################################################
########################################################################
    print ("Total predicted rows: ", len(predicted))

    # # y_list = [element.item() for element in y_test.flatten()]
    # xl = np.squeeze(X_norm).tolist()

    # # predicted.len = X.len (as desired)
    # # pred_df  = pd.DataFrame ([xl, y_list, predicted, uncertainty, index_df]).transpose()
    pred_df  = df.copy()    # make a copy, then we append the results
    pred_df[args.key] = predicted    
    pred_df["uncertainty"] = uncertainty    
    new_cols = pred_df.columns.values
    new_cols[0]="ID"
    # pred_df.columns = new_cols   # we rename the name of the column [0], which has empty
    # pd.DataFrame ([y_list, predicted, uncertainty, index_df]).transpose()
    # pred_df.columns = ['y', 'predicted', 'uncertainty', 'index']

    # output_name = "bnn_predictions_S" + str(n_samples) + "_E" + str(num_epochs) + "_H" + str(n_latents) + ".csv"
    print (pred_df.head())
    output_name = args.output
    Console.info("Exporting predictions to:", output_name)
    pred_df.to_csv(output_name)
    Console.warn("Done!")
    return 0

########################################################################
########################################################################
#####
# Older training step
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

    # for epoch in range(num_epochs):
    #     train_loss = []
    #     for i, (datapoints, labels) in enumerate(dataloader_train):
    #         optimizer.zero_grad()
            
    #         loss = regressor.sample_elbo(inputs=datapoints.to(device),
    #                         labels=labels.to(device),
    #                         criterion=criterion,    # MSELoss
    #                         sample_nbr=n_samples,
    #                         complexity_cost_weight=elbo_kld/X_train.shape[0])  # normalize the complexity cost by the number of input points
    #         loss.backward() # the returned loss is the combination of fit loss (MSELoss) and complexity cost (KL_div against the )
    #         optimizer.step()
    #         train_loss.append(loss.item())
            
    #     test_loss = []
    #     fit_loss = []

    #     for k, (test_datapoints, test_labels) in enumerate(dataloader_test):
    #         sample_loss = regressor.sample_elbo(inputs=test_datapoints.to(device),
    #                             labels=test_labels.to(device),
    #                             criterion=criterion,
    #                             sample_nbr=n_samples,
    #                             complexity_cost_weight=elbo_kld/X_test.shape[0])

    #         fit_loss_sample = regressor.sample_elbo(inputs=test_datapoints.to(device),
    #                             labels=test_labels.to(device),
    #                             criterion=criterion,
    #                             sample_nbr=n_samples,
    #                             complexity_cost_weight=0)   # we are interested in the reconstruction/prediction loss only (no KL cost)

    #         test_loss.append(sample_loss.item())
    #         fit_loss.append(fit_loss_sample.item())

    #     mean_test_loss = statistics.mean(test_loss)
    #     stdv_test_loss = statistics.stdev(test_loss)

    #     mean_train_loss = statistics.mean(train_loss)

    #     mean_fit_loss = statistics.mean(fit_loss)
    #     stdv_fit_loss = statistics.stdev(fit_loss)

    #     Console.info("Epoch [" + str(epoch) + "] Train loss: {:.4f}".format(mean_train_loss) + " Valid. loss: {:.4f}".format(mean_test_loss) + " Fit loss: {:.4f}  ***".format(mean_fit_loss) )
    #     Console.progress(epoch, num_epochs)

    #     test_hist.append(mean_test_loss)
    #     uncert_hist.append(stdv_test_loss)
    #     train_hist.append(mean_train_loss)

    #     fit_hist.append(mean_fit_loss)
    #     ufit_hist.append(stdv_fit_loss)

    #     # train_hist.append(statistics.mean(train_loss))

    #     # if (epoch % 50) == 0:   # every 50 epochs, we save a network snapshot
    #     #     temp_name = "bnn_model_" + str(epoch) + ".pth"
    #     #     torch.save(regressor.state_dict(), temp_name)

    # Console.info("Training completed!")
    # # torch.save(regressor.state_dict(), "bnn_model_N" + str (num_epochs) + ".pth")
    # torch.save(regressor.state_dict(), args.network)

    # export_df = pd.DataFrame([train_hist, test_hist, uncert_hist, fit_hist, ufit_hist]).transpose()
    # export_df.columns = ['train_error', 'test_error', 'test_error_stdev', 'test_loss', 'test_loss_stdev']

    # print ("head", export_df.head())
    # output_name = "bnn_training_S" + str(n_samples) + "_E" + str(num_epochs) + "_H" + str(n_latents) + ".csv"
    # export_df.to_csv(output_name)
    # # export_df.to_csv("bnn_train_report.csv")
    # # df = pd.read_csv(input_filename, index_col=0) # use 1t column as ID, the 2nd (relative_path) can be used as part of UUID


if __name__ == '__main__':
    main()
