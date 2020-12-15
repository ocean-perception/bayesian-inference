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
from sklearn.model_selection import train_test_split
# Helper lirbaries (viz)
import matplotlib.pyplot as plt
import matplotlib
from tools.console import Console
import statistics
import math

def add_arguments(obj):
    obj.add_argument(
        "-t", "--target",
        type=str,
        default='target_values.csv',
        help="Path to CSV file containing a list of images ('relative_path') and the measured ground truth value we wan to predict (e.g. 'mean_slope)"
    )
    obj.add_argument(
        "-l", "--latent",
        type=str,
        default='image_latents.csv',
        help="Path to CSV containing the latent representation for each image. The 'relative_path' will be used to match against the targe_values.csv"
    )
    obj.add_argument(
        "-o", "--output",
        default='inferred.csv',
        type=str,
        help="File containing the expected and inferred value for each input image (validation + training datasets can be configured)"
    )
    obj.add_argument(
        "-n", "--network",
        default='bnn_trained.pth',
        type=str,
        help="Output path to write the trained Bayesian Neural Network, PyTorch compatible format."
    )


def load_dataset (input_filename, target_filename, matching_key='relative_path', target_key ='mean_slope', latent_name_prefix= 'latent_'):
    Console.info("load_dataset called for: ", input_filename)

    df = pd.read_csv(input_filename, index_col=0) # use 1t column as ID, the 2nd (relative_path) can be used as part of UUID
    # 1) Data validation, remove invalid entries (e.g. NaN)
    df = df.dropna()
    Console.info("Total valid entries: ", len(df))
    # df.reset)index(drop = True) # not sure if we prefer to reset index, as column index was externallly defined

    # 2) Let's determine number of latent-space dimensions
    # The number of 'features' are defined by those columns labeled as 'relative_path'xxx, where xx is 0-based index for the h-latent space vector
    # Example: (8 dimensions: h0, h1, ... , h7)
    # relative_path northing [m] easting [m] ... latitude [deg] longitude [deg] recon_loss h0 h1 h2 h3 h4 h5 h6 h7
    n_latents = len(df.filter(regex=latent_name_prefix).columns)
    Console.info ("Latent dimensions: ", n_latents)

    # 3) Key matching
    # each 'relative_path' entry has the format  slo/20181121_depthmap_1050_0251_no_slo.tif
    # where the filename is composed by [date_type_tilex_tiley_mod_type]. input and target tables differ only in 'type' field
    # let's use regex 
    df['filename_base'] = df[matching_key].str.extract('(?:\/)(.*_)')   # I think it is possible to do it in a single regex
    df['filename_base'] = df['filename_base'].str.rstrip('_')
    # print (df['filename_base'].head())

    tdf = pd.read_csv(target_filename) # expected header: relative_path	mean_slope [ ... ] mean_rugosity
    tdf = tdf.dropna()
    # target_key='mean_rugosity'
    tdf['filename_base'] = tdf[matching_key].str.extract('(?:\/)(.*_)')   # I think it is possible to do it in a single regex
    tdf['filename_base'] = tdf['filename_base'].str.rstrip('_r002')

    # print (tdf.head())    
    Console.info("Target entries: ", len(tdf))
    merged_df = pd.merge(df, tdf, how='right', on='filename_base')
    merged_df = merged_df.dropna()

    latent_df = merged_df.filter(regex=latent_name_prefix)
    Console.info ("Latent size: ", latent_df.shape)
    target_df = merged_df[target_key]

    np_latent = latent_df.to_numpy(dtype='float')
    np_target = target_df.to_numpy(dtype='float')
    # input-output datasets are linked using the key provided by matching_key
    return np_latent, np_target, merged_df['filename_base']
@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # simple 2-layer fully connected linear regressor
        #self.linear = nn.Linear(input_dim, output_dim)
        self.linear1  = nn.Linear(input_dim, 128)
        self.linear2  = nn.Linear(128, 128)
        self.linear3  = nn.Linear(128, output_dim)
        
        self.blinear1 = BayesianLinear(input_dim, 128)
        self.elu1     = nn.ELU()
        self.elu2     = nn.ELU()
        self.elu3     = nn.ELU()
        self.blinear3 = BayesianLinear(128, 128)
        self.blinear4 = BayesianLinear(128, 128)
        self.sigmoid1 = nn.Sigmoid()
        self.blinear2 = BayesianLinear(128, output_dim)
        
    def forward(self, x):
        x_ = self.blinear1(x)
        # x_ = self.sigmoid1 (x_)
        # x_ = self.blinear3(x_)
        # x_ = self.elu1 (x_)
        # x_ = self.blinear4(x_)
        x_ = self.elu2 (x_)
        x_ = self.blinear2(x_)
        return x_
        # x_ = self.blinear1(x)
        # x_ = self.blinear3(x_)
        # # x_ = self.sigmoid1(x_)
        # # x_ = self.blinear4(x_)
        # # x_ = self.elu2 (x_)
        # return self.blinear2(x_)


def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100):

    # we need to draw k-samples for each x-input entry. Posterior sampling is done to obtain the E[y] over a Gaussian distribution
    # The maximum likelihood estimates: meand & stdev of the sample vector (large enough for a good approximation)
    # If sample vector is large enough biased and unbiased estimators will converge (samples >> 1)
    errors = [] # list containing the error as e = y - E[f(x)] for all the x in X. y_pred = f(x)
                # E[f(x)] is the expected value, computed as the mean(x) as the MLE for a Gaussian distribution (expected)
    uncert = []
    y_list = y.tolist()
    for i in range(len(X)): # for each input x[i] (that should be the latent enconding of the image)
        y_samples = []
        for k in range(samples): # draw k-samples.
            y_tensor = regressor(X[i])
            y_samples.append(y_tensor[0].tolist()) # Each call to regressor(x = X[i]) will return a different value
        # print ("y_samples.len", len(y_samples))
        # print ("y_samples", y_samples)
        e_y = statistics.mean(y_samples)        # mean(y_samples) as MLE for E[f(x)]
        u_y = statistics.stdev(y_samples)        # mean(y_samples) as MLE for E[f(x)]
        error = e_y - y_list[i][0]                      # error = (expected - target)^2
        errors.append(error*error)
        uncert.append(u_y)

    errors_mean = math.sqrt(statistics.mean(errors)) # single axis list, eq: (axis = 0)
    uncert_mean = statistics.mean(uncert)
    return errors_mean, uncert_mean

def main(args=None):
    parser = argparse.ArgumentParser()
    add_arguments(parser)

    if len(sys.argv) == 1 and args is None: # no arggument passed? error, some parameters were expected
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(args)  # retrieve parsed arguments

    Console.info("Bayesian Neural Network for hi-res inference from low res acoustic priors (LGA-Bathymetry)")
    # let's check if input files exist
    if os.path.isfile(args.target):
        Console.info("Target input file: ", args.target)
    else:
        Console.error("Target input file [" + args.target + "] not found. Please check the provided input path (-t, --target)")

    if os.path.isfile(args.latent):
        Console.info("Latent input file: ", args.latent)
    else:
        Console.error("Latent input file [" + args.latent + "] not found. Please check the provided input path (-l, --latent)")
    # check for pre-trained network
    # if output file exists, warn user
    if os.path.isfile(args.network):
        Console.warn("Destination trained network file [", args.network, "] already exists. It will be overwritten (default action)")
    else:
        Console.info("Destination trained network: ", args.network)

    if os.path.isfile(args.output):
        Console.warn("Output file [", args.output, "] already exists. It will be overwritten (default action)")
    else:
        Console.info("Output file: ", args.output)
    # it can be "none"

    # // TODO : add arg parser, admit input file (dataset), config file, validation dataset file, mode (train, validate, predict)
    Console.info("Geotech landability/measurability predictor from low-res acoustics. Uses Bayesian Neural Networks as predictive engine")
    dataset_filename = args.latent # dataset containing the predictive input. e.g. the latent vector
    target_filename = args.target  # output variable to be predicted, e.g. mean_slope
    # dataset_filename = "data/output-201811-merged-h14.xls"     # dataset containing the predictive input
    # target_filename = "data/target/koyo20181121-stat-r002-slo.csv"  # output variable to be predicted
    Console.info("Loading dataset: " + dataset_filename)

    X, y, index_df = load_dataset(dataset_filename, target_filename, matching_key='relative_path', target_key = 'mean_slope')    # relative_path is the common key in both tables
    Console.info("Data loaded...")
    y = y/10    #some rescale    WARNING
 
    n_sample = X.shape[0]
    n_latents = X.shape[1]
    # X = StandardScaler().fit_transform(X)
    # y = StandardScaler().fit_transform(np.expand_dims(y, -1)) # this is resizing the array so it can match Size (D,1) expected by pytorch
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.25, # 4:1 ratio
                                                        shuffle = True) 

    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

    y_train = torch.unsqueeze(y_train, -1)  # PyTorch will complain if we feed the (N) tensor rather than a (NX1) tensor
    y_test = torch.unsqueeze(y_test, -1)    # we add an additional dummy dimension
    # sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regressor = BayesianRegressor(n_latents, 1).to(device)  # Single output being predicted
    optimizer = optim.Adam(regressor.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # print("Model's state_dict:")
    # for param_tensor in regressor.state_dict():
    #     print(param_tensor, "\t", regressor .state_dict()[param_tensor].size())

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

    ds_test = torch.utils.data.TensorDataset(X_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=True)

    iteration = 0
    num_epochs = 30
    # Training time
    test_hist = []
    uncert_hist = []
    train_hist = []
    fit_hist = []
    ufit_hist = []

    for epoch in range(num_epochs):
        train_loss = []
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            
            loss = regressor.sample_elbo(inputs=datapoints.to(device),
                            labels=labels.to(device),
                            criterion=criterion,    # MSELoss
                            sample_nbr=20,
                            complexity_cost_weight=0.2/X_train.shape[0])  # normalize the complexity cost by the number of input points
            loss.backward() # the returned loss is the combination of fit loss (MSELoss) and complexity cost (KL_div against the )
            optimizer.step()
            train_loss.append(loss.item())
            
        test_loss = []
        fit_loss = []

        for k, (test_datapoints, test_labels) in enumerate(dataloader_test):
            sample_loss = regressor.sample_elbo(inputs=test_datapoints.to(device),
                                labels=test_labels.to(device),
                                criterion=criterion,
                                sample_nbr=20,
                                complexity_cost_weight=0.2/X_test.shape[0])

            fit_loss_sample = regressor.sample_elbo(inputs=test_datapoints.to(device),
                                labels=test_labels.to(device),
                                criterion=criterion,
                                sample_nbr=20,
                                complexity_cost_weight=0)   # we are interested in the reconstruction/prediction loss only (no KL cost)

            test_loss.append(sample_loss.item())
            fit_loss.append(fit_loss_sample.item())

        mean_test_loss = statistics.mean(test_loss)
        stdv_test_loss = statistics.stdev(test_loss)
        mean_train_loss = statistics.mean(train_loss)

        mean_fit_loss = statistics.mean(fit_loss)
        stdv_fit_loss = statistics.stdev(fit_loss)

        Console.info("Epoch [" + str(epoch) + "] Train loss: {:.4f}".format(mean_train_loss) + " Validation loss: {:.4f}".format(mean_test_loss) )
        Console.progress(epoch, num_epochs)

        test_hist.append(mean_test_loss)
        uncert_hist.append(stdv_test_loss)
        train_hist.append(mean_train_loss)

        fit_hist.append(mean_fit_loss)
        ufit_hist.append(stdv_fit_loss)

        # train_hist.append(statistics.mean(train_loss))

        if (epoch % 50) == 0:   # every 50 epochs, we save a network snapshot
            temp_name = "bnn_model_" + str(epoch) + ".pth"
            torch.save(regressor.state_dict(), temp_name)

    Console.info("Training completed!")
    # torch.save(regressor.state_dict(), "bnn_model_N" + str (num_epochs) + ".pth")
    torch.save(regressor.state_dict(), args.network)

    export_df = pd.DataFrame([train_hist, test_hist, uncert_hist, fit_hist, ufit_hist]).transpose()
    export_df.columns = ['train_error', 'test_error', 'test_error_stdev', 'test_loss', 'test_loss_stdev']

    print ("head", export_df.head())
    export_df.to_csv("train_report_N" + str(num_epochs) + "_H" + str(n_latents) + ".csv")
    # export_df.to_csv("bnn_train_report.csv")
    # df = pd.read_csv(input_filename, index_col=0) # use 1t column as ID, the 2nd (relative_path) can be used as part of UUID

    # Once trained, we start inferring
    expected = []
    uncertainty = []
    predicted = [] # == y

    Console.info("testing predictions...")
    n_samples = 25
    idx = 0 
    # for x in X_test:
    Xp_ = torch.tensor(X).float()

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

    # print ("predicted:" , predicted)
    # print ("predicted.type", type(predicted))
    # print ("predicted.len", len(predicted))
    # print ("X.len:" , len(X_test))
    y_list = y_train.squeeze().tolist()
    y_list = y.squeeze().tolist()
    # y_list = y_test.squeeze().tolist()

    # y_list = [element.item() for element in y_test.flatten()]

    # print ("y_list.len", len(y_list))
    # predicted.len = X.len (as desired)
    pred_df  = pd.DataFrame ([y_list, predicted, uncertainty, index_df.values.tolist() ]).transpose()
    pred_df.columns = ['y', 'predicted', 'uncertainty', 'index']
    pred_df.to_csv("bnn_predictions_N" + str(num_epochs) + "_S" + str(n_samples) + "_H" + str(n_latents) + ".csv")
    print (pred_df.head())

if __name__ == '__main__':
    main()
