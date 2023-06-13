# Create class that stores default and user defined configuration parameters for both training and evaluation of BNNs.
#

import sys
import os
# import numpy as np
# import pandas as pd
# import argparse
from .console import Console

class BNNConfiguration:
    def __init__(self):
        # Set a collection of default keys
        self.latent_key = 'latent_'
        self.output_key = 'measurability'
        self.UUID = 'relative_path'
        self.num_epochs = 100
        self.n_samples = 10
        self.learning_rate = 0.001
        self.lambda_recon = 10.0
        self.lambda_elbo = 1.0
        self.xratio = 0.9
        self.scale_factor = 1.0
        self.target = []
        self.input = []
        self.device_index = 0

        self.predictions_name = None
        self.logfile_name = None
        self.network_name = None
        self.filename_suffix = None


    def set_filenames(self, args, n_latents):
            # for each output file, we check if user defined name is provided. If not, use default naming convention
        filename_suffix = "H" + str(n_latents) + "_E" + str(self.num_epochs) + "_S" + str(self.n_samples)
        # Console.warn("Suffix:", filename_suffix)
        if (args.output is None):
            self.predictions_name = "bnn_predictions_" + filename_suffix +  ".csv"
        else:
            self.predictions_name = args.output
        if os.path.isfile(self.predictions_name):
            Console.warn("Output file [", self.predictions_name, "] already exists. It will be overwritten (default action)")
        else:
            Console.info("Output file:   \t", self.predictions_name)
        if (args.logfile is None):
            self.logfile_name = "bnn_logfile_" + filename_suffix +  ".csv"
        else:
            self.logfile_name = args.logfile
        if os.path.isfile(self.logfile_name):
            Console.warn("Log file [", self.logfile_name, "] already exists. It will be overwritten (default action)")
        else:
            Console.info("Log file:      \t", self.logfile_name)
        if (args.network is None):
            self.network_name = "bnn_" + filename_suffix +  ".pth"   # PyTorch compatible network definition file
        else:
            self.network_name = args.network
        if os.path.isfile(self.network_name):
            Console.warn("Trained output [", self.network_name, "] already exists. It will be overwritten (default action)")
        else:
            Console.info("Trained output:\t", self.network_name)


    def load_from_parser(self, args):

        if (args.config):
            Console.warn("Configuration file provided:\t", args.config, " will be ignored (usage not implemented yet)")

        # Start verifying mandatory arguments
        # [mandatory] Check if input file (latents) exists
        if os.path.isfile(args.input):
            self.input = args.input
            Console.info("Latent input file:\t", self.input)
        else:
            Console.error("Latent input file [" + args.input + "] not found. Please check the provided input path (-i, --input)")
        # [mandatory] target file
        if os.path.isfile(args.target):
            self.target = args.target
            Console.info("Target input file:\t", self.target)
        else:
            Console.error("Target input file [" + args.target + "] not found. Please check the provided input path (-t, --target)")

        # this is the key that is used to identity the target output (single) or the column name for the predictions
        if (args.key):
            self.output_key = args.key
            Console.info("Using user-defined target key:\t[", self.output_key, "]")
        else:
            self.output_key = 'measurability'
            Console.warn("Using default target key:     \t[", self.output_key, "]")
        # user defined keyword (prefix) employed to detect the columns containing our input values (latent space representation of the bathymetry images)
        if (args.latent):
            self.latent_key = args.latent
            Console.info("Using user-defined latent key:\t[", self.latent_key, "]")
        else:
            self.latent_key = 'latent_'
            Console.info("Using default latent key:     \t[", self.latent_key, "]")
        # user defined keyword (prefix) employed to detect the columns containing our input values (latent space representation of the bathymetry images)
        if (args.uuid):
            self.UUID = args.uuid
            Console.info("Using user-defined UUID:\t[", self.UUID, "]")
        else:
            self.UUID = 'relative_path'
            Console.info("Using default UUID:     \t[", self.UUID, "]")
        # these parameters are only used in training mode
        if (args.epochs):
            self.num_epochs = args.epochs
        else:
            self.num_epochs = 100    # default
        # number of random samples used by sample_elbo to estimate the mean/std for each gp_inference epoch
        if (args.samples):
            # Verify the number of samples is larger than 2, otherwise Monte Carlo sampling is not possible (won't make sense)
            if (args.samples > 2):
                self.n_samples = args.samples
            else:
                # If the number of samples is not larger than 2, show an error and exit
                Console.error("The number of MC samples must be larger than 2. Please provide a number larger than 2 (-s, --samples)")
                sys.exit(1)
        else:
            self.n_samples = 10      # default

        # Check if user specified a learning rate
        if (args.lr):
            self.learning_rate = args.lr
            Console.info("Using user-defined learning rate:\t[", self.learning_rate, "]")
        else:
            self.learning_rate = 0.001 # Default value

        # Check if user specified a lambda for reconstruction loss
        if (args.lambda_recon):
            self.lambda_recon = args.lambda_recon
            Console.info("Using user-defined lambda for reconstruction loss:\t[", self.lambda_recon, "]")
        else:
            self.lambda_recon = 10.0

        # Check if user specified a lambda for ELBO KL loss
        if (args.lambda_elbo):
            self.lambda_elbo = args.lambda_elbo
            Console.info("Using user-defined lambda for ELBO KL loss:\t[", self.lambda_elbo, "]")
        else:
            self.lambda_elbo = 1.0

        # Check if user specified a xratio for T:V ratio
        if (args.xratio):
            self.xratio = args.xratio
            Console.info("Using user-defined xratio:\t[", self.xratio, "]")
        else:
            self.xratio = 0.9 # Default value 80:20 for training/validation

        if (args.scale):
            self.scale_factor = args.scale
            Console.info("Using user-defined scale:\t[", self.scale_factor, "]")
        else:
            self.scale_factor = 1.0

        if (args.gpu):
            Console.info("User-defined GPU index: \t", args.gpu)
            self.device_index = args.gpu # to be used if CUDA is available
