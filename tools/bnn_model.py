# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, Ocean Perception Lab, Univ. of Southampton
All rights reserved.
Licensed under GNU General Public License v3.0
See LICENSE file in the project root for full license information.
"""
# Author: Jose Cappelletto (j.cappelletto@soton.ac.uk) 

import statistics
import math
import torch.nn as nn
import torch.nn.functional as F
# Import blitz (BNN) modules
from submodules.blitz.blitz.modules import BayesianLinear
from submodules.blitz.blitz.utils import variational_estimator

@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Per-layer dimensions could be defined at construction time, store it as class attributes and save it in the state_dict
        # Layer transfer function type (relu, sigmoid, tanh, etc) are static (sames as the calculation DAG) for PyTorch definitions
        # ONNX definitions are dynamic (differentiable)
        # DIM1 = 16
        # DIM2 = 4
        # DIM3 = 2
        DIM1 = 256
        DIM2 = 128
        DIM3 = 64

        self.linear_input  = nn.Linear(input_dim, DIM1, bias=True)

        self.blinear1      = BayesianLinear(DIM1, DIM1, bias=True, prior_sigma_1=0.5, prior_sigma_2=0.5)
        self.silu1         = nn.SiLU()

        self.linear2       = nn.Linear(DIM1, DIM2, bias=True)
        # self.silu2         = nn.Hardtanh(0.0,1.0)                          # consider using Softsign
        self.silu2         = nn.Softsign()                          # consider using Softsign
        # self.silu2         = nn.SiLU()                          # consider using Softsign

        self.linear3       = nn.Linear(DIM2, DIM3, bias=True)

        # self.linear2       = nn.Linear(128, 128, bias=True)
        self.linear_output = nn.Linear(DIM3, output_dim, bias=True)

        self.last_layer = nn.Softsign()
        # self.relu          = nn.LeakyReLU()
        # self.relu2         = nn.LeakyReLU()

    # Oceans2021 architecture: 256 x SiLU | 521 x SiLU | 128 x Lin | 64 x Lin | y: output 


    def forward(self, x):
        x_ =            self.linear_input (x)
        x_ = self.silu1(self.blinear1     (x_))
        x_ = self.silu2(self.linear2      (x_))
        x_ =            self.linear3      (x_)
        x_ =            self.linear_output(x_)
        x_ = self.last_layer(x_)
        # normalize output using L1 norm
        x_ = F.normalize (x_, p=1, dim=-1)
        return x_
        
def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 15):

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
        e_y = statistics.mean(y_samples)        # mean(y_samples) as MLE for E[f(x)]
        u_y = statistics.stdev(y_samples)       # mean(y_samples) as MLE for E[f(x)]
        error = e_y - y_list[i][0]              # error = (expected - target)^2
        errors.append(error*error)
        uncert.append(u_y)

    errors_mean = math.sqrt(statistics.mean(errors)) # single axis list, eq: (axis = 0)
    uncert_mean = statistics.mean(uncert)
    return errors_mean, uncert_mean