# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, Ocean Perception Lab, Univ. of Southampton
All rights reserved.
Licensed under GNU General Public License v3.0
See LICENSE file in the project root for full license information.
"""
# Author: Jose Cappelletto (j.cappelletto@soton.ac.uk)

import math
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import blitz (BNN) modules
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator




@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # We can define at construction time the type of last layer: linear, softmax or softmin
        # Default is linear, which is unbounded and suitable for regression. We can convert this into
        # a 'mode' option switching betwee 'regression' to 'classification'

        # Per-layer dimensions could be defined at construction time, store it as class attributes and save it in the state_dict
        # Layer transfer function type (relu, sigmoid, tanh, etc) are static (sames as the calculation DAG) for PyTorch definitions
        # ONNX definitions are dynamic (differentiable)
        # DIM1 = 16
        # DIM2 = 4
        # DIM3 = 2

        # Complex network
        # DIM1 = 256
        # DIM2 = 128
        # DIM3 = 64

        # Medium network
        DIM1 = 64
        DIM2 = 32
        DIM3 = 8

        self.linear_input = nn.Linear(input_dim, DIM1, bias=True)

        self.blinear1 = BayesianLinear(
            DIM1, DIM1, bias=True, prior_sigma_1=0.5, prior_sigma_2=0.5
        )
        self.silu1 = nn.SiLU()

        self.linear2 = nn.Linear(DIM1, DIM2, bias=True)
        # self.silu2         = nn.Hardtanh(0.0,1.0)                          # consider using Softsign
        # self.silu2         = nn.Softsign()                          # consider using Softsign
        self.silu2 = nn.SiLU()  # consider using Softsign

        self.linear3 = nn.Linear(DIM2, DIM3, bias=True)
        self.linear_output = nn.Linear(DIM3, output_dim, bias=True)

        self.last_layer = nn.Softmin(dim=0) 
        # It can be Sotfmax, depending on boththe loss function and
        # the underlying distribution of your clas probabilities. Please read about the differences between both
        # and the use of each one in the context of your problem.

    # Oceans2021 architecture: 256 x SiLU | 521 x SiLU | 128 x Lin | 64 x Lin | y: output

    def forward(self, x):
        x_ = self.linear_input(x)
        x_ = self.silu1(self.blinear1(x_))
        x_ = self.silu2(self.linear2(x_))
        x_ = self.linear3(x_)
        x_ = self.linear_output(x_)
        x_ = self.last_layer(x_)    # last layer is Softmin to produce that output Tensor lie in the range [0, 1] and sum to 1
        # normalize output using L1 norm
        # x_ = F.normalize (x_, p=1, dim=-1)
        return x_

    def sample_elbo_weighted_mse(
        self,
        inputs,
        labels,
        criterion,
        sample_nbr,
        criterion_loss_weight=1,
        complexity_cost_weight=1,
    ):
        """Samples the ELBO Loss for a batch of data, consisting of inputs and corresponding-by-index labels
            The ELBO Loss consists of the sum of the KL Divergence of the model
                (explained above, interpreted as a "complexity part" of the loss)
                with the actual criterion - (loss function) of optimization of our model
                (the performance part of the loss).
            As we are using variational inference, it takes several (quantified by the parameter sample_nbr) Monte-Carlo
                samples of the weights in order to gather a better approximation for the loss.
        Parameters:
            inputs: torch.tensor -> the input data to the model
            labels: torch.tensor -> label data for the performance-part of the loss calculation
                    The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
            criterion: torch.nn.Module, custom criterion (loss) function, torch.nn.functional function -> criterion to gather
                        the performance cost for the model
            sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to
                        gather the loss to be .backwarded in the optimization of the model.
        """

        loss = 0
        criterion_loss = 0
        kldiverg_loss = 0
        # y_target = torch.ones(labels.shape[0], device=torch.device("cuda"))

        for _ in range(sample_nbr):
            outputs = self(inputs)
            criterion_loss += criterion(outputs, labels)
            kldiverg_loss += self.nn_kl_divergence()

        criterion_loss = criterion_loss_weight * criterion_loss / sample_nbr
        kldiverg_loss = complexity_cost_weight * kldiverg_loss / sample_nbr
        loss = criterion_loss + kldiverg_loss

        return loss, criterion_loss, kldiverg_loss


    setattr(variational_estimator, "sample_elbo_weighted_mse", sample_elbo_weighted_mse)

    def sample_elbo_weighted_cos_sim(
        self,
        inputs,
        labels,
        criterion,
        sample_nbr,
        criterion_loss_weight=1,
        complexity_cost_weight=1,
    ):
        """Samples the ELBO Loss for a batch of data, consisting of inputs and corresponding-by-index labels
            The ELBO Loss consists of the sum of the KL Divergence of the model
                (explained above, interpreted as a "complexity part" of the loss)
                with the actual criterion - (loss function) of optimization of our model
                (the performance part of the loss).
            As we are using variational inference, it takes several (quantified by the parameter sample_nbr) Monte-Carlo
                samples of the weights in order to gather a better approximation for the loss.
        Parameters:
            inputs: torch.tensor -> the input data to the model
            labels: torch.tensor -> label data for the performance-part of the loss calculation
                    The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
            criterion: torch.nn.Module, custom criterion (loss) function, torch.nn.functional function -> criterion to gather
                        the performance cost for the model
            sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to
                        gather the loss to be .backwarded in the optimization of the model.
        """

        loss = 0
        criterion_loss = 0
        kldiverg_loss = 0
        # TODO (@cappelletto please verify) for Cosine Similarity, uses 1 - cos(x1,x2)
        y_target = torch.ones(
            labels.shape[0], device=torch.device("cuda")
        ) - torch.cosine_similarity(inputs, labels, dim=1)

        for _ in range(sample_nbr):
            outputs = self(inputs)
            criterion_loss += criterion(outputs, labels, y_target)  # use this for cosine
            kldiverg_loss += self.nn_kl_divergence()

        criterion_loss = criterion_loss_weight * criterion_loss / sample_nbr
        kldiverg_loss = complexity_cost_weight * kldiverg_loss / sample_nbr
        loss = criterion_loss + kldiverg_loss

        return loss, criterion_loss, kldiverg_loss


    setattr(
        variational_estimator, "sample_elbo_weighted_cos_sim", sample_elbo_weighted_cos_sim
    )


def evaluate_regression(regressor, X, y, samples=15):

    # we need to draw k-samples for each x-input entry. Posterior sampling is done to obtain the E[y] over a Gaussian distribution
    # The maximum likelihood estimates: meand & stdev of the sample vector (large enough for a good approximation)
    # If sample vector is large enough biased and unbiased estimators will converge (samples >> 1)
    errors = (
        []
    )  # list containing the error as e = y - E[f(x)] for all the x in X. y_pred = f(x)
    # E[f(x)] is the expected value, computed as the mean(x) as the MLE for a Gaussian distribution (expected)
    uncert = []
    y_list = y.tolist()
    for i in range(
        len(X)
    ):  # for each input x[i] (that should be the latent enconding of the image)
        y_samples = []
        for k in range(samples):  # draw k-samples.
            y_tensor = regressor(X[i])
            y_samples.append(
                y_tensor[0].tolist()
            )  # Each call to regressor(x = X[i]) will return a different value
        e_y = statistics.mean(y_samples)  # mean(y_samples) as MLE for E[f(x)]
        u_y = statistics.stdev(y_samples)  # mean(y_samples) as MLE for E[f(x)]
        error = e_y - y_list[i][0]  # error = (expected - target)^2
        errors.append(error * error)
        uncert.append(u_y)

    errors_mean = math.sqrt(statistics.mean(errors))  # single axis list, eq: (axis = 0)
    uncert_mean = statistics.mean(uncert)
    return errors_mean, uncert_mean
