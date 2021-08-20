import statistics
import math
import torch.nn as nn
import torch.nn.functional as F
# Import blitz (BNN) modules
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # simple 2-layer fully connected linear regressor
        self.blinear1 = BayesianLinear(64, 64, bias=True, prior_sigma_1=0.5, prior_sigma_2=0.5)
        # self.blinear2 = BayesianLinear(64, 64, bias=True)
        # self.blinear2 = BayesianLinear(64, output_dim, bias=True)
        self.linear_input  = nn.Linear(input_dim, 64, bias=True)
        self.linear1       = nn.Linear(64, 64, bias=True)
        # self.linear2       = nn.Linear(64, 64, bias=True)
        self.linear_output = nn.Linear(64, output_dim, bias=True)
        self.relu          = nn.LeakyReLU()

    def forward(self, x):
        x_ = self.linear_input(x)
        x_ = self.blinear1(x_)
        x_ = self.relu(x_)
        x_ = self.linear1(x_)
        # x_ = F.relu(self.linear1(x_))
        # x_ = F.relu(self.linear2(x_))
        # x_ = self.blinear2(x_)
        # x_ = self.relu2(x_);
        x_ = self.linear_output(x_)
        return x_

def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 25):

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
        u_y = statistics.stdev(y_samples)       # mean(y_samples) as MLE for E[f(x)]
        error = e_y - y_list[i][0]              # error = (expected - target)^2
        errors.append(error*error)
        uncert.append(u_y)

    errors_mean = math.sqrt(statistics.mean(errors)) # single axis list, eq: (axis = 0)
    uncert_mean = statistics.mean(uncert)
    return errors_mean, uncert_mean
