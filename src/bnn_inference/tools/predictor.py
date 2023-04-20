# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import os
import timeit

import pandas as pd

from bnn_inference.tools.console import BColors, Console


class PredictiveEngine:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ""

    def loadData(input_filename, input_key_prefix="latent_"):
        Console.info("PredictiveEngine.predict called for: ", input_filename)

        # Check if input_filename exists
        if not os.path.isfile(input_filename):
            Console.error("Input file does not exist: ", input_filename)
            return
        df = pd.read_csv(
            input_filename, index_col=0
        )  # use 1st column as ID, the 2nd (relative_path) can be used as part of UUID
        # 1) Data validation, remove invalid entries (e.g. NaN)
        # print (df.head())
        df = df.dropna()
        Console.info("Total valid entries: ", len(df))
        # 2) Let's determine number of latent-space dimensions
        # The number of 'features' are defined by those columns labeled as 'relative_path'xxx, where xx is 0-based index for the h-latent space vector
        # Example: (8 dimensions: h0, h1, ... , h7)
        # relative_path northing [m] easting [m] ... latitude [deg] longitude [deg] recon_loss h0 h1 h2 h3 h4 h5 h6 h7
        n_latents = len(df.filter(regex=input_key_prefix).columns)
        Console.info("Latent dimensions: ", n_latents)

        latent_df = df.filter(regex=input_key_prefix)
        Console.info("Latent size: ", latent_df.shape)

        np_latent = latent_df.to_numpy(dtype=np.float64)
        return np_latent, n_latents, df

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print(
            BColors.OKBLUE
            + self.name
            + " took ▸ "
            + BColors.ENDC
            + str(self.took)
            + " ms"
        )
