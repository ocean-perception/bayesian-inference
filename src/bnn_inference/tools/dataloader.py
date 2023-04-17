# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
"""Utility class to print messages to the console
"""
from bnn_inference.tools.console import Console
import pandas as pd
import os


class CustomDataloader:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    """
    Dataset loader for oplab pipeline compatible CSV files. The 'matching_key' is used to build the input-output paris (e.g. label for each input row)
    The name of the target label is defined using 'target_key_prefix'
    """

    def load_dataset(input_filename,
                     target_filename,
                     matching_key='relative_path',
                     target_key_prefix='mean_slope',
                     input_key_prefix='latent_'):

        # check if input_filename exists
        if not os.path.isfile(input_filename):
            Console.quit("Input file does not exist: ", input_filename)

        df = pd.read_csv(
            input_filename
        )  # remove index_col=0 when using toy dataset (otherwise it's used as df index and won't be available for query)
        # df = pd.read_csv(input_filename, index_col=0) # use 1st column as ID, the 2nd (relative_path) can be used as part of UUID

        # 1) Data validation, remove invalid entries (e.g. NaN)
        df = df.dropna()
        # print (df.head()) # Enable for debug purposes
        Console.info("Input entries (NaN removed): ", len(df))

        # 2) Let's determine number of latent-space dimensions
        # The number of 'features' are defined by those columns labeled as 'relative_path'xxx, where xx is 0-based index for the h-latent space vector
        # Example: (8 dimensions: h0, h1, ... , h7)
        # relative_path northing [m] easting [m] ... latitude [deg] longitude [deg] recon_loss h0 h1 h2 h3 h4 h5 h6 h7
        n_latents = len(df.filter(regex=input_key_prefix).columns)
        Console.info("Input latent entries: ", n_latents)

        # If the number of latent dimensions is zero, show error message and exit
        if n_latents == 0:
            Console.error(
                "No columns matching the input_key [" + input_key_prefix +
                "] found in input file: ", input_filename)
            return

        # Check if the matching_key column exists in the dataframe df
        if matching_key not in df.columns:
            Console.error(
                "Matching key [" + matching_key +
                "] not found in input file: ", input_filename)
            return
        df['matching_key'] = df[
            matching_key]  # create a new dataframe column with the matching key

        # 3) Key matching
        # each 'relative_path' entry has the format  slo/20181121_depthmap_1050_0251_no_slo.tif
        # where the filename is composed by [date_type_tilex_tiley_mod_type]. input and target tables differ only in 'type' field
        # print ("matching_key: ", matching_key)
        # print ("target_key_prefix: ", target_key_prefix)
        # print ("input_key_prefix: ", input_key_prefix)
        ###############################################
        # Let's repeat the process with the target file
        # Check if target_filename exists
        if not os.path.isfile(target_filename):
            Console.error("Target file does not exist: ", target_filename)
            return
        tdf = pd.read_csv(
            target_filename
        )  # expected header: relative_path	mean_slope [ ... ] mean_rugosity
        tdf = tdf.dropna()

        n_targets = len(tdf.filter(regex=target_key_prefix).columns)
        Console.info("Dimension of targets (y): ", n_targets)
        # If the number of target dimensions is zero, show error message and exit
        if n_targets == 0:
            Console.error(
                "No columns matching the target_key_prefix [" +
                target_key_prefix + "] found in target file: ",
                target_filename)
            return
        # Check if the matching_key column exists in the target dataframe tdf
        if matching_key not in tdf.columns:
            Console.error("Matching key not found in target file: ",
                          matching_key)
            return
        tdf['matching_key'] = tdf[
            matching_key]  # create the dataframe containing the target values
        ###############################################

        Console.info("Total loaded targets(y): ", len(tdf))

        merged_df = pd.merge(
            df, tdf, how='right', on='matching_key'
        )  # join on right, so that we can use the target values
        merged_df = merged_df.dropna(
        )  # drop any stray NaN values. There should be none

        latent_df = merged_df.filter(
            regex=input_key_prefix
        )  # remove all columns not starting with input_key_prefix ('latent_')
        Console.info("Latent vector list (after filter): ", latent_df.shape)

        target_df = merged_df.filter(
            regex=target_key_prefix
        )  # remove all columns not starting with input_key_prefix ('latent_')
        #        target_df = merged_df[target_key_prefix]
        # latent_np = latent_df.to_numpy(dtype='float')   # Explicit numeric data conversion to avoid silent bugs with implicit string conversion
        # target_np = target_df.to_numpy(dtype='float')   # Apply to both target and latent data
        # input-output datasets are linked using the key provided by matching_key
        return latent_df, target_df, merged_df['matching_key']

    def load_toydataset(input_filename,
                        target_key_prefix='mean_slope',
                        input_prefix='latent_',
                        matching_key='relative_path'):
        Console.info("load_toydataset called for: ", input_filename)

        df = pd.read_csv(
            input_filename, index_col=0
        )  # use 1st column as ID, the 2nd (relative_path) can be used as part of UUID
        # 1) Data validation, remove invalid entries (e.g. NaN)
        print(df.head())
        df = df.dropna()
        Console.info("Total valid entries: ", len(df))
        # df.reset)index(drop = True) # not sure if we prefer to reset index, as column index was externallly defined

        # 2) Let's determine number of latent-space dimensions
        # The number of 'features' are defined by those columns labeled as 'relative_path'xxx, where xx is 0-based index for the h-latent space vector
        # Example: (8 dimensions: h0, h1, ... , h7)
        # relative_path northing [m] easting [m] ... latitude [deg] longitude [deg] recon_loss h0 h1 h2 h3 h4 h5 h6 h7
        n_latents = len(df.filter(regex=input_prefix).columns)
        Console.info("Latent dimensions: ", n_latents)

        latent_df = df.filter(regex=input_prefix)
        target_df = df[target_key_prefix]
        Console.info("Latent size: ", latent_df.shape)

        latent_np = latent_df.to_numpy(dtype='float')
        target_np = target_df.to_numpy(dtype='float')
        uuid_np = df[matching_key].to_numpy(
        )  # UUIDs are retrieved from the matching_key column
        # input-output datasets are linked using the key provided by matching_key
        return latent_np, target_np, uuid_np

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print(BColors.OKBLUE + self.name + ' took > ' + BColors.ENDC +
              str(self.took) + ' ms')
