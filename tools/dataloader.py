# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

"""Utility class to print messages to the console
"""
from tools.console import Console
import pandas as pd

class CustomDataloader:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''


    def load_dataset (input_filename, target_filename, matching_key='relative_path', target_key ='mean_slope', latent_name_prefix= 'latent_'):
        Console.info("load_dataset called for: ", input_filename)

        df = pd.read_csv(input_filename, index_col=0) # use 1st column as ID, the 2nd (relative_path) can be used as part of UUID
        # 1) Data validation, remove invalid entries (e.g. NaN)
        print (df.head())
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
        df['filename_base'] = df[matching_key]# I think it is possible to do it in a single regex
        # df['filename_base'] = df[matching_key].str.extract('(?:\/)(.*_)')   # I think it is possible to do it in a single regex
        # df['filename_base'] = df['filename_base'].str.rstrip('_')


        tdf = pd.read_csv(target_filename) # expected header: relative_path	mean_slope [ ... ] mean_rugosity
        tdf = tdf.dropna()
        # target_key='mean_rugosity'
        tdf['filename_base'] = tdf[matching_key]   # I think it is possible to do it in a single regex
        # tdf['filename_base'] = tdf[matching_key].str.extract('(?:\/)(.*_)')   # I think it is possible to do it in a single regex
        # tdf['filename_base'] = tdf['filename_base'].str.rstrip('_r002')

        # print (tdf.head())    
        Console.info("Target entries:    ", len(tdf))
        merged_df = pd.merge(df, tdf, how='right', on='filename_base')
        merged_df = merged_df.dropna()

        latent_df = merged_df.filter(regex=latent_name_prefix)
        Console.info ("Latent size:       ", latent_df.shape)
        target_df = merged_df[target_key]

        np_latent = latent_df.to_numpy(dtype='float')
        np_target = target_df.to_numpy(dtype='float')
        # input-output datasets are linked using the key provided by matching_key
        return np_latent, np_target, merged_df['filename_base']


    def load_toydataset (input_filename, target_key ='mean_slope', input_prefix= 'latent_', matching_key='relative_path'):
        Console.info("load_toydataset called for: ", input_filename)

        df = pd.read_csv(input_filename, index_col=0) # use 1st column as ID, the 2nd (relative_path) can be used as part of UUID
        # 1) Data validation, remove invalid entries (e.g. NaN)
        print (df.head())
        df = df.dropna()
        Console.info("Total valid entries: ", len(df))
        # df.reset)index(drop = True) # not sure if we prefer to reset index, as column index was externallly defined

        # 2) Let's determine number of latent-space dimensions
        # The number of 'features' are defined by those columns labeled as 'relative_path'xxx, where xx is 0-based index for the h-latent space vector
        # Example: (8 dimensions: h0, h1, ... , h7)
        # relative_path northing [m] easting [m] ... latitude [deg] longitude [deg] recon_loss h0 h1 h2 h3 h4 h5 h6 h7
        n_latents = len(df.filter(regex=input_prefix).columns)
        Console.info ("Latent dimensions: ", n_latents)

        latent_df = df.filter(regex=input_prefix)
        target_df = df[target_key]
        Console.info ("Latent size: ", latent_df.shape)

        np_latent = latent_df.to_numpy(dtype='float')
        np_target = target_df.to_numpy(dtype='float')
        np_uuid   = df[matching_key].to_numpy()
        # input-output datasets are linked using the key provided by matching_key
        return np_latent, np_target, np_uuid

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print(BColors.OKBLUE + self.name + ' took ▸ ' + BColors.ENDC + str(self.took) + ' ms')

