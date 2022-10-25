import pandas as pd
import numpy as np
import argparse
import sys
import os
import tools.parser as par
from tools.console import Console

def handler(signum, frame):
    Console.warn ("CTRL + C pressed. Stopping...")
    sys.exit(1)


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
    Console.info("Postprocessing tool for predictions generated with BNN. Merges predicted entries with target values by key (uuid) and export as a single file")

    if os.path.isfile(args.target):
        Console.info("Target file:\t", args.target)
    else:
        Console.error("Targetfile [" + args.target + "] not found. Please check the provided input path (-t, --target)")

    if os.path.isfile(args.input):
        Console.info("Prediction file:\t", args.input)
    else:
        Console.error("Prediction file [" + args.input + "] not found. Please check the provided input path (-i, --input)")

    if os.path.isfile(args.output):
        Console.warn("Output file [", args.output, "] already exists. It will be overwritten (default action)")
    else:
        Console.info("Output file: ", args.output)


    if (args.key):
        index_key = args.key
        Console.info("Using output key [", index_key,"]")
    else:
        index_key = "predicted"
        Console.warn("Using default output key [", index_key,"]")

    if (not os.path.isfile(args.target)):
        Console.error("Target file [" + args.target + "] not found. Please check the provided input path (-t, --target)")
        return -1

    if (not os.path.isfile(args.input)):
        Console.error("Prediction file [" + args.input + "] not found. Please check the provided input path (-i, --input)")
        return -1

    df1 = pd.read_csv(args.target, index_col = 0)       # <------- ground truth
    df2 = pd.read_csv(args.input, index_col = 0)        # <------- predictions

    df1 = df1.dropna()
    df2 = df2.dropna()

    # Typical name/header for target (ground truth file)
    # Name: M3_direct_r020_TR_ALL.csv
    # Header:[empty] | uuid | northing [m] | easting [m] | landability
    # Data format is pretty clean, northing/easting is expected to be uuid-format compatible (no trailing decimals)

    # Typical header format for predicted values (exhaustive list format)
    # Name: all_dM3h1631.csv
    # Header:
    # uuid  valid_ratio northing [m]    easting [m] depth [m]   latitude [deg]  longitude [deg] altitude [m]    roll [deg]  pitch [deg] heading [deg]   timestamp [s]   recon_loss  landability   uncertainty

    # Columns we need for the output join
    # [index/empty] | uuid | northing [m] from target | easting [m] from target | [score: measurability/landability] | [predicted score]

    # We trim the prediction dataframe, we only need 'uuid' and the prediction + uncertainty columns
    dfx = df2[["uuid", index_key, "uncertainty"]]
    merged_df = pd.merge(df1, dfx, on = 'uuid', how = 'inner')
    Console.info ("Exporting merged dataframes to ", args.output)
    merged_df.index.names=['index']
    merged_df.to_csv(args.output)
    Console.info ("... done!")

if __name__ == '__main__':
    main()
