import pandas as pd
import os
from bnn_inference.tools.console import Console



def join_predictions_impl(
        latent_csv,
        target_csv,
        target_key,
        output_csv):
    Console.info(
        "Postprocessing tool for predictions generated with BNN. Merges predicted entries with target values by key (uuid) and export as a single file"
    )

    if os.path.isfile(target_csv):
        Console.info("Target file:\t", target_csv)
    else:
        Console.error(
            "Targetfile [" + target_csv +
            "] not found. Please check the provided input path (-t, --target)")

    if os.path.isfile(latent_csv):
        Console.info("Prediction file:\t", latent_csv)
    else:
        Console.error(
            "Prediction file [" + latent_csv +
            "] not found. Please check the provided input path (-i, --input)")

    if os.path.isfile(output_csv):
        Console.warn(
            "Output file [", output_csv,
            "] already exists. It will be overwritten (default action)")
    else:
        Console.info("Output file: ", output_csv)

    if (target_key):
        index_key = target_key
        Console.info("Using output key [", index_key, "]")
    else:
        index_key = "predicted"
        Console.warn("Using default output key [", index_key, "]")

    if (not os.path.isfile(target_csv)):
        Console.error(
            "Target file [" + target_csv +
            "] not found. Please check the provided input path (-t, --target)")
        return -1

    if (not os.path.isfile(latent_csv)):
        Console.error(
            "Prediction file [" + latent_csv +
            "] not found. Please check the provided input path (-i, --input)")
        return -1

    df1 = pd.read_csv(target_csv, index_col=0)  # <------- ground truth
    df2 = pd.read_csv(latent_csv, index_col=0)  # <------- predictions

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
    merged_df = pd.merge(df1, dfx, on='uuid', how='inner')
    Console.info("Exporting merged dataframes to ", output_csv)
    merged_df.index.names = ['index']
    merged_df.to_csv(output_csv)
    Console.info("... done!")
