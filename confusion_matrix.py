# Python script to read CSV file containin output for a multi-class classifier:
# Filename: valid_ce_loss_test_elbo0001_recon100.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import os
import yaml


# Create main (entry point)
def main():

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Confusion matrix: post-processing of CSV file containing output for a multi-class classifier')

    # Add arguments to the parser
    parser.add_argument('--input', type=str, help='CSV file containing output for a multi-class classifier')
    # Add argument for output filename. If detects and use he user-provided filename. If not, use the same as the input file
    parser.add_argument('--output', type=str, help='Output filename for the confusion matrix plot')
    # We will export the output in SVG format, unless --png is specified
    parser.add_argument('--png', action='store_true', help='Export plots in PNG format instead of SVG')
    # Flag to show the plot
    parser.add_argument('--show', action='store_true', help='Show the confusion matrix plot')
    # Output filename containing a summary of the confusion matrix, Brier score and accuracy table
    parser.add_argument('--summary', action='store_true', help='Exports confusion matrix & scores summary')

    # Parse the arguments
    args = parser.parse_args()

    # Get the filename from the command line argument
    filename = args.input
    # CHeck if the input file exists (use os module)
    if not os.path.isfile(filename):
        print("[error] Input file does not exist: ", filename)
        # Exit with error
        sys.exit(1)
    
    # Use bash colour when printing [info] and [warning] messages
    info_str="\033[1;33m[info]\033[0m"
    warning_str="\033[1;31m[warning]\033[0m"

    # Get the output filename from the command line argument
    output_filename = args.output
    # Check if the output filename is specified. If not, we use the same as the inut and append .svg
    if output_filename is None:
        # Strip the extension from the filename
        # print message informing we are exporting the plot to the same filename
        print(info_str, " Output filename not specified. Using the same as the input file")
        output_filename = filename.split('.')[0]
        # Append the extension
        if args.png:
            output_filename += '.png'
        else:
            output_filename += '.svg'

    # Check if the file exists. If true, we print a warning message indicating we will overwrite it
    # use os to check if the file exists
    if os.path.isfile(output_filename):
        print(warning_str, " File exists. Will overwrite", output_filename)

    # Read CSV file
    df = pd.read_csv(filename)

    # the target (ground truth) labels are the columns starting with "target_"
    # the predicted labels are the columns starting with "pred_"

    # Get the target labels
    target_labels = [col for col in df.columns if col.startswith('target_')]
    # Get the predicted labels
    pred_labels = [col for col in df.columns if col.startswith('pred_')]
    # Get the number of classes
    num_classes = len(target_labels)
    print("Number of classes: ", num_classes)
    # Get the number of samples
    num_samples = len(df)
    print("Number of samples: ", num_samples)

    # We want to create a confusion matrix of size num_classes x num_classes
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    # We also calculate the confusion_matrix using the raw values (not argmax)
    # This is useful for the Brier score
    confusion_matrix_raw = np.zeros((num_classes, num_classes))

    # We also want to calculate the Brier score (MSE between the target and predicted labels)
    # We have one Brier score for one-hot (argmax) encoding and one for the raw values
    brier_score_onehot = 0.0
    brier_score_raw = 0.0

    print ("Processing samples...\n")
    # Iterate over the samples
    for i in range(num_samples):
        # Get the target label
        target_label = df.iloc[i][target_labels].to_numpy().argmax()
        # Get the predicted label
        pred_label = df.iloc[i][pred_labels].to_numpy().argmax()
        # Update the confusion matrix
        confusion_matrix[target_label, pred_label] += 1

        # Update the confusion matrix, using the raw values
        confusion_matrix_raw += np.outer(df.iloc[i][target_labels].to_numpy(), df.iloc[i][pred_labels].to_numpy())

        # Update the Brier score, using the argmax (one-hot encoding)
        brier_score_onehot += (target_label - pred_label) ** 2
        # Update the Brier score, using the raw values
        brier_score_raw += (df.iloc[i][target_labels].to_numpy() - df.iloc[i][pred_labels].to_numpy()) ** 2

    # Normalize the confusion matrices
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix_raw = confusion_matrix_raw / confusion_matrix_raw.sum(axis=1)[:, np.newaxis]
    # Print the content of the confusion matrix
    print(confusion_matrix)
    print(confusion_matrix_raw)
    brier_score_onehot /= num_samples
    brier_score_raw /= num_samples
    # Print the Brier score
    print("Brier score (one-hot): ", brier_score_onehot)
    print("Brier score (one-raw): ", brier_score_raw)

    # Calculate the accuracy for each class
    accuracy = np.diag(confusion_matrix)
    # Print the accuracy for each class
    for i in range(num_classes):
        print("Accuracy for class {}: {:.2f}".format(i, accuracy[i]))

    # Export the confusion matrix and scores summary to a TXT file
    # We will use the same filename as the output file, but with the extension .summary.txt (we could use JSON or YAML for machine readability)
    summary_filename = output_filename.split('.')[0] + '.summary.txt'
    print("Exporting summary to: ", summary_filename)
    # Information to be exported:
    # Input filename
    # Number of classes
    # Number of samples
    # Brier score (one-hot)
    # Brier score (raw)
    # Accuracy for each class
    # Confusion matrix (one-hot)
    # Confusion matrix (raw)

    # Open the file for writing
    with open(summary_filename, 'w') as f:
        # Write the input filename
        f.write("Input filename:\t{}\n".format(filename))
        # Write the number of classes
        f.write("Number of classes:\t{}\n".format(num_classes))
        # Write the number of samples
        f.write("Number of samples:\t{}\n".format(num_samples))
        # Write the Brier score (one-hot)
        f.write("Brier score (one-hot):\n{}\n".format(brier_score_onehot))
        # Write the Brier score (raw)
        f.write("Brier error (MSE):\n{}\n".format(brier_score_raw))
        # Write the accuracy for each class
        for i in range(num_classes):
            f.write("Accuracy for class {}: {:.2f}\n".format(i, accuracy[i]))
        # Write the confusion matrix (one-hot)
        f.write("Confusion matrix (one-hot):\n")
        f.write(str(confusion_matrix))
        f.write("\n")
        # Write the confusion matrix (raw)
        f.write("Confusion matrix (raw):\n")
        f.write(str(confusion_matrix_raw))
        f.write("\n")

    # Export summary as YAML too
    summary_filename = output_filename.split('.')[0] + '.summary.yaml'
    print("Exporting YAML summary to: ", summary_filename)
    # Open the YAML file for writing (using 'yaml' package)
    # Write the information to the YAML file
    with open(summary_filename, 'w') as f:
        # Write the information to the YAML file
        yaml.dump({'input_filename': filename,
                   'num_classes': num_classes,
                   'num_samples': num_samples,
                   'brier_score_onehot': brier_score_onehot,
                   'brier_score_raw': brier_score_raw,
                   'accuracy': accuracy,
                   'confusion_matrix_onehot': confusion_matrix.tolist(),
                   'confusion_matrix_raw': confusion_matrix_raw.tolist()}, f)

    # Generate a figure to plot the confusion matrix
    print ("Plotting confusion matrix...\n")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot the confusion matrix
    cax = ax.matshow(confusion_matrix)
    fig.colorbar(cax)
    # Set the labels for the x-axis
    ax.set_xticklabels([''] + target_labels)
    # Set the labels for the y-axis
    ax.set_yticklabels([''] + pred_labels)
    # Rotate the labels for the x-axis
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    # Set the title
    plt.title('Confusion matrix')
    # Add subtitle with the Brier scores
    plt.suptitle('Brier score (one-hot):' + str(brier_score_onehot) + '\n Brier score (raw): ' + str(brier_score_raw) + '\n Confusion matrix raw:' + str(confusion_matrix_raw))
    # Set the x-axis label
    plt.xlabel('Target')
    # Set the y-axis label
    plt.ylabel('Predicted')

    # Check if we want to show the plot
    if args.show:
        # Show the figure
        plt.show()

    # Save the figure
    print ("Saving figure to: ", output_filename)
    fig.savefig(output_filename)


# Call main
if __name__ == "__main__":
    main()