# Python script to read CSV file containin output for a multi-class classifier:
# Filename: valid_ce_loss_test_elbo0001_recon100.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import os


# Create main (entry point)
def main():

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Confusion matrix: post-processing of CSV file containing output for a multi-class classifier')

    # Add arguments to the parser
    parser.add_argument('--input', type=str, help='CSV file containing output for a multi-class classifier')
    # Add argument for output filename
    parser.add_argument('--output', type=str, help='Output filename for the confusion matrix plot')
    # We will export the output in both PNG and SVG format

    # Flag to show the plot
    parser.add_argument('--show', action='store_true', help='Show the confusion matrix plot')

    # Parse the arguments
    args = parser.parse_args()

    # Get the filename from the command line argument
    filename = args.input
    # CHeck if the input file exists (use os module)
    if not os.path.isfile(filename):
        print("[error] Input file does not exist: ", filename)
        # Exit with error
        sys.exit(1)
    
    # Get the output filename from the command line argument
    output_filename = args.output
    # Check if the output filename is specified. If not, we use the same as the inut and append .svg
    if output_filename is None:
        # Strip the extension from the filename
        output_filename = filename.split('.')[0]
        # Append the extension
        output_filename += '.svg'

    # Check if the file exists. If true, we print a warning message indicating we will overwrite it
    # use os to check if the file exists
    if os.path.isfile(output_filename):
        print("[warning] File exists. Will overwrite", output_filename)

    # Read CSV file
    # filename = 'valid_ce_loss_test_elbo0001_recon100.csv'
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

    # Generate a figure to plot the confusion matrix
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