import argparse
import os
import sys

from bnn_inference.tools.console import Console
from bnn_inference.tools import parser as par
from bnn_inference.predict import predict
from bnn_inference.train import train
from bnn_inference.join_predictions import join_predictions

def main(args=None):
    # enable VT100 Escape Sequence for WINDOWS 10 for Console outputs
    # https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences
    os.system("")

    Console.banner()
    Console.info("Running bnn_inference version " + str(Console.get_version()))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser(
        "train",
        help="Train a BNN model",
    )
    par.add_arguments(parser_train)
    parser_train.set_defaults(func=train)


    parser_predict = subparsers.add_parser(
        "predict",
        help="Predict terrain data from priors using BNN inference",
    )
    par.add_arguments(parser_predict)
    parser_predict.set_defaults(func=predict)

    parser_join_predictions = subparsers.add_parser(
        "join_predictions",
        help="Join predictions with target values",
    )
    par.add_arguments(parser_join_predictions)
    parser_predict.set_defaults(func=join_predictions)

    args = parser.parse_args(args)

    if len(sys.argv) == 1 and args is None:  # no argument passed? error, some parameters were expected
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)

    args.func(args)
