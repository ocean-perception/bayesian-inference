def add_arguments(obj):
    # input #########################
    obj.add_argument(
        "-i", "--input",
        type=str,
        # default='input_latents.csv',
        help="Path to CSV containing the latent representation vector for each input entry (image). The 'UUID' is used to match against the target file entries"
    )
    # latent
    obj.add_argument(
        "-l", "--latent",
        type=str,
        default='latent_',
        help="Name of the key used for the columns containing the latent vector. For example, a h=8 vector should be read as 'latent_0,latent_1,...,latent_7'"
    )
    # target #########################
    obj.add_argument(
        "-t", "--target",
        type=str,
        # default='target_file.csv',
        help="Path to CSV containing the target entries to be used for training/validation. The 'UUID' is used to match against the input file entries"
    )
    # key #########################
    obj.add_argument(
        "-k", "--key",
#        default='key',
        type=str,
        help="Keyword that defines the field to be learnt/predicted. It must match the column name in the target file"
    )
    # output #########################
    obj.add_argument(
        "-o", "--output",
        # default='inferred.csv',
        type=str,
        help="File containing the expected and inferred value for each input entry. It preserves the input file columns and appends the corresponding prediction"
    )
    # uuid #########################
    obj.add_argument(
        "-u", "--uuid",
        default='UUID',
        type=str,
        help="Unique identifier string used as key for input/target example matching. The UUID string must match for both the input (latent) file and the target file column identifier"
    )
    # network #########################
    obj.add_argument(
        "-n", "--network",
        default='bnn_trained.pth',
        type=str,
        help="Output path to write the trained Bayesian Neural Network in PyTorch compatible format."
    )
    # logfile #########################
    obj.add_argument(
        "-g", "--logfile",
#        default='training_log.csv',
        type=str,
        help="Output path to the logfile with the training / validation error for each epoch. Used to inspect the training performance"
    )

    # config #########################
    obj.add_argument(
        "-c", "--config",
    #    default='configuration.yaml',
        type=str,
        help="Path to YAML configuration file (optional)"
    )
    # epochs #########################
    obj.add_argument(
        "-e", "--epochs",
        default='100',
        type=int,
        help="Define the number of training epochs"
    )
    # samples #########################
    obj.add_argument(
        "-s", "--samples",
        default='10',
        type=int,
        help="Define the number of samples for sample_elbo based posterior estimation"
    )
    # xvalitaion #########################
    obj.add_argument(
        "-x", "--xratio",
        default='0.8',
        type=float,
        help="Define the training (T) ratio as the proportion of the complete dataset used for training. T + V = 1.0"
    )
    # output/target scaling
    obj.add_argument(
        "--scale",
        default='1.0',
        type=float,
        help="Define the outputtarget scaling factor. Default: 1.0 (no scaling))"
    )

    # user defined learning rate
    obj.add_argument(
        "--lr",
        # default='0.001',
        type=float,
        help="Define the learning rate for the optimizer. Default: 0.001"
    )

    # user defined lamba reconstruction loss
    obj.add_argument(
        "--lambda_recon",
        # default='10.0',
        type=float,
        help="Define the lambda value for the reconstruction loss. Default: 10.0"
    )

    # User defined lambda value for ELBO KL divergence cost
    obj.add_argument(
        "--lambda_elbo",
        # default='1.0',
        type=float,
        help="Define the lambda value for the ELBO KL divergence cost. Default: 1.0"
    )