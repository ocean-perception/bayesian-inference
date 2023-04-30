import os

import typer
import yaml

from bnn_inference.join_predictions import join_predictions_impl
from bnn_inference.predict import predict_impl
from bnn_inference.tools.console import Console
from bnn_inference.train import train_impl

app = typer.Typer(
    add_completion=False, context_settings={"help_option_names": ["-h", "--help"]}
)


def config_cb(ctx: typer.Context, param: typer.CallbackParam, value: str):
    """Callback function to load a YAML configuration file before parsing the CLI

    Args:
        ctx (typer.Context): Typer context
        param (typer.CallbackParam): Typer callback parameter
        value (str): Path to the YAML configuration file

    """
    if value:
        Console.info(f"Loading config file: {value}")
        try:
            with open(value, "r") as f:  # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)  # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return value


@app.command()
def train(
    config: str = typer.Option(
        "",
        help="[future] Path to a YAML configuration file. You can use the file exclusively or overwrite any arguments via CLI.",
        callback=config_cb,
        is_eager=True,
    ),
    latent_csv: str = typer.Option(
        ...,
        help="Path to CSV containing the latent representation vector for each input entry (image). The 'UUID' is used to match against the target file entries",
    ),
    latent_key: str = typer.Option(
        "latent_",
        help="Name of the key used for the columns containing the latent vector. For example, a h=8 vector should be read as 'latent_0,latent_1,...,latent_7'",
    ),
    target_csv: str = typer.Option(
        ...,
        help="Path to CSV containing the target entries to be used for training/validation. The 'UUID' is used to match against the input file entries",
    ),
    target_key: str = typer.Option(
        ...,
        help="Keyword that defines the field to be learnt/predicted. It must match the column name in the target file",
    ),
    uuid_key: str = typer.Option(
        "relative_path",
        help="Unique identifier string used as key for input/target example matching. The UUID string must match for both the input (latent) file and the target file column identifier",
    ),
    output_csv: str = typer.Option(
        "",
        help="Generated file containing the expected and predicted value for each input entry. It preserves the input file columns and appends the predicted columns",
    ),
    output_network_filename: str = typer.Option(
        "",
        help="Output path for the trained Bayesian NN in PyTorch compatible format.",
    ),
    logfile_name: str = typer.Option(
        "",
        help="Output path to the logfile with the training / validation error for each epoch. It can be used to monitor the training process",
    ),
    num_epochs: int = typer.Option(
        100,
        help="Number of training epochs"
    ),
    num_samples: int = typer.Option(
        10,
        help="Number of Monte Carlo samples for ELBO based posterior estimation",
    ),
    xratio: float = typer.Option(
        0.9,
        help="Ratio of dataset samples to be used for training (T). The validatio (V) is calculated as V = 1 - T",
    ),
    scale_factor: float = typer.Option(
        1.0, help="Scaling factor to apply to the output target. Default: 1.0 (no scaling))"
    ),
    learning_rate: float = typer.Option(
        1e-3, help="Optimizer learning rate"
    ),
    lambda_recon: float = typer.Option(
        10.0, help="Reconstruction loss lambda value (hyperparameter)"
    ),
    lambda_elbo: float = typer.Option(
        1.0, help="ELBO KL divergence cost lamba value (hyperparameter)"
    ),
    loss_method: str = typer.Option(
        "mse", help="Defines the loss method. Can be 'mse' or 'cosine_similarity'"
    ),
    gpu_index: int = typer.Option(0, help="Index of CUDA device to be used."),
    cpu_only: bool = typer.Option(
        False,
        help="If set, the training will be performed on the CPU. This is useful for debugging purposes and low-spec computers.",
    ),
):
    Console.info("Training")
    if config == "":
        Console.info("Using command line arguments only.")
    train_impl(
        latent_csv=latent_csv,
        latent_key=latent_key,
        target_csv=target_csv,
        target_key=target_key,
        uuid_key=uuid_key,
        output_csv=output_csv,
        output_network_filename=output_network_filename,
        logfile_name=logfile_name,
        num_epochs=num_epochs,
        num_samples=num_samples,
        xratio=xratio,
        scale_factor=scale_factor,
        learning_rate=learning_rate,
        lambda_recon=lambda_recon,
        lambda_elbo=lambda_elbo,
        loss_method=loss_method,
        gpu_index=gpu_index,
        cpu_only=cpu_only,
    )


@app.command()
def predict(
    config: str = typer.Option(
        "",
        help="[future] Path to a YAML configuration file. You can use the file exclusively or overwrite any arguments via CLI.",
        callback=config_cb,
        is_eager=True,
    ),
    latent_csv: str = typer.Option(
        ...,
        help="Path to CSV containing the latent representation vector for each input entry (image). The 'UUID' is used to match against the target file entries",
    ),
    latent_key: str = typer.Option(
        "latent_",
        help="Name of the key used for the columns containing the latent vector. For example, a h=8 vector should be read as 'latent_0,latent_1,...,latent_7'",
    ),
    target_key: str = typer.Option(
        ...,
        help="Keyword that defines the field to be learnt/predicted. It must match the column name in the target file",
    ),
    output_csv: str = typer.Option(
        "",
        help="File containing the expected and inferred value for each input entry. It preserves the input file columns and appends the corresponding prediction",
    ),
    output_network_filename: str = typer.Option(
        ..., help="Trained Bayesian Neural Network in PyTorch compatible format."
    ),
    num_samples: int = typer.Option(
        10,
        help="Number of Monte Carlo samples for ELBO based posterior estimation",
    ),
    scale_factor: float = typer.Option(
        1.0, help="Output scaling factor. Default: 1.0 (no scaling))"
    ),
    gpu_index: int = typer.Option(0, help="Index of CUDA device to be used."),
    cpu_only: bool = typer.Option(
        False,
        help="If set, the training will be performed on the CPU. This is useful for debugging purposes.",
    ),
):
    Console.info("Predicting")
    if config == "":
        Console.info("Using command line arguments only.")
    predict_impl(
        latent_csv,
        latent_key,
        target_key,
        output_csv,
        output_network_filename,
        num_samples,
        scale_factor,
        gpu_index,
        cpu_only,
    )


@app.command("join_predictions")
def join_predictions(
    config: str = typer.Option(
        "",
        help="Path to a YAML configuration file. You can use the file exclusively or overwrite any arguments via CLI.",
        callback=config_cb,
        is_eager=True,
    ),
    latent_csv: str = typer.Option(
        ...,
        help="Path to CSV containing the latent representation vector for each input entry (image). The 'UUID' is used to match against the target file entries",
    ),
    target_csv: str = typer.Option(
        ...,
        help="Path to CSV containing the target entries to be used for training/validation. The 'UUID' is used to match against the input file entries",
    ),
    target_key: str = typer.Option(
        ...,
        help="Keyword that defines the field to be learnt/predicted. It must match the column name in the target file",
    ),
    output_csv: str = typer.Option(
        "",
        help="File containing the expected and inferred value for each input entry. It preserves the input file columns and appends the corresponding prediction",
    ),
):
    Console.info("Joining predictions")
    join_predictions_impl(latent_csv, target_csv, target_key, output_csv)


def main(args=None):
    # enable VT100 Escape Sequence for WINDOWS 10 for Console outputs
    # https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences
    os.system("")
    Console.banner()
    Console.info("Running bnn_inference version " + str(Console.get_version()))
    app()


if __name__ == "__main__":
    main()
