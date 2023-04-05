from pydantic import BaseModel, validator
from typing import List
import yaml
from pathlib import Path

from bnn_inference.tools.console import Console

class BNNConfiguration(BaseModel):
    input_file: List[str]
    latent_key: str = 'latent_'
    output_key: str = 'measurability'
    uuid: str = 'relative_path'
    target: List[str]
    predictions_name: str
    network_name: str
    logfile_name: str
    num_epochs: int = 100
    num_samples: int = 10
    xratio: float = 0.9
    scale_factor: float = 1.0
    learning_rate: float = 1e-3
    lambda_recon: float = 10.0
    lambda_elbo: float = 1.0
    gpu_index: int = 0

    def __init__(self, args=None, config_file = None):
        if args.config:
            config_file = args.config
        if config_file:
            Console.info("Loading configuration from file", config_file)
            config_file = Path(config_file)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file {config_file} not found")
            stream = config_file.open("r")
            data = yaml.safe_load(stream)
            super().__init__(**data)
        if args:
            Console.info("Loading configuration from command line")
            self.from_comand_line(args)
        print(self)

    def __str__(self):
        return str(self.dict())

    def from_comand_line(self, args):
        if args.input:
            self.input_file = args.input
        if args.latent:
            self.latent_key = args.latent
        if args.target:
            self.target = args.target
        if args.key:
            self.output_key = args.key
        if args.output:
            self.predictions_name = args.output
        if args.uuid:
            self.uuid = args.uuid
        if args.network:
            self.network_name = args.network
        if args.logfile:
            self.logfile_name = args.logfile
        if args.epochs:
            self.num_epochs = args.epochs
        if args.samples:
            self.num_samples = args.samples
        if args.xratio:
            self.xratio = args.xratio
        if args.scale:
            self.scale_factor = args.scale
        if args.lr:
            self.learning_rate = args.lr
        if args.lambda_recon:
            self.lambda_recon = args.lambda_recon
        if args.lambda_elbo:
            self.lambda_elbo = args.lambda_elbo
        if args.gpu:
            self.gpu_index = args.gpu

    @validator('num_samples')
    def num_samples_must_be_greater_than_two(cls, v):
        assert v > 2, 'num_samples must be greater than 2'
        return v

    @validator('xratio')
    def xratio_must_be_between_zero_and_one(cls, v):
        assert 0.0 < v < 1.0, 'xratio must be between 0.0 and 1.0'
        return v

    @validator('scale_factor')
    def scale_factor_must_be_greater_than_zero(cls, v):
        assert v > 0.0, 'scale_factor must be greater than 0.0'
        return v

    @validator('learning_rate')
    def learning_rate_must_be_greater_than_zero(cls, v):
        assert v > 0.0, 'learning_rate must be greater than 0.0'
        return v

    @validator('lambda_recon')
    def lambda_recon_must_be_greater_than_zero(cls, v):
        assert v > 0.0, 'lambda_recon must be greater than 0.0'
        return v

    @validator('lambda_elbo')
    def lambda_elbo_must_be_greater_than_zero(cls, v):
        assert v > 0.0, 'lambda_elbo must be greater than 0.0'
        return v

    @validator('gpu_index')
    def gpu_index_must_be_greater_than_zero(cls, v):
        assert v >= 0, 'gpu_index must be greater than or equal to 0'
        return v

    @validator('input_file')
    def input_file_list_must_exist(cls, v):
        for file in v:
            assert Path(file).exists(), f"Input file {file} not found"
        return v

    @validator('target')
    def target_file_list_must_exist(cls, v):
        for file in v:
            assert Path(file).exists(), f"Target file {file} not found"
        return v

