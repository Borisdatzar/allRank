import pickle
from attr import asdict
import numpy as np
import os
import torch
import torch.nn as nn
from allrank.config import Config, FCConfig, TransformerConfig, PostModelConfig, PositionalEncoding
from allrank.data.dataset_loading import create_data_loaders, load_test_libsvm_dataset
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.eval_utils import eval_model
from allrank.utils.command_executor import execute_command
from allrank.utils.file_utils import PathsContainer
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from pprint import pformat

import torch
import torch.nn.functional as F

from types import SimpleNamespace


def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")

    return parser.parse_args()


def run():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    logger = init_logger(paths.output_dir, log_name='serilazation.log')

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))


    # gpu support
    dev = get_torch_device()
    logger.info("Model testing will execute on {}".format(dev.type))

    # load model
    model_path = os.path.join(paths.output_dir, 'model.pkl')

    # Load the model from the pickle file
    with open(model_path, 'rb') as file:
        state_dict = torch.load(model_path)

    model = make_model(n_features=29, **asdict(config.model, recurse=False))

    params = model.serialize_params()
    print(params)
    params['transformer']['positional_encoding'] = None if params['transformer']['positional_encoding'] is None else PositionalEncoding(**params['transformer']['positional_encoding'])

    params['transformer'] = TransformerConfig(**params['transformer'])

    model = make_model(**params)


    # Create the test data with shape [25, 29] for 25 items, each with 29 features
    test_data = [[0 for _ in range(29)] for _ in range(25)]

    # Convert test data to a tensor and add batch dimension
    test_data_tensor = torch.tensor(test_data).unsqueeze(0)  # Shape becomes [1, 25, 29]

    # Pad the test data to expand from 25 items to 250 items (adding 225 padding rows)
    padded_data = F.pad(test_data_tensor, (0, 0, 0, 225), mode='constant', value=0)  # Shape: [1, 250, 29]

    # Create the mask: True for actual items, False for padding
    mask = torch.cat([torch.ones(1, 25, dtype=torch.bool), torch.zeros(1, 225, dtype=torch.bool)], dim=1)  # Shape: [1, 250]

    # Create the indices, 1..250
    indices = torch.arange(1, 251).unsqueeze(0)  # Shape: [1, 250]

    # Set the model to evaluation mode
    model.eval()

    # Forward pass with no gradient calculation
    with torch.no_grad():
        output = model(padded_data, mask, indices)



    print(output)

    print('success!')

    print(model.serialize_params())




if __name__ == "__main__":
    run()
