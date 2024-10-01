import pickle
from attr import asdict
import numpy as np
import os
import torch
import torch.nn as nn
from allrank.config import Config, FCConfig, TransformerConfig, PostModelConfig
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

    model = make_model(n_features=30, **asdict(config.model, recurse=False))

    params = model.serialize_params()
    print(params)

    params['transformer'] = TransformerConfig(**params['transformer'])

    model = make_model(**params)

    print('success!')

    print(model.serialize_params())




if __name__ == "__main__":
    run()
