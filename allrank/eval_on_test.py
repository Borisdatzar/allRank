import pickle

import numpy as np
import os
import torch
from allrank.config import Config
from allrank.data.dataset_loading import create_data_loaders, load_test_libsvm_dataset
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.eval_utils import eval_model
from allrank.utils.command_executor import execute_command
from allrank.utils.file_utils import PathsContainer
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from pprint import pformat


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

    logger = init_logger(paths.output_dir, log_name='evaluation.log')

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config_for_eval.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # train_ds, val_ds
    test_ds = load_test_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        test_ds_role=config.data.test_ds_role,
    )

    test_dl, _ = create_data_loaders(
        test_ds, test_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    dev = get_torch_device()
    logger.info("Model testing will execute on {}".format(dev.type))

    # load model
    model_path = os.path.join(paths.output_dir, 'results', args.run_id, 'model.pkl')

    # Load the model from the pickle file
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model testing will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)


    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run testing
        result = eval_model(
            model=model,
            config=config,
            test_dl=test_dl,
            device=dev
        )



if __name__ == "__main__":
    run()
