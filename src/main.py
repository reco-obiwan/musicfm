import os
import argparse

from torch.utils.data import DataLoader

from transformers.utils import logging

from utilities import Config
from trainer import MusicFMTrainer
from datasets import TrainDataset, ValidationDataset

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

workdir = os.environ["WORKDIR"]


def get_arguments():
    parser = argparse.ArgumentParser(description="Train Mulan")
    parser.add_argument("--task", "-t", help="choose a task", required=True)
    parser.add_argument("--config", "-c", help="config yaml file path", required=True)
    return parser.parse_args()


def train(config):
    batch_size = config["params"]["batch_size"]

    train_loader = DataLoader(
        dataset=TrainDataset(config=config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        prefetch_factor=4,
    )

    valid_loader = DataLoader(
        dataset=ValidationDataset(config=config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        prefetch_factor=4,
    )

    MusicFMTrainer(
        workdir=workdir,
        train_loader=train_loader,
        valid_loader=valid_loader,
        version=config["version"],
        epoches=config["params"]["epoches"],
        save_interval=config["params"]["save_interval"],
        log_interval=config["params"]["log_interval"],
        accelerate_kwargs={
            "cpu": False,
        },
    ).start_train()


def main():
    args = get_arguments()
    logger.info("args: %s", args)

    config = Config(config_path=args.config)
    logger.info("config: %s ", config)

    if args.task == "train":
        train(config)
    else:
        raise ValueError()


if __name__ == "__main__":
    main()
