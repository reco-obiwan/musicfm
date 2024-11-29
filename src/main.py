import os
import argparse

from torch.utils.data import DataLoader

from transformers.utils import logging

from config import Config
from trainer import MusicFMTrainer
from datasets import TrainDataset, ValidationDataset
from musicfm import MusicFM25Hz

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
    num_train_steps = config["params"]["num_train_steps"]

    # load MusicFM
    musicfm = MusicFM25Hz(
        stat_path=os.path.join(workdir, "res", "msd_stats.json"),
        model_path=os.path.join(workdir, "res", "pretrained_msd.pt"),
    )

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
        model=musicfm,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_train_steps=num_train_steps,
        accelerate_kwargs={
            "cpu": False,
            "log_with": ["tensorboard"],
            "project_dir": "tensorboard_log",
        },
    ).start()


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
