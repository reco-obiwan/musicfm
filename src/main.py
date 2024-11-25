import os
import sys
import torch

from model.musicfm_25hz import MusicFM25Hz
from trainer import MusicFMTrainer

workdir = os.environ.get("WORKDIR")

def main():
    # load MusicFM
    musicfm = MusicFM25Hz(
        stat_path=os.path.join(workdir, "res", "msd_stats.json"),
        model_path=os.path.join(workdir, "res", "pretrained_msd.pt"),
    )

    MusicFMTrainer(model=musicfm).train()


if __name__ == "__main__":
    main()
    