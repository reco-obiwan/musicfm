import os
import sys
import torch

from model.musicfm_25hz import MusicFM25Hz

# dummy audio (30 seconds, 24kHz)
wav = (torch.rand(4, 24000 * 30) - 0.5) * 2
workdir = os.environ.get("WORKDIR")

# load MusicFM
musicfm = MusicFM25Hz(
    is_flash=False,
    stat_path=os.path.join(workdir, "res", "msd_stats.json"),
    model_path=os.path.join(workdir, "res", "pretrained_msd.pt"),
)

# to GPUs
wav = wav.cuda()
musicfm = musicfm.cuda()

# get embeddings
musicfm.eval()
emb = musicfm.get_latent(wav, layer_ix=7)
