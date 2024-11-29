import os
import sys
import torch

from model.musicfm_25hz import MusicFM25Hz

# dummy audio (30 seconds, 24kHz)
wav = (torch.rand(4, 24000 * 30) - 0.5) * 2
workdir = os.environ.get("WORKDIR")

# load MusicFM
musicfm = MusicFM25Hz(
    stat_path=os.path.join(workdir, "res", "msd_stats.json"),
    model_path=os.path.join(workdir, "res", "pretrained_msd.pt"),
)

# to GPUs


print(f"wav: {wav.shape}")
wav = wav.cuda()
musicfm = musicfm.cuda()

# get embeddings
musicfm.eval()
emb = musicfm.get_latent(wav, layer_ix=12)
print(f"emb: {emb.shape}")

# logits, hidden_emb = musicfm.get_predictions(wav)


# print(f"logits: {logits['melspec_2048'].shape}")
# print(f"hidden_states: {hidden_emb[0].shape}")
# print(f"hidden_states: {hidden_emb[1].shape}")
