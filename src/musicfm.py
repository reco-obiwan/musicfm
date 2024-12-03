# pylint: disable=too-many-locals, too-many-arguments
import json
import random

import torch
from torch import nn, einsum
import torchaudio
from einops import rearrange

from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder,
    Wav2Vec2ConformerConfig,
)

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


class RandomProjectionQuantizer(nn.Module):
    """
    Random projection and codebook lookup module

    Some code is borrowed from:
     https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/random_projection_quantizer.py
    But I did normalization using pre-computed global mean & variance instead of using layer norm.
    """

    def __init__(
        self,
        input_dim,
        codebook_dim=16,
        codebook_size=4096,
        seed=142,
    ):
        super().__init__()

        # random seed
        torch.manual_seed(seed)

        # randomly initialized projection
        random_projection = torch.empty(input_dim, codebook_dim)
        nn.init.xavier_normal_(random_projection)
        self.register_buffer("random_projection", random_projection)

        # randomly initialized codebook
        codebook = torch.empty(codebook_size, codebook_dim)
        nn.init.normal_(codebook)
        nn.functional.normalize(codebook, dim=1, p=2)

        self.register_buffer("codebook", codebook)

    def codebook_lookup(self, x):
        # reshape
        b = x.shape[0]
        x = rearrange(x, "b n e -> (b n) e")
        # [3000, 16]

        # L2 normalization
        normalized_x = nn.functional.normalize(x, dim=1, p=2)
        # normalized_codebook = nn.functional.normalize(self.codebook, dim=1, p=2)

        # compute distances
        print(f"codebook shape: {self.codebook.shape}")
        print(f"normalized_x shape: {normalized_x.shape}")
        distances = torch.cdist(self.codebook, normalized_x)
        print(f"distances shape: {distances.shape}")

        # get nearest
        nearest_indices = torch.argmin(distances, dim=0)
        # 3000

        # reshape
        xq = rearrange(nearest_indices, "(b n) -> b n", b=b)

        return xq

    @torch.no_grad()
    def forward(self, x):
        # always eval
        self.eval()

        # random projection [batch, length, input_dim] -> [batch, length, codebook_dim]
        # [4, 750, 512]
        x = einsum("b n d, d e -> b n e", x, self.random_projection)
        # [4, 750, 16]

        # codebook lookup
        xq = self.codebook_lookup(x)

        return xq


class MelSTFT(nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=2048,
        hop_length=240,
        n_mels=128,
    ):
        super().__init__()

        # spectrogram
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        # amplitude to decibel
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
        return self.amplitude_to_db(self.mel_stft(waveform))


class Res2dModule(nn.Module):
    def __init__(self, idim, odim, stride=(2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(idim, odim, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(odim)
        self.conv2 = nn.Conv2d(odim, odim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(odim)
        self.relu = nn.ReLU()

        # residual
        self.diff = False
        if (idim != odim) or (stride[0] > 1):
            self.conv3 = nn.Conv2d(idim, odim, 3, padding=1, stride=stride)
            self.bn3 = nn.BatchNorm2d(odim)
            self.diff = True

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.diff:
            x = self.bn3(self.conv3(x))
        out = x + out
        out = self.relu(out)
        return out


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        hdim (int): Hidden dimension.
        odim (int): Output dimension.
        strides (list): Sizes of strides.
        n_bands (int): Number of frequency bands.
    """

    def __init__(self, idim, hdim, odim, strides=[2, 2], n_bands=64):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()

        self.conv = nn.Sequential(
            Res2dModule(idim, hdim, (2, strides[0])),
            Res2dModule(hdim, hdim, (2, strides[1])),
        )
        self.linear = nn.Linear(hdim * n_bands // 2 // 2, odim)

    def forward(self, x):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, idim, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
        """

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, f, t)
        x = self.conv(x)
        x = rearrange(x, "b c f t -> b t (c f)")
        x = self.linear(x)
        return x


class MusicFM25Hz(nn.Module):
    """
    MusicFM (Music Foundation Model)

    Input: 128-band mel spectrogram
    Frontend: 2-layer Residual convolution
    Backend: 12-layer Conformer
    Quantizer: a codebook for mel spectrogram
    """

    def __init__(
        self,
        num_codebooks=1,
        codebook_dim=16,
        codebook_size=4096,
        features=["melspec_2048"],
        hop_length=240,
        n_mels=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=12,
        mask_hop=0.4,
        mask_prob=0.6,
        # is_flash=False,
        stat_path="./res/msd_stats.json",
        model_path="./res/pretrained_msd.pt",
    ):
        super().__init__()

        # global variables
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.features = features

        # load feature mean / std stats
        with open(stat_path, "r") as f:
            self.stat = json.load(f)

        # feature extractor
        self.preprocessor_melspec_2048 = MelSTFT(n_fft=2048, hop_length=hop_length)

        # random quantizer
        seed = 142
        for feature in self.features:
            for i in range(num_codebooks):
                setattr(
                    self,
                    f"quantizer_{feature}_{i}",
                    RandomProjectionQuantizer(
                        n_mels * 4, codebook_dim, codebook_size, seed=seed + i
                    ),
                )

        # two residual convolution layers + one projection layer
        self.conv = Conv2dSubsampling(
            1, conv_dim, encoder_dim, strides=[2, 2], n_bands=n_mels
        )

        config = Wav2Vec2ConformerConfig.from_pretrained(
            "facebook/wav2vec2-conformer-rope-large-960h-ft"
        )

        config.num_hidden_layers = encoder_depth
        config.hidden_size = encoder_dim

        # logger.info(config)

        self.conformer = Wav2Vec2ConformerEncoder(config)

        # projection
        self.linear = nn.Linear(in_features=encoder_dim, out_features=codebook_size)

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # cls token (used for sequence classification)
        random.seed(seed)
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))

        # load model
        if model_path:
            S = torch.load(model_path, weights_only=False)["state_dict"]
            SS = {k[6:]: v for k, v in S.items()}
            self.load_state_dict(SS, strict=True)

    def masking(self, x):
        """random masking of 400ms with given probability"""
        # logger.info(f"x: {x.shape}")
        mx = x.clone()
        b, t = mx.shape
        len_masking_raw = int(24000 * self.mask_hop)  # 토큰 길이 400ms
        len_masking_token = int(
            24000 / self.hop_length / 2 / 2 * self.mask_hop
        )  # self.hop_length = 240, self.mask_hop = 0.4
        # len_masking_raw = 9600, len_masking_token = 10

        # get random mask indices
        start_indices = torch.rand(b, t // len_masking_raw) < self.mask_prob
        # [4, 75]

        a = start_indices.repeat_interleave(len_masking_raw, dim=1)
        # logger.info(f"a: {a.shape}")

        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_raw, dim=1)
        )
        # logger.info(f"time_domain_masked_indices: {time_domain_masked_indices.shape}")

        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_token, dim=1)
        )

        logger.info(f"token_domain_masked_indices: {token_domain_masked_indices.shape}")

        # mask with random values
        masking_noise = (
            torch.randn(time_domain_masked_indices.shape[0], dtype=x.dtype) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise.to(x.device)

        return mx, token_domain_masked_indices

    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
        return out

    def encoder(self, x):
        """2-layer conv + w2v-conformer"""
        logger.info(f"conv_input: {x.shape}")
        x = self.conv(x)
        logger.info(f"conv_output: {x.shape}")
        out = self.conformer(x, output_hidden_states=True)
        logger.info(f"conformer_out: {out['last_hidden_state'].shape}")
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        logits = {
            key: logits[:, :, i * self.codebook_size : (i + 1) * self.codebook_size]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        for key in x.keys():
            x[key] = (x[key] - self.stat["%s_mean" % key]) / self.stat["%s_std" % key]
        return x

    @torch.no_grad()
    def rearrange(self, x):
        """rearrange the batch to flatten every 4 steps"""
        for key in x.keys():
            if key == "chromagram":
                x[key] = rearrange(x[key], "b f t -> b t f")
            else:
                # [4, 128, 3000]
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=4)
                # [4, 750, 512]
        return x

    @torch.no_grad()
    def tokenize(self, x):
        out = {}
        for i, key in enumerate(x.keys()):
            logger.info("tokenize: %s", x[key].shape)
            layer = getattr(self, f"quantizer_{key}_{i}")
            out[key] = layer(x[key])
        return out

    def get_targets(self, x):
        x = self.preprocessing(x, features=self.features)
        x = self.normalize(x)
        x = self.rearrange(x)
        # [4, 750, 512]
        target_tokens = self.tokenize(x)
        # [4, 750]
        return target_tokens

    def get_predictions(self, x):
        # preprocessing
        x = self.preprocessing(x, features=self.features)
        x = self.normalize(x)

        # encoding
        logits, hidden_emb = self.encoder(x["melspec_2048"])

        return logits, hidden_emb

    def get_latent(self, x, layer_ix=12):
        _, hidden_states = self.get_predictions(x)
        logger.info(len(hidden_states))
        emb = hidden_states[layer_ix]
        return emb

    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            masked_logits = logits[key][tuple(masked_indices.t())]
            masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
        return losses, accuracies

    def forward(self, x):
        # get target feature tokens
        target_tokens = self.get_targets(x)

        # masking
        x, masked_indices = self.masking(x)

        # forward
        logits, hidden_emb = self.get_predictions(x)

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)

        return logits, hidden_emb, losses, accuracies
