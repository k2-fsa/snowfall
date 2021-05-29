#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import humanfriendly
import librosa
import numpy as np
import torch

from pathlib import Path
from typeguard import check_argument_types
from typing import Optional, Tuple, Union


# Modified from:
# https://github.com/espnet/espnet/blob/08feae5bb93fa8f6dcba66760c8617a4b5e39d70/espnet/nets/pytorch_backend/frontends/feature_transform.py#L135
class GlobalMVN(torch.nn.Module):
    """Apply global mean and variance normalization

    TODO(kamo): Make this class portable somehow

    Args:
        stats_file: npy file
        norm_means: Apply mean normalization
        norm_vars: Apply var normalization
        eps:
    """

    def __init__(
        self,
        stats_file: Union[Path, str],
        norm_means: bool = True,
        norm_vars: bool = True,
        eps: float = 1.0e-20,
    ):
        assert check_argument_types()
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps
        stats_file = Path(stats_file)

        self.stats_file = stats_file
        stats = np.load(stats_file)
        if isinstance(stats, np.ndarray):
            # Kaldi like stats
            count = stats[0].flatten()[-1]
            mean = stats[0, :-1] / count
            var = stats[1, :-1] / count - mean * mean
        else:
            # New style: Npz file
            count = stats["count"]
            sum_v = stats["sum"]
            sum_square_v = stats["sum_square"]
            mean = sum_v / count
            var = sum_square_v / count - mean * mean
        std = np.sqrt(np.maximum(var, eps))

        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("std", torch.from_numpy(std))

    def forward(
            self,
            x: torch.Tensor,
            ilens: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)
        """
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1))
        norm_means = self.norm_means
        norm_vars = self.norm_vars
        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)

        # feat: (B, T, D)
        if norm_means:
            if x.requires_grad:
                x = x - self.mean
            else:
                x -= self.mean

        if norm_vars:
            x /= self.std

        return x, ilens


# Modified from:
# https://github.com/espnet/espnet/blob/08feae5bb93fa8f6dcba66760c8617a4b5e39d70/espnet2/layers/stft.py#L14:7
class Stft(torch.nn.Module):

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(self.win_length,
                                 dtype=input.dtype,
                                 device=input.device)
        else:
            window = None
        output = torch.stft(
            input,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            normalized=self.normalized,
            onesided=self.onesided,
        )
        output = output.transpose(1, 2)

        if self.center:
            pad = self.win_length // 2
            ilens = ilens + 2 * pad

        olens = (ilens - self.win_length) // self.hop_length + 1

        return output, olens


# Modified from:
# https://github.com/espnet/espnet/blob/08feae5bb93fa8f6dcba66760c8617a4b5e39d70/espnet2/asr/frontend/default.py#L19
class Fbank(torch.nn.Module):
    """

    Stft -> Power-spec -> Mel-Fbank
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            window=window,
            normalized=normalized,
            onesided=onesided,
        )

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        # _mel_options = {'sr': 16000, 'n_fft': 512, 'n_mels': 80, 'fmin': 0, 'fmax': 8000.0, 'htk': False}
        melmat = librosa.filters.mel(**_mel_options)

        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def forward(
            self, input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_stft, feats_lens = self.stft(input, input_lengths)
        input_stft = torch.complex(input_stft[..., 0], input_stft[..., 1])

        input_power = input_stft.real**2 + input_stft.imag**2

        mel_feat = torch.matmul(input_power, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        input_feats = mel_feat.log()

        return input_feats, feats_lens
