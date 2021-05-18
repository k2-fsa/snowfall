#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#
import logging

import torch
import torch.nn as nn
from snowfall.models.transformer import PositionalEncoding
from snowfall.models.transformer import TransformerEncoderLayer
from snowfall.models.transformer import generate_square_subsequent_mask

import k2


def _compute_padding_mask(len_per_seq: torch.Tensor):
    '''Sequences are of different lengths in number of frames
    and they are padded to the longest length. This function
    returns a mask to exclude the padded positions for attention.

    The returned mask is called `key_padding_mask` in PyTorch's
    implementation of multihead attention.

    Args:
      len_per_seq:
        A 1-D tensor of dtype torch.int32 containing the number
        of entries per seq before padding.
    Returns:
      Return a bool tensor of shape (len_per_seq.shape[0], max(len_per_seq)).
      The masked positions contain True, while non-masked positions contain
      False.
    '''
    assert len_per_seq.ndim == 1
    num_seqs = len_per_seq.shape[0]

    device = len_per_seq.device

    max_len = len_per_seq.max().item()

    # [0, 1, 2, ..., max_len - 1]
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=device)

    # [
    #   [0, 1, 2, ..., max_len - 1]
    #   [0, 1, 2, ..., max_len - 1]
    #   [0, 1, 2, ..., max_len - 1]
    #   ...
    # ]
    seq_range_expanded = seq_range.unsqueeze(0).expand(num_seqs, max_len)

    # [
    #   [x]
    #   [x]
    #   [x]
    #   ...
    # ]
    len_per_seq = len_per_seq.unsqueeze(-1)

    # It counts from zero, so >= instead of > is used.
    #
    # Padding positions are set to True
    mask = seq_range_expanded >= len_per_seq
    return mask


class SecondPassModel(nn.Module):
    '''
    The second pass model accepts two inputs:

        - The encoder memory output of the first pass model
        - The decoding denominator lattice of the first pass model

    For each sequence in the lattice, it computes the best path of it.
    Then the labels of the best path are extracted, which are phone IDs.
    Therefore, for each input frame, we can get its corresponding phone
    ID, i.e., its alignment.

    The phone IDs of each best path is used as input to an encoder
    model. The encoder model uses masked multihead attention.

    The output of the encoder model is concatenated with the encoder memory
    output from the first pass model. The concatenated result is fed into
    a linear layer. log_softmax is called on the output of the linear layer
    and the output of log_softmax is the output of the second pass model.
    '''

    def __init__(
            self,
            max_phone_id: int,
            d_model: int = 256,
            dropout: float = 0.1,
            nhead: int = 4,
            dim_feedforward: int = 2048,
            num_encoder_layers: int = 6,
    ):
        super().__init__()
        normalize_before = True  # True to use pre_LayerNorm

        num_classes = max_phone_id + 1  # +1 for the blank symbol

        self.encoder_embed = nn.Embedding(num_classes, d_model)

        self.encoder_pos = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout)

        if normalize_before:
            encoder_norm = nn.LayerNorm(d_model)
        else:
            encoder_norm = None

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_encoder_layers,
                                             norm=encoder_norm)

        # *2 because we concatenate the outputs of the two encoders from the first
        # pass and the second pass model
        self.output_linear = nn.Linear(d_model * 2, num_classes)

    def forward(self, encoder_memory: torch.Tensor, den_lats: k2.Fsa,
                supervision_segments: torch.Tensor):
        '''
        Args:
          encoder_memory:
            The output of the first network before applying log-softmax.
            If you're using Transformer/Conformer, it is the encoder output.
            Its shape is (T, batch_size, d_model)
        '''
        debug = True
        device = encoder_memory.device

        # Get the best path of each sequence. Only the labels of each path
        # are used in the following code. 0s in labels are not removed.
        best_paths = k2.shortest_path(den_lats, use_double_scores=True)

        # Note that len_per_seq does not count -1
        len_per_seq = supervision_segments[:, 2].to(device)

        if debug:
            # offset indicates the arc start index of each seq
            offset = k2.index(best_paths.arcs.row_splits(2),
                              best_paths.arcs.row_splits(1))

            # number of phones per seq.
            # minus 1 to exclude the label -1
            expected_len_per_seq = offset[1:] - offset[:-1] - 1

            assert torch.all(torch.eq(len_per_seq, expected_len_per_seq))

        # Note: `phones` also contains -1, for the arcs entering the final state
        phones = best_paths.labels.clone()

        # remove label -1
        phones = phones[phones != -1]

        # torch.split requires a tuple/list for sizes, so we use tolist() here.
        phones_per_seq = torch.split(phones, len_per_seq.tolist())

        # default padding value is 0
        padded_phones = nn.utils.rnn.pad_sequence(phones_per_seq,
                                                  batch_first=True)
        # padded_phones is of shape (num_seqs, T)
        # encoder_memory is of shape (T, num_batches, F)

        # Number of frames T should be equal
        assert padded_phones.shape[1] == encoder_memory.shape[0]
        # Caution: number of seqs is not necessarily equal to number of batches
        #  assert padded_phones.shape[0] == encoder_memory.shape[1]

        encoder_memory = encoder_memory.permute(1, 0, 2)
        # Now encoder_memory is (num_batches, T, F)

        acoustic_out = []
        for segment in supervision_segments.tolist():
            batch_idx, start, duration = segment
            end = start + duration
            acoustic_tensor = encoder_memory[batch_idx, start:end]
            acoustic_out.append(acoustic_tensor)

        padded_acoustics = nn.utils.rnn.pad_sequence(acoustic_out,
                                                     batch_first=True)
        # padded_acoustics is of shape (num_seqs, T, F)

        x2 = self.encoder_embed(padded_phones.long())
        # x2 is (num_seqs, T, F)
        x2 = self.encoder_pos(x2)

        assert x2.shape == padded_acoustics.shape

        # (B, T, F) -> (T, B, F)
        x2 = x2.permute(1, 0, 2)

        # compute two masks
        # (1) padding_mask
        # (2) src_mask for masked self-attention

        key_padding_mask = _compute_padding_mask(len_per_seq)
        # key_padding_mask is of shape (B, T)

        attn_mask = generate_square_subsequent_mask(x2.shape[0]).to(device)
        # attn_mask is of shape (T, T)

        x2 = self.encoder(src=x2,
                          mask=attn_mask,
                          src_key_padding_mask=key_padding_mask)

        # x2 is (T, B, F)

        x2 = x2.permute(1, 0, 2)

        # now x2 is (B, T, F)

        x_concat = torch.cat((padded_acoustics, x2), dim=-1)

        out = self.output_linear(x_concat)

        out = nn.functional.log_softmax(out, dim=2)  # (num_seqs, T, F)

        return out


def _test_compute_padding_mask():
    len_per_seq = torch.tensor([3, 5, 1, 2, 4])
    mask = _compute_padding_mask(len_per_seq)
    expected_mask = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, False, False],
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, False, True],
    ])
    assert torch.all(torch.eq(mask, expected_mask))


if __name__ == '__main__':
    _test_compute_padding_mask()
