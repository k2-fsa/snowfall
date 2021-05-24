from typing import Any
from typing import List
from typing import Tuple

import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from snowfall.models.transformer import generate_square_subsequent_mask
from snowfall.models.transformer import TransformerEncoderLayer


# modified from:
# https://github.com/espnet/espnet/blob/dab2092bc9c8e184c48cc6e603037333bd97dcd1/espnet/nets/pytorch_backend/nets_utils.py#L64
def make_pad_mask(lengths):
    """Make mask tensor containing indices of padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).

    Returns:
        Tensor: Mask tensor containing indices of padded part.
    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()

    maxlen = int(max(lengths))
    bs = int(len(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(
                    0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores,
                                      dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
            )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, attn_mask), None


class LMEncoder(nn.Module):

    def __init__(
        self,
        attention_dim=512,
        attention_heads=8,
        attention_dropout_rate=0.0,
        num_blocks=16,
        dim_feedforward=2048,
        normalize_before=True,
    ):
        super().__init__()

        self.normalize_before = normalize_before

        encoder_layer = TransformerEncoderLayer(
            d_model=attention_dim,
            custom_attn=MultiHeadedAttention(attention_heads, attention_dim,
                                             attention_dropout_rate),
            nhead=attention_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=attention_dropout_rate,
        )

        self.encoders = nn.TransformerEncoder(encoder_layer, num_blocks, None)

        if self.normalize_before:
            self.after_norm = nn.LayerNorm(attention_dim)

    def forward(self, xs, masks):
        # xs: [batch_size, max_seq_len]
        # masks: [1, max_seq_len, max_seq_len], looks like
        #   tensor([[[ True, False, False,  ..., False, False, False],
        #            [ True,  True, False,  ..., False, False, False],
        #            [ True,  True,  True,  ..., False, False, False],
        #            ...,
        #            [ True,  True,  True,  ...,  True, False, False],
        #            [ True,  True,  True,  ...,  True,  True, False],
        #            [ True,  True,  True,  ...,  True,  True,  True]]])

        import numpy as np
        np.save('xs_before_encoders', xs.cpu().numpy())
        xs = self.encoders(xs, masks)
        np.save('xs_after_encoders', xs.cpu().numpy())
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        pos_enc: str = None,
        embed_unit: int = 128,
        att_unit: int = 512,
        head: int = 8,
        unit: int = 2048,
        layer: int = 16,
        dropout_rate: float = 0.0,
        ignore_id: int = 0,
    ):
        super().__init__()

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.ignore_id = ignore_id

        self.embed = nn.Embedding(vocab_size, embed_unit)
        self.input_embed = nn.Sequential(
            nn.Linear(embed_unit, att_unit),
            nn.LayerNorm(att_unit),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.encoder = LMEncoder(attention_dim=att_unit,
                                 attention_heads=head,
                                 num_blocks=layer,
                                 dim_feedforward=unit)
        self.decoder = nn.Linear(att_unit, vocab_size)

    def forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        import numpy as np
        # input: [batch_size, max_seq_len]
        x = self.embed(input)
        np.save('embed_x', x.cpu().numpy())
        x = self.input_embed(x)
        np.save('input_embed_x', x.cpu().numpy())
        mask = (generate_square_subsequent_mask(
            input.shape[-1]) == 0).unsqueeze(0).to(x.device)
        h = self.encoder(x, mask)
        np.save('h', h.cpu().numpy())
        y = self.decoder(h)
        # y: [batch_size, max_seq_len, vocab_size]
        return y

    def nll(self, xs_pad, target_pad, token_lens):
        # xs_pad/ys_pad: [batch_size, max_seq_len]
        # max_seq_len == max(len([<sos> token token token ... token])
        #             == max(len([token token token ... token <eos>])
        y = self.forward(xs_pad)
        # nll: (batch_size * max_seq_len,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]),
                              target_pad.view(-1),
                              reduction="none")
        # assign padded postion with 0.0
        nll.masked_fill_(
            make_pad_mask(token_lens).to(nll.device).view(-1), 0.0)

        # nll: (batch_size * max_seq_len,) -> (batch_size, max_seq_len)
        nll = nll.view(xs_pad.size(0), -1)
        return nll
