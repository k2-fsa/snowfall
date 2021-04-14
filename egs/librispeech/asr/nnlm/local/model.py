# reference:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        # x: [seq_len, batch_size, d_model]
        # self.pe: [max_len, 1, d_model]
        # add with broadcasting
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
                 ntoken: int,
                 embed_unit: int,
                 attention_heads: int,
                 linear_units: int,
                 nlayers: int,
                 dropout: float = 0.5):
        '''
        ntoken: usually vocab_size + 3; 1 for <bos>, 1 for <eos>, 1 for <pad>
        embed_unit: the number of input channels
        attention_heads: parallel attention attention_headss
        linear_units: the dimension of the feedforward network model. 
              feedforward contains two Linear modules.
              self.linear1 = Linear(d_model, dim_feedforward)
              self.linear2 = Linear(dim_feedforward, d_model)
              so for a torch.nn.TransformerEncoder layer, the output dimension equals to input_dimension.

        '''
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError(
                'TransformerEncoder module does not exist in PyTorch 1.1 or lower.'
            )
        attention_head_dim = embed_unit / attention_heads
        assert attention_head_dim * attention_heads == embed_unit, "embed_dim must be divisible by num_attention_headss"

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embed_unit, dropout)
        encoder_layers = TransformerEncoderLayer(embed_unit, attention_heads,
                                                 linear_units, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, embed_unit, padding_idx=ntoken - 1)
        self.embed_unit = embed_unit
        self.decoder = nn.Linear(embed_unit, ntoken)

        self.init_weights()

        # used by evaluator
        self.ntoken = ntoken

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        # src: [seq_len, batch_size]
        # len(src) is seq_len
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(
                    len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        # mask: [seq_len, seq_len]
        # looks like:
        #     tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
        #             [0., 0., -inf,  ..., -inf, -inf, -inf],
        #             [0., 0., 0.,  ..., -inf, -inf, -inf],
        #             ...,
        #             [0., 0., 0.,  ..., 0., -inf, -inf],
        #             [0., 0., 0.,  ..., 0., 0., -inf],
        #             [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')

        # after self.encoder
        # src: [seq_len, batch_size, embed_unit]
        src = self.encoder(src) * math.sqrt(self.embed_unit)
        src = self.pos_encoder(src)

        # output: [seq_len, batch_size, ntoken]
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
