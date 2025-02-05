from torch import Tensor
import torch
import torch.nn as nn
import math
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int = 2000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# 3D BERT
class BERTCoords(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout: float = 0.1,
        add_coords=True,
    ):
        super(BERTCoords, self).__init__()

        # dtype=torch.double
        dtype=torch.float
        self.emb_size = emb_size
        self.add_coords = add_coords
        self.embed = nn.Embedding(src_vocab_size, emb_size)
        self.coords_embed = nn.Linear(4, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size,
            dropout=dropout
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dtype=dtype,
            dropout=dropout,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.norm = nn.LayerNorm(emb_size)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
        src_pos: Tensor,
        mask: Tensor,
        return_emb=False,
    ):
        src_pos = src_pos / 16 # Rescale
        src_emb = self.embed(src)
        if self.add_coords:
            src_emb = src_emb + self.coords_embed(src_pos)
        src_emb = self.positional_encoding(src_emb)
        outs = self.encoder(src_emb, src_key_padding_mask=mask)
        if return_emb:
            return self.norm(outs)
        logits = self.generator(self.norm(outs))
        return logits
