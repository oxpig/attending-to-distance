from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import math

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PartialAttention(nn.Module):
    def __init__(
        self,
        emb_size,
        num_heads,
    ):
        super(PartialAttention, self).__init__()

        self.key_map = nn.Linear(emb_size, int(emb_size/num_heads))
        self.query_map = nn.Linear(emb_size, int(emb_size/num_heads))
        self.norm = nn.LayerNorm(emb_size)

    def forward(
        self,
        src_emb,
    ):
        src_emb = src_emb.transpose(0,1)
        src_emb = self.norm(src_emb)
        k = self.key_map(src_emb)
        q = self.query_map(src_emb)
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = attn_output_weights
        max_attn = torch.max(attn_output_weights, dim=-1, keepdim=True).values
        attn_output_weights = attn_output_weights - max_attn
        attn_output_weights = torch.exp(attn_output_weights)
        return attn_output_weights


# 3D Transformer Partial Encoder
class StructLangReg(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation='relu',
    ):
        super(StructLangReg, self).__init__()

        # dtype=torch.double
        dtype=torch.float
        self.embed = nn.Linear(src_vocab_size, emb_size)
        encoder_layer = TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dtype=dtype,
            dropout=dropout,
            norm_first=True,
            dim_feedforward=dim_feedforward,
            activation=activation,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pa = PartialAttention(emb_size, nhead)
        self.d_model = emb_size

    def forward(
        self,
        src: Tensor,
        src_pos: Tensor,
        test=False,
    ):
        s = 16
        src_pos = src_pos / s
        src_embed = self.embed(src_pos)
        outs = self.encoder(src_embed)
        return self.pa(outs)

