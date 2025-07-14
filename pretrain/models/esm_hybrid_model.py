from torch import Tensor
import torch
import torch.nn as nn
import math
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer

import esm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 3D BERT
class BERTCoords(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        add_coords=True,
    ):
        super(BERTCoords, self).__init__()

        dtype=torch.float

        self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm_model.requires_grad_(False)
        self.emb_size = self.esm_model.embed_dim
        self.num_layers = self.esm_model.num_layers

        self.encoder = esm.model.esm2.ESM2(
            num_layers = self.num_layers,
            embed_dim = self.emb_size,
            attention_heads = self.esm_model.attention_heads,
            alphabet = self.alphabet,
            token_dropout = dropout,
        )

        self.add_coords = add_coords
        self.coords_embed = nn.Linear(3, self.emb_size)
        self.connector = nn.Linear(self.emb_size, self.emb_size)

        self.norm = nn.LayerNorm(self.emb_size)
        self.generator = nn.Linear(self.emb_size, len(self.alphabet.all_toks))

    def forward(
        self,
        src: Tensor,
        src_pos: Tensor,
        mask: Tensor,
        return_emb=False,
        return_contacts=False,
    ):
        src_pos = src_pos / 16 # Rescale
        layer_idx = self.num_layers
        esm_result = self.esm_model(src, repr_layers=[layer_idx])
        src_emb = self.connector(esm_result['representations'][layer_idx]) # Map from ESM last layer
        if self.add_coords:
            src_emb = src_emb + self.coords_embed(src_pos)
        src_emb = src_emb.transpose(0,1)
        if return_contacts:
            attn_weights = []
        for layer in self.encoder.layers:
            src_emb, attn = layer(
                src_emb,
                self_attn_padding_mask=mask,
                need_head_weights=return_contacts,
            )
            if return_contacts:
                attn_weights.append(attn.transpose(0,1))
        src_emb = src_emb.transpose(0,1)

        if return_contacts:
            attentions = torch.stack(attn_weights, 1)
            if mask is not None:
                attention_mask = 1 - mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            contacts = self.encoder.contact_head(src, attentions)
            return contacts

        if return_emb:
            return self.norm(src_emb)
        logits = self.generator(self.norm(src_emb))
        return logits
