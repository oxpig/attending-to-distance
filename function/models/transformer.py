import torch
from torch import nn


class TransformerModel(nn.Module):

    def __init__(self, input_dim, num_onts, device, transformer):
        super().__init__()
        self.transformer = transformer
        self.out_layer = nn.Linear(input_dim, num_onts)

    def forward(self, src, src_pos, mask):
        features = self.transformer(src, src_pos, mask, return_emb=True)
        return self.out_layer(features[0])
