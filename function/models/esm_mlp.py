# Adapted from https://github.com/bio-ontology-research-group/deepgo2/tree/main/deepgo

import torch
from torch import nn


class Residual(nn.Module):
    """
    A residual layer that adds the output of a function to its input.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class MLPBlock(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) block with one fully connected layer.
    """

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class MLPModel(nn.Module):
    """
    Baseline MLP model with two fully connected layers with residual connection
    """
    
    def __init__(self, input_dim, num_onts, device, nodes=[1024,]):
        super().__init__()
        self.num_onts = num_onts
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_dim, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
        net.append(nn.Linear(hidden_dim, num_onts))
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        inputs = features.nan_to_num()
        return self.net(inputs)
