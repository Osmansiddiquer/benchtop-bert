import torch
from torch import nn

def initialize_normal_torch_weights(M, N, std = 0.02) -> nn.Parameter:
    """Initialize weights with a normal distribution"""
    return nn.Parameter(torch.randn(M,N) * std)    # std = 0.02. from gpt. leaf tensor 