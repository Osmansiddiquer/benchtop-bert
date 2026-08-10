import torch
from torch import nn

class LayerNormalization(nn.Module):
    """
    Normalizes the input across the features dimension (last dimension) for each sample 
    in the batch. Additionally applies learnable scaling and shifting parameters 
    (gamma and beta) to the normalized output.

    Different from Batch Normalization, which normalizes across the batch dimension, 
    Layer Normalization is applied independently to each sample in the batch. 
    This makes it more suitable for tasks where the batch size may vary or be small, 
    such as in sequence modeling tasks.
    """
    def __init__(self, d_model: int, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon # ensures that this doesn't blow up when the variance is very small leading to floating point instability. also division by zero.
        self.gamma = nn.Parameter(torch.ones(d_model)) # per-feature learnable scale
        self.beta = nn.Parameter(torch.zeros(d_model)) # per-feature learnable shift

    def forward(self, x:torch.Tensor):
        # x is (d1, d2, d3, ..., d_n). normalize across d_n
        mean = x.mean(dim = -1, keepdim=True)
        # Biased variance (population), matching torch.nn.LayerNorm; sqrt(var + eps)
        # keeps the divisor away from zero instead of dividing by a bare std.
        var = x.var(dim = -1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.epsilon) + self.beta


        