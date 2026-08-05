from mini_enc_transformer.model.utils import initialize_normal_torch_weights
import torch
from torch import nn

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff = None):
        """
        d_model: dimension of the model (input and output)
        d_ff: dimension of the feedforward layer (hidden layer). If None, defaults to
        4 * d_model, as commonly used in transformer architectures.
        """
        super().__init__()
        self.d_model = d_model
        self.l1 = initialize_normal_torch_weights(d_model, d_model * 4) # the first layer weights
        self.b1 = nn.Parameter(torch.zeros(d_model * 4)) # the first layer bias
        self.relu_1 = nn.ReLU() # the activation function
        self.l2 = initialize_normal_torch_weights(d_model * 4, d_model) # the second layer weights connecting the hidden layer back to the output dimension. Nice design choice to have the same dimension as the input and output, allowing for residual connections.
        self.b2 = nn.Parameter(torch.zeros(d_model)) # the second layer bias

    def forward(self, x):
        assert x.shape[-1] == self.d_model

        x = self.relu_1(x @ self.l1 + self.b1) @ self.l2 + self.b2
        return x
        