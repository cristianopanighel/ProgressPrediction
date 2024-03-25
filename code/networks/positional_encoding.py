import math
import torch
import numpy as np

from torch import nn, Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # random
        # idx = torch.randperm(x.shape[0])
        # y = x[idx].view(x.size())
        # x = y + self.pe[:x.size(0)]

        # reverse
        # y = torch.flip(x, (1, 0, 2))
        # x = y + self.pe[:x.size(0)]

        # normal
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
