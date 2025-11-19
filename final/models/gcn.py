
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    """Simple GCN layer: output = activation(A @ X @ W) [+ residual]."""
    def __init__(self, in_units: int, out_units: int, residual: bool=False, use_bias: bool=False, activation="relu"):
        super().__init__()
        self.linear = nn.Linear(in_units, out_units, bias=use_bias)
        self.residual = residual
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def forward(self, features: torch.Tensor, adj: torch.Tensor):
        agg = adj @ features
        out = self.linear(agg)
        if self.activation is not None:
            out = self.activation(out)
        if self.residual and out.shape == features.shape:
            out = out + features
        return out
