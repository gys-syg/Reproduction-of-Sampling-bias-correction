
import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(sizes, activation=nn.ReLU, last_bn=False):
    layers = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(activation())
        elif last_bn:
            layers.append(nn.BatchNorm1d(sizes[i+1]))
    return nn.Sequential(*layers)

class QueryTower(nn.Module):
    def __init__(self, in_dim: int, hidden=(1024, 512), out_dim=128, l2norm=True):
        super().__init__()
        sizes = [in_dim] + list(hidden) + [out_dim]
        self.net = mlp(sizes)
        self.l2norm = l2norm

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        if self.l2norm:
            z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z

class ItemTower(nn.Module):
    def __init__(self, in_dim: int, hidden=(1024, 512), out_dim=128, l2norm=True):
        super().__init__()
        sizes = [in_dim] + list(hidden) + [out_dim]
        self.net = mlp(sizes)
        self.l2norm = l2norm

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        if self.l2norm:
            z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z
