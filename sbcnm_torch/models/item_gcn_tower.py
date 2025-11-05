
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCN

class ItemGCNTower(nn.Module):
    """GCN-based item encoder operating on the full item graph."""
    def __init__(self, in_dim, hidden=(256,), out_dim=128, l2norm=True):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(GCN(last, h, residual=False, use_bias=False, activation="relu"))
            last = h
        self.gcn_layers = nn.ModuleList(layers)
        self.proj = nn.Linear(last, out_dim, bias=False)
        self.l2norm = l2norm
        self.register_buffer("_all_embs", None, persistent=False)

    def encode_all(self, item_feats: torch.Tensor, adj: torch.Tensor):
        x = item_feats
        for gcn in self.gcn_layers:
            x = gcn(x, adj)
        z = self.proj(x)
        if self.l2norm:
            z = F.normalize(z, p=2, dim=-1)
        self._all_embs = z
        return z

    def forward_indices(self, indices: torch.Tensor):
        assert self._all_embs is not None, "Call encode_all(...) before forward_indices"
        return self._all_embs[indices]
