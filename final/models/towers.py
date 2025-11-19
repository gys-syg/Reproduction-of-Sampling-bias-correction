
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


# ========= 从这里开始加 DCN =========

class CrossLayer(nn.Module):
    """DCN v1 的单层 cross：x_{l+1} = x0 * (w^T x_l) + b + x_l"""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0, xl: [B, d]
        # (xl * w) 按维度相乘后求和 → [B, 1]
        xw = torch.sum(xl * self.weight, dim=1, keepdim=True)  # [B,1]
        return x0 * xw + self.bias + xl                        # [B,d]


class DCNTower(nn.Module):
    """
    通用 DCN 塔：可以用在 query 或 item 一侧。
    - num_cross: cross 网络层数
    - deep_hidden: deep 部分的 MLP 隐层
    """
    def __init__(
        self,
        in_dim: int,
        num_cross: int = 3,
        deep_hidden=(256, 128),
        out_dim: int = 64,
        l2norm: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.l2norm = l2norm

        # Cross 部分
        self.cross_layers = nn.ModuleList(
            [CrossLayer(in_dim) for _ in range(num_cross)]
        )

        # Deep 部分 = 一个普通 MLP
        if deep_hidden and len(deep_hidden) > 0:
            sizes = [in_dim] + list(deep_hidden)
            self.deep = mlp(sizes)
            deep_out_dim = deep_hidden[-1]
        else:
            # 不要 deep，直接用 x0
            self.deep = None
            deep_out_dim = in_dim

        # Cross 输出维度仍然是 in_dim
        cross_out_dim = in_dim

        # 把 cross 和 deep concat，再投到 out_dim
        self.out = nn.Linear(cross_out_dim + deep_out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        x0 = x
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)          # [B, in_dim]

        if self.deep is not None:
            xd = self.deep(x0)          # [B, deep_out_dim]
        else:
            xd = x0                     # [B, in_dim]

        h = torch.cat([xc, xd], dim=-1) # [B, in_dim + deep_out_dim]
        z = self.out(h)                 # [B, out_dim]
        if self.l2norm:
            z = F.normalize(z, p=2, dim=-1)
        return z