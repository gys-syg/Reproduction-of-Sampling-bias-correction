# sbcnm_torch/models/query_embed_tower.py
# ==========================================
# 稳定版 QueryEmbedTower:
# - 支持 ID / 性别 / 年龄桶 / 职业 embedding
# - 加 LayerNorm + Dropout 提升泛化
# - 输出向量 L2 归一化
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation=nn.ReLU, dropout=0.2, use_layernorm=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if use_layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class QueryEmbedTower(nn.Module):
    """
    用户塔: [ID, gender, age, occupation] Embedding + (可选稠密特征)
    输出: L2-normalized embedding
    """
    def __init__(
        self,
        num_users: int,
        gender_vocab: int = 2,
        age_buckets: int = 7,
        occ_vocab: int = 21,
        id_dim: int = 32,
        g_dim: int = 4,
        a_dim: int = 8,
        o_dim: int = 8,
        dense_in: int = 0,
        mlp_hidden=(128,),
        out_dim: int = 64,
        l2norm=True,
        dropout=0.2,
    ):
        super().__init__()

        # ---- Embedding 层 ----
        self.user_emb = nn.Embedding(num_users, id_dim)
        self.gender_emb = nn.Embedding(gender_vocab, g_dim)
        self.age_emb = nn.Embedding(age_buckets, a_dim)
        self.occ_emb = nn.Embedding(occ_vocab, o_dim)

        # ---- 拼接后 MLP ----
        in_dim = id_dim + g_dim + a_dim + o_dim + dense_in
        self.mlp = mlp([in_dim] + list(mlp_hidden) + [out_dim],
                       activation=nn.ReLU, dropout=dropout)
        self.l2norm = l2norm

        # 初始化更稳定
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, user_row, gender_idx, age_idx, occ_idx, query_dense=None):
        e_id = self.user_emb(user_row)
        e_g = self.gender_emb(gender_idx)
        e_a = self.age_emb(age_idx)
        e_o = self.occ_emb(occ_idx)

        feats = [e_id, e_g, e_a, e_o]
        if query_dense is not None:
            feats.append(query_dense)
        x = torch.cat(feats, dim=-1)
        z = self.mlp(x)

        if self.l2norm:
            z = F.normalize(z, p=2, dim=-1)
        return z
