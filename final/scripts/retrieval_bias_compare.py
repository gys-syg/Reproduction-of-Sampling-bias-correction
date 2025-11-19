import sys
from pathlib import Path

p = Path.cwd()
# 向上查找直到找到 sbcnm_torch 目录或达到文件系统根
while not (p / "sbcnm_torch").exists():
    if p.parent == p:
        raise RuntimeError("未在父目录中找到 sbcnm_torch，无法自动设置 sys.path，请手动指定项目根路径")
    p = p.parent
# 将项目根插入到 sys.path 顶部（优先级高）
sys.path.insert(0, str(p))
print("Added to sys.path:", str(p))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import math

from sbcnm_torch.utils.real_dataset import RealTwoTowerDataset
from sbcnm_torch.models.towers import QueryTower, ItemTower, DCNTower
from sbcnm_torch.models.query_embed_tower import QueryEmbedTower
from sbcnm_torch.models.item_gcn_tower import ItemGCNTower


@torch.no_grad()
def build_item_index(item_tower, i_feats_full, batch_size=1024, device="cpu"):
    """MLP/任意塔版本 item 塔：逐 batch 前向，拼成全量 item embedding。"""
    item_tower.eval()
    embs = []
    for i in range(0, i_feats_full.shape[0], batch_size):
        e = item_tower(i_feats_full[i:i+batch_size].to(device))
        embs.append(e.cpu())
    embs = torch.cat(embs, dim=0)                     # [Ni, d]
    embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
    return embs                                       


def recall_at_k(pred_ids, true_ids, k):
    # pred_ids: [B, K]  true_ids: [B]
    hits = (pred_ids[:, :k] == true_ids.view(-1, 1)).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(pred_ids, true_ids, k):
    """二分类 NDCG：命中在位置 r，则 DCG = 1/log2(r+2)，IDCG=1。"""
    B = pred_ids.shape[0]
    pos = (pred_ids[:, :k] == true_ids.view(-1, 1)).float()
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2).float())
    dcg = (pos * discounts.to(pos.device)).max(dim=1).values  # 一行最多一个1
    return dcg.mean().item()


def auc_score(pos_score: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    pos_score: 标量 tensor，正样本分数
    neg_scores: [N_neg]，负样本分数
    返回单个用户的 AUC = P( pos > neg )
    """
    return (pos_score > neg_scores).float().mean().item()


def load_model_from_ckpt(
    ckpt_path,
    ds,
    device,
    use_user_embed=False,
    user_embed_cfg=None,
    use_dcn_item=False,      # item 侧是否 DCN
):
    """
    从 ckpt 里恢复 query_tower 和 item_tower。

    - use_user_embed=True  → 用 QueryEmbedTower
    - use_user_embed=False → 用 QueryTower 或 DCNTower（自动根据 ckpt 判断）
    - use_dcn_item=True    → item 侧用 DCNTower（与你的 dcn_ckpt 对应）
    """
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["config"]

    # === item tower ===
    if use_dcn_item:
        # DCN 参数尽量和训练时保持一致
        item_tower = DCNTower(
            in_dim=ds.i_feats.shape[1],
            num_cross=cfg.get("cross_layers", 3),
            deep_hidden=(cfg.get("h1", 256), cfg.get("h2", 128)),
            out_dim=cfg.get("out_dim", 64),
            l2norm=True,
        ).to(device)
    else:
        item_tower = ItemTower(
            in_dim=ds.i_feats.shape[1],
            hidden=(cfg.get("h1", 256), cfg.get("h2", 128)),
            out_dim=cfg.get("out_dim", 64),
            l2norm=True
        ).to(device)

    item_tower.load_state_dict(state["item_state"])
    item_tower.eval()

    # === query tower ===
    q_state = state["query_state"]

    if use_user_embed:
        # 这一支是你 user-embed 那条线（QueryEmbedTower）
        assert user_embed_cfg is not None, "use_user_embed=True 时必须提供 user_embed_cfg"
        qe = QueryEmbedTower(
            num_users=ds.q_feats.shape[0],
            gender_vocab=user_embed_cfg["gender_vocab"],
            age_buckets=user_embed_cfg["age_buckets"],
            occ_vocab=user_embed_cfg["occ_vocab"],
            id_dim=user_embed_cfg["id_dim"],
            g_dim=user_embed_cfg["g_dim"],
            a_dim=user_embed_cfg["a_dim"],
            o_dim=user_embed_cfg["o_dim"],
            dense_in=ds.q_feats.shape[1] if user_embed_cfg["use_query_dense"] else 0,
            mlp_hidden=(cfg.get("h1", 256),),
            out_dim=cfg.get("out_dim", 64),
            l2norm=True
        ).to(device)
        qe.load_state_dict(q_state)
        qe.eval()
        query_tower = qe
    else:
        # 这一支是「不带 user-embed」的版本，有两种可能：
        # - 旧的 MLP QueryTower（state 里的 key 叫 net.*）
        # - DCN 版 Query（state 里的 key 叫 cross_layers./deep./out.）
        q_keys = list(q_state.keys())
        has_dcn = any(
            k.startswith("cross_layers") or k.startswith("deep.") or k.startswith("out.")
            for k in q_keys
        )

        if has_dcn:
            # 👉 说明 ckpt 里的 query_tower 其实是 DCNTower
            qt = DCNTower(
                in_dim=ds.q_feats.shape[1],
                num_cross=cfg.get("cross_layers", 3),
                deep_hidden=(cfg.get("h1", 256), cfg.get("h2", 128)),
                out_dim=cfg.get("out_dim", 64),
                l2norm=True,
            ).to(device)
        else:
            # 👉 普通的 MLP QueryTower
            qt = QueryTower(
                in_dim=ds.q_feats.shape[1],
                hidden=(cfg.get("h1", 256), cfg.get("h2", 128)),
                out_dim=cfg.get("out_dim", 64),
                l2norm=True
            ).to(device)

        qt.load_state_dict(q_state)
        qt.eval()
        query_tower = qt

    return query_tower, item_tower, cfg


@torch.no_grad()
def evaluate_sampled(
    ckpt_path,
    args,
    split="valid",
    use_user_embed=False,
    user_embed_cfg=None,
    use_dcn_item=False,
    num_neg=99,
):
    """
    论文/比赛常用评估方式：
    对每个 user，在 {1 个正样本 + num_neg 个随机负样本} 的候选集合上，
    计算 Recall@K / NDCG@K / AUC。

    注意：这里不再用“全库 topK”，而是“采样 topK”。
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ds = RealTwoTowerDataset(
        args.q_feats_npy,
        args.q_idmap_csv,
        args.i_feats_npy,
        args.i_idmap_csv,
        args.inter_csv,
        split=split,
        q_side_cat_npy=(args.q_side_cat_npy if use_user_embed else None),
    )
    print(f"✅ RealTwoTowerDataset[{split}] 样本数: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 加载模型
    query_tower, item_tower, cfg = load_model_from_ckpt(
        ckpt_path,
        ds,
        device,
        use_user_embed=use_user_embed,
        user_embed_cfg=user_embed_cfg,
        use_dcn_item=use_dcn_item,
    )

    # 预计算全量 item embedding
    i_feats_full = torch.tensor(ds.i_feats, dtype=torch.float32, device=device)
    item_embs = build_item_index(item_tower, i_feats_full, batch_size=2048, device=device)
    Ni = item_embs.shape[0]

    Ks = [5, 10, 50]
    sums_recall = {k: 0.0 for k in Ks}
    sums_ndcg = {k: 0.0 for k in Ks}
    sum_auc = 0.0
    n_user = 0

    for batch in loader:
        # 计算 query embedding
        if use_user_embed:
            q_emb = query_tower(
                user_row=batch["q_row"].to(device),
                gender_idx=batch["q_gender_idx"].to(device),
                age_idx=batch["q_age_idx"].to(device),
                occ_idx=batch["q_occ_idx"].to(device),
                query_dense=(
                    batch["query_feat"].to(device)
                    if user_embed_cfg["use_query_dense"]
                    else None
                ),
            )  # [B,d]
        else:
            q_emb = query_tower(batch["query_feat"].to(device))  # [B,d]

        true_ids = batch["pos_item_row"].to(device).long()       # [B]
        B = q_emb.shape[0]

        for b in range(B):
            pos = true_ids[b].item()

            # === 采样 num_neg 个负样本（不包含 pos） ===
            # trick：先在 [0, Ni-1] 采样，再把 >= pos 的索引整体 +1，跳过正样本
            idx = torch.randint(0, Ni - 1, (num_neg,), device=device)
            neg_ids = idx + (idx >= pos).long()   # [num_neg]
            cand_ids = torch.cat(
                [torch.tensor([pos], device=device, dtype=torch.long), neg_ids],
                dim=0
            )  # [1 + num_neg]

            # 计算这 1+num_neg 个候选的分数
            scores = (q_emb[b:b+1] @ item_embs[cand_ids].to(device).T).squeeze(0)  # [C]
            pos_score = scores[0]
            neg_scores = scores[1:]

            # 计算 rank（在候选集合内部的名次，0 为 best）
            _, sorted_idx = scores.sort(descending=True)
            rank = (sorted_idx == 0).nonzero(as_tuple=False).item()  # 0-based

            # Recall & NDCG（在采样集合上的版本）
            for k in Ks:
                if rank < k:
                    sums_recall[k] += 1.0
                    sums_ndcg[k] += 1.0 / math.log2(rank + 2)  # rank 从0开始，所以 +2
            # AUC（pos vs neg_scores）
            sum_auc += auc_score(pos_score, neg_scores)

            n_user += 1

    metrics = {f"Recall@{k}": sums_recall[k] / n_user for k in Ks}
    metrics.update({f"NDCG@{k}": sums_ndcg[k] / n_user for k in Ks})
    metrics["AUC"] = sum_auc / n_user
    return metrics


def main():
    p = argparse.ArgumentParser()
    # 数据路径
    p.add_argument("--q_feats_npy", type=str, required=True)
    p.add_argument("--q_idmap_csv", type=str, required=True)
    p.add_argument("--i_feats_npy", type=str, required=True)
    p.add_argument("--i_idmap_csv", type=str, required=True)
    p.add_argument("--inter_csv", type=str, required=True)
    p.add_argument("--q_side_cat_npy", type=str, default="")  # 只有 user-embed 时用

    # ckpt 路径
    p.add_argument("--dcn_bias_ckpt", type=str, required=True)  # DCN 纠偏版本
    p.add_argument("--dcn_nobias_ckpt", type=str, required=True)  # DCN 不纠偏版本

    # 其它
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    # user-embed 配置（与训练时保持一致）
    user_embed_cfg = dict(
        gender_vocab=2,
        age_buckets=7,
        occ_vocab=21,
        id_dim=32,
        g_dim=4,
        a_dim=8,
        o_dim=8,
        use_query_dense=True,
    )

    # ========= 1. DCN (纠偏版本) =========
    print("== Evaluate DCN (Bias Correction) Version ==")
    m_dcn_bias = evaluate_sampled(
        args.dcn_bias_ckpt,
        args,
        split="valid",
        use_user_embed=False,  # 根据需要设定
        use_dcn_item=True,     # 使用 DCN item 塔
        num_neg=99,
    )
    for k, v in m_dcn_bias.items():
        print(f"{k}: {v:.4f}")

    # ========= 2. DCN (不纠偏版本) =========
    print("\n== Evaluate DCN (No Bias Correction) Version ==")
    m_dcn_nobias = evaluate_sampled(
        args.dcn_nobias_ckpt,
        args,
        split="valid",
        use_user_embed=False,  # 根据需要设定
        use_dcn_item=True,     # 使用 DCN item 塔
        num_neg=99,
    )
    for k, v in m_dcn_nobias.items():
        print(f"{k}: {v:.4f}")

    # ========= 差值对比 =========
    print("\n== Δ (DCN with Bias Correction - DCN without Bias Correction) ==")
    for k in m_dcn_bias.keys():
        print(f"{k}: {m_dcn_bias[k] - m_dcn_nobias[k]:+.4f}")


if __name__ == "__main__":
    main()
