# 双塔训练基础版，只用了用户和物品id，one-hot特征

# 把项目根目录自动加入 sys.path（更健壮的实现）
import sys
from pathlib import Path
import numpy as np


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
import torch
from torch.utils.data import DataLoader
from sbcnm_torch.models.towers import QueryTower, ItemTower, DCNTower
from sbcnm_torch.models.item_gcn_tower import ItemGCNTower
from sbcnm_torch.utils.real_dataset import RealTwoTowerDataset
from sbcnm_torch.utils.freq_estimator import FrequencyEstimator


def remove_accidental_negatives(scores, pos_rows, batch_pos_rows, very_neg=-1e9):
    B = scores.shape[0]
    cand_ids = batch_pos_rows.view(1, B).expand(B, B)
    row_pos_ids = pos_rows.view(B, 1).expand(B, B)
    mask = (cand_ids == row_pos_ids)
    eye = torch.eye(B, dtype=torch.bool, device=scores.device)
    mask = mask & (~eye)
    scores = scores.masked_fill(mask, very_neg)
    return scores

def hard_negative_mining(scores, labels, m: int):
    if m is None or m <= 0: return scores, labels
    augmented = scores + labels * 1e9
    vals, idx = torch.topk(augmented, k=min(m+1, scores.shape[1]), dim=1)
    gathered_labels = torch.gather(labels, 1, idx)
    return vals, gathered_labels

def sampling_bias_correction(scores, candidate_probs):
    pj = torch.clamp(candidate_probs, min=1e-9)
    corr = torch.log(pj)
    return scores - corr.view(1, -1)

def compute_loss(q_emb, i_emb, pos_rows, freq_estimator, global_step, temperature=0.05, hard_neg_m=20, use_bias_correction=True):
    batch_ids = pos_rows.detach().cpu().numpy().tolist()
    freq_estimator.update_batch(batch_ids, t=global_step)
    pj = torch.from_numpy(freq_estimator.get_probs(batch_ids)).to(q_emb.device, dtype=q_emb.dtype)
    scores = (q_emb @ i_emb.T)
    if use_bias_correction:                      # ⭐ 有开关才做纠偏
        scores = sampling_bias_correction(scores, pj)
    labels = torch.eye(scores.shape[0], device=scores.device)
    scores = remove_accidental_negatives(scores, pos_rows, pos_rows)
    if hard_neg_m is not None and hard_neg_m > 0:
        scores, labels = hard_negative_mining(scores, labels, hard_neg_m)
    if temperature is not None:
        scores = scores / float(temperature)
    log_probs = torch.log_softmax(scores, dim=-1)
    loss = -(labels * log_probs).sum(dim=1).mean()
    with torch.no_grad():
        topk = torch.topk(scores, k=min(10, scores.shape[1]), dim=-1).indices
        r_at_10 = labels.gather(1, topk).sum(dim=1).float().mean().item()
    return loss, {"recall@10(in-batch)": r_at_10}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds_train = RealTwoTowerDataset(args.q_feats_npy, args.q_idmap_csv, args.i_feats_npy, args.i_idmap_csv, args.inter_csv, split="train")
    ds_valid = RealTwoTowerDataset(args.q_feats_npy, args.q_idmap_csv, args.i_feats_npy, args.i_idmap_csv, args.inter_csv, split="valid")
    i_feats_full = torch.tensor(ds_train.i_feats, dtype=torch.float32, device=device)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, drop_last=True)
    query_tower = QueryTower(in_dim=ds_train.q_feats.shape[1], hidden=(args.h1, args.h2), out_dim=args.out_dim, l2norm=True).to(device)
    
    '''if args.use_gcn:
        if args.item_adj_npy and len(args.item_adj_npy)>0:
            adj = torch.tensor(np.load(args.item_adj_npy).astype("float32"), device=device)
        else:
            Ni = ds_train.i_feats.shape[0]
            adj = torch.eye(Ni, dtype=torch.float32, device=device)
        item_tower = ItemGCNTower(in_dim=ds_train.i_feats.shape[1], hidden=(args.gcn_h,), out_dim=args.out_dim, l2norm=True).to(device)
    else:
        item_tower = ItemTower(in_dim=ds_train.i_feats.shape[1], hidden=(args.h1, args.h2), out_dim=args.out_dim, l2norm=True).to(device)
        adj = None'''
    
        # === Query 塔：可选 MLP 或 DCN ===
    if args.use_dcn_query:
        query_tower = DCNTower(
            in_dim=ds_train.q_feats.shape[1],
            num_cross=args.cross_layers,
            deep_hidden=(args.h1, args.h2),
            out_dim=args.out_dim,
            l2norm=True
        ).to(device)
    else:
        query_tower = QueryTower(
            in_dim=ds_train.q_feats.shape[1],
            hidden=(args.h1, args.h2),
            out_dim=args.out_dim,
            l2norm=True
        ).to(device)

    # === Item 塔：可以是 GCN，也可以是 DCN 或 MLP ===
    if args.use_gcn:
        # 用 GCN 就不用 DCN 了
        if args.item_adj_npy and len(args.item_adj_npy) > 0:
            adj = torch.tensor(np.load(args.item_adj_npy).astype("float32"), device=device)
        else:
            Ni = ds_train.i_feats.shape[0]
            adj = torch.eye(Ni, dtype=torch.float32, device=device)
        item_tower = ItemGCNTower(
            in_dim=ds_train.i_feats.shape[1],
            hidden=(args.gcn_h,),
            out_dim=args.out_dim,
            l2norm=True
        ).to(device)
    else:
        if args.use_dcn_item:
            item_tower = DCNTower(
                in_dim=ds_train.i_feats.shape[1],
                num_cross=args.cross_layers,
                deep_hidden=(args.h1, args.h2),
                out_dim=args.out_dim,
                l2norm=True
            ).to(device)
        else:
            item_tower = ItemTower(
                in_dim=ds_train.i_feats.shape[1],
                hidden=(args.h1, args.h2),
                out_dim=args.out_dim,
                l2norm=True
            ).to(device)
        adj = None



    opt = torch.optim.Adagrad(list(query_tower.parameters()) + list(item_tower.parameters()), lr=args.lr)
    freq_est = FrequencyEstimator(H=args.freq_H, alpha=args.freq_alpha, init_gap=100.0)
    global_step = 0
    for epoch in range(args.epochs):
        query_tower.train(); item_tower.train()
        if args.use_gcn:
            _ = item_tower.encode_all(i_feats_full, adj)
        for batch in train_loader:
            global_step += 1
            q = batch["query_feat"].to(device)
            pos_row = batch["pos_item_row"].to(device)
            q_emb = query_tower(q)
            if args.use_gcn:
                i_emb = item_tower.forward_indices(pos_row)
            else:
                pos_item_feat = batch["pos_item_feat"].to(device)
                i_emb = item_tower(pos_item_feat)
            loss, metrics = compute_loss(q_emb, i_emb, pos_row, freq_estimator=freq_est,
                                         global_step=global_step, temperature=args.temperature,
                                         hard_neg_m=args.hard_negatives, use_bias_correction=(not args.no_bias_correction),)
            opt.zero_grad(); loss.backward(); opt.step()
        query_tower.eval(); item_tower.eval()
        
        with torch.no_grad():
            if args.use_gcn:
                _ = item_tower.encode_all(i_feats_full, adj)
            recs = []
            for batch in valid_loader:
                q = batch["query_feat"].to(device)
                pos_row = batch["pos_item_row"].to(device)
                q_emb = query_tower(q)
                i_emb = item_tower.forward_indices(pos_row) if args.use_gcn else item_tower(batch["pos_item_feat"].to(device))
                scores = (q_emb @ i_emb.T) / float(args.temperature)
                labels = torch.eye(scores.shape[0], device=device)
                topk = torch.topk(scores, k=min(10, scores.shape[1]), dim=-1).indices
                r_at_10 = labels.gather(1, topk).sum(dim=1).float().mean().item()
                recs.append(r_at_10)
            print(f"[epoch {epoch+1}] valid recall@10(in-batch) = {np.mean(recs):.4f}")
    save = {"query_state": query_tower.state_dict(),
            "item_state": item_tower.state_dict(),
            "config": vars(args),
            "item_gcn": bool(args.use_gcn),
            "i_feats_shape": tuple(ds_train.i_feats.shape)}
    torch.save(save, args.save_path); print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--q_feats_npy", type=str, required=True)
    p.add_argument("--q_idmap_csv", type=str, required=True)
    p.add_argument("--i_feats_npy", type=str, required=True)
    p.add_argument("--i_idmap_csv", type=str, required=True)
    p.add_argument("--inter_csv", type=str, required=True)
    p.add_argument("--item_adj_npy", type=str, default="")
    p.add_argument("--use_gcn", action="store_true")
    p.add_argument("--gcn_h", type=int, default=256)
    p.add_argument("--use_dcn_query", action="store_true",
                   help="如果设置，则 query 侧用 DCN 替代 MLP")
    p.add_argument("--use_dcn_item", action="store_true",
                   help="如果设置，则 item 侧用 DCN 替代 MLP（仅在未使用 GCN 时生效）")
    p.add_argument("--cross_layers", type=int, default=3,
                   help="DCN cross 网络的层数")
    p.add_argument("--h1", type=int, default=256)
    p.add_argument("--h2", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--hard_negatives", type=int, default=20)
    p.add_argument("--freq_H", type=int, default=1_000_003)
    p.add_argument("--freq_alpha", type=float, default=0.01)
    p.add_argument("--save_path", type=str, default="./data/sbcnm_real_ckpt.pt")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
            "--no_bias_correction",
            action="store_true",
            help="如果设置，则不减 log p_j，不做采样偏差纠偏（训练 naive 对照组）",
        )
    args = p.parse_args(); train(args)
