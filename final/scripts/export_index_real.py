
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sbcnm_torch.models.towers import QueryTower, ItemTower
from sbcnm_torch.models.item_gcn_tower import ItemGCNTower
from sbcnm_torch.utils.real_dataset import RealTwoTowerDataset
from sbcnm_torch.utils.index import build_item_index
from sbcnm_torch.utils.factorized_topk import BruteForceRetriever, FaissRetriever
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    state = torch.load(args.ckpt_path, map_location=device)
    cfg = state["config"]; use_gcn = state.get("item_gcn", False)
    ds = RealTwoTowerDataset(args.q_feats_npy, args.q_idmap_csv, args.i_feats_npy, args.i_idmap_csv, args.inter_csv, split="train")
    i_feats_full = torch.tensor(ds.i_feats, dtype=torch.float32, device=device)
    if use_gcn or args.force_gcn:
        if args.item_adj_npy and len(args.item_adj_npy)>0:
            adj = torch.tensor(np.load(args.item_adj_npy).astype("float32"), device=device)
        else:
            Ni = ds.i_feats.shape[0]; adj = torch.eye(Ni, dtype=torch.float32, device=device)
        item_tower = ItemGCNTower(in_dim=ds.i_feats.shape[1], hidden=(args.gcn_h,), out_dim=cfg["out_dim"], l2norm=True).to(device)
        _ = item_tower.encode_all(i_feats_full, adj)
        item_embs = item_tower._all_embs.detach().cpu()
    else:
        item_tower = ItemTower(in_dim=ds.i_feats.shape[1], hidden=(cfg["h1"], cfg["h2"]), out_dim=cfg["out_dim"], l2norm=True).to(device)
        item_embs = build_item_index(item_tower, i_feats_full, batch_size=1024, device=device)
    item_ids = torch.arange(item_embs.shape[0], dtype=torch.long)
    retriever = FaissRetriever(item_embs, item_ids) if args.use_faiss else BruteForceRetriever(item_embs, item_ids)
    print("Built", "FAISS" if args.use_faiss else "BruteForce", "index.")
    q = item_embs[:5, :].clone()
    sims, ids = retriever.query(q, k=10)
    print("Top-10 ids for first query:\n", ids[0].tolist())
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--q_feats_npy", type=str, required=True)
    p.add_argument("--q_idmap_csv", type=str, required=True)
    p.add_argument("--i_feats_npy", type=str, required=True)
    p.add_argument("--i_idmap_csv", type=str, required=True)
    p.add_argument("--inter_csv", type=str, required=True)
    p.add_argument("--item_adj_npy", type=str, default="")
    p.add_argument("--use_faiss", type=int, default=0)
    p.add_argument("--force_gcn", action="store_true")
    p.add_argument("--gcn_h", type=int, default=256)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(); main(args)
