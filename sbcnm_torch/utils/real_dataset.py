
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
def load_id_map_csv(path):
    id2row = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            _id = r.get("id")
            if _id is None:
                keys = list(r.keys())
                _id = r[keys[0]]
                row_idx = int(r[keys[1]])
            else:
                row_idx = int(r.get("row_idx"))
            id2row[_id] = row_idx
    return id2row
class RealTwoTowerDataset(Dataset):
    def __init__(self, q_feats_npy, q_idmap_csv, i_feats_npy, i_idmap_csv, inter_csv, split="train", seed=42, split_ratio=(0.8,0.1,0.1)):
        super().__init__()
        self.q_feats = np.load(q_feats_npy).astype("float32")
        self.i_feats = np.load(i_feats_npy).astype("float32")
        self.q_id2row = load_id_map_csv(q_idmap_csv)
        self.i_id2row = load_id_map_csv(i_idmap_csv)
        pairs = []
        with open(inter_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                qid = r["query_id"]; iid = r["item_id"]
                if qid in self.q_id2row and iid in self.i_id2row:
                    pairs.append((self.q_id2row[qid], self.i_id2row[iid]))
        rng = np.random.RandomState(seed); rng.shuffle(pairs)
        n = len(pairs); n_train = int(n*split_ratio[0]); n_valid = int(n*split_ratio[1])
        if split == "train":   self.pairs = pairs[:n_train]
        elif split == "valid": self.pairs = pairs[n_train:n_train+n_valid]
        else:                  self.pairs = pairs[n_train+n_valid:]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        q_row, i_row = self.pairs[idx]
        q_feat = self.q_feats[q_row]; i_feat = self.i_feats[i_row]
        return {"query_feat": torch.from_numpy(q_feat),
                "pos_item_feat": torch.from_numpy(i_feat),
                "pos_item_row": torch.tensor(i_row, dtype=torch.long)}
