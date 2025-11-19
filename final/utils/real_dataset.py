# real_dataset.py
# ==============================================
# 功能：
#   - 加载 query/item 特征、交互数据
#   - 自动按比例切分 train/valid/test
#   - 支持用户侧离散特征 (gender / age / occupation)
#     -> 从 q_side_cat.npy 加载
#   - 返回适配双塔模型的样本
# ==============================================

import csv
import numpy as np
import torch
from torch.utils.data import Dataset


def load_id_map_csv(path):
    """读取 id → row_idx 映射表"""
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
    """
    返回的样本（键）：
      query_feat: FloatTensor [Qdim]
      pos_item_feat: FloatTensor [Idim]
      pos_item_row: LongTensor []
      q_row: LongTensor []
      (可选) q_gender_idx / q_age_idx / q_occ_idx: LongTensor []
    """
    def __init__(
        self,
        q_feats_npy,
        q_idmap_csv,
        i_feats_npy,
        i_idmap_csv,
        inter_csv,
        split="train",
        seed=42,
        split_ratio=(0.8, 0.1, 0.1),
        q_side_cat_npy=None,
    ):
        super().__init__()

        # 先初始化，避免任何提前访问时报 AttributeError
        self.pairs = []

        # 1) 特征矩阵
        self.q_feats = np.load(q_feats_npy).astype("float32")
        self.i_feats = np.load(i_feats_npy).astype("float32")

        # 2) 映射
        self.q_id2row = load_id_map_csv(q_idmap_csv)
        self.i_id2row = load_id_map_csv(i_idmap_csv)

        # 3) 用户侧索引特征（可选）
        self.q_side_cat = None
        if q_side_cat_npy is not None and str(q_side_cat_npy):
            self.q_side_cat = np.load(q_side_cat_npy).astype("int64")
            if self.q_side_cat.shape[0] != self.q_feats.shape[0]:
                print(f"⚠️ q_side_cat.npy 行数 {self.q_side_cat.shape[0]} 与 q_feats {self.q_feats.shape[0]} 不一致")

        # 4) 读取交互构造 (q_row, i_row) 对
        pairs = []
        with open(inter_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # 兼容列名大小写或额外空格
            for r in reader:
                qid = r.get("query_id") or r.get("QueryId") or r.get("user_id") or r.get("UserId")
                iid = r.get("item_id")  or r.get("ItemId")  or r.get("itemID")
                if qid is None or iid is None:
                    continue
                if qid in self.q_id2row and iid in self.i_id2row:
                    pairs.append((self.q_id2row[qid], self.i_id2row[iid]))

        # 5) 切分
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"split 必须是 'train'|'valid'|'test'，当前为: {split}")

        rng = np.random.RandomState(seed)
        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * split_ratio[0])
        n_valid = int(n * split_ratio[1])

        if split == "train":
            self.pairs = pairs[:n_train]
        elif split == "valid":
            self.pairs = pairs[n_train:n_train + n_valid]
        else:
            self.pairs = pairs[n_train + n_valid:]

        print(f"✅ RealTwoTowerDataset[{split}] 样本数: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q_row, i_row = self.pairs[idx]
        q_feat = self.q_feats[q_row]
        i_feat = self.i_feats[i_row]

        sample = {
            "query_feat": torch.from_numpy(q_feat),
            "pos_item_feat": torch.from_numpy(i_feat),
            "pos_item_row": torch.tensor(i_row, dtype=torch.long),
            "q_row": torch.tensor(q_row, dtype=torch.long),
        }

        if self.q_side_cat is not None:
            g, a, o = self.q_side_cat[q_row].tolist()
            sample.update({
                "q_gender_idx": torch.tensor(g, dtype=torch.long),
                "q_age_idx": torch.tensor(a, dtype=torch.long),
                "q_occ_idx": torch.tensor(o, dtype=torch.long),
            })

        return sample
