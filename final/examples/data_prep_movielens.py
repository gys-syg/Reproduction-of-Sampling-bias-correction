# data_prep_movielens.py
# ==============================================
# 生成 MovieLens100K 的训练数据：
#   q_idmap.csv, i_idmap.csv, interactions.csv,
#   q_feats.npy, i_feats.npy,
#   q_side_cat.npy（新增：性别/年龄桶/职业索引）
# ==============================================

import numpy as np
import pandas as pd
import os

# === 路径配置 ===
root = "ml-100k"  # MovieLens-100K 解压后的目录
save_dir = "data/ml_small_prepared"
os.makedirs(save_dir, exist_ok=True)

# === 1. 读取交互数据 ===
data = pd.read_csv(os.path.join(root, "u.data"), sep="\t", header=None,
                   names=["user_id", "item_id", "rating", "timestamp"])

# 过滤低分交互，认为 rating >= 3 是“正样本”
data = data[data["rating"] >= 3]
print(f"交互数量: {len(data)}")

# === 2. 构造用户特征 ===
user_ids = sorted(data["user_id"].unique())
user_id2row = {u: i for i, u in enumerate(user_ids)}
num_users = len(user_ids)
user_feats = np.eye(num_users, dtype=np.float32)  # one-hot 用户特征

# === 2.b 读取用户侧元数据（性别 / 年龄 / 职业）===
user_meta = pd.read_csv(os.path.join(root, "u.user"), sep="|", header=None,
                        names=["user_id", "age", "gender", "occupation", "zip"])
user_meta = user_meta[user_meta["user_id"].isin(user_ids)]
user_meta = user_meta.sort_values("user_id")

# 1) 性别映射
gender_map = {"M": 0, "F": 1}
gender_idx = user_meta["gender"].map(gender_map).fillna(0).astype(int).values
num_gender = len(gender_map)

# 2) 年龄分桶
def age_bucket(a):
    a = int(a)
    if a < 18: return 0
    if a < 25: return 1
    if a < 35: return 2
    if a < 45: return 3
    if a < 55: return 4
    if a < 65: return 5
    return 6
age_idx = user_meta["age"].apply(age_bucket).astype(int).values
num_age_buckets = 7

# 3) 职业映射
occ_list = sorted(user_meta["occupation"].unique().tolist())
occ2idx = {o: i for i, o in enumerate(occ_list)}
occ_idx = user_meta["occupation"].map(occ2idx).astype(int).values
num_occ = len(occ2idx)

# 确保顺序和 user_ids 对齐
user_meta = user_meta.set_index("user_id").loc[user_ids]
gender_idx = user_meta["gender"].map(gender_map).fillna(0).astype(int).values
age_idx = user_meta["age"].apply(age_bucket).astype(int).values
occ_idx = user_meta["occupation"].map(occ2idx).astype(int).values

q_side_cat = np.stack([gender_idx, age_idx, occ_idx], axis=1).astype("int64")
np.save(os.path.join(save_dir, "q_side_cat.npy"), q_side_cat)

print(f"✅ 用户侧索引特征已保存: q_side_cat.npy, 形状={q_side_cat.shape}")
print(f"   gender_vocab={num_gender}, age_buckets={num_age_buckets}, occ_vocab={num_occ}")

# === 3. 构造物品特征 ===
item_ids = sorted(data["item_id"].unique())
item_id2row = {i: j for j, i in enumerate(item_ids)}
num_items = len(item_ids)
item_feats = np.eye(num_items, dtype=np.float32)

# === 4. 生成 ID 映射文件 ===
pd.DataFrame({"id": user_ids, "row_idx": [user_id2row[u] for u in user_ids]}).to_csv(
    os.path.join(save_dir, "q_idmap.csv"), index=False)
pd.DataFrame({"id": item_ids, "row_idx": [item_id2row[i] for i in item_ids]}).to_csv(
    os.path.join(save_dir, "i_idmap.csv"), index=False)

# === 5. 生成交互 CSV ===
interactions = data[["user_id", "item_id"]].rename(
    columns={"user_id": "query_id", "item_id": "item_id"})
interactions.to_csv(os.path.join(save_dir, "interactions.csv"), index=False)

# === 6. 保存特征矩阵 ===
np.save(os.path.join(save_dir, "q_feats.npy"), user_feats)
np.save(os.path.join(save_dir, "i_feats.npy"), item_feats)

print("✅ 数据准备完成，文件保存在:", save_dir)

# === 7. 构造 item 邻接矩阵 (item_adj.npy) ===
print("开始构造 item 邻接矩阵 ...")

num_items = len(item_ids)
adj = np.zeros((num_items, num_items), dtype=np.float32)

# 1) 用“同一用户点击过的 item”构建共现矩阵
#    data 里已经过滤过 rating >= 3，并且只包含在 user_ids / item_ids 内的交互
#    先按 user 分组
user_group = data.groupby("user_id")["item_id"].apply(list)

for items in user_group:
    # 把 item_id 转成 row_idx
    idxs = [item_id2row[i] for i in items]
    # 去重一下，防止一个用户对同一 item 多次交互
    idxs = list(set(idxs))
    # 所有两两组合打上共现
    for i in idxs:
        for j in idxs:
            if i == j:
                continue
            adj[i, j] += 1.0

# 2) 二值化 / 对称化
#    共现次数 > 0 的地方记为 1
adj = (adj > 0).astype(np.float32)
#    确保对称
adj = np.maximum(adj, adj.T)

# 3) 加 self-loop：A_hat = A + I
adj = adj + np.eye(num_items, dtype=np.float32)

# 4) 做 D^{-1/2} A D^{-1/2} 归一化 (GCN 标准做法)
deg = adj.sum(axis=1)  # 度
deg[deg == 0] = 1.0    # 防止除零
deg_inv_sqrt = np.power(deg, -0.5)
adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

# 5) 保存
np.save(os.path.join(save_dir, "item_adj.npy"), adj_norm)
print(f"✅ item_adj.npy 已保存，形状={adj_norm.shape}")
