import numpy as np
import pandas as pd
import os

# === 路径配置 ===
root = "ml-100k"  # 你解压的位置
save_dir = "data/ml_small_prepared"
os.makedirs(save_dir, exist_ok=True)

# === 1. 读取交互 ===
data = pd.read_csv(os.path.join(root, "u.data"), sep="\t", header=None,
                   names=["user_id", "item_id", "rating", "timestamp"])

# 简单过滤掉评分低于 3 的交互（认为高分为正样本）
data = data[data["rating"] >= 3]
print(f"交互数量: {len(data)}")

# === 2. 构造用户特征 ===
user_ids = sorted(data["user_id"].unique())
user_id2row = {u: i for i, u in enumerate(user_ids)}

# 用 one-hot 用户ID 当特征（非常简单）
num_users = len(user_ids)
user_feats = np.eye(num_users, dtype=np.float32)

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
