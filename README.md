# SBCNM-PyTorch (Two-Tower + Sampling-Bias Correction)

基于 PyTorch 的**双塔召回**实现，复现 Yi et al., RecSys 2019 《Sampling-Bias-Corrected Neural Modeling for Large-Corpus Item Recommendations》中的**采样偏差校正**，支持：
- **双塔** （Query / Item） + L2 归一化（余弦相似度）
- **Batch softmax + 采样偏差校正**（在线频率估计）
- **意外负样本屏蔽**、**Hard Negative Mining**、**温度缩放**
- **Factorized Top-K** 检索（BruteForce / 可选 FAISS）
- **可选 GCN item tower**（有物品图时开启）
- 真实数据接口（以 MovieLens 100K 为例）

## 1) 安装

```bash
pip install -r requirements.txt
```

## 2) 数据准备（MovieLens 100K 示例）

下载 [MovieLens 100K 数据集](https://files.grouplens.org/datasets/movielens/ml-100k.zip)，解压到项目根目录的 `ml-100k/`，然后执行：

```bash
python examples/data_prep_movielens.py
```

会在 data/ml_small_prepared/ 生成：

```
q_feats.npy, i_feats.npy, q_idmap.csv, i_idmap.csv, interactions.csv
```

## 3) 训练（MLP 版）

``` bash
python -m scripts.train_real \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv data/ml_small_prepared/interactions.csv \
  --epochs 3 --batch_size 128 --save_path movielens_ckpt.pt --cpu
```

## 4) 训练（GCN 版，需提供 item_adj.npy）

``` bash
python -m scripts.train_real --use_gcn \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv data/ml_small_prepared/interactions.csv \
  --item_adj_npy data/ml_small_prepared/item_adj.npy \
  --epochs 3 --batch_size 128 --save_path movielens_gcn_ckpt.pt --cpu
```

## 5) 导出索引并做检索 Demo

``` bash
python -m scripts.export_index_real \
  --ckpt_path movielens_ckpt.pt \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv data/ml_small_prepared/interactions.csv \
  --use_faiss 0
```

## 6) 项目结构

``` kotlin
sbcnm_torch/
  models/
    towers.py
    gcn.py
    item_gcn_tower.py
  utils/
    real_dataset.py
    factorized_topk.py
    freq_estimator.py
    index.py
    metrics.py
scripts/
  train_real.py
  export_index_real.py
examples/
  data_prep_movielens.py
data/
  (放本地数据，已在 .gitignore 中忽略)
```

## 7) 论文

Yi et al. (RecSys 2019) Sampling-Bias-Corrected Neural Modeling for Large-Corpus Item Recommendations.