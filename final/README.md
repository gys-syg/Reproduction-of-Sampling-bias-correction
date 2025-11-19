# Sampling-Bias-Corrected Neural Modeling for Large-Corpus Item Recommendations 论文复现 - pytorch


基于 PyTorch 的**双塔召回**实现，复现 Yi et al., RecSys 2019 《Sampling-Bias-Corrected Neural Modeling for Large-Corpus Item Recommendations》中的**采样偏差校正**与**概率估计**，支持：
- **双塔** （Query / Item） + L2 归一化（余弦相似度）
- **Batch softmax + 采样偏差校正**（在线频率估计）
- **意外负样本屏蔽**、**Hard Negative Mining**、**温度缩放**
- **Factorized Top-K** 检索（BruteForce / 可选 FAISS）
- **可选 GCN item tower**（有物品图时开启）
- 真实数据接口（以 MovieLens 100K 为例）

## 1 安装所需包

```bash
pip install -r requirements.txt
```

## 2 数据准备（MovieLens 100K 示例）

下载 [MovieLens 100K 数据集](https://files.grouplens.org/datasets/movielens/ml-100k.zip)，解压到项目根目录的 `ml-100k/`，然后执行：

```bash
python examples/data_prep_movielens.py
```

会在 data/ml_small_prepared/ 生成：

```
q_feats.npy, i_feats.npy, q_idmap.csv, i_idmap.csv, interactions.csv
```

## 3 训练（MLP 版）

``` bash
python -m scripts.train_real \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv data/ml_small_prepared/interactions.csv \
  --epochs 3 --batch_size 128 --save_path movielens_ckpt.pt --cpu
```

## 4 训练（GCN 版，需提供 item_adj.npy，改进中……）

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

## 5 导出索引并做检索 Demo

``` bash
python -m scripts.export_index_real \
  --ckpt_path movielens_gcn_ckpt.pt \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv data/ml_small_prepared/interactions.csv \
  --use_faiss 0
```

## item塔用GCN

``` bash
python -m scripts.retrieval_three \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv  data/ml_small_prepared/interactions.csv \
  --q_side_cat_npy data/ml_small_prepared/q_side_cat.npy \
  --baseline_ckpt  data/baseline_ckpt.pt \
  --userembed_ckpt data/userembed_ckpt_epoch10.pt \
  --gcn_ckpt       movielens_gcn_ckpt.pt \
  --item_adj_npy   data/ml_small_prepared/item_adj.npy \
  --batch_size 512
```

## 尝试两个塔都用DCN

``` bash
python -m scripts.train_real \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv  data/ml_small_prepared/interactions.csv \
  --use_dcn_query \
  --use_dcn_item \
  --cross_layers 3 \
  --h1 256 --h2 128 \
  --out_dim 64 \
  --batch_size 256 \
  --epochs 5   --save_path ./data/dcn_epoch{}.pt
```

## 比较baseline，userembed和dcn

``` bash
python -m scripts.retrieval_three_dcn \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv  data/ml_small_prepared/interactions.csv \
  --q_side_cat_npy data/ml_small_prepared/q_side_cat.npy \
  --baseline_ckpt  ./data/baseline_ckpt.pt \
  --userembed_ckpt ./data/userembed_ckpt_epoch10.pt \
  --dcn_ckpt       ./data/dcn_epoch{}.pt \
  --batch_size 512

```

## 训练纠偏

``` bash
python -m scripts.train_real \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv  data/ml_small_prepared/interactions.csv \
  --use_dcn_query \
  --use_dcn_item \
  --cross_layers 3 \
  --h1 256 --h2 128 \
  --out_dim 64 \
  --epochs 5 \
  --save_path ./data/dcn_bias.pt

```

## 训练不纠偏

``` bash
python -m scripts.train_real \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv  data/ml_small_prepared/interactions.csv \
  --use_dcn_query \
  --use_dcn_item \
  --cross_layers 3 \
  --h1 256 --h2 128 \
  --out_dim 64 \
  --epochs 5 \
  --save_path ./data/dcn_nobias.pt \
  --no_bias_correction 

```

## 比较是否纠偏

``` bash
python -m scripts.retrieval_bias_compare \
  --dcn_bias_ckpt ./data/dcn_bias.pt \
  --dcn_nobias_ckpt ./data/dcn_nobias.pt \
  --q_feats_npy data/ml_small_prepared/q_feats.npy \
  --q_idmap_csv data/ml_small_prepared/q_idmap.csv \
  --i_feats_npy data/ml_small_prepared/i_feats.npy \
  --i_idmap_csv data/ml_small_prepared/i_idmap.csv \
  --inter_csv data/ml_small_prepared/interactions.csv \
  --batch_size 512


```


## 6 项目结构

``` bash
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

## 7 论文

Yi et al. (RecSys 2019) Sampling-Bias-Corrected Neural Modeling for Large-Corpus Item Recommendations.