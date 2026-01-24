# FeaLearner-2.0：基于用户帖子序列的自杀风险等级预测

本仓库实现了一个**帖子级别**的风险等级预测模型：输入为用户的**帖子序列（BERT embedding）** + **手工特征**，通过“用户的历史后上下文建模 + 多视点自适应特征选择网络（MoE）”融合后做分类。

---

## 项目目录树

```text

├── data_analy/                                         # 特征存放目录
│ ├── feature_reddit_500.csv                            # Reddit 数据集提取的手工特征文件
│ ├── feature_sigir.csv                                 # sigir 数据集提取的手工特征文件
│ └── feature_bigdata.csv                               # bigdata提取的手工特征文件
├── data/                                               # 数据存放目录
│ ├── bert_embeddings.pkl                               # Reddit 数据集的 BERT 嵌入
│ ├── sigir_bert_embeddings.pkl                         # sigir 数据集的 BERT 嵌入
│ └── bigdata_bert_embeddings                           # bigdata 数据集的 BERT 嵌入
├── tools/                                              # 工具函数库
│ └── utils.py                                          # 包含评价指标计算 (gr_metrics) 等辅助工具
├── twomoe.py                                           # 核心模型组件：双层稀疏混合专家网络
├── reddit.py                                           # 主训练脚本：适配 Reddit、sigir、BigData 数据集
├── ablation_study.py                                   # 消融实验脚本：验证各模块有效性
├── run_experiments.sh                                  # 自动化脚本：用于超参数搜索与批量实验
├── ablation_results.csv                                # 实验产出：消融实验的结果记录表
└── bad_cases.csv                                       # 训练产出：模型预测错误的样本分析表
```

## 快速开始（复现/测试）

### reddit 数据集复现示例

```bash
python auto_select/reddit.py
```

### sigir 数据集复现示例

```bash
python reddit.py \
 --max_len 300 \
 --classnum 2 \
 --use_pretrain True \
 --data_embeddings "../data/sigir_bert_embeddings.pkl" \
 --data_features "../data_analy/feature_sigir.csv" 2>&1
```

实验结果：
Accuracy: 0.9489
test GP: 0.9766081871345029 GR: 0.9709302325581395 FS: 0.9737609329446064 OE: 0.0

### bigdata 数据集复现示例

```bash
python reddit.py \
 --cv_heads 8 \
 --cv_d_model 256 \
 --lr 1e-4 \
 --batch_size 4 \
 --max_len 5 \
--epochs 50 \
 --patience 10 \
 --classnum 4 \
 --use_pretrain True \
 --data_embeddings "../data/bigdata_bert_embeddings.pkl" \
 --data_features "../data_analy/feature_bigdata.csv" 2>&1
```

实验结果：
最佳结果：Accuracy: 0.5366  
 test GP: 0.7779960707269156 GR: 0.6336 FS: 0.6984126984126984 OE: 0.08130081300813008

### weibo数据集

weibo中文数据集：
特征维度：(7327, 130)

- POS: 57
- TF-IDF: 50
- NRC: 10
- SUI: 13
  train.py
  总体指标:
  Accuracy: 0.8499
  F1-Score: 0.8499
  Precision: 0.8499
  Recall: 0.8499

## 指标汇总

| 数据集  | 任务（classnum） | Accuracy |                 GP |                 GR |             FS(F1) |                  OE |
| ------- | ---------------: | -------: | -----------------: | -----------------: | -----------------: | ------------------: |
| reddit  |                5 |   0.6400 | 0.7804878048780488 | 0.7804878048780488 | 0.7804878048780488 |                 0.1 |
| bigdata |                4 |   0.5366 | 0.7779960707269156 |             0.6336 | 0.6984126984126984 | 0.08130081300813008 |
| sigir   |                2 |   0.9489 | 0.9766081871345029 | 0.9709302325581395 | 0.9737609329446064 |                 0.0 |

---

## 消融实验

运行脚本：`ablation_study.py`  
输出结果：`ablation_results.csv`

| 实验             |                     描述 | Accuracy |     F1 |     GP |     GR |
| ---------------- | -----------------------: | -------: | -----: | -----: | -----: |
| Exp-0_Full       |                 Baseline |     0.64 | 0.7805 | 0.7805 | 0.7805 |
| Exp-1_no_Temp    |        w/o Temporal Attn |     0.34 | 0.5075 | 0.5313 | 0.4857 |
| Exp-2_no_Cross   |       w/o Cross-Var Attn |     0.46 | 0.6301 | 0.6389 | 0.6216 |
| Exp-3_no_Dual    |             w/o All Attn |     0.46 | 0.6301 | 0.6765 | 0.5897 |
| Exp-4_Fixed      |                  α=β=1.0 |     0.48 | 0.6486 | 0.6857 | 0.6154 |
| Exp-5_MLP        |               MoE -> MLP |     0.50 | 0.6667 | 0.7576 | 0.5952 |
| Exp-6_1Layer_MoE |   2-Layer -> 1-Layer MoE |     0.42 | 0.5915 | 0.7000 | 0.5122 |
| Exp-7_UniLSTM    |          Bi- -> Uni-LSTM |     0.48 | 0.6486 | 0.7273 | 0.5854 |
| Exp-8_no_Feat    | w/o Handcrafted Features |     0.42 | 0.5915 | 0.6176 | 0.5676 |
| Exp-9_CE_Loss    |         Focal -> CE Loss |     0.46 | 0.6301 | 0.7931 | 0.5227 |
