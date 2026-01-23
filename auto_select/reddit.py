import argparse
import os
from pickletools import optimize
import random
import string
import time
from math import log
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from torch import Tensor, nn
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import pickle as pkl
import json
from tools import utils
from ordered_set import OrderedSet
from sklearn.model_selection import KFold
import twomoe as BS


# 将一个批次中的多个数据项按特定规则组合并填充（pad），以便它们可以被批量处理。
def pad_collate_reddit(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    lens = [len(x) for x in tweet]
    feature = [item[2] for item in batch]
    tweet = nn.utils.rnn.pad_sequence(tweet, batch_first=True, padding_value=0)
    target = torch.tensor(target)
    lens = torch.tensor(lens)
    feature = torch.stack(feature)
    return [target, tweet, lens, feature]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_size", type=int, default=768)
    parser.add_argument("--max_len", default=200, type=int)
    parser.add_argument("--hidden_size", type=int, default=128)
    # Cross-Variable Self-Attention 超参（变量路）
    parser.add_argument("--cv_d_model", type=int, default=128)
    parser.add_argument("--cv_heads", type=int, default=4)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=24, type=int)
    parser.add_argument("--classnum", default=5, type=int)
    parser.add_argument("--use_pretrain", default=False, type=bool)
    parser.add_argument("--patience", default=10, type=int, help="Early stopping patience")
    # 数据集路径参数
    parser.add_argument("--data_embeddings", type=str, default="../data/bert_embeddings.pkl",
                        help="BERT embeddings 数据文件路径（pkl格式）")
    parser.add_argument("--data_features", type=str, default="../data_analy/feature_reddit_500.csv",
                        help="特征数据文件路径（csv格式）")
    return parser.parse_args(args)


class RedditDataset(Dataset):
    def __init__(self, labels, tweets, days=200):
        super().__init__()
        self.labels = labels
        self.tweets = tweets  # 预训练的嵌入向量.
        # days代表是POST的数量还是其他东西？？？,用户发表的帖子的嵌入
        self.days = days

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        labels = torch.tensor(self.labels['labels'].iloc[item], dtype=torch.long)
        feature = torch.tensor(self.labels.iloc[item, :-1].values, dtype=torch.float32)
        if self.days >= len(self.tweets[item]):
            tweets = torch.tensor(self.tweets[item], dtype=torch.float32)
        else:
            tweets = torch.tensor(self.tweets[item][:self.days], dtype=torch.float32)
            print('进行了截取')
        return [labels, tweets, feature]


class SelfAttentionLayer(nn.Module):
    """
    Multi-head self-attention layer for contextual modeling
    """

    def __init__(self, hidden_size, num_heads=4):
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, key_padding_mask=None):
        """
        x: [batch_size, seq_len, hidden_size]
        key_padding_mask: [batch_size, seq_len] (True = padding)
        """
        attn_output, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask
        )
        x = self.layer_norm(x + attn_output) #残差连接
        return x

# 3. 添加学习率调度和优化器设置
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress))
        return max(min_lr / optimizer.defaults['lr'], cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 将768个通道（变量）作为"token"，在变量维度上做注意力，增强不同BERT维度间的交互。
    """
    Cross-Variable Self-Attention（跨变量/通道注意力）

    目标：输入为序列表示 h=[B,L,D]，把 768当作 D 个“变量 token”，
    先用 masked mean pooling 在时间维 L 上汇聚得到每个变量的 token，
    再在变量维 D 上做多头自注意力，最后把变量表征回写成 [B,L,D] 供融合。
    """

class AdaptiveCrossVariableSelfAttentionLayer(nn.Module):
    def __init__(self, num_variables: int, seq_len: int, d_model=128, num_heads=4):
        super().__init__()
        self.seq_len = seq_len  # L (fixed)
        self.num_variables = num_variables  # m (we use D channels as variables)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_k = d_model // num_heads

        # (14) WQv, WKv, WVv map from time length L to dk/dv per head.
        # We implement multi-head by producing h*dk and then reshaping.
        self.WQv = nn.Linear(self.seq_len, num_heads * self.d_k, bias=False)
        self.WKv = nn.Linear(self.seq_len, num_heads * self.d_k, bias=False)
        self.WVv = nn.Linear(self.seq_len, num_heads * self.d_k, bias=False)  # dv=dk

        # (15) WOv maps concatenated heads back to d_model
        self.WOv = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        self.time_norm = nn.LayerNorm(self.seq_len)

        # Map variable representation back to time domain so we can fuse with time-branch output
        # [B, m, d_model] -> [B, m, L] -> transpose -> [B, L, m]
        self.back_to_time = nn.Linear(d_model, self.seq_len, bias=False)

    def forward(self, h: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, L, D]
            padding_mask: [B, L]  (True=padding)
        Returns:
            h_var: [B, L, D]
        """
        B, L, D = h.shape
        if D != self.num_variables:
            raise ValueError(
                f"CrossVariableSelfAttentionLayer expects num_variables={self.num_variables} but got D={D}"
            )
        if L != self.seq_len:
            raise ValueError(
                f"CrossVariableSelfAttentionLayer expects fixed seq_len={self.seq_len} but got L={L}. "
                f"请确保 pad_packed_sequence(total_length=args.max_len) 让 L 固定。"
            )

        # (7) Flip time tokens into variable tokens:
        # h: [B, L, m] -> Xtrans: [B, m, L]
        Xtrans = h.transpose(1, 2)  # [B, m, L]

        # Mask out padded time steps before linear projections (so WQv/WKv/WVv don't see padding)
        valid_mask = (~padding_mask).unsqueeze(1).to(h.dtype)  # [B,1,L]
        Xtrans = Xtrans * valid_mask  # [B,m,L]

        # (14) Qv, Kv, Vv: [B, m, h*dk]
        Q = self.WQv(Xtrans)
        K = self.WKv(Xtrans)
        V = self.WVv(Xtrans)

        # reshape to heads: [B, h, m, dk]
        Q = Q.view(B, self.num_variables, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, self.num_variables, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, self.num_variables, self.num_heads, self.d_k).transpose(1, 2)

        # attention over variables (m)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # [B,h,m,m]
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B,h,m,dk]

        # concat heads: [B,m,h*dk]
        out = out.transpose(1, 2).contiguous().view(B, self.num_variables, self.num_heads * self.d_k)

        # (15) WOv -> d_model
        VMul = self.WOv(out)  # [B,m,d_model]
        X_time = self.back_to_time(VMul)
        X_time = self.time_norm(X_time)

        # back to time domain: [B,m,L] -> [B,L,m]
        # X_time = self.back_to_time(VMul)  # [B,m,L]
        h_var = X_time.transpose(1, 2)  # [B,L,m] where m==D
        return h_var


#序列分支 
class BiLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_size,
        num_layer,
        max_len: int,
        cv_d_model: int = 128,
        cv_heads: int = 4,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim  # 768
        self.max_len = max_len

        # ========== 1. 定义可学习参数 alpha 和 beta ==========
        # 论文 Eq.(16): Fmap = concat(alpha * Tmul, beta * Vmul)WF
        # 初始化为 1.0，让模型自己学习这两个分支的重要性
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # ========== 先做两路Attention ==========
        # 时间路：跨时间Self-Attention（输入768维）
        self.self_attention = SelfAttentionLayer(
            hidden_size=embedding_dim,  # 768
            num_heads=8  # 768可以被8整除
        )

        # 变量路：Cross-Variable Self-Attention（输入768维）
        self.cross_variable_attention = AdaptiveCrossVariableSelfAttentionLayer(
            num_variables=embedding_dim,  # 768维作为变量数
            seq_len=self.max_len,
            d_model=cv_d_model,
            num_heads=cv_heads,
        )

        # 融合：Concat(h_time, h_var) -> Linear(2*768 -> 768)
        # 这里的 Linear 对应论文中的 WF
        self.fuse = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fuse_norm = nn.LayerNorm(embedding_dim)

        # ========== 然后做LSTM ==========
        # LSTM输入是融合后的768维
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  # 融合后的768
            hidden_size=hidden_size,    # 64
            num_layers=num_layer,
            batch_first=True,
            bidirectional=True
        )

        # LSTM后的输出维度是 hidden_size * 2 = 128
        self.lstm_output_dim = hidden_size * 2

    def forward(self, inputs, x_len):
        """
        Args:
            inputs: [B, L, 768] - BERT embeddings
            x_len: [B] - 每个样本的有效长度
        Returns:
            representations: [B, 128] - 池化后的表示
        """
        # inputs是原始BERT embedding [B, L, 768]
        
        # 构造padding mask
        B, L_real, D = inputs.shape
        device = inputs.device

        # ===== pad / truncate 到固定长度 =====
        if L_real < self.max_len:
            pad_len = self.max_len - L_real
            pad_tensor = torch.zeros(B, pad_len, D, device=device, dtype=inputs.dtype)
            inputs = torch.cat([inputs, pad_tensor], dim=1)
        else:
            inputs = inputs[:, :self.max_len, :]

        # ===== 构造 padding mask =====
        padding_mask = (torch.arange(self.max_len, device=device).unsqueeze(0) 
                        >= x_len.unsqueeze(1).to(device))


        # ========== 第一阶段：并行两路Attention ==========
        # 时间路：跨时间注意力 (T_Mul)
        h_time = self.self_attention(inputs, key_padding_mask=padding_mask)  # [B,L,768]
        
        # 变量路：Cross-Variable Self-Attention (V_Mul)
        h_var = self.cross_variable_attention(inputs, padding_mask=padding_mask)  # [B,L,768]

        # ========== 第二阶段：加权融合 (论文 Eq. 16) ==========
        # 这里的 h_time 对应 T_Mul, h_var 对应 V_Mul
        # 分别乘以可学习参数 alpha 和 beta
        h_time_weighted = self.alpha * h_time
        h_var_weighted = self.beta * h_var

        # 融合：Concat + 线性降维回768
        h_cat = torch.cat([h_time_weighted, h_var_weighted], dim=-1)  # [B,L,1536]
        x_fused = self.fuse(h_cat)  # [B,L,768]
        
        # 残差：以原始BERT embedding为主干，融合结果为残差
        # 注意：论文中通常是 Add & Norm，这里你保留了原本的 ResNet 结构
        x_attended = self.fuse_norm(inputs + x_fused)  # [B,L,768]

        # ========== 第三阶段：LSTM处理 ==========
        # 将GPU上的lengths转移到CPU
        x_len_cpu = x_len.cpu()

        # Pack sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x_attended, x_len_cpu,
            batch_first=True,
            enforce_sorted=False
        )
        
        output, _ = self.lstm(packed)
        
        # Unpack sequence
        x, lengths = nn.utils.rnn.pad_packed_sequence(
            output, 
            batch_first=True, 
            total_length=self.max_len
        )
        # x: [B, L, 128] (hidden_size * 2)

        # ========== 第四阶段：Masked Mean Pooling ==========
        # 构造mask用于平均池化
        mask = torch.arange(x.size(1), device=x.device)[None, :] < x_len[:, None].to(x.device)
        mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
        
        # Masked mean pooling
        representations = (x * mask_expanded).sum(1) / (x_len[:, None].to(x.device) + 1e-8)
        # representations: [B, 128]
        
        return representations, None  # 返回None占位符保持接口一致


class MyLSTMATT(nn.Module):
    """主模型: 先Attention后BiLSTM + MoE"""

    def __init__(
        self,
        features_dic,
        class_num=5,
        engine_dim=100,
        embedding_dim=768,
        hidden_dim=64,
        lstm_layer=2,
        max_len: int = 200,
        cv_d_model: int = 128,
        cv_heads: int = 4,
    ):
        super(MyLSTMATT, self).__init__()
        self.embedding_dim = embedding_dim
        self.engine_dim = engine_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # BiLSTM输出维度 = hidden_dim * 2（双向）
        bilstm_output_dim = hidden_dim * 2

        # MoE输出维度
        moe_output_dim = 128

        # 总输入维度 = 128 + 128 = 256
        total_input_dim = bilstm_output_dim + moe_output_dim

        # 分类头 - 输入维度为256
        # 融合：concat[128+128] -> Linear(256->64) -> Linear(64->5)
        self.fc_1 = nn.Linear(total_input_dim, hidden_dim)  # 256 -> 64
        self.fc_2 = nn.Linear(hidden_dim, class_num)
        # 序列分支：BERT embeddings -> Attention融合 -> BiLSTM -> Pooling (128维)
        # 使用新的BiLSTM架构（先Attention后LSTM）
        self.historic_model = BiLSTM(
            self.embedding_dim,      # 768
            self.hidden_dim,         # 64
            lstm_layer,
            max_len=self.max_len,
            cv_d_model=cv_d_model,
            cv_heads=cv_heads,
        )
        
        # 特征分支：四类特征 -> 两层MoE (128维)
        self.moe = BS.TwoLayerMoE(
            input_dim=self.engine_dim,
            mid_dim=128,
            output_dim=128,
            num_experts_layer1=4,
            num_experts_layer2=4,
            k1=4,
            k2=2
        )

    def get_pred(self, bert_feat, features):
        """Get predictions from fused features"""
        # 仅将四类特征送入MoE
        moe_out = self.moe(features) 

        # 合并序列特征和MoE处理后的特征
        fused = torch.cat((bert_feat, moe_out), dim=1)  # [batch_size, 256]
        feat = self.fc_1(fused)
        logits = self.fc_2(feat)
        return logits

    def forward(self, tweets, lengths, labels, features):
        """
        Args:
            tweets: [B, L, 768] - BERT embeddings
            lengths: [B] - 有效长度
            labels: [B] - 标签
            features: [B, engine_dim] - 四类特征
        """
        # historic_model现在是：BERT(768) -> Attention融合 -> LSTM -> Pooling -> (128)
        h, _ = self.historic_model(tweets, lengths)
        if h.dim() == 1:
            h = h.unsqueeze(0)

        logits = self.get_pred(h, features)
        return logits

#  上述模型已完成，下面是训练和测试的代码
def focal_loss(logits, labels, class_weights=None, alpha=0.25, gamma=2.0, num_classes=5):
    """
    带类别权重的Focal Loss
    """
    # 计算交叉熵损失
    if class_weights is not None:
        # 使用带权重的交叉熵
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
    else:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

    pt = torch.exp(-ce_loss)  # 模型对正确类的预测概率

    # Focal Loss公式: -alpha * (1-p_t)^gamma * log(p_t)
    if isinstance(alpha, (list, np.ndarray, torch.Tensor)):
        # alpha作为类别权重
        alpha_t = torch.tensor(alpha, device=logits.device)[labels]
        focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss
    else:
        # alpha作为标量
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    return focal_loss.mean()

def read_reddit_embeddings(embeddings_path):
    """
    读取BERT embeddings数据
    
    Args:
        embeddings_path: embeddings文件路径（pkl格式）
    
    Returns:
        embeddings列表，每个元素包含 {'label': ..., 'embeddings': ...}
    """
    with open(embeddings_path, 'rb') as f:
        embeddings = pkl.load(f)
    return embeddings




def train(args):
    # 读取数据（支持通过命令行参数指定路径）
    bert_embeddings = read_reddit_embeddings(args.data_embeddings)
    labels = []
    posts = []
    for i in range(len(bert_embeddings)):
        labels.append(bert_embeddings[i]['label'])
        posts.append(bert_embeddings[i]['embeddings'])

    features = pd.read_csv(args.data_features)

    features_dic = {
        'pos': 36,
        'tidif': 50,
        'nrc': 10,
        'sui': 4
    }

    features_dim = features.shape[1]
    labels = pd.DataFrame(labels, columns=['labels'])

    # 设备（CPU/GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_labels = pd.concat([features, labels], axis=1)

    # 开始划分数据集，进行80%的训练集和20%的测试集
    train_data, test_data, train_labels, test_labels = train_test_split(posts, features_labels, test_size=0.2,
                                                                        random_state=args.seed,
                                                                        stratify=features_labels['labels'].values)
    # train_data, val_data,train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=args.seed, stratify=train_labels['labels'].values)
    test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5,
                                                                    random_state=args.seed,
                                                                    stratify=test_labels['labels'].values)

    # 将数据转换为Dataset，并传入 args.max_len
    train_dataset = RedditDataset(train_labels, train_data, days=args.max_len)
    val_dataset = RedditDataset(val_labels, val_data, days=args.max_len)
    test_dataset = RedditDataset(test_labels, test_data, days=args.max_len)


    # 将数据转换为DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_reddit)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_reddit)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_reddit)

    # 初始化模型
    model = MyLSTMATT(features_dic=features_dic, class_num=args.classnum, engine_dim=features_dim,
                      embedding_dim=args.embed_size, hidden_dim=args.hidden_size, lstm_layer=2,
                      max_len=args.max_len, cv_d_model=args.cv_d_model, cv_heads=args.cv_heads)
    model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 添加学习率调度
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10%作为warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=1e-6
    )

    # 添加早停机制
    patience = args.patience
    best_f1 = 0
    early_stop_counter = 0
    model_save_path = './my_reddit_model.pth'

    if args.use_pretrain:
        print("Using pre-trained model")
        model.load_state_dict(torch.load('./my_reddit_model.pth'))
    else:
        for epoch in range(args.epochs):
            model.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch_idx, (labels, tweets, lengths, features) in enumerate(pbar):
                labels = labels.to(device)
                tweets = tweets.to(device)
                features = features.to(device)
                optimizer.zero_grad()

                outputs = model(tweets, lengths, labels, features)

                classification_loss = focal_loss(
                    logits=outputs,
                    labels=labels,
                    alpha=0.25,
                    gamma=2.0,
                    num_classes=args.classnum
                )

                total_batch_loss = classification_loss 

                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()


                # 更新进度条
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{classification_loss.item():.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
            model.eval()
            val_preds_list = []  # 改为新的变量名
            val_labels_list = []  # 改为新的变量名
            val_loss = 0.0
            with torch.no_grad():
                for labels, tweets, lengths, features in val_loader:
                    labels = labels.to(device)
                    tweets = tweets.to(device)
                    features = features.to(device)
                    outputs = model(tweets, lengths, labels, features)

                    classification_loss = focal_loss(
                        logits=outputs,
                        labels=labels,
                        alpha=0.25,
                        gamma=2.0,
                        num_classes=args.classnum
                    )

                    total_loss = classification_loss
                    val_loss += total_loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    val_preds_list.extend(preds.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            # 转换为numpy数组进行计算
            val_preds = np.array(val_preds_list)
            val_labels = np.array(val_labels_list)

            M = utils.gr_metrics(val_preds, val_labels)

            # 计算accuracy
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            accuracy = np.mean(val_preds == val_labels)
            print(f"Epoch {epoch} Validation Accuracy: {accuracy}")
            f1 = M[2]

            # 1. 保存最佳模型
            if f1 > best_f1:
                best_f1 = f1
                early_stop_counter = 0
                # torch.save(model.state_dict(), 'my_best_model.pth')
            else:
                early_stop_counter += 1

            # 2. 无论好坏，都在循环最后保存一个
            torch.save(model.state_dict(), 'my_reddit_model.pth')

            # 3. 检查早停
            if early_stop_counter >= patience:
                break

    model.load_state_dict(torch.load('./my_reddit_model.pth'))

    model.eval()
    test_preds_list = []  # 新的变量名
    test_labels_list = []  # 新的变量名
    test_loss = 0.0

    with torch.no_grad():
        for labels, tweets, lengths, features in test_loader:
            labels = labels.to(device)
            tweets = tweets.to(device)
            features = features.to(device)
            outputs = model(tweets, lengths, labels, features)

            classification_loss = focal_loss(
                logits=outputs,
                labels=labels,
                alpha=0.25,
                gamma=2.0,
                num_classes=args.classnum
            )

            total_loss = classification_loss 

            test_loss += total_loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_preds_list.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)

    # 转换为numpy数组
    fin_outputs = np.array(test_preds_list)
    fin_targets = np.array(test_labels_list)
    # 详细结果输出
    print("\n[1] Classification Report (按类别统计):")
    print(classification_report(fin_targets, fin_outputs, digits=4, zero_division=0))
    # 找到所有预测错误的索引
    misclassified_mask = fin_outputs != fin_targets
    misclassified_indices = np.where(misclassified_mask)[0]

    # 收集错误数据
    bad_cases = {
        'Sample_Index': misclassified_indices,
        'True_Label': fin_targets[misclassified_mask],
        'Pred_Label': fin_outputs[misclassified_mask]
    }

    df_bad = pd.DataFrame(bad_cases)
    df_bad.to_csv('bad_cases.csv', index=False)

    M = utils.gr_metrics(fin_outputs, fin_targets)
    accuracy = np.mean(fin_outputs == fin_targets)
    print(f"Accuracy: {accuracy:.4f}")
    print(f" test GP: {M[0]} GR: {M[1]} FS: {M[2]} OE: {M[3]}")


def set_seed(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # 关键：锁定cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False  
    np.random.seed(args.seed)
    random.seed(args.seed)
    # 设置Python哈希种子（终端执行或代码开头）
    os.environ['PYTHONHASHSEED'] = str(args.seed)


def main():
    args = parse_args()
    set_seed(args)
    train(args)


if __name__ == '__main__':
    main()
