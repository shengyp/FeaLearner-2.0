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
import base_model_moe as BS


# from torchfm.layer import MultiLayerPerceptron
# from transformers import (AdamW, get_cosine_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup)


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
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embed_size", type=int, default=768)
    parser.add_argument("--max_len", default=200, type=int)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=24, type=int)
    parser.add_argument("--classnum", default=5, type=int)
    parser.add_argument("--use_pretrain", default=False, type=bool)
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
        if self.days > len(self.tweets[item]):
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
        x = self.layer_norm(x + attn_output)
        return x


# 增强模型对输入数据的某些部分的关注度
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first  # 指示输入数据的第一个维度是否是批次大小。
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        # self.SelfAttention = SelfAttention(hidden_size, batch_first=True)
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze(-1)), dim=-1)
        mask = torch.ones(attentions.size(), requires_grad=True).to(attentions.device)

        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                # 排除填充为零的部分的注意力
                mask[i, l:] = 0
        # 这块目的去除填充为零的部分,然后重新计算每个帖子的权重
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        attentions = masked.div(_sums)
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()
        return representations, attentions


class Attention1(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention1, self).__init__()
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.batch_first = batch_first
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
        u = torch.tanh(torch.matmul(inputs, self.w_omega))  # [batch, seq_len, hidden_dim]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        attentions = torch.softmax(F.relu(att.squeeze(-1)), dim=-1)
        mask = torch.ones(attentions.size(), requires_grad=True).to(attentions.device)

        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                # 排除填充为零的部分的注意力
                mask[i, l:] = 0
        # 这块目的去除填充为零的部分,然后重新计算每个帖子的权重
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        attentions = masked.div(_sums)
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()
        return representations, attentions

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layer):
        super().__init__()
        self.embedding_dim = embedding_dim

        # hidden_size=64
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            bidirectional=True
        )

        # 添加SelfAttention层
        self.self_attention = SelfAttentionLayer(
            hidden_size=hidden_size * 2,
            num_heads=4
        )

        self.attention = Attention1(hidden_size * 2, batch_first=True)

    def forward(self, inputs, x_len):

        # 将GPU上的lengths转移到CPU
        x_len_cpu = x_len.cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, x_len_cpu,
            batch_first=True,
            enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        x, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # 构造padding mask
        max_len = x.size(1)
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= x_len.to(x.device)[:, None]

        # Self-Attention
        x = self.self_attention(x, key_padding_mask=padding_mask)

        # Attention pooling
        representations, attentions = self.attention(x, lengths)
        return representations, attentions


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


class MyLSTMATT(nn.Module):
    """主模型: BiLSTM + Attention + MoE"""

    def __init__(self, features_dic, class_num=5, engine_dim=100, embedding_dim=768, hidden_dim=64,
                 lstm_layer=2):
        super(MyLSTMATT, self).__init__()
        self.embedding_dim = embedding_dim
        self.engine_dim = engine_dim
        self.hidden_dim = hidden_dim

        # BiLSTM输出维度 = hidden_dim * 2
        bilstm_output_dim = hidden_dim * 2

        # MoE输出维度
        moe_output_dim = 128

        # 总输入维度 = 128 + 128 = 256
        total_input_dim = bilstm_output_dim + moe_output_dim

        # 分类头 - 输入维度为256
        self.fc_1 = nn.Linear(total_input_dim, hidden_dim)  # 256 -> 64
        self.fc_2 = nn.Linear(hidden_dim, class_num)

        # BiLSTM用于序列建模，传入隐藏层大小64
        self.historic_model = BiLSTM(self.embedding_dim, self.hidden_dim, lstm_layer)
        # MoE用于特征融合
        self.moe = BS.TwoLayerMoE(
            input_dim=self.engine_dim,  # 仅使用特征维度作为输入
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
        fused = torch.cat((bert_feat, moe_out), dim=1)  # [batch_size, 128 + 128 = 256]
        feat = self.fc_1(fused)
        logits = self.fc_2(feat)
        return logits

    def forward(self, tweets, lengths, labels, features):
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


def read_reddit_embeddings():
    # Load the reddit embeddings
    with open('../data/bert_embeddings.pkl', 'rb') as f:
        # with open('../data/reddit_clean.pkl', 'rb') as f:

        reddit_embeddings = pkl.load(f)
    return reddit_embeddings


def train(args):
    bert_embeddings = read_reddit_embeddings()
    labels = []
    posts = []
    for i in range(len(bert_embeddings)):
        labels.append(bert_embeddings[i]['label'])
        posts.append(bert_embeddings[i]['embeddings'])

    features = pd.read_csv('../data_analy/feature.csv')

    features_dic = {
        'pos': 36,
        'tidif': 50,
        'nrc': 10,
        'sui': 4
    }

    features_dim = features.shape[1]
    labels = pd.DataFrame(labels, columns=['labels'])

    # 计算类别权重用于focal loss（保留以备选，但移到 device）
    class_counts = labels['labels'].value_counts().sort_index().values
    total = len(labels)
    class_weights = total / (len(class_counts) * class_counts)
    print(f"类别权重: {class_weights}")

    features_labels = pd.concat([features, labels], axis=1)

    # 开始划分数据集，进行80%的训练集和20%的测试集
    train_data, test_data, train_labels, test_labels = train_test_split(posts, features_labels, test_size=0.2,
                                                                        random_state=args.seed,
                                                                        stratify=features_labels['labels'].values)
    # train_data, val_data,train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=args.seed, stratify=train_labels['labels'].values)
    test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5,
                                                                    random_state=args.seed,
                                                                    stratify=test_labels['labels'].values)
    # print(train_data)

    # print(train_labels)

    # 将数据转换为Dataset，并传入 args.max_len
    train_dataset = RedditDataset(train_labels, train_data, days=args.max_len)
    val_dataset = RedditDataset(val_labels, val_data, days=args.max_len)
    test_dataset = RedditDataset(test_labels, test_data, days=args.max_len)

    # 设备（CPU/GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将数据转换为DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_reddit)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_reddit)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_reddit)

    # 初始化模型
    model = MyLSTMATT(features_dic=features_dic, class_num=args.classnum, engine_dim=features_dim,
                      embedding_dim=args.embed_size, hidden_dim=args.hidden_size, lstm_layer=2)
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
    patience = 10
    best_f1 = 0
    early_stop_counter = 0
    model_save_path = './my_best_model.pth'

    best_f1 = 0

    if args.use_pretrain:
        print("Using pre-trained model")
        model.load_state_dict(torch.load('./my_best_model.pth'))
    else:
        for epoch in range(args.epochs):
            model.train()
            total_focal_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch_idx, (labels, tweets, lengths, features) in enumerate(pbar):
                labels = labels.to(device)
                tweets = tweets.to(device)
                features = features.to(device)
                optimizer.zero_grad()

                # outputs, moe_loss = model(tweets, lengths, labels, features)
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
            else:
                early_stop_counter += 1

            # 2. 无论好坏，都在循环最后保存一个
            torch.save(model.state_dict(), 'my_best_model.pth')

            # 3. 检查早停
            if early_stop_counter >= patience:
                break

    model.load_state_dict(torch.load('./my_best_model.pth'))
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
    """
    :param args:
    :return:
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def main():
    args = parse_args()
    set_seed(args)
    train(args)


if __name__ == '__main__':
    main()

