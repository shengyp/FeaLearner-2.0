import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl
import twomoe as BS
from tools import utils

# 直接从 train.py 引用核心组件，确保基础环境绝对对齐
from auto_select.weibo_train import (
    set_seed, 
    pad_collate_weibo, 
    WeiboDataset, 
    BiLSTM, 
    MyLSTMATT, 
    focal_loss, 
    get_cosine_schedule_with_warmup
)

# ==================== 1. 定义消融变体组件 ====================

class BiLSTM_Ablation_Variants(BiLSTM):
    """
    序列建模变体 (Exp-1, 2, 3)
    """
    def __init__(self, embedding_dim, hidden_size, num_layer, max_len, 
                 cv_d_model=128, cv_heads=2, variant='full'):
        super().__init__(embedding_dim, hidden_size, num_layer, max_len, cv_d_model, cv_heads)
        self.variant = variant
        
        # 移除对应路，但保留参数槽位以防初始化顺序偏移
        if variant == 'no_temporal':
            self.self_attention = None 
        elif variant == 'no_crossvar':
            self.cross_variable_attention = None
        elif variant == 'no_dual':
            self.self_attention = None
            self.cross_variable_attention = None

    def forward(self, inputs, x_len):
        B, L_real, D = inputs.shape
        device = inputs.device

        if L_real < self.max_len:
            pad_len = self.max_len - L_real
            inputs = torch.cat([inputs, torch.zeros(B, pad_len, D, device=device, dtype=inputs.dtype)], dim=1)
        else:
            inputs = inputs[:, :self.max_len, :]

        padding_mask = (torch.arange(self.max_len, device=device).unsqueeze(0) 
                        >= x_len.unsqueeze(1).to(device))

        # 根据变体计算分支
        h_time = self.alpha * self.self_attention(inputs, key_padding_mask=padding_mask) if self.self_attention else 0
        h_var = self.beta * self.cross_variable_attention(inputs, padding_mask=padding_mask) if self.cross_variable_attention else 0

        if self.variant == 'no_dual':
            x_attended = self.fuse_norm(inputs)
        else:
            # 兼容单路情况
            t = h_time if self.self_attention else torch.zeros_like(inputs)
            v = h_var if self.cross_variable_attention else torch.zeros_like(inputs)
            x_fused = self.fuse(torch.cat([t, v], dim=-1))
            x_attended = self.fuse_norm(inputs + x_fused)

        packed = nn.utils.rnn.pack_padded_sequence(x_attended, x_len.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=self.max_len)
        mask_expanded = (torch.arange(x.size(1), device=x.device)[None, :] < x_len[:, None].to(x.device)).unsqueeze(-1).float()
        return (x * mask_expanded).sum(1) / (x_len[:, None].to(x.device) + 1e-8), None

class BiLSTM_FixedWeights(BiLSTM):
    """固定权重变体 (Exp-4)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.alpha
        del self.beta
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('beta', torch.tensor(1.0))

class AblationModel(MyLSTMATT):
    def __init__(self, features_dic, class_num, engine_dim, embedding_dim, hidden_dim,
                 lstm_layer, max_len, cv_d_model, cv_heads, 
                 exp_type='full', use_moe=True, use_features=True, moe_layers=2):
        super().__init__(features_dic, class_num, engine_dim, embedding_dim, hidden_dim,
                         lstm_layer, max_len, cv_d_model, cv_heads)
        
        self.use_features = use_features

        # 1. 序列分支消融
        if exp_type in ['no_temporal', 'no_crossvar', 'no_dual']:
            self.historic_model = BiLSTM_Ablation_Variants(
                embedding_dim, hidden_dim, lstm_layer, max_len, cv_d_model, cv_heads, variant=exp_type
            )
        elif exp_type == 'fixed_weights':
            self.historic_model = BiLSTM_FixedWeights(embedding_dim, hidden_dim, lstm_layer, max_len, cv_d_model, cv_heads)

        # 2. 特征分支消融
        if use_features:
            if not use_moe: # Exp-5
                self.moe = nn.Sequential(nn.Linear(engine_dim, 128), nn.ReLU(), nn.Linear(128, 128))
            elif moe_layers == 1: # Exp-6
                self.moe = BS.SparseMoELayer(input_dim=engine_dim, output_dim=128, num_experts=4, k=2)
        
        # 3. 维度校准 (Exp-7)
        moe_out_dim = 128 if use_features else 0
        self.fc_1 = nn.Linear(hidden_dim * 2 + moe_out_dim, hidden_dim)

    def forward(self, tweets, lengths, labels, features):
        h, _ = self.historic_model(tweets, lengths)
        if h.dim() == 1: h = h.unsqueeze(0)
        if self.use_features:
            moe_out = self.moe(features)
            fused = torch.cat((h, moe_out), dim=1)
        else:
            fused = h
        return self.fc_2(self.fc_1(fused))

# ==================== 2. 实验核心逻辑 ====================

def train_single_experiment(args, exp_config, loaders, device, features_dim, captured_state):
    # 【核心】恢复 RNG 状态，确保参数初始化和 Shuffle 顺序与 train.py 完全一致
    random.setstate(captured_state['random'])
    np.random.set_state(captured_state['numpy'])
    torch.set_rng_state(captured_state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(captured_state['cuda'])

    train_loader, val_loader, test_loader = loaders
    features_dic = {'pos': 57, 'tfidf': 50, 'nrc': 10, 'sui': 13}

    # Exp-0 必须直接实例化原类，避免 AblationModel 构造逻辑干扰 RNG
    if exp_config['name'] == 'Exp-0_Full':
        model = MyLSTMATT(features_dic, args.classnum, features_dim, args.embed_size, 
                          args.hidden_size, 2, args.max_len, args.cv_d_model, args.cv_heads).to(device)
    else:
        model = AblationModel(features_dic, args.classnum, features_dim, args.embed_size, 
                             args.hidden_size, 2, args.max_len, args.cv_d_model, args.cv_heads,
                             **exp_config.get('model_args', {})).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * num_steps), num_steps)
    
    best_f1, early_stop_counter = 0, 0
    os.makedirs("./ablation_weibo", exist_ok=True)
    model_save_path = f"./ablation_weibo/{exp_config['name']}.pth"

    for epoch in range(args.epochs):
        model.train()
        for labels, tweets, lengths, features in train_loader:
            labels, tweets, features = labels.to(device), tweets.to(device), features.to(device)
            optimizer.zero_grad()
            out = model(tweets, lengths, labels, features)
            loss = focal_loss(out, labels, alpha=0.25, gamma=2.0) if exp_config.get('use_focal_loss', False) else F.cross_entropy(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for labels, tweets, lengths, features in val_loader:
                out = model(tweets.to(device), lengths, labels, features.to(device))
                v_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                v_labels.extend(labels.numpy())
        
        _, _, _, f1 = utils.binary_metrics(v_preds, v_labels)
        if f1 > best_f1:
            best_f1, early_stop_counter = f1, 0
            torch.save(model.state_dict(), model_save_path)
        else:
            early_stop_counter += 1
        if early_stop_counter >= args.patience: break

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    t_preds, t_labels = [], []
    with torch.no_grad():
        for labels, tweets, lengths, features in test_loader:
            out = model(tweets.to(device), lengths, labels, features.to(device))
            t_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            t_labels.extend(labels.numpy())
    
    acc, prec, rec, f1 = utils.binary_metrics(t_preds, t_labels)
    print(f"[{exp_config['name']}] F1: {f1:.4f}, Acc: {acc:.4f}")
    return {'exp': exp_config['name'], 'Acc': acc, 'F1': f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embed_size", type=int, default=768)
    parser.add_argument("--max_len", default=100, type=int)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--cv_d_model", type=int, default=128)
    parser.add_argument("--cv_heads", type=int, default=2)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=24, type=int)
    parser.add_argument("--classnum", default=2, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--data_embeddings", type=str, default="../data/user_post_embeddings_bert_wwm.pkl")
    parser.add_argument("--data_features", type=str, default="../data_analy/feature_weibo.csv")
    args = parser.parse_args()

    set_seed(args) # 数据划分前设定种子

    # 数据加载完全同步 train.py
    with open(args.data_embeddings, 'rb') as f: bert_embeddings = pkl.load(f)
    posts = [x['embeddings'] for x in bert_embeddings]
    labels_list = [x['label'] for x in bert_embeddings]
    features_df = pd.read_csv(args.data_features)
    df_combined = pd.concat([features_df, pd.DataFrame(labels_list, columns=['labels'])], axis=1)
    
    # 划分逻辑镜像
    tr_d, te_d, tr_l, te_l = train_test_split(posts, df_combined, test_size=0.2, random_state=args.seed, stratify=df_combined['labels'])
    te_d, va_d, te_l, va_l = train_test_split(te_d, te_l, test_size=0.5, random_state=args.seed, stratify=te_l['labels'])

    loaders = (
        DataLoader(WeiboDataset(tr_l, tr_d, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_weibo),
        DataLoader(WeiboDataset(va_l, va_d, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_weibo),
        DataLoader(WeiboDataset(te_l, te_d, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_weibo)
    )

    # 【重要】在模型初始化前捕捉 RNG 状态
    captured_state = {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

    experiments = [
        {'name': 'Exp-0_Full', 'model_args': {'exp_type': 'full'}}, 
        {'name': 'Exp-1_no_Temp', 'model_args': {'exp_type': 'no_temporal'}},
        {'name': 'Exp-2_no_Cross', 'model_args': {'exp_type': 'no_crossvar'}},
        {'name': 'Exp-3_no_Dual', 'model_args': {'exp_type': 'no_dual'}},
        {'name': 'Exp-4_Fixed', 'model_args': {'exp_type': 'fixed_weights'}},
        {'name': 'Exp-5_MLP', 'model_args': {'use_moe': False}},
        {'name': 'Exp-6_1Layer_MoE', 'model_args': {'moe_layers': 1}},
        {'name': 'Exp-7_no_Feat', 'model_args': {'use_features': False}},
        {'name': 'Exp-8_Focal_Loss', 'use_focal_loss': True},
    ]

    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for config in experiments:
        res = train_single_experiment(args, config, loaders, device, features_df.shape[1], captured_state)
        all_results.append(res)
        
    pd.DataFrame(all_results).to_csv('weibo_ablation_results.csv', index=False)

if __name__ == '__main__':
    main()