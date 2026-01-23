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
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import pickle as pkl
from sklearn.metrics import classification_report
from tools import utils
import twomoe as BS

# 直接从主程序导入模型组件，确保 Exp-0 完全一致
from reddit import (
    MyLSTMATT, 
    BiLSTM, 
    SelfAttentionLayer, 
    AdaptiveCrossVariableSelfAttentionLayer,
    RedditDataset,
    pad_collate_reddit,
    focal_loss,
    get_cosine_schedule_with_warmup,
    set_seed
)

# ==================== 消融专用组件实现 ====================

# Exp-1/2/3/7: 定制化的序列建模变体
class BiLSTM_Ablation_Variants(BiLSTM):
    def __init__(self, embedding_dim, hidden_size, num_layer, max_len, 
                 cv_d_model=128, cv_heads=2, variant='full'):
        super().__init__(embedding_dim, hidden_size, num_layer, max_len, cv_d_model, cv_heads)
        self.variant = variant
        
        if variant == 'no_temporal':
            # 移除时间路，仅保留变量路
            self.self_attention = None 
        elif variant == 'no_crossvar':
            # 移除变量路
            self.cross_variable_attention = None
        elif variant == 'no_dual':
            # 移除所有注意力
            self.self_attention = None
            self.cross_variable_attention = None
        elif variant == 'uni_lstm':
            # 变更为单向LSTM
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layer,
                batch_first=True,
                bidirectional=False
            )
            self.lstm_output_dim = hidden_size

    def forward(self, inputs, x_len):
        B, L_real, D = inputs.shape
        device = inputs.device

        # Padding/Truncating 逻辑与基类一致
        if L_real < self.max_len:
            pad_len = self.max_len - L_real
            pad_tensor = torch.zeros(B, pad_len, D, device=device, dtype=inputs.dtype)
            inputs = torch.cat([inputs, pad_tensor], dim=1)
        else:
            inputs = inputs[:, :self.max_len, :]

        padding_mask = (torch.arange(self.max_len, device=device).unsqueeze(0) 
                        >= x_len.unsqueeze(1).to(device))

        # 根据变体执行不同的 Attention 分支
        h_time = 0
        h_var = 0
        
        if self.self_attention is not None:
            h_time = self.alpha * self.self_attention(inputs, key_padding_mask=padding_mask)
        
        if self.cross_variable_attention is not None:
            h_var = self.beta * self.cross_variable_attention(inputs, padding_mask=padding_mask)

        if self.variant == 'no_dual':
            x_attended = self.fuse_norm(inputs) # 无注意力，直接残差
        else:
            # 这里兼容只有单路或双路的情况
            h_cat = torch.cat([h_time if isinstance(h_time, torch.Tensor) else torch.zeros_like(inputs), 
                               h_var if isinstance(h_var, torch.Tensor) else torch.zeros_like(inputs)], dim=-1)
            x_fused = self.fuse(h_cat)
            x_attended = self.fuse_norm(inputs + x_fused)

        # LSTM 处理
        x_len_cpu = x_len.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x_attended, x_len_cpu, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=self.max_len)

        # Pooling
        mask = torch.arange(x.size(1), device=x.device)[None, :] < x_len[:, None].to(x.device)
        mask_expanded = mask.unsqueeze(-1).float()
        representations = (x * mask_expanded).sum(1) / (x_len[:, None].to(x.device) + 1e-8)
        
        return representations, None

# Exp-4: 固定权重的变体
class BiLSTM_FixedWeights(BiLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 将 alpha, beta 设为不可学习的 buffer
        del self.alpha
        del self.beta
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('beta', torch.tensor(1.0))

# ==================== 统一消融接口模型 ====================
class AblationModel(MyLSTMATT):
    def __init__(self, features_dic, class_num, engine_dim, embedding_dim, hidden_dim,
                 lstm_layer, max_len, cv_d_model, cv_heads, 
                 exp_type='full', use_moe=True, use_features=True, moe_layers=2):
        super().__init__(features_dic, class_num, engine_dim, embedding_dim, hidden_dim,
                         lstm_layer, max_len, cv_d_model, cv_heads)
        
        self.use_features = use_features
        self.exp_type = exp_type

        # 1. 序列分支消融处理
        if exp_type == 'full':
            pass # 保持基类中的 BiLSTM 不变 (Exp-0)
        elif exp_type in ['no_temporal', 'no_crossvar', 'no_dual', 'uni_lstm']:
            self.historic_model = BiLSTM_Ablation_Variants(
                embedding_dim, hidden_dim, lstm_layer, max_len, cv_d_model, cv_heads, variant=exp_type
            )
        elif exp_type == 'fixed_weights':
            self.historic_model = BiLSTM_FixedWeights(
                embedding_dim, hidden_dim, lstm_layer, max_len, cv_d_model, cv_heads
            )

        # 2. 特征分支消融处理
        if use_features:
            if not use_moe:
                # Exp-5: MLP替换MoE
                self.moe = nn.Sequential(
                    nn.Linear(engine_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128)
                )
            elif moe_layers == 1:
                # Exp-6: 单层MoE
                self.moe = BS.SparseMoELayer(input_dim=engine_dim, output_dim=128, num_experts=4, k=2)
        
        # 更新全连接层输入维度（处理 Exp-8: 无特征, Exp-7: Uni-LSTM）
        bilstm_out_dim = self.historic_model.lstm_output_dim
        moe_out_dim = 128 if use_features else 0
        self.fc_1 = nn.Linear(bilstm_out_dim + moe_out_dim, hidden_dim)

    def forward(self, tweets, lengths, labels, features):
        h, _ = self.historic_model(tweets, lengths)
        if h.dim() == 1: h = h.unsqueeze(0)
        
        if self.use_features:
            moe_out = self.moe(features)
            fused = torch.cat((h, moe_out), dim=1)
        else:
            fused = h # Exp-8
            
        feat = self.fc_1(fused)
        logits = self.fc_2(feat)
        return logits

# ==================== 训练核心逻辑 ====================
def train_single_experiment(args, exp_config, train_loader, val_loader, test_loader, device):
    # 1. 严格重置种子，确保每个消融实验的起点完全一致
    set_seed(args) 
    
    # 2. 模型初始化分支处理
    if exp_config['name'] == 'Exp-0_Full':
        model = MyLSTMATT(
            features_dic={'pos': 36, 'tidif': 50, 'nrc': 10, 'sui': 4}, 
            class_num=args.classnum, 
            engine_dim=100, # 需根据具体特征维度对齐
            embedding_dim=args.embed_size, 
            hidden_dim=args.hidden_size, 
            lstm_layer=2,
            max_len=args.max_len, 
            cv_d_model=args.cv_d_model, 
            cv_heads=args.cv_heads
        ).to(device)
    else:
        # 其他消融实验 AblationModel
        model = AblationModel(
            features_dic={'pos': 36, 'tidif': 50, 'nrc': 10, 'sui': 4}, 
            class_num=args.classnum, 
            engine_dim=100,
            embedding_dim=args.embed_size, 
            hidden_dim=args.hidden_size, 
            lstm_layer=2,
            max_len=args.max_len, 
            cv_d_model=args.cv_d_model, 
            cv_heads=args.cv_heads,
            **exp_config['model_args']
        ).to(device)

    # 3. 优化器配置
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 4. 学习率调度器配置
    num_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * num_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps, 
        num_steps,
        min_lr=1e-6 # 显式对齐 dual_bigdata.py 的参数
    )
    
    patience = args.patience
    best_f1 = 0
    early_stop_counter = 0
    os.makedirs("./ablation", exist_ok=True)
    model_save_path = f"./ablation/{exp_config['name']}.pth"

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        for labels, tweets, lengths, features in train_loader:
            labels, tweets, features = labels.to(device), tweets.to(device), features.to(device)
            optimizer.zero_grad()
            
            outputs = model(tweets, lengths, labels, features)
            
            # 关键：严格对齐 dual_bigdata.py 的损失函数调用参数
            if exp_config.get('use_focal_loss', True):
                loss = focal_loss(
                    logits=outputs,
                    labels=labels,
                    alpha=0.25,      # 显式传递
                    gamma=2.0,       # 显式传递
                    num_classes=args.classnum
                )
            else:
                loss = F.cross_entropy(outputs, labels) 
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪对齐
            optimizer.step()
            scheduler.step()

        # 6. 验证逻辑
        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for labels, tweets, lengths, features in val_loader:
                out = model(tweets.to(device), lengths, labels, features.to(device))
                v_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                v_labels.extend(labels.numpy())
        
        metrics = utils.gr_metrics(np.array(v_preds), np.array(v_labels))
        f1 = metrics[2]
        
        # 保存最佳模型逻辑对齐
        if f1 > best_f1:
            best_f1 = f1
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        torch.save(model.state_dict(), model_save_path) 
        if early_stop_counter >= patience:
            break

    # 7. 测试逻辑
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    t_preds, t_labels = [], []
    with torch.no_grad():
        for labels, tweets, lengths, features in test_loader:
            out = model(tweets.to(device), lengths, labels, features.to(device))
            t_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            t_labels.extend(labels.numpy())
    
    t_preds, t_labels = np.array(t_preds), np.array(t_labels)
    M = utils.gr_metrics(t_preds, t_labels)
    acc = np.mean(t_preds == t_labels)
    
    print(f"[{exp_config['name']}] Acc: {acc:.4f}, F1: {M[2]:.4f}")
    return {'exp': exp_config['name'], 'Acc': acc, 'F1': M[2], 'GP': M[0], 'GR': M[1]}
def run_ablation_study(args):
    # 数据加载完全对齐 dual_bigdata.py
    with open(args.data_embeddings, 'rb') as f:
        data = pkl.load(f)
    posts = [x['embeddings'] for x in data]
    labels = pd.DataFrame([x['label'] for x in data], columns=['labels'])
    features = pd.read_csv(args.data_features)
    df_combined = pd.concat([features, labels], axis=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 划分逻辑必须与主程序一致
    tr_d, te_d, tr_l, te_l = train_test_split(posts, df_combined, test_size=0.2, random_state=args.seed, stratify=df_combined['labels'])
    te_d, va_d, te_l, va_l = train_test_split(te_d, te_l, test_size=0.5, random_state=args.seed, stratify=te_l['labels'])

    train_loader = DataLoader(RedditDataset(tr_l, tr_d, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_reddit)
    val_loader = DataLoader(RedditDataset(va_l, va_d, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_reddit)
    test_loader = DataLoader(RedditDataset(te_l, te_d, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_reddit)

    experiments = [
        {'name': 'Exp-0_Full', 'desc': 'Baseline', 'model_args': {'exp_type': 'full'}},
        {'name': 'Exp-1_no_Temp', 'desc': 'w/o Temporal Attn', 'model_args': {'exp_type': 'no_temporal'}},
        {'name': 'Exp-2_no_Cross', 'desc': 'w/o Cross-Var Attn', 'model_args': {'exp_type': 'no_crossvar'}},
        {'name': 'Exp-3_no_Dual', 'desc': 'w/o All Attn', 'model_args': {'exp_type': 'no_dual'}},
        {'name': 'Exp-4_Fixed', 'desc': 'α=β=1.0', 'model_args': {'exp_type': 'fixed_weights'}},
        {'name': 'Exp-5_MLP', 'desc': 'MoE -> MLP', 'model_args': {'exp_type': 'full', 'use_moe': False}},
        {'name': 'Exp-6_1Layer_MoE', 'desc': '2-Layer -> 1-Layer MoE', 'model_args': {'exp_type': 'full', 'moe_layers': 1}},
        {'name': 'Exp-7_UniLSTM', 'desc': 'Bi- -> Uni-LSTM', 'model_args': {'exp_type': 'uni_lstm'}},
        {'name': 'Exp-8_no_Feat', 'desc': 'w/o Handcrafted Features', 'model_args': {'exp_type': 'full', 'use_features': False}},
        {'name': 'Exp-9_CE_Loss', 'desc': 'Focal -> CE Loss', 'model_args': {'exp_type': 'full'}, 'use_focal_loss': False},
    ]

    all_res = []
    for config in experiments:
        res = train_single_experiment(args, config, train_loader, val_loader, test_loader, device)
        all_res.append(res)
    
    pd.DataFrame(all_res).to_csv('ablation_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_size", type=int, default=768)
    parser.add_argument("--max_len", default=200, type=int)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--cv_d_model", type=int, default=128)
    parser.add_argument("--cv_heads", type=int, default=2)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=24, type=int)
    parser.add_argument("--classnum", default=5, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--data_embeddings", type=str, default="../data/bert_embeddings.pkl")
    parser.add_argument("--data_features", type=str, default="../data_analy/feature_reddit_500.csv")
    args = parser.parse_args()
    
    run_ablation_study(args)
