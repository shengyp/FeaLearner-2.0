import argparse
import os
from pickletools import optimize
import random
import string
import time
from math import log
from pathlib import Path
import numpy as np
import scipy.sparse as sp
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


# 统一的路径解析：不依赖运行时的当前工作目录（CWD）
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../auto_select
_REPO_ROOT = _SCRIPT_DIR.parent                        # 仓库根目录

def _default_repo_path(*parts: str) -> str:
    return str(_REPO_ROOT.joinpath(*parts))

def _default_result_path(*parts: str) -> str:
    return str(_SCRIPT_DIR.joinpath(*parts))


# 将一个批次中的多个数据项按特定规则组合并填充(pad),以便它们可以被批量处理。
def pad_collate_weibo(batch):
    """Weibo数据集的collate函数"""
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
        description='Training and Testing on Weibo Dataset',
        usage='weibo.py [<args>] [-h | --help]'
    )
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embed_size", type=int, default=768)
    # Weibo数据集调整: 每个用户最多200条微博
    parser.add_argument("--max_len", default=100, type=int, help="Maximum number of posts per user")
    parser.add_argument("--hidden_size", type=int, default=128)
    # Cross-Variable Self-Attention 超参(变量路)
    parser.add_argument("--cv_d_model", type=int, default=128)
    parser.add_argument("--cv_heads", type=int, default=2)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--seed", default=24, type=int)
    # Weibo数据集调整: 二分类任务
    parser.add_argument("--classnum", default=2, type=int, help="Number of classes (2 for Weibo)")
    parser.add_argument("--use_pretrain", default=False, type=bool)
    parser.add_argument("--patience", default=10, type=int, help="Early stopping patience")
    # Weibo数据集路径参数
    parser.add_argument("--data_embeddings", type=str, default=_default_repo_path("data", "user_post_embeddings_bert_wwm.pkl"),
                        help="Weibo BERT embeddings数据文件路径(pkl格式)")
    parser.add_argument("--data_features", type=str, default=_default_repo_path("data_analy", "feature_weibo_2.csv"),
                        help="Weibo特征数据文件路径(csv格式)")
    parser.add_argument("--save_path", type=str, default=_default_result_path("results", "weibo", "weibo_best_model.pth"),
                        help="模型保存路径")
    return parser.parse_args(args)

class WeiboDataset(Dataset):
    """Weibo数据集类"""
    def __init__(self, labels, tweets, days=200):
        super().__init__()
        self.labels = labels
        self.tweets = tweets  # 预训练的嵌入向量
        self.days = days  # 用户发表的帖子的嵌入

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        labels = torch.tensor(self.labels['labels'].iloc[item], dtype=torch.long)
        feature = torch.tensor(self.labels.iloc[item, :-1].values, dtype=torch.float32)
        
        # 处理tweets嵌入 - 需要处理numpy数组和tensor两种情况
        tweets_list = self.tweets[item]
        
        # 截断或保留
        if len(tweets_list) > self.days:
            tweets_list = tweets_list[:self.days]
        
        if len(tweets_list) == 0:
            # 防止空数据导致的错误，填充一个全0向量
            tweets = torch.zeros((1, 768), dtype=torch.float32)
        else:
            # 确保所有元素都是torch张量（处理numpy数组的情况）
            emb_tensors = []
            for emb in tweets_list:
                if isinstance(emb, np.ndarray):
                    emb_tensors.append(torch.from_numpy(emb).float())
                elif isinstance(emb, torch.Tensor):
                    emb_tensors.append(emb.detach().float())
                else:
                    emb_tensors.append(torch.tensor(emb, dtype=torch.float32))
            tweets = torch.stack(emb_tensors)  # 已经是float类型
        
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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress))
        return max(min_lr / optimizer.defaults['lr'], cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class AdaptiveCrossVariableSelfAttentionLayer(nn.Module):
    """Cross-Variable Self-Attention层"""
    def __init__(self, num_variables: int, seq_len: int, d_model=128, num_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.num_variables = num_variables
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_k = d_model // num_heads

        self.WQv = nn.Linear(self.seq_len, num_heads * self.d_k, bias=False)
        self.WKv = nn.Linear(self.seq_len, num_heads * self.d_k, bias=False)
        self.WVv = nn.Linear(self.seq_len, num_heads * self.d_k, bias=False)
        self.WOv = nn.Linear(num_heads * self.d_k, d_model, bias=False)
        self.time_norm = nn.LayerNorm(self.seq_len)
        self.back_to_time = nn.Linear(d_model, self.seq_len, bias=False)

    def forward(self, h: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        if D != self.num_variables:
            raise ValueError(
                f"CrossVariableSelfAttentionLayer expects num_variables={self.num_variables} but got D={D}"
            )
        if L != self.seq_len:
            raise ValueError(
                f"CrossVariableSelfAttentionLayer expects fixed seq_len={self.seq_len} but got L={L}."
            )

        Xtrans = h.transpose(1, 2)
        valid_mask = (~padding_mask).unsqueeze(1).to(h.dtype)
        Xtrans = Xtrans * valid_mask

        Q = self.WQv(Xtrans)
        K = self.WKv(Xtrans)
        V = self.WVv(Xtrans)

        Q = Q.view(B, self.num_variables, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, self.num_variables, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, self.num_variables, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, self.num_variables, self.num_heads * self.d_k)
        VMul = self.WOv(out)
        X_time = self.back_to_time(VMul)
        X_time = self.time_norm(X_time)
        h_var = X_time.transpose(1, 2)
        return h_var


class BiLSTM(nn.Module):
    """序列分支 - 先Attention后LSTM"""
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
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.self_attention = SelfAttentionLayer(
            hidden_size=embedding_dim,
            num_heads=8
        )

        self.cross_variable_attention = AdaptiveCrossVariableSelfAttentionLayer(
            num_variables=embedding_dim,
            seq_len=self.max_len,
            d_model=cv_d_model,
            num_heads=cv_heads,
        )

        self.fuse = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fuse_norm = nn.LayerNorm(embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_output_dim = hidden_size * 2

    def forward(self, inputs, x_len):
        B, L_real, D = inputs.shape
        device = inputs.device

        if L_real < self.max_len:
            pad_len = self.max_len - L_real
            pad_tensor = torch.zeros(B, pad_len, D, device=device, dtype=inputs.dtype)
            inputs = torch.cat([inputs, pad_tensor], dim=1)
        else:
            inputs = inputs[:, :self.max_len, :]

        padding_mask = (torch.arange(self.max_len, device=device).unsqueeze(0) 
                        >= x_len.unsqueeze(1).to(device))

        h_time = self.self_attention(inputs, key_padding_mask=padding_mask)
        h_var = self.cross_variable_attention(inputs, padding_mask=padding_mask)

        h_time_weighted = self.alpha * h_time
        h_var_weighted = self.beta * h_var

        h_cat = torch.cat([h_time_weighted, h_var_weighted], dim=-1)
        x_fused = self.fuse(h_cat)
        x_attended = self.fuse_norm(inputs + x_fused)

        x_len_cpu = x_len.cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            x_attended, x_len_cpu,
            batch_first=True,
            enforce_sorted=False
        )
        
        output, _ = self.lstm(packed)
        
        x, lengths = nn.utils.rnn.pad_packed_sequence(
            output, 
            batch_first=True, 
            total_length=self.max_len
        )

        mask = torch.arange(x.size(1), device=x.device)[None, :] < x_len[:, None].to(x.device)
        mask_expanded = mask.unsqueeze(-1).float()
        
        representations = (x * mask_expanded).sum(1) / (x_len[:, None].to(x.device) + 1e-8)
        
        return representations, None


class MyLSTMATT(nn.Module):
    """主模型: 先Attention后BiLSTM + MoE"""

    def __init__(
        self,
        features_dic,
        class_num=2,
        engine_dim=130,  # Weibo特征总维度
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

        bilstm_output_dim = hidden_dim * 2
        moe_output_dim = 128
        total_input_dim = bilstm_output_dim + moe_output_dim

        self.fc_1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, class_num)

        self.historic_model = BiLSTM(
            self.embedding_dim,
            self.hidden_dim,
            lstm_layer,
            max_len=self.max_len,
            cv_d_model=cv_d_model,
            cv_heads=cv_heads,
        )
        
        self.moe = BS.TwoLayerMoE(
            input_dim=self.engine_dim,
            mid_dim=128,
            output_dim=128,
            num_experts_layer1=4,
            num_experts_layer2=4,
            k1=3,
            k2=3
        )

    def get_pred(self, bert_feat, features):
        moe_out = self.moe(features)
        fused = torch.cat((bert_feat, moe_out), dim=1)
        feat = self.fc_1(fused)
        logits = self.fc_2(feat)
        return logits

    def forward(self, tweets, lengths, labels, features):
        h, _ = self.historic_model(tweets, lengths)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logits = self.get_pred(h, features)
        return logits


def focal_loss(logits, labels, class_weights=None, alpha=0.25, gamma=2.0, num_classes=2):
    """带类别权重的Focal Loss - 适配二分类"""
    if class_weights is not None:
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
    else:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

    pt = torch.exp(-ce_loss)

    if isinstance(alpha, (list, np.ndarray, torch.Tensor)):
        alpha_t = torch.tensor(alpha, device=logits.device)[labels]
        focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss
    else:
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    return focal_loss.mean()


def read_weibo_embeddings(embeddings_path):
    """
    读取Weibo BERT embeddings数据
    
    Args:
        embeddings_path: embeddings文件路径(pkl格式)
    
    Returns:
        embeddings列表,每个元素包含 {'label': ..., 'embeddings': ...}
    """
    with open(embeddings_path, 'rb') as f:
        embeddings = pkl.load(f)
    return embeddings


def train(args):
    # 读取Weibo数据
    print("正在加载Weibo数据集...")
    bert_embeddings = read_weibo_embeddings(args.data_embeddings)
    
    labels = []
    posts = []
    for i in range(len(bert_embeddings)):
        labels.append(bert_embeddings[i]['label'])
        posts.append(bert_embeddings[i]['embeddings'])

    features = pd.read_csv(args.data_features)

    # Weibo特征字典: POS(57) + TFIDF(50) + NRC(10) + SUI(13) = 130
    features_dic = {
        'pos': 57,
        'tfidf': 50,
        'nrc': 10,
        'sui': 13
    }

    features_dim = features.shape[1]
    print(f"特征维度: {features_dim} (预期130维)")
    
    labels = pd.DataFrame(labels, columns=['labels'])

    # 计算类别权重 - 二分类
    class_counts = labels['labels'].value_counts().sort_index().values
    total = len(labels)
    raw_weights = total / (len(class_counts) * class_counts)
    
    # 对数平滑
    class_weights = np.log1p(raw_weights)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"类别分布: {class_counts}")
    print(f"类别权重: {class_weights}")

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    features_labels = pd.concat([features, labels], axis=1)

    # 划分数据集: 80% train, 10% val, 10% test
    train_data, test_data, train_labels, test_labels = train_test_split(
        posts, features_labels, test_size=0.2,
        random_state=args.seed,
        stratify=features_labels['labels'].values
    )
    
    test_data, val_data, test_labels, val_labels = train_test_split(
        test_data, test_labels, test_size=0.5,
        random_state=args.seed,
        stratify=test_labels['labels'].values
    )

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    # 创建Dataset
    train_dataset = WeiboDataset(train_labels, train_data, days=args.max_len)
    val_dataset = WeiboDataset(val_labels, val_data, days=args.max_len)
    test_dataset = WeiboDataset(test_labels, test_data, days=args.max_len)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_weibo)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_weibo)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_weibo)

    # 初始化模型
    model = MyLSTMATT(
        features_dic=features_dic,
        class_num=args.classnum,
        engine_dim=features_dim,
        embedding_dim=args.embed_size,
        hidden_dim=args.hidden_size,
        lstm_layer=2,
        max_len=args.max_len,
        cv_d_model=args.cv_d_model,
        cv_heads=args.cv_heads
    )
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=1e-6
    )

    # 早停机制
    patience = args.patience
    best_f1 = 0
    early_stop_counter = 0
    model_save_path = args.save_path

    # 确保保存目录存在（避免因为目录调整/运行目录不同导致保存失败）
    save_dir = os.path.dirname(model_save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if args.use_pretrain:
        print("使用预训练模型")
        model.load_state_dict(torch.load(model_save_path))
    else:
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch_idx, (labels, tweets, lengths, features) in enumerate(pbar):
                labels = labels.to(device)
                tweets = tweets.to(device)
                features = features.to(device)
                optimizer.zero_grad()

                outputs = model(tweets, lengths, labels, features)

                classification_loss = F.cross_entropy(outputs, labels)

                classification_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += classification_loss.item()

                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{classification_loss.item():.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
            
            # 验证
            model.eval()
            val_preds_list = []
            val_labels_list = []
            val_loss = 0.0
            
            with torch.no_grad():
                for labels, tweets, lengths, features in val_loader:
                    labels = labels.to(device)
                    tweets = tweets.to(device)
                    features = features.to(device)
                    outputs = model(tweets, lengths, labels, features)
                    
                    classification_loss = F.cross_entropy(outputs, labels)

                    val_loss += classification_loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_preds_list.extend(preds.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_preds = np.array(val_preds_list)
            val_labels = np.array(val_labels_list)

            accuracy, precision, recall, f1 = utils.binary_metrics(val_preds_list, val_labels_list)

            print(f"\nEpoch {epoch+1} - Val Loss: {val_loss/len(val_loader):.4f}, "
                f"Acc: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                early_stop_counter = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                early_stop_counter += 1
            
            # torch.save(model.state_dict(), model_save_path)

            if early_stop_counter >= patience:
                print(f"早停触发! 最佳F1: {best_f1:.4f}")
                break

    # 测试
    print("\n开始测试...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    test_preds_list = []
    test_labels_list = []
    test_loss = 0.0

    with torch.no_grad():
        for labels, tweets, lengths, features in test_loader:
            labels = labels.to(device)
            tweets = tweets.to(device)
            features = features.to(device)
            outputs = model(tweets, lengths, labels, features)
            
            classification_loss = F.cross_entropy(outputs, labels)

            test_loss += classification_loss.item()
            preds = torch.argmax(outputs, dim=1)
            test_preds_list.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    fin_outputs = np.array(test_preds_list)
    fin_targets = np.array(test_labels_list)

    # 调用统一的指标函数
    accuracy, precision, recall, f1 = utils.binary_metrics(fin_outputs, fin_targets)
    
    print(f"\n总体指标:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("="*60)

    # 保存错误样本
    misclassified_mask = fin_outputs != fin_targets
    misclassified_indices = np.where(misclassified_mask)[0]

    bad_cases = {
        'Sample_Index': misclassified_indices,
        'True_Label': fin_targets[misclassified_mask],
        'Pred_Label': fin_outputs[misclassified_mask]
    }

    df_bad = pd.DataFrame(bad_cases)
    bad_cases_path = _default_result_path("results", "weibo", "weibo_bad_cases.csv")
    os.makedirs(os.path.dirname(bad_cases_path), exist_ok=True)
    df_bad.to_csv(bad_cases_path, index=False)
    print(f"\n错误样本已保存至: {bad_cases_path} ({len(df_bad)} 个)")


def set_seed(args):
    """设置随机种子确保可重复性"""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)


def main():
    args = parse_args()
    set_seed(args)
    train(args)


if __name__ == '__main__':
    main()