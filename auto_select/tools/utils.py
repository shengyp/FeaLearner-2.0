import csv
import numpy as np
import pandas as pd
import datetime
import os
from pathlib import Path
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm,time

_TOOLS_DIR = Path(__file__).resolve().parent          # .../auto_select/tools
_AUTO_SELECT_DIR = _TOOLS_DIR.parent                  # .../auto_select
_REPO_ROOT = _AUTO_SELECT_DIR.parent                  # 仓库根目录

PATH = str(_TOOLS_DIR)  # 保持兼容：部分旧代码可能依赖 PATH 为字符串

def _first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    attempted = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(
        "未找到所需数据文件。已尝试以下路径：\n" + attempted +
        "\n如果你调整过目录结构，请把文件放到上述任一路径，或在调用方传入正确路径。"
    )


# 得到data文件夹的路径
def get_data_path():
    candidates = [
        _REPO_ROOT / "data" / "reddit_500.csv",
        _REPO_ROOT / "data_analy" / "reddit_500.csv",
        # 兼容旧相对路径写法（以 tools 目录为基准）
        _TOOLS_DIR / ".." / ".." / "data" / "reddit_500.csv",
    ]
    return str(_first_existing([c.resolve() for c in candidates]))


def get_s_d():
    candidates = [
        _REPO_ROOT / "data" / "Reddit_Suicide_Dictionary.csv",
        _REPO_ROOT / "data_analy" / "Reddit_Suicide_Dictionary.csv",
        _REPO_ROOT / "data_analy" / "tools_dataset" / "Reddit_Suicide_Dictionary.csv",
        # 兼容旧相对路径写法（以 tools 目录为基准）
        _TOOLS_DIR / ".." / ".." / "data" / "Reddit_Suicide_Dictionary.csv",
    ]
    return str(_first_existing([c.resolve() for c in candidates]))

# 读取数据
def load_df(dataset_name):
    if dataset_name in ['reddit_500', 'reddit']:
        file_name = get_data_path()
        df = pd.read_csv(file_name)
        return df
    else:
        raise ValueError("Error: unrecognized dataset")

# 加载自杀字典数据
def load_SD():
    file_name = get_s_d()
    df = pd.read_csv(file_name)
    return df



# 定义损失函数



# 这个函数的目的是将真实标签转化为软变迁
# 生成一个定制化的损失函数，它考虑到了类别之间的距离或者差异,有序数回归问题
def true_metric_loss(true, no_of_classes, scale=1):
    # true: 真实的标签向量，其中每个元素代表一个样本的类别标签。
    # no_of_classes: 数据集中总的类别数量。
    # scale: 一个可选的缩放因子，默认值为1，用于调节类别之间差异的影响。
    batch_size = true.size(0) # 批次中样本的数量。

    true = true.view(batch_size,1) # 将真实标签向量转换为(batch_size, 1)的形状。
    # 将真实标签向量转换为LongTensor类型，并在列方向上重复no_of_classes次，形成一个矩阵，然后转换为浮点数。这个矩阵的每一行都是相同的真实标签。
    # true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
    true_labels = true.clone().detach().to(device='cuda', dtype=torch.long) \
                    .repeat(1, no_of_classes) \
                    .float()
    # class_labels = torch.arange(no_of_classes).float().cuda()：生成一个从0到no_of_classes-1的连续整数向量，然后转换为浮点数并移动到CUDA设备上。
    class_labels = torch.arange(no_of_classes).float().cuda()
    # 计算class_labels向量和true_labels矩阵之间的绝对差值，然后乘以缩放因子scale
    phi = (scale * torch.abs(class_labels - true_labels)).cuda()
    # 对phi矩阵的每一行进行softmax操作，得到一个概率分布。
    y = nn.Softmax(dim=1)(-phi) # 用-phi是为了让距离较小（即类别接近真实标签）的类别有较高的概率值。
    return y


def loss_function(output, labels, loss_type, expt_type=5, scale=1.8):
    targets = true_metric_loss(labels, expt_type, scale)
    return torch.sum(- targets * F.log_softmax(output, -1), -1).mean()


def gr_metrics(op, t):
    op = np.array(op)  # 确保 op 是 NumPy 数组
    t = np.array(t)    # 确保 t 是 NumPy 数组
    # 
    TP = (op==t).sum()
    # FN（False Negative）：实际为正例，但预测为负例的数量。
    FN = (t>op).sum()
    # FP（False Positive）：实际为负例，但预测为正例的数量。
    FP = (t<op).sum()

    GP = TP/(TP + FP)
    GR = TP/(TP + FN)

    FS = 2 * GP * GR / (GP + GR)

    # 过估计错误率（OE, Overestimation Error）：

    OE = (t-op > 1).sum() # 计算预测值与真实标签之差大于1的次数，即模型严重过估计的情况。
    OE = OE / op.shape[0] # 过估计错误率。

    return GP, GR, FS, OE




def test(model, data_loader, device):
    model.eval()
    targets, predicts, infer_time  = list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            start = time.time()
            outputs = model(fields)
            _, y = torch.max(outputs.data, 1)
            infer_cost = time.time() - start
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            infer_time.append(infer_cost)

    fin_outputs =  np.hstack(predicts)
    fin_targets =  np.hstack(targets)
    m = gr_metrics(fin_outputs, fin_targets)
    return m[0],m[1],m[2]


# 根据指定的分布（dist_values）从给定的数据框（df）中分割出一个测试集，并且返回剩余的数据作为训练集
def splits(df, dist_values):
    # dist_values：一个列表，指定了每个类别（标签）在测试集中应该有多少样本。

    # 使用sample(frac=1)函数随机打乱df中的行，其中frac=1意味着选择全部数据（但顺序被随机化）。
    # reset_index(drop=True)用于重置索引并删除旧的索引列。

    df = df.sample(frac=1).reset_index(drop=True)
    # 然后，按照label列的值对数据进行排序，并再次重置索引。
    df = df.sort_values(by='Label').reset_index(drop=True)

    df_test = df[df['Label']==0][0:dist_values[0]].reset_index(drop=True)

    for i in range(1,5):
        df_test = df_test.append(df[df['Label']==i][0:dist_values[i]], ignore_index=True)

    # 通过一个循环，对每个标签，从原数据df中移除已经加入测试集的样本：
    for i in range(5):
        # 使用drop方法删除这些行，并通过inplace=True直接在原数据框上进行修改。
        df.drop(df[df['Label']==i].index[0:dist_values[i]], inplace=True)
    
    df = df.reset_index(drop=True)
    return df, df_test


def binary_metrics(op, t):
    """
    针对二分类任务（如 Weibo, SuicidEmoji）的评价指标函数。
    计算标准 Accuracy, Precision, Recall 和 F1-score。
    """
    op = np.array(op)
    t = np.array(t)

    # 1. 计算 Accuracy (与 gr_metrics 中的 TP/(TP+FN+FP) 逻辑等价)
    acc = metrics.accuracy_score(t, op)

    # 2. 计算标准的 Precision, Recall, F1-score (针对正类，即标签 1)
    # 使用 zero_division=0 处理分母为 0 的情况
    precision = metrics.precision_score(t, op, average='binary', zero_division=0)
    recall = metrics.recall_score(t, op, average='binary', zero_division=0)
    f1 = metrics.f1_score(t, op, average='binary', zero_division=0)

    # 返回四元组以保持与 gr_metrics 的返回结构一致：
    # (Accuracy, Precision, Recall, F1)
    return acc, precision, recall, f1


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算 softmax 概率
        logp = F.log_softmax(inputs, dim=1)
        p = torch.exp(logp)
        
        # 将 targets 转换为 one-hot 编码
        targets = torch.zeros_like(logp).scatter_(1, targets.unsqueeze(1), 1)
        
        # 计算每个类别的权重
        if self.weight is not None:
            self.weight = self.weight
            logp = logp * self.weight
        
        # 计算 Focal Loss
        loss = -1 * targets * (1 - p) ** self.gamma * logp  # 核心 focal loss 计算公式
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
