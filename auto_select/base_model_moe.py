import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torchfm.layer import CrossNetwork, FeaturesEmbedding


class PWLayer(nn.Module):
    """每个专家就是一个简单的线性层，但有中心化预处理。Expert module"""

    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)  # 中心化（类似批归一化） 线性变换
        # #引入非线性
        # return F.relu(self.lin(self.dropout(x) - self.bias))


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts Layer with Load Balancing
    - Top-K Noisy Gating
    - Load balancing loss to prevent expert collapse
    """

    def __init__(self, input_dim, output_dim, num_experts=4, k=2, dropout=0.2, noise=True):
        super().__init__()

        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.noisy_gating = noise

        # Expert modules
        self.experts = nn.ModuleList([
            PWLayer(input_dim, output_dim, dropout) for _ in range(num_experts)
        ])  # 创建num_experts个专家

        # Gating network parameters
        self.w_gate = nn.Parameter(torch.empty(input_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.empty(input_dim, num_experts), requires_grad=True)

        # Initialize weights
        nn.init.xavier_normal_(self.w_gate)  # 门控网络权重
        nn.init.xavier_normal_(self.w_noise)  # 噪声权重

        # For load balancing loss computation
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_counts', torch.tensor(0.0))

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        Noisy Top-K Gating with load balancing
        Returns: gates, top_k_indices, load_balancing_loss
        """
        # 1. 计算基础门控分数
        clean_logits = x @ self.w_gate  # (B, num_experts)

        # 2. 训练时添加噪声
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits

        # 3. 选择Top-K专家
        top_k_logits, top_k_indices = logits.topk(self.k, dim=-1)  # (B, k) 每个样本选择k个专家

        # 4. 创建稀疏门控（非Top-K的位置设为负无穷）
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(1, top_k_indices, top_k_logits)

        # 5. Softmax归一化（只在Top-K上）
        gates = F.softmax(sparse_logits, dim=-1)  # (B, num_experts)

        # 6. 计算负载均衡损失
        if train:
            load_loss = self._compute_load_balancing_loss(gates, top_k_indices)
        else:
            load_loss = torch.tensor(0.0, device=x.device)

        return gates, top_k_indices, load_loss

    def _compute_load_balancing_loss(self, gates, top_k_indices):
        """
        Compute load balancing loss to encourage uniform expert usage
        Uses importance loss + load loss from Switch Transformer paper
        """
        batch_size = gates.size(0)

        # 重要性：每个专家的总门控权重
        importance = gates.sum(dim=0)  # (num_experts,)

        # 负载：每个专家被选中的次数
        load = torch.zeros(self.num_experts, device=gates.device)
        for i in range(self.num_experts):
            load[i] = (top_k_indices == i).float().sum()

        # 归一化
        importance = importance / importance.sum()
        load = load / load.sum()

        # 计算变异系数平方（CV^2），越小表示分布越均匀
        cv_importance = (importance.std() / (importance.mean() + 1e-10)) ** 2
        cv_load = (load.std() / (load.mean() + 1e-10)) ** 2

        # 总负载均衡损失
        load_balancing_loss = cv_importance + cv_load

        return load_balancing_loss

    def forward(self, x):
        """
        Forward pass with load balancing
        Returns: output, load_balancing_loss
        """
        # 1. 计算门控权重
        gates, top_k_indices, load_loss = self.noisy_top_k_gating(x, self.training)

        # 2. 计算所有专家输出
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.num_experts)]
        expert_stack = torch.cat(expert_outputs, dim=-2)  # (B, num_experts, D)

        # 3. 加权求和（稀疏）
        weighted_output = gates.unsqueeze(-1) * expert_stack  # (B, num_experts, D)

        # 4. 最终输出
        output = weighted_output.sum(dim=-2)  # (B, D)

        return output, load_loss


class TwoLayerMoE(nn.Module):
    """
    Two-layer MoE with load balancing and residual connections
    """

    def __init__(self, input_dim, mid_dim, output_dim,
                 num_experts_layer1=4, num_experts_layer2=4,
                 k1=2, k2=2, dropout=0.2, noise=True):
        super().__init__()

        self.layer1 = SparseMoELayer(
            input_dim=input_dim,
            output_dim=mid_dim,
            num_experts=num_experts_layer1,
            k=k1,
            dropout=dropout,
            noise=noise
        )

        self.layer2 = SparseMoELayer(
            input_dim=mid_dim,
            output_dim=output_dim,
            num_experts=num_experts_layer2,
            k=k2,
            dropout=dropout,
            noise=noise
        )

        self.act = nn.ReLU()

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(mid_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        # Residual projection if dimensions don't match
        self.use_residual = (input_dim == output_dim)
        if not self.use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        """
        Forward pass with residual connections
        Returns: output, total_load_balancing_loss
        """
        identity = x  # 残差连接

        # Layer 1
        x, load_loss1 = self.layer1(x)
        x = self.norm1(x)  # 层归一化
        x = self.act(x)  # ReLU激活

        # Layer 2
        x, load_loss2 = self.layer2(x)
        x = self.norm2(x)  # 层归一化

        # 残差连接
        if self.use_residual:
            x = x + identity
        elif self.residual_proj is not None:
            x = x + self.residual_proj(identity)

        # 总负载损失
        total_load_loss = load_loss1 + load_loss2

        return x, total_load_loss


# 三层设计
class ThreeLayerMoE(nn.Module):
    """
    Three-layer Sparse MoE with load balancing and residual connections
    """

    def __init__(self,
                 input_dim,
                 mid_dim1,
                 mid_dim2,
                 output_dim,
                 num_experts_layer1=4,
                 num_experts_layer2=6,
                 num_experts_layer3=4,
                 k1=2,
                 k2=2,
                 k3=2,
                 dropout=0.2,
                 noise=True):
        super().__init__()

        self.layer1 = SparseMoELayer(
            input_dim=input_dim,
            output_dim=mid_dim1,
            num_experts=num_experts_layer1,
            k=k1,
            dropout=dropout,
            noise=noise
        )

        self.layer2 = SparseMoELayer(
            input_dim=mid_dim1,
            output_dim=mid_dim2,
            num_experts=num_experts_layer2,
            k=k2,
            dropout=dropout,
            noise=noise
        )

        self.layer3 = SparseMoELayer(
            input_dim=mid_dim2,
            output_dim=output_dim,
            num_experts=num_experts_layer3,
            k=k3,
            dropout=dropout,
            noise=noise
        )

        self.act = nn.ReLU()

        self.norm1 = nn.LayerNorm(mid_dim1)
        self.norm2 = nn.LayerNorm(mid_dim2)
        self.norm3 = nn.LayerNorm(output_dim)

        # Residual
        self.use_residual = (input_dim == output_dim)
        if not self.use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        identity = x

        x, load1 = self.layer1(x)
        x = self.norm1(x)
        x = self.act(x)

        x, load2 = self.layer2(x)
        x = self.norm2(x)
        x = self.act(x)

        x, load3 = self.layer3(x)
        x = self.norm3(x)

        if self.use_residual:
            x = x + identity
        else:
            x = x + self.residual_proj(identity)

        total_load_loss = load1 + load2 + load3
        return x, total_load_loss


# Legacy classes for compatibility (not used in new architecture)
class ExpertMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def forward(self, x):
        return self.net(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, expert_hidden=128, k=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.experts = nn.ModuleList(
            [ExpertMLP(input_dim, expert_hidden, output_dim, dropout=dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        nn.init.xavier_normal_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_scores = torch.softmax(gate_logits, dim=1)
        expert_outs = []
        for e in self.experts:
            expert_outs.append(e(x).unsqueeze(2))
        expert_stack = torch.cat(expert_outs, dim=2)
        weighted = torch.bmm(expert_stack, gate_scores.unsqueeze(2)).squeeze(2)
        return weighted

#一致性设计
class UCR(nn.Module):
    def __init__(self, temp=0.1, dropout_prob=0.1) -> None:
        super().__init__()
        self.temp = temp
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        # 构造正样本：对同一输入应用随机 Dropout 扰动
        inputs_enhanced = self.dropout(inputs)

        # 计算余弦相似度矩阵 (Batch_size, Batch_size)
        similarity = F.cosine_similarity(inputs.unsqueeze(1), inputs_enhanced.unsqueeze(0), dim=-1)
        sim_tau = similarity / self.temp

        # 对角线元素为正样本对，其余为负样本
        # InfoNCE Loss 实现
        logits = sim_tau
        labels = torch.arange(inputs.size(0)).to(inputs.device)
        return F.cross_entropy(logits, labels)
