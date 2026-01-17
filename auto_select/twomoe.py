import torch.nn.functional as F
import torch.nn as nn
import torch

class PWLayer(nn.Module):
    """
    专家模块：线性层 + 中心化预处理
    每个专家负责对输入进行特定的特征变换。
    """
    def __init__(self, input_size, output_size):
        super(PWLayer, self).__init__()
        # 中心化参数：允许专家学习输入的平移偏置
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        # 核心投影层：不带偏置
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 权重初始化
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(x - self.bias) 

class SparseMoELayer(nn.Module):
    """
    稀疏混合专家层：负责计算门控权重并将输入路由给专家
    """
    def __init__(self, input_dim, output_dim, num_experts=4, k=2, noise=True):
        super().__init__()

        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.noisy_gating = noise

        # 初始化专家列表
        self.experts = nn.ModuleList([
            PWLayer(input_dim, output_dim) for _ in range(num_experts)
        ])
        # 路由权重：决定输入分配给哪个专家
        self.w_gate = nn.Parameter(torch.empty(input_dim, num_experts), requires_grad=True)
        # 噪声权重：用于计算动态噪声强度
        self.w_noise = nn.Parameter(torch.empty(input_dim, num_experts), requires_grad=True)

        nn.init.xavier_normal_(self.w_gate)
        nn.init.xavier_normal_(self.w_noise)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        计算 Top-K 门控权重，并在训练时引入噪声以平衡专家利用率
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            # 计算噪声的标准差，使用 softplus 确保为正数
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            # 在原始 Logits 上叠加高斯噪声
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits

        top_k_logits, top_k_indices = logits.topk(self.k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(1, top_k_indices, top_k_logits)
        gates = F.softmax(sparse_logits, dim=-1)
        return gates, top_k_indices

    def forward(self, x):
        gates, top_k_indices = self.noisy_top_k_gating(x, self.training)
        
        # 计算专家输出
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.num_experts)]
        expert_stack = torch.cat(expert_outputs, dim=-2)
        
        # 加权求和
        weighted_output = gates.unsqueeze(-1) * expert_stack
        output = weighted_output.sum(dim=-2)
        return output

class TwoLayerMoE(nn.Module):
    """
    双层 MoE 架构：由两个连续的稀疏 MoE 层组成
    """
    def __init__(self, input_dim, mid_dim, output_dim,
                 num_experts_layer1=4, num_experts_layer2=4,
                 k1=2, k2=2, noise=True):
        super().__init__()

        self.layer1 = SparseMoELayer(
            input_dim=input_dim,
            output_dim=mid_dim,
            num_experts=num_experts_layer1,
            k=k1,
            noise=noise
        )

        self.layer2 = SparseMoELayer(
            input_dim=mid_dim,
            output_dim=output_dim,
            num_experts=num_experts_layer2,
            k=k2,
            noise=noise
        )

        self.act = nn.ReLU()
        self.norm1 = nn.LayerNorm(mid_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        self.use_residual = (input_dim == output_dim)
        if not self.use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
    def forward(self, x):
        identity = x # 保存原始输入用于残差连接
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.norm2(x)

        # 残差融合
        if self.use_residual:
            x = x + identity  # 维度相同时直接相加
        elif self.residual_proj is not None:
            x = x + self.residual_proj(identity)  # 维度不同时通过投影层对齐后再加
        return x
