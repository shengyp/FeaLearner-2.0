import torch.nn.functional as F
import torch.nn as nn
import torch

class PWLayer(nn.Module):
    """专家模块：线性层 + 中心化预处理 (无 Dropout)"""
    def __init__(self, input_size, output_size):
        super(PWLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(x - self.bias) 

class SparseMoELayer(nn.Module):
    """稀疏混合专家层 (无负载均衡 Loss，无 Dropout)"""
    def __init__(self, input_dim, output_dim, num_experts=4, k=2, noise=True):
        super().__init__()

        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.noisy_gating = noise

        self.experts = nn.ModuleList([
            PWLayer(input_dim, output_dim) for _ in range(num_experts)
        ])

        self.w_gate = nn.Parameter(torch.empty(input_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.empty(input_dim, num_experts), requires_grad=True)

        nn.init.xavier_normal_(self.w_gate)
        nn.init.xavier_normal_(self.w_noise)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
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
    """双层 MoE 结构 (无 Dropout)"""
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
        if not self.use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.norm2(x)

        if self.use_residual:
            x = x + identity
        elif self.residual_proj is not None:
            x = x + self.residual_proj(identity)
        return x
