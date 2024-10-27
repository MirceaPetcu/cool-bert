import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import GateConfig


class Gate(nn.Module):
    def __init__(self,
                 config: GateConfig) -> None:
        super(Gate, self).__init__()
        self.num_experts_per_token = config.num_experts_per_token
        self.linear = nn.Linear(config.d_model, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.linear(x)  # (batch_size x seq_len, num_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # (batch_size x seq_len, num_experts) normalized
        routing_topk_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_token, dim=-1)
        # (batch_size x seq_len, k); first probas non-normalized, second indices of selected experts
        routing_topk_weights /= routing_weights.sum(dim=-1, keepdim=True)  # (batch_size x seq_len, k) normalized
        return routing_topk_weights, routing_weights, selected_experts, router_logits
