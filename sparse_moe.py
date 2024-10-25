import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    def __init__(self, num_experts_per_token: int = 2, num_experts: int = 8, d_model: int = 512) -> None:
        super(Gate, self).__init__()
        self.num_experts_per_token = num_experts_per_token
        self.linear = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.linear(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_token, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights, selected_experts


class Expert(nn.Module):
    def __init__(self, d_model: int = 512, d_expert: int = 64) -> None:
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(d_model, d_expert)
        self.fc2 = nn.Linear(d_expert, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MoEFFN(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 d_expert: int = 64,
                 num_experts: int = 8,
                 num_experts_per_token: int = 2) -> None:
        super(MoEFFN, self).__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.experts = nn.ModuleList([
            Expert(d_model, d_expert) for _ in range(num_experts)
        ])
        self.gate = Gate(num_experts_per_token, num_experts, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = x.size()
        # reshape input so you get the (tokens, embeddings of the tokens) because tokens are caclulated
        # independently of each other
        x = x.view(-1, hidden_dim)
        # get experts weights and selected experts for each token
        routing_weights, selected_experts = self.gate(x)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device
        )

        # One hot encode the for expert idx
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Get the position of the expert in the
            # selected experts tensor and the tokens assigned to the current expert
            # torch where --> returns indices for each dimension where the condition is True
            expert_pos_in_selected_experts, tokens_for_current_expert = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # I think the first line is redundant but whatever
            current_state = x[None, tokens_for_current_expert].reshape(-1, hidden_dim)
            current_hidden_states = (expert_layer(current_state) *
                                     routing_weights[tokens_for_current_expert, expert_pos_in_selected_experts, None])

            # Add the computed hidden states to the final hidden states tensor (but cleverly, inplace)
            # to final_hidden_states at dimension 0, at the indices of tokens_for_current_expert the
            # current_hidden_states tensor (meaning the weighted hidden states for the current  expert)
            # equivalent to final_hidden_states[tokens_for_current_expert] += current_hidden_states
            # but more efficient
            final_hidden_states.index_add_(0, tokens_for_current_expert, current_hidden_states.to(x.dtype))
        # reshape back to the original shape (batch_size, sequence_length, hidden_dim) and return
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states
