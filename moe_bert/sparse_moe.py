import torch
import torch.nn as nn

from expert import Expert
from gate import Gate
from utils.config import MoEConfig


class MoE(nn.Module):
    def __init__(self,
                 config: MoEConfig
                 ) -> None:
        super(MoE, self).__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([
            Expert(config.expert_config) for _ in range(config.num_experts)
        ])
        self.gate = Gate(config.gate_config)
        self.jitter_noise = config.jitter_noise

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Add jitter noise to the input for address expert load balancing
        if self.jitter_noise > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        batch_size, sequence_length, hidden_dim = x.size()
        # reshape input so you get the (tokens, embeddings of the tokens) because tokens are caclulated
        # independently of each other
        x = x.view(-1, hidden_dim)
        # get experts weights and selected experts for each token
        routing_topk_weights, routing_weights, selected_experts, routing_logits = self.gate(x)

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

            current_state = x[None, tokens_for_current_expert].reshape(-1, hidden_dim)
            current_hidden_states = (expert_layer(current_state) *
                                     routing_topk_weights[
                                         tokens_for_current_expert, expert_pos_in_selected_experts, None])

            # Add the computed hidden states to the final hidden states tensor (but cleverly, inplace)
            # to final_hidden_states at dimension 0, at the indices of tokens_for_current_expert the
            # current_hidden_states tensor (meaning the weighted hidden states for the current  expert)
            # equivalent to final_hidden_states[tokens_for_current_expert] += current_hidden_states
            # but more efficient
            final_hidden_states.index_add_(0, tokens_for_current_expert, current_hidden_states.to(x.dtype))

        # reshape back to the original shape (batch_size, sequence_length, hidden_dim) and return
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, routing_weights
