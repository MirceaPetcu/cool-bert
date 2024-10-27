import torch
import torch.nn as nn

from attention import MultiHeadAttention
from sparse_moe import MoE
from utils.config import EncoderBlockConfig


class EncoderBlock(nn.Module):
    def __init__(self,
                 config: EncoderBlockConfig,
                 ) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(config.attention_config)
        self.sparse_moe = MoE(config.moe_config)
        self.rms_norm1 = nn.RMSNorm(config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        self.rms_norm2 = nn.RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Residual connection after attention
        attn_output = self.attention(x, x, x)
        x = self.rms_norm1(x + self.dropout(attn_output))

        # Apply MoE and residual connection
        moe_output, routing_weights = self.sparse_moe(x)  # Unpack hidden states and routing logits
        x = self.rms_norm2(x + self.dropout(moe_output))  # Residual connection after MoE output

        return x, routing_weights  # Return both final hidden states and routing logits
