import torch
import torch.nn as nn
from attention import MultiHeadAttention
from input_embeddings import InputEmbeddings
from ffn import FFN
import torch.nn.functional as F
from sparse_moe import MoEFFN


class EncoderBlock(nn.Module):
    def __init__(self, dropout: float = 0.0,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(dropout, d_model, d_ff)
        self.sparse_moe = MoEFFN()
        self.rms_norm1 = nn.RMSNorm(512)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=512)
        self.rms_norm2 = nn.RMSNorm(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rms_norm1(x + self.dropout(self.attention(x, x, x)))
        x = self.rms_norm2(x + self.dropout(self.sparse_moe(x)))
        return x
