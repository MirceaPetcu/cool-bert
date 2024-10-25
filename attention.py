import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import DEVICE


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512,
                 num_heads: int = 8,
                 pe: str = 'rotary') -> None:
        super(MultiHeadAttention, self).__init__()
        assert pe in ['sinusoidal', 'rotary'], \
            "Positional encoding must be either 'sinusoidal' or 'rotary'"
        self.pe = pe
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)

    def split_input_in_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ Splits the last dimension of the input in self.num_heads pieces.
        Every piece (head) will operate on the on the entire sequence, but with a reduced dimension.
        The output will have the shape (batch_size, num_heads, seq_len, head_dimension)."""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def head_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes the scaled dot-product attention for the given query, key and value tensors."""

        # transpose the last two dimensions of the key for the matrix multiplication
        scores = q.matmul(k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        # softmax over the last dimension to get the weigths for every token
        attention_states = F.softmax(scores, dim=-1).matmul(v)
        return attention_states

    def concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenates the last two dimensions of the input."""
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    @torch.no_grad()
    def add_sinusoidal_position_embedding(self, x: torch.Tensor, theta: int = 10000) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        # Ensure d_model is even
        assert d_model % 2 == 0, \
            "Cannot apply sinusoidal position embedding to a model with an odd number of dimensions"

        # shape (seq_len, 1) (for broadcasting) -> 0, 1,2 ,.. seq_len-1
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()  # Shape: (seq_len, 1)
        # shape (d_model // 2) -> 0, 2, 4, 6, 8, ... d_model
        dim = torch.arange(0, d_model, 2, device=x.device).float()  # Shape: (d_model // 2)

        # shape; (d_model // 2)
        div_term = theta ** (dim / d_model)  # Shape: (d_model // 2)
        # shape: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model, device=x.device)
        # set sin to even indices (repeat the pos because i need it for every dimension of the word embedding)
        pe[:, 0::2] = torch.sin(pos.repeat(1, 256) / div_term.unsqueeze(0))
        pe[:, 1::2] = torch.cos(pos.repeat(1, 256) / div_term.unsqueeze(0))

        # repeat this for every batch because sin pe is not dependent on the semantic,
        # but with this implementation I can have different lengths for sequences
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)  # shape: (batch_size, seq_len, d_model)
        return x + pe

    @torch.no_grad()
    def get_sin_cos_for_rotation_matrix(self, x: torch.Tensor, base: float = 10000.0) \
            -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.size()
        # Ensure d_model is even
        assert d_model % 2 == 0, \
            "Cannot apply rotary position embedding to a model with an odd number of dimensions"
        # rotations matrix
        positions = torch.arange(1, seq_len + 1, device=x.device).unsqueeze(1)
        slice_i = torch.arange(0, d_model // 2, device=x.device)

        # theta = base (10000) ^ (-2i / d_model)
        theta = base ** (-2.0 * (slice_i.float()) / d_model)

        # m_theta = m (position) * theta --> shape: (seq_len, d_model // 2) --> cate un theta pentru
        # fiecare pereche de 2 pozitii, pe urma sin si cos pentru rotatie
        m_theta = positions * theta

        cos_values = torch.cos(m_theta)
        sin_values = torch.sin(m_theta)

        return cos_values, sin_values

    def add_rotary_position_embedding(self, x: torch.Tensor,
                                      rot_matrix: torch.Tensor,
                                      ) -> torch.Tensor:
        """Rotary position embedding for the given input tensor, but inefficient."""

        slice_i = torch.arange(0, x.size(-1) // 2, device=x.device)
        cos_values, sin_values = self.get_sin_cos_for_rotation_matrix(x)

        rot_matrix[:, 2 * slice_i, 2 * slice_i] = cos_values
        rot_matrix[:, 2 * slice_i, 2 * slice_i + 1] = -sin_values
        rot_matrix[:, 2 * slice_i + 1, 2 * slice_i] = sin_values
        rot_matrix[:, 2 * slice_i + 1, 2 * slice_i + 1] = cos_values

        x = (x.transpose(0, 1) @ rot_matrix).transpose(0, 1)

        return x

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        # make -x2, x1, -x4, x3, -x6, x5, ... to multiple with sin matrix rotation for efficiency
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @torch.no_grad()
    def add_rotary_position_embedding_efficient(self, x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
        cos_values, sin_values = self.get_sin_cos_for_rotation_matrix(x)
        # apply the rotation to the input (efficient implementation :) )
        x = (x * cos_values.repeat_interleave(2, dim=1) +
             self.rotate_half(x) * sin_values.repeat_interleave(2, dim=1))
        return x


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.pe == 'sinusoidal':
            # add position embedding
            q = self.add_sinusoidal_position_embedding(q)
            k = self.add_sinusoidal_position_embedding(k)
            v = self.add_sinusoidal_position_embedding(v)
        elif self.pe == 'rotary':
            q = self.add_rotary_position_embedding_efficient(q)
            k = self.add_rotary_position_embedding_efficient(k)

        Q = self.split_input_in_heads(self.W_q(q))
        K = self.split_input_in_heads(self.W_k(k))
        V = self.split_input_in_heads(self.W_v(v))

        attn_probs = self.head_attention(Q, K, V)
        # combine heads (and make continuous, because of the view and transpose)
        multi_head_attn_probs = self.concat_heads(attn_probs)

        # apply the output linear layer
        return self.W_o(multi_head_attn_probs)
