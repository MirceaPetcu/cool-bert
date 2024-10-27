import torch

from utils.config import EmbeddConfig


class InputEmbeddings(torch.nn.Module):
    def __init__(self,
                 config: EmbeddConfig) -> None:
        super(InputEmbeddings, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=config.embedding_dim,
                                            padding_idx=config.padding_idx,
                                            scale_grad_by_freq=config.scale_grad_by_freq,
                                            max_norm=config.max_norm,
                                            norm_type=config.norm_type)
        self.embedding_dim = config.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (batch_size, seq_len) with dtype=torch.long. It outputs (batch_size, seq_len, embedding_dim) """
        assert x.max() < self.vocab_size, \
            f"Token ID  is greater than vocab_size: {x.max(dim=1)} > {self.vocab_size}"
        return self.embedding(x)
