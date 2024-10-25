import torch


class InputEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size: int,
                 embedding_dim: int,
                 padding_idx: int = 0,
                 scale_grad_by_freq: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.0,

                 ) -> None:
        super(InputEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=embedding_dim,
                                            padding_idx=padding_idx,
                                            scale_grad_by_freq=scale_grad_by_freq,
                                            max_norm=max_norm,
                                            norm_type=norm_type)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (batch_size, seq_len) with dtype=torch.long. It outputs (batch_size, seq_len, embedding_dim) """
        assert x.max() < self.vocab_size, \
            f"Input value is greater than vocab_size: {x.max(dim=1)} > {self.vocab_size}"
        return self.embedding(x)
