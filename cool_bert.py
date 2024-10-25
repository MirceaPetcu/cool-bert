import torch
import torch.nn as nn
from attention import MultiHeadAttention
from input_embeddings import InputEmbeddings
from encoder import EncoderBlock


class CoolBert(nn.Module):
    def __init__(self,
                 task: str = 'mlm',
                 num_classes: int = 1,
                 vocab_size: int = 1000,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.0,
                 num_layers: int = 6
                 ) -> None:
        super(CoolBert, self).__init__()
        assert task in {'mlm', 'nsp', 'custom'}, \
            "Task must be 'mlm', 'nsp' or 'custom' (for fine-tuning on a custom task)"
        self.task = task
        self.embeddings = InputEmbeddings(vocab_size, d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(dropout, d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        if task == 'mlm':
            self.lm_head = nn.Linear(d_model, vocab_size)
        elif task == 'nsp':
            self.lm_head = nn.Linear(d_model, 2)
        elif task == 'custom':
            self.lm_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        if self.task == 'custom':
            cls_token_output = x[:, 0, :]
            x = self.lm_head(cls_token_output)
        else:
            x = self.lm_head(x)
        return x


if __name__ == "__main__":
    model = CoolBert(task='custom', num_classes=1).to("cuda")
    x = torch.randint(0, 1000, (16, 54)).to("cuda")
    y = model(x)
    print(y.size())