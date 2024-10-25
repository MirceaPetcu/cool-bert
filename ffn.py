import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, dropout: float = 0.0,
                 d_model: int = 512,
                 d_ff: int = 2048) -> None:
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
