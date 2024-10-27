import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import ExpertConfig


class Expert(nn.Module):
    def __init__(self,
                 config: ExpertConfig) -> None:
        super(Expert, self).__init__()
        d_expert = config.d_model // config.num_experts
        self.fc1 = nn.Linear(config.d_model, d_expert)
        self.fc2 = nn.Linear(d_expert, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
