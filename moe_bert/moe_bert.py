import torch
import torch.nn as nn

from encoder import EncoderBlock
from input_embeddings import InputEmbeddings
from utils.config import ModelConfig


class MoeBert(nn.Module):
    def __init__(self,
                 config: ModelConfig,
                 ) -> None:
        super(MoeBert, self).__init__()
        assert config.task in {'mlm', 'nsp', 'custom'}, \
            "Task must be 'mlm', 'nsp' or 'custom' (for fine-tuning on a downstream task)"
        self.task = config.task
        self.embeddings = InputEmbeddings(config.embedd_config)
        self.layers = nn.ModuleList([
            EncoderBlock(config.encoder_block_config) for _ in range(config.num_layers)
        ])
        if config.task == 'mlm':
            self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        elif config.task == 'nsp':
            self.lm_head = nn.Linear(config.d_model, 2)
        elif config.task == 'custom':
            self.lm_head = nn.Linear(config.d_model, config.num_classes)
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embeddings(x)
        routing_weights_list = []
        for layer in self.layers:
            x, routing_weight = layer(x)
            routing_weights_list.append(routing_weight)

        if self.task == 'custom':
            cls_token_output = x[:, 0, :]
            x = self.lm_head(cls_token_output)
        else:
            x = self.lm_head(x)

        return x, routing_weights_list
