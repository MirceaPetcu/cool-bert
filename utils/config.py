from dataclasses import dataclass
from typing import Optional


@dataclass
class GateConfig:
    num_experts_per_token: int = 2
    num_experts: int = 8
    d_model: int = 512


@dataclass
class ExpertConfig:
    d_model: int = 512
    num_experts: int = 8


@dataclass
class MoEConfig:
    gate_config: GateConfig
    expert_config: ExpertConfig
    num_experts: int = 8
    jitter_noise: float = 0.1


@dataclass
class AttentionConfig:
    pe: str = 'rotary'
    num_heads: int = 8
    d_model: int = 512


@dataclass
class EncoderBlockConfig:
    attention_config: AttentionConfig
    moe_config: MoEConfig
    dropout: float = 0.1
    d_model: int = 512


@dataclass
class EmbeddConfig:
    vocab_size: int = 30000
    embedding_dim: int = 512
    padding_idx: int = 0
    scale_grad_by_freq: bool = True
    max_norm: Optional[float] = None
    norm_type: float = 2.0


@dataclass
class ModelConfig:
    encoder_block_config: EncoderBlockConfig
    embedd_config: EmbeddConfig
    task: str = 'mlm'
    num_classes: int = 1
    vocab_size: int = 30000
    d_model: int = 512
    num_layers: int = 6
