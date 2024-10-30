import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from pyngrok import ngrok
from getpass import getpass

from config import (MoEConfig,
                    ModelConfig,
                    EncoderBlockConfig,
                    AttentionConfig,
                    EmbeddConfig,
                    GateConfig,
                    ExpertConfig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth_token", default='1')
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wd", type=float, default=0.001)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=0.03)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--balance_loss", type=bool, default=True)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--eval_steps", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_experts_per_token", type=int, default=2)
    parser.add_argument("--jitter_noise", type=float, default=0.1)
    parser.add_argument("--pe", type=str, default='rotary')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--scale_grad_by_freq", type=bool, default=True)
    parser.add_argument("--norm_type", type=float, default=2.0)

    return parser.parse_args()


def setup_tunneling(args):
    # IMP: please create a auth token from https://dashboard.ngrok.com/auth by creating an account.
    # the below auth ticket will not work for anyone re-running the notebook.

    # Terminate open tunnels if exist
    ngrok.kill()

    # Setting the authtoken (optional)
    # Get your authtoken from https://dashboard.ngrok.com/auth
    NGROK_AUTH_TOKEN = args.auth_token
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    # Open an HTTPs tunnel on port 5000 for http://localhost:5000
    ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
    print("MLflow Tracking UI:", ngrok_tunnel.public_url)


def get_model_config(args):
    expert_config = ExpertConfig(
        d_model=args.d_model,
        num_experts=args.num_experts
    )
    gate_config = GateConfig(
        num_experts_per_token=args.num_experts_per_token,
        num_experts=args.num_experts,
        d_model=args.d_model
    )
    moe_config = MoEConfig(
        gate_config=gate_config,
        expert_config=expert_config,
        num_experts=args.num_experts,
        jitter_noise=args.jitter_noise
    )
    attention_config = AttentionConfig(
        pe=args.pe,
        num_heads=args.num_heads,
        d_model=args.d_model
    )
    encoder_block_config = EncoderBlockConfig(
        attention_config=attention_config,
        moe_config=moe_config,
        dropout=args.dropout,
        d_model=args.d_model
    )
    embedd_config = EmbeddConfig(
        vocab_size=args.vocab_size,
        embedding_dim=args.d_model,
        padding_idx=args.padding_idx,
        scale_grad_by_freq=args.scale_grad_by_freq,
        norm_type=args.norm_type
    )
    config = ModelConfig(
        encoder_block_config=encoder_block_config,
        embedd_config=embedd_config,
        task='mlm',
        num_classes=args.num_classes,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers
    )

    return config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def tokenize_function(examples, tokenizer, args):
    return tokenizer(examples["text"],
                     return_tensors="pt",
                     truncation=True,
                     max_length=args.max_length,
                     padding='max_length')


def prepare_dataset(tokenizer, args):
    dataset = load_dataset('allenai/c4', 'en', streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10_000)
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer, args), batched=True, batch_size=10000,
                          remove_columns=["text", "timestamp", "url"])
    return dataset


def prepare_dataloaders(dataset, mlm_collator, args):
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   collate_fn=mlm_collator,
                                                   pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  collate_fn=mlm_collator,
                                                  pin_memory=True)
    return train_dataloader, eval_dataloader


def get_chosen_experts(routing_weights_list, k=2):
    num_layers = len(routing_weights_list)
    num_tokens, num_experts = routing_weights_list[0].size()
    tokens_dispatch = torch.zeros(num_experts, device=routing_weights_list[0].device)
    for routing_weights in routing_weights_list:
        _, selected_experts = torch.topk(routing_weights, k, dim=-1)
        # fi
        for i in range(num_experts):
            tokens_dispatch[i] += torch.sum(selected_experts == i)

    tokens_dispatch /= (num_tokens * num_layers * 2)
    return tokens_dispatch.cpu().numpy()
