import unittest
from dataclasses import dataclass
import torch

from moe_bert.moe_bert import MoeBert
from utils.losses import mlm_loss, balance_loss_switch_transformer, total_loss
from utils.config import (GateConfig,
                          ExpertConfig,
                          MoEConfig,
                          AttentionConfig,
                          EncoderBlockConfig,
                          EmbeddConfig,
                          ModelConfig)


class MoEBertUnitTests(unittest.TestCase):
    @staticmethod
    def get_config():
        expert_config = ExpertConfig(
            d_model=512,
            num_experts=8
        )
        gate_config = GateConfig(
            num_experts_per_token=4,
            num_experts=8,
            d_model=512
        )
        moe_config = MoEConfig(
            gate_config=gate_config,
            expert_config=expert_config,
            num_experts=8,
            jitter_noise=0.1
        )
        attention_config = AttentionConfig(
            pe='sinusoidal',
            num_heads=8,
            d_model=512
        )
        encoder_block_config = EncoderBlockConfig(
            attention_config=attention_config,
            moe_config=moe_config,
            dropout=0.1,
            d_model=512
        )
        embedd_config = EmbeddConfig(
            vocab_size=30_000,
            embedding_dim=512,
            padding_idx=0,
            scale_grad_by_freq=True,
            norm_type=2.0
        )
        config = ModelConfig(
            encoder_block_config=encoder_block_config,
            embedd_config=embedd_config,
            task='mlm',
            num_classes=2,
            vocab_size=30_000,
            d_model=512,
            num_layers=6
        )
        return config

    def test_instantiation(self):
        config = self.get_config()
        model = MoeBert(config)
        self.assertIsInstance(model, MoeBert)

    def test_inference(self):
        config = self.get_config()
        model = (MoeBert(config).
                 to('cuda' if torch.cuda.is_available() else 'cpu'))
        x = torch.randint(low=0, high=29999, size=(8, 256)).to('cuda' if torch.cuda.is_available() else 'cpu')
        output, routing_weights_list = model(x)
        self.assertEqual(output.size(), torch.Size([8, 256, 30000]))
        self.assertEqual(len(routing_weights_list), 6)
        self.assertEqual(routing_weights_list[0].size(), torch.Size([8 * 256, 8]))

    def test_backward(self):
        config = self.get_config()
        model = (MoeBert(config).
                 to('cuda' if torch.cuda.is_available() else 'cpu'))
        x = torch.randint(low=0, high=29999, size=(8, 256)).to('cuda' if torch.cuda.is_available() else 'cpu')
        y = torch.randint(low=0, high=30000, size=(8, 256)).to('cuda' if torch.cuda.is_available() else 'cpu')
        output, routing_weights_list = model(x)
        loss = mlm_loss(output, y, torch.nn.CrossEntropyLoss(ignore_index=-100))
        loss.backward()
        self.assertEqual(loss.shape, torch.Size([]))

    def test_losses(self):
        routing_weights_all_layers = tuple(
            [torch.randn((2 * 128, 8), requires_grad=True).softmax(dim=-1) for _ in range(6)])
        k = 2
        alpha = 1e-2
        loss = balance_loss_switch_transformer(routing_weights_all_layers, k, alpha)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.requires_grad, True)

        output = torch.randn((8, 256, 30000), requires_grad=True)
        labels = torch.randint(low=0, high=30000, size=(8, 256))
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        @dataclass
        class Args:
            balance_loss = True
            alpha = 1e-2

        args = Args()
        loss, mlm_loss_value, balancing_loss = total_loss(output, labels, criterion, routing_weights_all_layers, args)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(mlm_loss_value.shape, torch.Size([]))
        self.assertEqual(balancing_loss.shape, torch.Size([]))
        self.assertEqual(loss.requires_grad, True)
        self.assertEqual(mlm_loss_value.requires_grad, True)
        self.assertEqual(balancing_loss.requires_grad, True)
        self.assertNotEquals(loss.grad_fn, None)
        self.assertNotEquals(mlm_loss_value.grad_fn, None)
        self.assertNotEquals(balancing_loss.grad_fn, None)


if __name__ == '__main__':
    unittest.main()
