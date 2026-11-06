import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from experiments.run_benchmark_experiment import config_from_args
from identitypfn.model import NanoERPFNModel


class PassThroughEncoder(torch.nn.Module):
    def forward(self, tokenized_cells):
        return tokenized_cells["cells"]


class NoRecordAttentionTests(unittest.TestCase):
    def test_other_record_cannot_change_pair_score(self):
        torch.manual_seed(0)
        model = NanoERPFNModel(
            embedding_size=8,
            text_embedding_size=8,
            num_attention_heads=2,
            mlp_hidden_size=16,
            num_layers=2,
            record_representation="mean_pool",
            adjacency_decoder="pair_mlp",
            cross_record_attention=False,
        ).eval()
        model.feature_encoder = PassThroughEncoder()
        cells = torch.randn(1, 3, 2, 8)
        changed = cells.clone()
        changed[:, 2] += 20

        with torch.no_grad():
            original_score = model({"cells": cells})[0, 0, 1]
            changed_score = model({"cells": changed})[0, 0, 1]

        self.assertEqual(original_score.item(), changed_score.item())

    def test_control_flag_rejects_attention_on_resume(self):
        checkpoint = {"label": "latest", "run_config": {"cross_record_attention": True}}
        with patch("experiments.run_benchmark_experiment.torch.load", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "attention-on checkpoint"):
                config_from_args(Namespace(resume="original.pt", no_cross_record_attention=True))


if __name__ == "__main__":
    unittest.main()
