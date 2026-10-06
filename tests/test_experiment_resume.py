import random
from argparse import Namespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.run_benchmark_experiment import (
    ExperimentConfig,
    config_from_args,
    run_experiment,
)


class ToyTokenizer:
    def __init__(self, **kwargs):
        pass

    def __call__(self, records, field_types):
        return {"x": records[0]}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.3))

    def forward(self, cells):
        x = cells["x"]
        return self.weight * (x[:, :, None] + x[:, None, :])


class ToyPrior:
    fail_at = None

    def __init__(self, num_steps, seed, **kwargs):
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)
        self.generated = 0

    def __iter__(self):
        for _ in range(self.num_steps):
            self.generated += 1
            if self.generated == self.fail_at:
                raise RuntimeError("simulated interruption")
            values = torch.tensor(self.rng.random((1, 3)), dtype=torch.float32)
            values += torch.rand((1, 3)) + random.random() + np.random.random()
            yield {
                "records": [values],
                "field_types": [{}],
                "adjacency": torch.tensor(
                    [[[0, 1, 0], [1, 0, 0], [0, 0, 0]]], dtype=torch.bool
                ),
            }


def toy_checkpoint_frame(model, tokenizer, benchmarks, device, step):
    return pd.DataFrame(
        {
            "step": [step],
            "benchmark": ["toy"],
            "pr_auc": [float(model.weight.detach())],
        }
    )


class ExperimentResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def config(name):
                return ExperimentConfig(
                    num_steps=4,
                    eval_every=2,
                    batch_size=1,
                    benchmark_names=[],
                    frozen_dgp_configs={},
                    results_directory=root / name / "results",
                    checkpoint_directory=root / name / "checkpoints",
                    save_models=True,
                )

            with (
                patch("experiments.run_benchmark_experiment.Tokenizer", ToyTokenizer),
                patch("experiments.run_benchmark_experiment.make_model", side_effect=lambda *args: ToyModel()),
                patch("experiments.run_benchmark_experiment.SyntheticWorldDataLoader", ToyPrior),
                patch("experiments.run_benchmark_experiment.load_experiment_benchmarks", return_value=[{}]),
                patch("experiments.run_benchmark_experiment.checkpoint_frame", toy_checkpoint_frame),
                patch("experiments.run_benchmark_experiment.get_default_device", return_value=torch.device("cpu")),
            ):
                ToyPrior.fail_at = None
                full_results = run_experiment(config("full"))
                ToyPrior.fail_at = 3
                interrupted_config = config("resumed")
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_experiment(interrupted_config)
                latest_path = next(interrupted_config.checkpoint_directory.glob("*_latest.pt"))
                self.assertEqual(
                    torch.load(latest_path, weights_only=False)["step"], 2
                )
                ToyPrior.fail_at = None
                resumed_config = config_from_args(Namespace(resume=latest_path))
                self.assertEqual(resumed_config.resume_checkpoint, latest_path)
                resumed_results = run_experiment(resumed_config)

            full_latest = torch.load(
                next((root / "full" / "checkpoints").glob("*_latest.pt")),
                weights_only=False,
            )
            resumed_latest = torch.load(latest_path, weights_only=False)
            self.assertTrue(
                torch.equal(
                    full_latest["training_model_state_dict"]["weight"],
                    resumed_latest["training_model_state_dict"]["weight"],
                )
            )
            full_log = pd.read_csv(next((root / "full" / "results").glob("*_loss.csv")))
            resumed_log = pd.read_csv(
                next((root / "resumed" / "results").glob("*_loss.csv"))
            )
            pd.testing.assert_frame_equal(full_log, resumed_log)
            pd.testing.assert_frame_equal(
                pd.read_csv(full_results)[["step", "pr_auc", "weighted_training_loss"]],
                pd.read_csv(resumed_results)[["step", "pr_auc", "weighted_training_loss"]],
            )


if __name__ == "__main__":
    unittest.main()
