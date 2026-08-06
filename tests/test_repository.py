import csv
import json
import unittest
from pathlib import Path

import pandas as pd
import torch

from identitypfn import (
    infer_field_type,
    infer_field_types,
    load_model,
    nearest_neighbors,
    pair_review_table,
    top_pairs,
)
from identitypfn.dgp.people.wag import generate_worlds
from identitypfn.train import load_benchmarks


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RepositorySmokeTests(unittest.TestCase):
    def test_included_synthetic_benchmarks_load(self):
        benchmarks = load_benchmarks(
            REPOSITORY_ROOT / "benchmark_data" / "manifest.json",
            ["fake_1000", "sim_er_data"],
        )

        self.assertEqual([item["name"] for item in benchmarks], [
            "fake_1000",
            "sim_er_data",
        ])
        self.assertTrue(all(len(item["records"]) > 0 for item in benchmarks))

    def test_wag_generation_does_not_require_ollama(self):
        worlds = generate_worlds(
            n_worlds=1,
            n_records=[24],
            n_fields=[6],
            seed=1337,
            allow_ollama=False,
        )

        self.assertEqual(len(worlds), 1)
        self.assertEqual(len(worlds[0].records), 24)
        self.assertEqual(worlds[0].adjacency.shape, (24, 24))

    def test_main_checkpoint_contains_release_metadata(self):
        checkpoint_path = (
            REPOSITORY_ROOT
            / "results"
            / "model_checkpoints"
            / "20260715_195803_seed1337_20005214_best_step1500.pt"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        self.assertEqual(
            checkpoint["run_id"],
            "20260715_195803_seed1337_20005214",
        )
        self.assertEqual(checkpoint["step"], 1500)
        self.assertEqual(
            checkpoint["run_config"]["text_backend"],
            "sentence_transformer",
        )

    def test_result_tables_reference_released_checkpoints(self):
        table_directory = REPOSITORY_ROOT / "results" / "paper_benchmark_tables"
        referenced_paths = set()
        for table_path in table_directory.glob("*.csv"):
            with table_path.open(newline="") as file:
                for row in csv.DictReader(file):
                    checkpoint_path = row.get("checkpoint_path")
                    if checkpoint_path:
                        referenced_paths.add(checkpoint_path)

        self.assertTrue(referenced_paths)
        for checkpoint_path in referenced_paths:
            self.assertTrue(
                (REPOSITORY_ROOT / checkpoint_path).is_file(),
                checkpoint_path,
            )

    def test_manifest_is_valid_json(self):
        manifest_path = REPOSITORY_ROOT / "benchmark_data" / "manifest.json"
        with manifest_path.open() as file:
            manifest = json.load(file)

        self.assertIn("_schema", manifest)
        self.assertIn("amazon_google_price_numeric", manifest)

    def test_public_pair_helpers_sort_scores(self):
        records = pd.DataFrame(
            [
                {"name": "Ada Lovelace", "email": "ada@example.com"},
                {"name": "A. Lovelace", "email": "ada@example.com"},
                {"name": "Grace Hopper", "email": "grace@example.com"},
            ],
            index=["r0", "r1", "r2"],
        )
        scores = pd.DataFrame(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ],
            index=records.index,
            columns=records.index,
        )

        self.assertEqual(top_pairs(scores, k=1).iloc[0]["right_index"], "r1")
        self.assertEqual(nearest_neighbors(scores, k=1).iloc[0]["neighbor_index"], "r1")
        review = pair_review_table(scores, records, k=1, columns=["name", "email"])
        self.assertIn("left_name", review.columns)
        self.assertEqual(review.iloc[0]["score"], 0.9)

    def test_public_field_type_inference(self):
        records = pd.DataFrame(
            {
                "email": ["ada@example.com", "grace@example.org"],
                "contact_no": ["+1 (865) 555-5555", "865-555-5556"],
                "sku": ["ABC-123", "XYZ-999"],
                "created_at": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "amount": [10.5, 22.0],
                "active": [True, False],
                "notes": ["met at event", "prefers email"],
            }
        )

        self.assertEqual(
            infer_field_types(records),
            [
                ("email", "column_name"),
                ("phone", "value_pattern"),
                ("identifier", "column_name"),
                ("date", "dtype"),
                ("numeric", "dtype"),
                ("categorical", "dtype"),
                ("text", "fallback"),
            ],
        )
        self.assertEqual(
            infer_field_type(pd.Series(["a@example.com", "not email"])),
            ("text", "fallback"),
        )
        self.assertEqual(
            infer_field_type(pd.Series(["a@example.com", "b@example.org"])),
            ("email", "value_pattern"),
        )
        self.assertEqual(
            infer_field_type(pd.Series(["", "   ", "a@example.com"], name="contact")),
            ("email", "value_pattern"),
        )

    def test_public_load_model_scores_records(self):
        checkpoint_path = (
            REPOSITORY_ROOT
            / "results"
            / "model_checkpoints"
            / "20260715_195803_seed1337_20005214_best_step1500.pt"
        )
        records = pd.DataFrame(
            [
                {
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                },
                {
                    "first_name": "A.",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                },
                {
                    "first_name": "Grace",
                    "last_name": "Hopper",
                    "email": "grace@example.com",
                },
            ],
            index=["r0", "r1", "r2"],
        )

        linker = load_model(checkpoint_path, device="cpu")
        scores = linker.predict_proba(records, ["text", "text", "email"])

        self.assertEqual(scores.shape, (3, 3))
        self.assertEqual(list(scores.index), list(records.index))
        self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).to_numpy().all())

    def test_public_load_model_infers_field_types(self):
        checkpoint_path = (
            REPOSITORY_ROOT
            / "results"
            / "model_checkpoints"
            / "20260715_195803_seed1337_20005214_best_step1500.pt"
        )
        records = pd.DataFrame(
            [
                {
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                },
                {
                    "first_name": "A.",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                },
                {
                    "first_name": "Grace",
                    "last_name": "Hopper",
                    "email": "grace@example.com",
                },
            ],
            index=["r0", "r1", "r2"],
        )

        linker = load_model(checkpoint_path, device="cpu")
        scores = linker.predict_proba(records, field_types="infer", verbose=False)

        self.assertEqual(scores.shape, (3, 3))
        self.assertEqual(list(scores.index), list(records.index))


if __name__ == "__main__":
    unittest.main()
