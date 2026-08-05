import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_MODEL_NAMES = {
    "generic_exact": "Splink generic exact",
    "all_pairs_small": "Splink all-pairs small",
    "standard_blocking_clean_weighted_edge_pruning_jaccard": "pyJedAI standard WEP Jaccard",
}


def load_benchmark_metadata(manifest_path: Path) -> pd.DataFrame:
    with manifest_path.open() as file:
        manifest = json.load(file)

    rows = []
    for name, config in manifest.items():
        field_types = config.get("field_types", {})
        rows.append(
            {
                "benchmark": name,
                "short_description": config.get(
                    "short_description", config.get("description", name)
                ),
                "text_profile": config.get("text_profile", pd.NA),
                "field_summary": "; ".join(
                    f"{field}:{field_type}" for field, field_type in field_types.items()
                ),
                "description": config.get("description", ""),
                "n_fields": len(field_types),
            }
        )
    return pd.DataFrame(rows)


def convert_baseline_rows(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows = rows.rename(columns={"name": "benchmark"})
    rows["model_name"] = rows["policy"].map(POLICY_MODEL_NAMES).fillna(
        rows["baseline"].astype(str) + ":" + rows["policy"].astype(str)
    )
    rows["model_kind"] = rows["baseline"].astype(str) + "_competitor"
    rows["checkpoint_path"] = pd.NA
    rows["run_id"] = rows["baseline"] + ":" + rows["policy"]
    rows["text_backend"] = pd.NA
    rows["record_representation"] = pd.NA
    rows["adjacency_decoder"] = pd.NA
    rows["train_generator"] = pd.NA
    rows["checkpoint_step"] = pd.NA
    rows["checkpoint_label"] = pd.NA
    rows["n_pairs"] = rows["n_records"] * (rows["n_records"] - 1) // 2
    rows["eval_seconds"] = rows["seconds"]
    rows["rss_mb_before"] = np.nan
    rows["rss_mb_after"] = np.nan
    rows["rss_mb_delta"] = np.nan
    rows["device_memory_mb_before"] = np.nan
    rows["device_memory_mb_after"] = np.nan
    rows["device_memory_mb_delta"] = np.nan
    rows["pair_precision_at_0_5"] = rows["pair_precision"]
    rows["pair_recall_at_0_5"] = rows["pair_recall"]
    rows["pair_f1_at_0_5"] = rows["pair_f1"]
    rows["oracle_pair_f1"] = rows["best_pair_f1"]
    rows["oracle_pair_threshold"] = rows["best_pair_threshold"]
    rows["adjusted_rand"] = rows["adjusted_rand"]

    rows = metadata.merge(rows, on="benchmark", how="right", suffixes=("", "_raw"))
    if "n_fields_raw" in rows:
        rows["n_fields"] = rows["n_fields"].fillna(rows["n_fields_raw"])
        rows = rows.drop(columns=["n_fields_raw"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=Path("benchmark_data/manifest.json"), type=Path
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            Path("results/baselines/splink_generic_exact_paper_benchmarks.csv"),
            Path("results/baselines/splink_all_pairs_small_paper_benchmarks.csv"),
            Path("results/baselines/pyjedai_standard_wep_jaccard_paper_benchmarks.csv"),
        ],
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("results/paper_benchmark_tables/competitor_baseline_table.csv"),
        type=Path,
    )
    parser.add_argument(
        "--existing-paper-table",
        default=Path("results/paper_benchmark_tables/benchmark_table.csv"),
        type=Path,
    )
    parser.add_argument(
        "--combined-output",
        default=Path("results/paper_benchmark_tables/benchmark_table_with_competitors.csv"),
        type=Path,
    )
    args = parser.parse_args()

    metadata = load_benchmark_metadata(args.manifest)
    baseline_frames = [pd.read_csv(path) for path in args.inputs]
    baselines = convert_baseline_rows(pd.concat(baseline_frames, ignore_index=True), metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    baselines.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    if args.existing_paper_table.exists():
        existing = pd.read_csv(args.existing_paper_table)
        combined = pd.concat([existing, baselines], ignore_index=True, sort=False)
        combined.to_csv(args.combined_output, index=False)
        print(f"Wrote {args.combined_output}")

    summary = baselines[
        [
            "model_name",
            "benchmark",
            "pr_auc",
            "oracle_pair_f1",
            "pair_f1_at_0_5",
            "adjusted_rand",
            "candidate_fraction",
            "true_pair_candidate_recall",
        ]
    ].sort_values(["benchmark", "model_name"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
