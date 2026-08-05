import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from model import NanoERPFNLinker
from run_field_evidence_diagnostics import (
    DEFAULT_MODELS,
    CROSS_SOURCE_BENCHMARKS,
    load_model_bundle,
    pair_indices_and_targets,
    prepare_benchmark_tasks,
)
from utils import get_default_device


DEFAULT_BENCHMARKS = [
    "dblp_acm",
    "dblp_google_scholar",
    "fodors_zagats",
    "walmart_amazon",
    "amazon_google_price_numeric",
    "amazon_google_price_both",
]

BASE_TYPES = ["text", "identifier", "categorical"]


def parse_model_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be formatted as name=checkpoint.pt")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("model name cannot be empty")
    return name, Path(path)


def numeric_like(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    parsed = pd.to_numeric(values, errors="coerce")
    return float(parsed.notna().mean()) >= 0.95


def date_like(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= 0.95


def allowed_types_for_column(name: str, series: pd.Series) -> list[str]:
    allowed = list(BASE_TYPES)
    if numeric_like(series):
        allowed.append("numeric")
    if date_like(series):
        allowed.append("date")
    lower_name = name.lower()
    if "email" in lower_name:
        allowed.append("email")
    if "phone" in lower_name:
        allowed.append("phone")
    return list(dict.fromkeys(allowed))


def config_label(field_types: list[str], columns: list[str]) -> str:
    return "; ".join(f"{column}:{field_type}" for column, field_type in zip(columns, field_types))


def preset_configs(original_types: list[str], records: pd.DataFrame):
    columns = list(records.columns)
    configs = [("manifest", original_types)]
    configs.append(("all_text", ["text"] * len(columns)))
    configs.append(("all_identifier", ["identifier"] * len(columns)))
    configs.append(("all_categorical", ["categorical"] * len(columns)))
    configs.append(
        (
            "non_numeric_text",
            [
                original if original == "numeric" else "text"
                for original in original_types
            ],
        )
    )
    configs.append(
        (
            "numeric_as_text",
            [
                "text" if original == "numeric" else original
                for original in original_types
            ],
        )
    )
    configs.append(
        (
            "categorical_as_text",
            [
                "text" if original == "categorical" else original
                for original in original_types
            ],
        )
    )
    configs.append(
        (
            "identifier_as_text",
            [
                "text" if original in {"identifier", "email", "phone"} else original
                for original in original_types
            ],
        )
    )
    seen = set()
    deduped = []
    for name, field_types in configs:
        key = tuple(field_types)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, field_types))
    return deduped


def one_at_a_time_configs(original_types: list[str], records: pd.DataFrame):
    columns = list(records.columns)
    configs = preset_configs(original_types, records)
    seen = {tuple(field_types) for _, field_types in configs}
    for index, column in enumerate(columns):
        for field_type in allowed_types_for_column(column, records[column]):
            field_types = list(original_types)
            field_types[index] = field_type
            key = tuple(field_types)
            if key in seen:
                continue
            seen.add(key)
            configs.append((f"{column}_as_{field_type}", field_types))
    return configs


def exhaustive_configs(original_types: list[str], records: pd.DataFrame):
    columns = list(records.columns)
    choices = [allowed_types_for_column(column, records[column]) for column in columns]
    configs = []
    for field_types in itertools.product(*choices):
        field_types = list(field_types)
        name = "manifest" if field_types == original_types else "exhaustive"
        configs.append((name, field_types))
    return configs


def candidate_configs(mode: str, original_types: list[str], records: pd.DataFrame):
    if mode == "presets":
        return preset_configs(original_types, records)
    if mode == "one_at_a_time":
        return one_at_a_time_configs(original_types, records)
    if mode == "exhaustive":
        return exhaustive_configs(original_types, records)
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_config(linker, benchmark: dict, field_types: list[str]):
    records = benchmark["records"]
    left_indices, right_indices, targets = pair_indices_and_targets(benchmark)
    logits = linker.decision_function(records, field_types)
    probabilities = 1 / (1 + np.exp(-logits[left_indices, right_indices]))
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    threshold_f1 = (
        2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best_threshold_index = int(np.argmax(threshold_f1))
    return {
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "oracle_pair_f1": float(threshold_f1[best_threshold_index]),
        "oracle_pair_threshold": float(thresholds[best_threshold_index]),
    }


def write_partial(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep benchmark field-type overrides for saved ERPFN checkpoints."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_arg,
        default=None,
        help="Model checkpoint as name=path. May be repeated.",
    )
    parser.add_argument("--benchmark-name", action="append", default=None)
    parser.add_argument(
        "--mode",
        choices=["presets", "one_at_a_time", "exhaustive"],
        default="one_at_a_time",
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=Path("benchmark_data/manifest.json")
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/field_type_sweeps/field_type_sweep.csv"),
    )
    parser.add_argument("--limit-configs", type=int, default=None)
    args = parser.parse_args()

    model_specs = args.model if args.model is not None else [
        (name, Path(path)) for name, path in DEFAULT_MODELS
    ]
    benchmark_names = args.benchmark_name or DEFAULT_BENCHMARKS
    benchmarks = prepare_benchmark_tasks(args.manifest_path, benchmark_names)
    device = get_default_device()
    print(f"device: {device}", flush=True)
    print(f"mode: {args.mode}", flush=True)
    print(f"output_path: {args.output_path}", flush=True)

    rows = []
    for model_name, checkpoint_path in model_specs:
        print(f"loading model {model_name}: {checkpoint_path}", flush=True)
        model, tokenizer, checkpoint = load_model_bundle(checkpoint_path, device)
        linker = NanoERPFNLinker(model, tokenizer, device=device)
        for benchmark in benchmarks:
            records = benchmark["records"]
            columns = list(records.columns)
            original_types = list(benchmark["field_types"])
            configs = candidate_configs(args.mode, original_types, records)
            if args.limit_configs is not None:
                configs = configs[: args.limit_configs]
            print(
                f"{model_name} :: {benchmark['name']} "
                f"({benchmark['pair_scope']}) configs={len(configs)}",
                flush=True,
            )
            for config_index, (variant, field_types) in enumerate(configs, start=1):
                try:
                    metrics = evaluate_config(linker, benchmark, field_types)
                    error = None
                except Exception as exc:
                    metrics = {
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                        "oracle_pair_f1": np.nan,
                        "oracle_pair_threshold": np.nan,
                    }
                    error = repr(exc)
                row = {
                    "model": model_name,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_step": checkpoint.get("step"),
                    "benchmark": benchmark["name"],
                    "pair_scope": benchmark["pair_scope"],
                    "variant": variant,
                    "config_index": config_index,
                    "n_configs": len(configs),
                    "field_types": config_label(field_types, columns),
                    "manifest_field_types": config_label(original_types, columns),
                    "candidate_pairs": len(pair_indices_and_targets(benchmark)[2]),
                    "error": error,
                    **metrics,
                }
                rows.append(row)
                if config_index % 10 == 0 or config_index == len(configs):
                    best = pd.DataFrame(rows)
                    best = best[
                        (best["model"] == model_name)
                        & (best["benchmark"] == benchmark["name"])
                    ].sort_values("pr_auc", ascending=False, na_position="last")
                    print(
                        f"  {config_index}/{len(configs)} best="
                        f"{best.iloc[0]['pr_auc']:.4f} "
                        f"{best.iloc[0]['field_types']}",
                        flush=True,
                    )
                    write_partial(rows, args.output_path)
    write_partial(rows, args.output_path)
    print(f"wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
