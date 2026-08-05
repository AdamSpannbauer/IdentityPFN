import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import duckdb
from sklearn.metrics import average_precision_score

from identitypfn.model import NanoERPFNLinker, NanoERPFNModel
from identitypfn.tokenizer import Tokenizer
from identitypfn.utils import get_default_device


DEFAULT_MODELS = [
    (
        "default_wag",
        "results/model_checkpoints/20260715_195803_seed1337_20005214_best_step1500.pt",
    ),
    (
        "hard_negative_wag",
        "results/model_checkpoints/20260721_200947_seed1337_bda60282_best_step1500.pt",
    ),
]

DEFAULT_BENCHMARKS = [
    "nc_voters_shared",
    "nc_voters_changed",
    "bpid_matching_balanced",
    "bpid_matching_paired",
    "ice_id_people_200",
    "dblp_acm",
    "dblp_google_scholar",
    "fodors_zagats",
    "walmart_amazon",
    "amazon_google_price_numeric",
    "amazon_google_price_both",
]

CROSS_SOURCE_BENCHMARKS = {
    "dblp_acm",
    "dblp_google_scholar",
    "fodors_zagats",
    "walmart_amazon",
    "amazon_google_price_numeric",
    "amazon_google_price_both",
}


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sample_entity_frame(
    csv_path: Path,
    columns: list[str],
    entity_id_column: str,
    sample_config: dict,
) -> pd.DataFrame:
    quoted_columns = [quote_identifier(column) for column in columns]
    row_signature = "md5(concat_ws(chr(31), {}))".format(", ".join(quoted_columns))
    group_filters = []
    if sample_config["require_positive_pairs"]:
        group_filters.append("count(*) >= 2")
    if sample_config["require_variation"]:
        group_filters.append(f"count(distinct {row_signature}) > 1")
    having_clause = f"having {' and '.join(group_filters)}" if group_filters else ""

    selected_columns = [entity_id_column, *columns]
    select_clause = ", ".join(quote_identifier(column) for column in selected_columns)
    entity_id = quote_identifier(entity_id_column)
    csv_literal = quote_literal(str(csv_path))
    sample_size = sample_config["n_entities"]
    seed = sample_config["seed"]

    query = f"""
        with raw as (
            select {select_clause}
            from read_csv_auto({csv_literal}, all_varchar=true)
        ),
        eligible_entities as (
            select {entity_id} as entity_id
            from raw
            group by {entity_id}
            {having_clause}
        ),
        sampled_entities as (
            select entity_id
            from eligible_entities
            order by hash(entity_id, {seed})
            limit {sample_size}
        ),
        sampled as (
            select raw.*
            from raw
            join sampled_entities
                on raw.{entity_id} = sampled_entities.entity_id
        )
        select *
        from sampled
        order by hash({entity_id}, {seed}), hash(concat_ws(chr(31), {select_clause}), {seed})
    """
    frame = duckdb.sql(query).df()
    n_entities = frame[entity_id_column].nunique()
    if n_entities < sample_size:
        raise ValueError(
            f"Requested {sample_size} entities but only {n_entities} are eligible"
        )
    return frame


def load_benchmark_frame(
    manifest_path: Path, config: dict, columns: list[str]
) -> pd.DataFrame:
    csv_path = manifest_path.parent / config["file"]
    if "entity_sample" in config:
        return sample_entity_frame(
            csv_path, columns, config["entity_id_column"], config["entity_sample"]
        )
    return pd.read_csv(
        csv_path, usecols=[config["entity_id_column"], *columns], dtype=str
    )


def load_benchmarks(manifest_path: Path, names: list[str]) -> list[dict]:
    with manifest_path.open() as file:
        manifest = json.load(file)
    benchmarks = []
    for name in names:
        config = manifest[name]
        columns = list(config["field_types"])
        data = load_benchmark_frame(manifest_path, config, columns)
        benchmarks.append(
            {
                "name": name,
                "description": config["description"],
                "kind": config["kind"],
                "entity_sample": config.get("entity_sample"),
                "records": data[columns],
                "field_types": list(config["field_types"].values()),
                "entity_ids": data[config["entity_id_column"]].to_numpy(),
            }
        )
    return benchmarks


def config_value(config: dict, key: str, default):
    value = config.get(key, default)
    return default if value is None else value


def load_model_bundle(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["run_config"]
    tokenizer = Tokenizer(
        text_backend=config_value(config, "text_backend", "sentence_transformer"),
        identifier_backend=config.get("identifier_backend"),
        normalize_identifiers=config_value(config, "normalize_identifiers", False),
        default_phone_region=config_value(config, "default_phone_region", "US"),
    )
    model = NanoERPFNModel(
        embedding_size=config_value(config, "embedding_size", 96),
        text_embedding_size=tokenizer.text_embedding_dim,
        identifier_embedding_size=tokenizer.identifier_embedding_dim,
        num_attention_heads=config_value(config, "num_attention_heads", 4),
        mlp_hidden_size=config_value(config, "mlp_hidden_size", 192),
        num_layers=config_value(config, "num_layers", 3),
        max_categories=config_value(config, "max_categories", 512),
        record_representation=config_value(
            config, "record_representation", "entity_target_column"
        ),
        adjacency_decoder=config_value(config, "adjacency_decoder", "pair_mlp"),
        entity_slot_count=config_value(config, "entity_slot_count", 300),
        num_slot_attention_layers=config_value(
            config, "num_slot_attention_layers", 1
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer, checkpoint


def benchmark_metadata(manifest_path: Path, benchmark_name: str, config: dict):
    columns = list(config["field_types"])
    metadata_columns = [*columns]
    ignored_columns = config.get("ignored_columns", [])
    if "source_table" in ignored_columns:
        metadata_columns.insert(0, "source_table")
    if benchmark_name == "bpid_matching_balanced":
        metadata_columns = ["pair_id", "source_table", "match_label", *columns]
    return load_benchmark_frame(manifest_path, config, metadata_columns)


def prepare_benchmark_tasks(manifest_path: Path, requested_names: list[str]):
    with manifest_path.open() as file:
        manifest = json.load(file)

    base_names = [
        name if name != "bpid_matching_paired" else "bpid_matching_balanced"
        for name in requested_names
    ]
    base_benchmarks = {
        benchmark["name"]: benchmark
        for benchmark in load_benchmarks(manifest_path, names=sorted(set(base_names)))
    }

    tasks = []
    for requested_name in requested_names:
        base_name = (
            "bpid_matching_balanced"
            if requested_name == "bpid_matching_paired"
            else requested_name
        )
        benchmark = base_benchmarks[base_name].copy()
        config = manifest[base_name]
        metadata = benchmark_metadata(manifest_path, base_name, config)

        if requested_name == "bpid_matching_paired":
            paired_indices = []
            paired_targets = []
            for _, group in metadata.groupby("pair_id", sort=False):
                if set(group["source_table"]) != {"profile1", "profile2"}:
                    continue
                if len(group) != 2:
                    continue
                left_index = int(group.index[group["source_table"] == "profile1"][0])
                right_index = int(group.index[group["source_table"] == "profile2"][0])
                paired_indices.append(
                    (min(left_index, right_index), max(left_index, right_index))
                )
                paired_targets.append(
                    str(group["match_label"].iloc[0]).lower() == "true"
                )
            benchmark["name"] = requested_name
            benchmark["pair_scope"] = "paired_rows"
            benchmark["paired_row_indices"] = np.asarray(paired_indices, dtype=int)
            benchmark["paired_targets"] = np.asarray(paired_targets, dtype=bool)
        elif requested_name in CROSS_SOURCE_BENCHMARKS and "source_table" in metadata:
            benchmark["pair_scope"] = "cross_source"
            benchmark["source_table_values"] = metadata["source_table"].to_numpy()
        else:
            benchmark["pair_scope"] = "all_pairs"
        tasks.append(benchmark)
    return tasks


def pair_indices_and_targets(benchmark: dict):
    entity_ids = benchmark["entity_ids"]
    if benchmark["pair_scope"] == "paired_rows":
        pair_indices = benchmark["paired_row_indices"]
        return pair_indices[:, 0], pair_indices[:, 1], benchmark["paired_targets"]

    left_indices, right_indices = np.triu_indices(len(entity_ids), k=1)
    if benchmark["pair_scope"] == "cross_source":
        source = benchmark["source_table_values"]
        keep = source[left_indices] != source[right_indices]
        left_indices = left_indices[keep]
        right_indices = right_indices[keep]

    targets = entity_ids[left_indices] == entity_ids[right_indices]
    return left_indices, right_indices, targets


def format_differing_patterns(equal_cells: np.ndarray, columns: list[str]) -> np.ndarray:
    unique_rows, inverse = np.unique(equal_cells, axis=0, return_inverse=True)
    labels = []
    for row in unique_rows:
        differing = [column for column, is_equal in zip(columns, row) if not is_equal]
        labels.append(", ".join(differing) if differing else "<none>")
    return np.asarray(labels, dtype=object)[inverse]


def summarize_pairs(model_name: str, checkpoint_path: Path, checkpoint: dict, benchmark: dict):
    records = benchmark["records"].reset_index(drop=True)
    columns = list(records.columns)
    left_indices, right_indices, targets = pair_indices_and_targets(benchmark)
    probabilities = benchmark["probabilities"][left_indices, right_indices]

    left = records.iloc[left_indices].reset_index(drop=True)
    right = records.iloc[right_indices].reset_index(drop=True)
    equal_cells = (left.eq(right) | (left.isna() & right.isna())).to_numpy()
    matching_fields = equal_cells.sum(axis=1)
    differing_fields = format_differing_patterns(equal_cells, columns)

    pairs = pd.DataFrame(
        {
            "model": model_name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": checkpoint.get("step"),
            "benchmark": benchmark["name"],
            "pair_scope": benchmark["pair_scope"],
            "target": targets,
            "probability": probabilities,
            "matching_fields": matching_fields,
            "differing_fields": differing_fields,
            "left_index": left_indices,
            "right_index": right_indices,
        }
    )
    return pairs, left, right


def quantile_10(values):
    return values.quantile(0.10)


def quantile_25(values):
    return values.quantile(0.25)


def quantile_75(values):
    return values.quantile(0.75)


def quantile_90(values):
    return values.quantile(0.90)


def probability_summary(frame: pd.DataFrame, group_columns: list[str]):
    return (
        frame.groupby(group_columns, dropna=False)["probability"]
        .agg(
            count="size",
            mean="mean",
            median="median",
            q10=quantile_10,
            q25=quantile_25,
            q75=quantile_75,
            q90=quantile_90,
            min="min",
            max="max",
        )
        .reset_index()
    )


def metric_summary(pairs: pd.DataFrame, n_records: int, n_entities: int, n_fields: int):
    targets = pairs["target"].to_numpy()
    probabilities = pairs["probability"].to_numpy()
    pr_auc = (
        average_precision_score(targets, probabilities)
        if np.unique(targets).size == 2
        else np.nan
    )
    positives = pairs[pairs["target"]]
    negatives = pairs[~pairs["target"]]
    return {
        "model": pairs["model"].iloc[0],
        "checkpoint_path": pairs["checkpoint_path"].iloc[0],
        "checkpoint_step": pairs["checkpoint_step"].iloc[0],
        "benchmark": pairs["benchmark"].iloc[0],
        "pair_scope": pairs["pair_scope"].iloc[0],
        "n_records": n_records,
        "n_entities": n_entities,
        "n_fields": n_fields,
        "candidate_pairs": len(pairs),
        "prevalence": float(targets.mean()),
        "pr_auc": float(pr_auc),
        "exact_positive_pair_rate": float(
            (positives["matching_fields"] == n_fields).mean()
        )
        if len(positives)
        else np.nan,
        "exact_negative_pair_rate": float(
            (negatives["matching_fields"] == n_fields).mean()
        )
        if len(negatives)
        else np.nan,
        "positive_median_probability": float(positives["probability"].median())
        if len(positives)
        else np.nan,
        "negative_median_probability": float(negatives["probability"].median())
        if len(negatives)
        else np.nan,
    }


def same_agreement_contrast(agreement_summary: pd.DataFrame):
    id_columns = [
        "model",
        "checkpoint_path",
        "checkpoint_step",
        "benchmark",
        "pair_scope",
        "matching_fields",
    ]
    wide = agreement_summary.pivot_table(
        index=id_columns,
        columns="target",
        values=["count", "median", "mean"],
        aggfunc="first",
    )
    wide.columns = [
        f"{metric}_{'positive' if target else 'negative'}"
        for metric, target in wide.columns
    ]
    wide = wide.reset_index()
    if {"median_positive", "median_negative"} <= set(wide.columns):
        wide["median_gap_positive_minus_negative"] = (
            wide["median_positive"] - wide["median_negative"]
        )
    return wide


def top_false_positives(
    pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, top_k: int
):
    false_pairs = pairs[~pairs["target"]].nlargest(top_k, "probability").copy()
    for column in left.columns:
        false_pairs[f"left_{column}"] = left.loc[false_pairs.index, column].to_numpy()
        false_pairs[f"right_{column}"] = right.loc[false_pairs.index, column].to_numpy()
    return false_pairs.reset_index(drop=True)


def parse_model_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be formatted as name=checkpoint.pt")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("model name cannot be empty")
    return name, Path(path)


def write_outputs(
    output_dir: Path,
    metric_rows: list[dict],
    agreement_frames: list[pd.DataFrame],
    pattern_frames: list[pd.DataFrame],
    contrast_frames: list[pd.DataFrame],
    top_false_frames: list[pd.DataFrame],
):
    metrics_path = output_dir / "metrics.csv"
    agreement_path = output_dir / "agreement_count_summary.csv"
    pattern_path = output_dir / "field_pattern_summary.csv"
    contrast_path = output_dir / "same_agreement_contrast.csv"
    top_false_path = output_dir / "top_false_positives.csv"

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.concat(agreement_frames, ignore_index=True).to_csv(agreement_path, index=False)
    pd.concat(pattern_frames, ignore_index=True).to_csv(pattern_path, index=False)
    pd.concat(contrast_frames, ignore_index=True).to_csv(contrast_path, index=False)
    if top_false_frames:
        pd.concat(top_false_frames, ignore_index=True).to_csv(top_false_path, index=False)
    else:
        pd.DataFrame().to_csv(top_false_path, index=False)

    return {
        "metrics": metrics_path,
        "agreement": agreement_path,
        "patterns": pattern_path,
        "contrast": contrast_path,
        "top_false": top_false_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize field-level evidence patterns for IdentityPFN checkpoints."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_arg,
        default=None,
        help="Model checkpoint as name=path. May be repeated.",
    )
    parser.add_argument(
        "--benchmark-name",
        action="append",
        default=None,
        help="Benchmark/scope to evaluate. May be repeated.",
    )
    parser.add_argument(
        "--manifest-path", default="benchmark_data/manifest.json", type=Path
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/field_evidence_diagnostics"),
        type=Path,
    )
    parser.add_argument("--field-pattern-top-n", default=80, type=int)
    parser.add_argument("--top-k-false", default=20, type=int)
    args = parser.parse_args()

    model_specs = args.model if args.model is not None else [
        (name, Path(path)) for name, path in DEFAULT_MODELS
    ]
    benchmark_names = args.benchmark_name or DEFAULT_BENCHMARKS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_default_device()
    print(f"device: {device}", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)
    benchmarks = prepare_benchmark_tasks(args.manifest_path, benchmark_names)

    metric_rows = []
    agreement_frames = []
    pattern_frames = []
    contrast_frames = []
    top_false_frames = []

    for model_name, checkpoint_path in model_specs:
        print(f"loading model {model_name}: {checkpoint_path}", flush=True)
        model, tokenizer, checkpoint = load_model_bundle(checkpoint_path, device)
        linker = NanoERPFNLinker(model, tokenizer, device=device)

        for benchmark in benchmarks:
            print(
                f"scoring {model_name} on {benchmark['name']} "
                f"({benchmark['pair_scope']})",
                flush=True,
            )
            linker.fit(benchmark["records"], benchmark["field_types"])
            benchmark = benchmark.copy()
            benchmark["probabilities"] = linker.adjacency_proba_

            pairs, left, right = summarize_pairs(
                model_name, checkpoint_path, checkpoint, benchmark
            )
            metric_rows.append(
                metric_summary(
                    pairs,
                    n_records=len(benchmark["entity_ids"]),
                    n_entities=len(np.unique(benchmark["entity_ids"])),
                    n_fields=benchmark["records"].shape[1],
                )
            )

            agreement = probability_summary(
                pairs,
                [
                    "model",
                    "checkpoint_path",
                    "checkpoint_step",
                    "benchmark",
                    "pair_scope",
                    "matching_fields",
                    "target",
                ],
            )
            agreement_frames.append(agreement)
            contrast_frames.append(same_agreement_contrast(agreement))

            pattern = probability_summary(
                pairs,
                [
                    "model",
                    "checkpoint_path",
                    "checkpoint_step",
                    "benchmark",
                    "pair_scope",
                    "target",
                    "differing_fields",
                ],
            )
            pattern = (
                pattern.sort_values(
                    ["model", "benchmark", "target", "count"],
                    ascending=[True, True, True, False],
                )
                .groupby(["model", "benchmark", "target"], group_keys=False)
                .head(args.field_pattern_top_n)
            )
            pattern_frames.append(pattern)

            if args.top_k_false > 0:
                top_false_frames.append(
                    top_false_positives(pairs, left, right, args.top_k_false)
                )
            paths = write_outputs(
                args.output_dir,
                metric_rows,
                agreement_frames,
                pattern_frames,
                contrast_frames,
                top_false_frames,
            )
            print(f"refreshed partial outputs through {benchmark['name']}", flush=True)

    paths = write_outputs(
        args.output_dir,
        metric_rows,
        agreement_frames,
        pattern_frames,
        contrast_frames,
        top_false_frames,
    )
    for path in paths.values():
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
