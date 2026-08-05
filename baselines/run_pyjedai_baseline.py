import argparse
import contextlib
import json
import os
import time
import warnings
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


VALID_FIELD_TYPES = {"text", "categorical", "numeric", "date", "identifier", "phone"}


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sample_entity_frame(
    csv_path: Path,
    columns: list[str],
    entity_id_column: str,
    sample_config: dict,
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    metadata_columns = metadata_columns or []
    quoted_columns = [quote_identifier(column) for column in columns]
    row_signature = "md5(concat_ws(chr(31), {}))".format(", ".join(quoted_columns))
    group_filters = []
    if sample_config["require_positive_pairs"]:
        group_filters.append("count(*) >= 2")
    if sample_config["require_variation"]:
        group_filters.append(f"count(distinct {row_signature}) > 1")
    having_clause = f"having {' and '.join(group_filters)}" if group_filters else ""

    selected_columns = [entity_id_column, *columns, *metadata_columns]
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
    return duckdb.sql(query).df()


def load_benchmarks(
    manifest_path: Path, names: list[str], pair_scope: str
) -> list[dict]:
    with manifest_path.open() as file:
        manifest = json.load(file)

    benchmarks = []
    for name in names:
        config = manifest[name]
        if config["kind"] != "single_table":
            raise ValueError(f"Unsupported benchmark kind for {name}: {config['kind']}")

        invalid_field_types = set(config["field_types"].values()) - VALID_FIELD_TYPES
        if invalid_field_types:
            raise ValueError(
                f"Benchmark {name} has invalid field types: {sorted(invalid_field_types)}"
            )

        columns = list(config["field_types"])
        metadata_columns = []
        if pair_scope == "cross_source":
            if "source_table" not in config.get("ignored_columns", []):
                raise ValueError(
                    f"Benchmark {name} does not define source_table metadata"
                )
            metadata_columns.append("source_table")

        csv_path = manifest_path.parent / config["file"]
        if "entity_sample" in config:
            data = sample_entity_frame(
                csv_path,
                columns,
                config["entity_id_column"],
                config["entity_sample"],
                metadata_columns,
            )
        else:
            data = pd.read_csv(
                csv_path,
                usecols=[config["entity_id_column"], *columns, *metadata_columns],
                dtype=str,
            )

        benchmarks.append(
            {
                "name": name,
                "records": data[columns].fillna("").astype(str).reset_index(drop=True),
                "field_types": list(config["field_types"].values()),
                "entity_ids": data[config["entity_id_column"]].to_numpy(),
                "pair_scope": pair_scope,
                "source_table_values": (
                    data["source_table"].to_numpy()
                    if pair_scope == "cross_source"
                    else None
                ),
            }
        )
    return benchmarks


def import_pyjedai():
    try:
        from pyjedai.block_building import StandardBlocking
        from pyjedai.block_cleaning import BlockFiltering, BlockPurging
        from pyjedai.comparison_cleaning import WeightedEdgePruning
        from pyjedai.datamodel import Data
        from pyjedai.matching import EntityMatching
    except ImportError as error:
        raise SystemExit(
            "pyJedAI is not importable. Install it first, for example with "
            "`uv pip install -e competitor_repos/pyJedAI`."
        ) from error
    return Data, StandardBlocking, BlockPurging, BlockFiltering, WeightedEdgePruning, EntityMatching


def scores_from_pyjedai_graph(graph, n_records: int) -> np.ndarray:
    scores = np.zeros((n_records, n_records), dtype=float)
    for left, right, data in graph.edges(data=True):
        probability = float(data.get("weight", 1.0))
        scores[int(left), int(right)] = probability
        scores[int(right), int(left)] = probability
    return scores


def connected_component_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    graph = nx.Graph()
    graph.add_nodes_from(range(scores.shape[0]))
    left, right = np.triu_indices(scores.shape[0], k=1)
    for i, j in zip(left[scores[left, right] >= threshold], right[scores[left, right] >= threshold]):
        graph.add_edge(int(i), int(j))

    labels = np.empty(scores.shape[0], dtype=int)
    for label, component in enumerate(nx.connected_components(graph)):
        for node in component:
            labels[node] = label
    return labels


def evaluate_scores(
    name: str,
    entity_ids: np.ndarray,
    scores: np.ndarray,
    source_table_values: np.ndarray | None,
) -> dict:
    true_adjacency = entity_ids[:, None] == entity_ids[None, :]
    pair_mask = np.triu(np.ones_like(true_adjacency, dtype=bool), k=1)
    if source_table_values is not None:
        pair_mask &= source_table_values[:, None] != source_table_values[None, :]
    targets = true_adjacency[pair_mask]
    probabilities = scores[pair_mask]
    predictions = probabilities >= 0.5
    candidate_mask = probabilities > 0.0

    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    threshold_f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best_threshold_index = int(np.argmax(threshold_f1))

    predicted_labels = connected_component_labels(scores, threshold=0.5)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%.*",
            category=UserWarning,
        )
        adjusted_rand = adjusted_rand_score(entity_ids, predicted_labels)

    return {
        "name": name,
        "n_records": len(entity_ids),
        "n_entities": len(np.unique(entity_ids)),
        "prevalence": float(targets.mean()),
        "candidate_pairs": int(candidate_mask.sum()),
        "candidate_fraction": float(candidate_mask.mean()),
        "true_pair_candidate_recall": float(candidate_mask[targets].mean()),
        "roc_auc": roc_auc_score(targets, probabilities),
        "pr_auc": average_precision_score(targets, probabilities),
        "pair_precision": precision_score(targets, predictions, zero_division=0),
        "pair_recall": recall_score(targets, predictions, zero_division=0),
        "pair_f1": f1_score(targets, predictions, zero_division=0),
        "best_pair_f1": float(threshold_f1[best_threshold_index]),
        "best_pair_threshold": float(thresholds[best_threshold_index]),
        "adjusted_rand": adjusted_rand,
    }


def run_pyjedai_benchmark(benchmark: dict, metric: str, comparison_cleaning: str) -> dict:
    (
        Data,
        StandardBlocking,
        BlockPurging,
        BlockFiltering,
        WeightedEdgePruning,
        EntityMatching,
    ) = import_pyjedai()
    import ray

    records = benchmark["records"].copy()
    records.insert(0, "_record_id", np.arange(len(records)).astype(str))

    start = time.perf_counter()
    status = "completed"
    error = ""
    try:
        ray.is_initialized = lambda: True
        data = Data(
            dataset_1=records,
            id_column_name_1="_record_id",
            attributes_1=list(benchmark["records"].columns),
        )
        blocks = StandardBlocking(disable_ray=True).build_blocks(data, tqdm_disable=True)
        blocks = BlockPurging().process(blocks, data, tqdm_disable=True)
        blocks = BlockFiltering(ratio=0.9).process(blocks, data, tqdm_disable=True)
        if comparison_cleaning == "weighted_edge_pruning":
            blocks = WeightedEdgePruning(weighting_scheme="CBS").process(
                blocks, data, tqdm_disable=True
            )
        if blocks:
            graph = EntityMatching(
                metric=metric,
                similarity_threshold=0.0,
                attributes=list(benchmark["records"].columns),
            ).predict(blocks, data, tqdm_disable=True)
            scores = scores_from_pyjedai_graph(graph, len(records))
        else:
            scores = np.zeros((len(records), len(records)), dtype=float)
        metrics = evaluate_scores(
            benchmark["name"],
            benchmark["entity_ids"],
            scores,
            benchmark["source_table_values"],
        )
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        metrics = {
            "name": benchmark["name"],
            "n_records": len(records),
            "n_entities": len(np.unique(benchmark["entity_ids"])),
            "prevalence": np.nan,
            "candidate_pairs": np.nan,
            "candidate_fraction": np.nan,
            "true_pair_candidate_recall": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "pair_precision": np.nan,
            "pair_recall": np.nan,
            "pair_f1": np.nan,
            "best_pair_f1": np.nan,
            "best_pair_threshold": np.nan,
            "adjusted_rand": np.nan,
        }

    metrics.update(
        {
            "baseline": "pyjedai_standard_blocking",
            "policy": f"standard_blocking_clean_{comparison_cleaning}_{metric}",
            "pair_scope": benchmark["pair_scope"],
            "status": status,
            "error": error,
            "seconds": time.perf_counter() - start,
            "blocking_policy": "StandardBlocking + BlockPurging + BlockFiltering(0.9)",
            "comparison_cleaning": comparison_cleaning,
            "matching_policy": f"EntityMatching(metric={metric}, similarity_threshold=0.0)",
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="benchmark_data/manifest.json", type=Path
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["fodors_zagats"],
    )
    parser.add_argument(
        "--output",
        default=Path("results/baselines/pyjedai_standard_blocking.csv"),
        type=Path,
    )
    parser.add_argument("--metric", default="jaccard")
    parser.add_argument(
        "--comparison-cleaning",
        choices=["none", "weighted_edge_pruning"],
        default="none",
    )
    parser.add_argument(
        "--pair-scope",
        choices=["all_pairs", "cross_source"],
        default="all_pairs",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    benchmarks = load_benchmarks(
        args.manifest, args.benchmarks, pair_scope=args.pair_scope
    )
    rows = []
    with open(os.devnull, "w") as devnull:
        output_context = (
            contextlib.redirect_stdout(devnull)
            if args.quiet
            else contextlib.nullcontext()
        )
        error_context = (
            contextlib.redirect_stderr(devnull)
            if args.quiet
            else contextlib.nullcontext()
        )
        with output_context, error_context:
            for benchmark in benchmarks:
                rows.append(
                    run_pyjedai_benchmark(
                        benchmark,
                        metric=args.metric,
                        comparison_cleaning=args.comparison_cleaning,
                    )
                )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
