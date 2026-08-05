import json
import time
import warnings
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import schedulefree
import torch

from .model import NanoERPFNLinker, NanoERPFNModel
from .data_loader import SyntheticWorldDataLoader
from .tokenizer import Tokenizer

from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import get_default_device, set_randomness_seed

set_randomness_seed(0)


VALID_BENCHMARK_KINDS = {"single_table"}
VALID_FIELD_TYPES = {
    "text",
    "categorical",
    "numeric",
    "date",
    "identifier",
    "email",
    "phone",
}
VALID_ENTITY_SAMPLE_KEYS = {
    "n_entities",
    "seed",
    "require_positive_pairs",
    "require_variation",
}


def validate_benchmark_config(name: str, config: dict) -> None:
    required_keys = {"kind", "description", "file", "entity_id_column", "field_types"}
    missing_keys = required_keys - set(config)
    if missing_keys:
        raise ValueError(f"Benchmark {name!r} is missing keys: {sorted(missing_keys)}")

    if config["kind"] not in VALID_BENCHMARK_KINDS:
        raise ValueError(f"Unsupported benchmark kind for {name!r}: {config['kind']!r}")

    invalid_field_types = {
        field_type
        for field_type in config["field_types"].values()
        if field_type not in VALID_FIELD_TYPES
    }
    if invalid_field_types:
        raise ValueError(
            f"Benchmark {name!r} has invalid field types: {sorted(invalid_field_types)}"
        )

    ignored_columns = set(config.get("ignored_columns", []))
    model_columns = set(config["field_types"])
    overlap = ignored_columns & model_columns
    if overlap:
        raise ValueError(
            f"Benchmark {name!r} has ignored columns also used as fields: {sorted(overlap)}"
        )

    sample_config = config.get("entity_sample")
    if sample_config is not None:
        unknown_keys = set(sample_config) - VALID_ENTITY_SAMPLE_KEYS
        if unknown_keys:
            raise ValueError(
                f"Benchmark {name!r} has unknown entity_sample keys: {sorted(unknown_keys)}"
            )
        if not isinstance(sample_config.get("n_entities"), int):
            raise ValueError(f"Benchmark {name!r} entity_sample.n_entities must be int")
        if not isinstance(sample_config.get("seed"), int):
            raise ValueError(f"Benchmark {name!r} entity_sample.seed must be int")
        for key in ("require_positive_pairs", "require_variation"):
            if not isinstance(sample_config.get(key), bool):
                raise ValueError(f"Benchmark {name!r} entity_sample.{key} must be bool")


def sample_entity_frame(
    csv_path: Path,
    columns: list[str],
    entity_id_column: str,
    sample_config: dict,
) -> pd.DataFrame:
    """Apply deterministic entity-level CSV sampling with DuckDB."""
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


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


def load_benchmarks(
    manifest_path: str | Path, names: list[str] | None = None
) -> list[dict]:
    manifest_path = Path(manifest_path)
    with manifest_path.open() as file:
        manifest = json.load(file)

    benchmarks = []
    for name, config in manifest.items():
        if name.startswith("_"):
            continue
        if names is not None and name not in names:
            continue

        validate_benchmark_config(name, config)
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


def evaluate_benchmark(linker, benchmark):
    entity_ids = benchmark["entity_ids"]
    true_adjacency = entity_ids[:, None] == entity_ids[None, :]
    pair_mask = np.triu(np.ones_like(true_adjacency, dtype=bool), k=1)

    predicted_labels = linker.fit_predict(
        benchmark["records"], benchmark["field_types"]
    )
    probabilities = linker.adjacency_proba_[pair_mask]
    targets = true_adjacency[pair_mask]
    if np.unique(targets).size < 2:
        raise ValueError(
            f"Benchmark {benchmark['name']!r} must contain positive and negative pairs"
        )
    predictions = probabilities >= linker.threshold

    threshold_precision, threshold_recall, thresholds = precision_recall_curve(
        targets, probabilities
    )
    threshold_f1 = (
        2
        * threshold_precision[:-1]
        * threshold_recall[:-1]
        / np.maximum(threshold_precision[:-1] + threshold_recall[:-1], 1e-12)
    )
    best_threshold_index = np.argmax(threshold_f1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%.*",
            category=UserWarning,
        )
        adjusted_rand = adjusted_rand_score(entity_ids, predicted_labels)

    return {
        "name": benchmark["name"],
        "n_records": len(entity_ids),
        "n_entities": len(np.unique(entity_ids)),
        "prevalence": float(targets.mean()),
        "roc_auc": roc_auc_score(targets, probabilities),
        "pr_auc": average_precision_score(targets, probabilities),
        "pair_precision": precision_score(targets, predictions, zero_division=0),
        "pair_recall": recall_score(targets, predictions, zero_division=0),
        "pair_f1": f1_score(targets, predictions, zero_division=0),
        "best_pair_f1": float(threshold_f1[best_threshold_index]),
        "best_pair_threshold": float(thresholds[best_threshold_index]),
        "adjusted_rand": adjusted_rand,
    }


def eval_frame(linker, benchmarks):
    return pd.DataFrame(
        [evaluate_benchmark(linker, benchmark) for benchmark in benchmarks]
    ).set_index("name")


def eval(linker, benchmarks):
    scores = eval_frame(linker, benchmarks)
    return {
        metric: float(scores[metric].mean())
        for metric in scores
        if metric not in {"n_records", "n_entities"}
    }


def train(
    model: NanoERPFNModel,
    tokenizer: Tokenizer,
    prior: SyntheticWorldDataLoader,
    lr: float = 1e-4,
    device: torch.device = None,
    steps_per_eval=10,
    steps_per_print=5,
    eval_func=None,
):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoERPFNModel) our PyTorch model
        prior: (SyntheticWorldDataLoader) er data generating dataloader
        lr: (float) learning rate
        device: (torch.device) the device we are using
        steps_per_eval: (int) how many steps we wait before running evaluation again
        steps_per_print: (int) how many steps we wait before printing progress
        eval_func: a function that takes in a classifier and returns a dict containing the average scores
                   for some metrics and datasets

    Returns:
        (model) our trained numpy model
        (list) a list containing our eval history, each entry is the real time used for training so far together
               with a dict mapping metric names to their average values accross a list of datasets
    """
    if not device:
        device = get_default_device()

    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=lr, weight_decay=0.0
    )

    model.train()
    optimizer.train()

    train_time = 0
    eval_history = []
    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()

            tokenized_cells = tokenizer(full_data["records"], full_data["field_types"])
            tokenized_cells = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_cells.items()
            }

            adjacency_target = full_data["adjacency"].to(device).float()

            adjacency_logits = model(tokenized_cells)

            pair_mask = torch.triu(
                torch.ones_like(adjacency_target, dtype=torch.bool), diagonal=1
            )
            pair_logits = adjacency_logits[pair_mask]
            pair_targets = adjacency_target[pair_mask]

            num_positive = pair_targets.sum()
            if num_positive == 0:
                continue
            num_negative = pair_targets.numel() - num_positive

            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pair_logits,
                pair_targets,
                pos_weight=num_negative / num_positive,
            )
            loss.backward()
            total_loss = loss.cpu().detach().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            status = f"step {step + 1:04d} | time {train_time:7.1f}s | loss {total_loss:7.4f}"
            should_print = (step + 1) % steps_per_print == 0
            should_eval = eval_func is not None and (step + 1) % steps_per_eval == 0

            if should_eval:
                model.eval()
                optimizer.eval()

                linker = NanoERPFNLinker(model, tokenizer, device=device)
                scores = eval_func(linker)

                eval_history.append((train_time, scores))
                score_str = " | ".join([f"{k} {v:7.4f}" for k, v in scores.items()])
                status += f" | {score_str}"

                model.train()
                optimizer.train()

            if should_print or should_eval:
                print(status)
    except KeyboardInterrupt:
        pass

    return model, eval_history


if __name__ == "__main__":
    print("Running test pass...")
    device = get_default_device()

    print("\tLoading text tokenizer...")
    tokenizer = Tokenizer(text_backend="fasttext")

    print("\tLoading eval benchmark data...")
    benchmarks = load_benchmarks(
        "benchmark_data/manifest.json",
        names=[
            "fake_1000",
            "sim_er_data",
        ],
    )
    eval_func = partial(eval, benchmarks=benchmarks)

    for record_representation in ["mean_pool", "entity_target_column"]:
        for adjacency_decoder in ["pair_mlp", "slot_coassignment", "slot_pair_mlp"]:
            print(
                f"\nrecord_representation={record_representation}, "
                f"adjacency_decoder={adjacency_decoder}"
            )
            print("\tInstantiating model...")
            model = NanoERPFNModel(
                embedding_size=96,
                text_embedding_size=tokenizer.text_embedding_dim,
                num_attention_heads=4,
                mlp_hidden_size=192,
                num_layers=3,
                max_categories=20,
                record_representation=record_representation,
                adjacency_decoder=adjacency_decoder,
            )

            prior = SyntheticWorldDataLoader(
                num_steps=8,
                batch_size=8,
            )

            print("\tStarting training...")
            model, history = train(
                model,
                tokenizer,
                prior,
                lr=4e-3,
                steps_per_eval=4,
                steps_per_print=1,
                eval_func=eval_func,
            )

            print("\tFinal evaluation:")
            print(
                eval_frame(NanoERPFNLinker(model, tokenizer, device=device), benchmarks)
            )
