import argparse
import csv
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


VALID_FIELD_TYPES = {"text", "categorical", "numeric", "date", "identifier"}


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
    return duckdb.sql(query).df()


def load_benchmark(manifest_path: Path, name: str) -> dict:
    with manifest_path.open() as file:
        manifest = json.load(file)

    config = manifest[name]
    if config["kind"] != "single_table":
        raise ValueError(f"Unsupported benchmark kind for {name}: {config['kind']}")

    invalid_field_types = set(config["field_types"].values()) - VALID_FIELD_TYPES
    if invalid_field_types:
        raise ValueError(
            f"Benchmark {name} has invalid field types: {sorted(invalid_field_types)}"
        )

    columns = list(config["field_types"])
    csv_path = manifest_path.parent / config["file"]
    if "entity_sample" in config:
        data = sample_entity_frame(
            csv_path, columns, config["entity_id_column"], config["entity_sample"]
        )
    else:
        data = pd.read_csv(
            csv_path, usecols=[config["entity_id_column"], *columns], dtype=str
        )

    return {
        "name": name,
        "records": data[columns].fillna("").astype(str).reset_index(drop=True),
        "field_types": list(config["field_types"].values()),
        "entity_ids": data[config["entity_id_column"]].to_numpy(),
    }


def serialize_record(row: pd.Series) -> str:
    pieces = []
    for column, value in row.items():
        value = str(value).replace("\t", " ").replace("\n", " ").strip()
        pieces.append(f"COL {column} VAL {value}")
    return " ".join(pieces)


def build_pair_indices(entity_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(len(entity_ids), k=1)
    labels = (entity_ids[left] == entity_ids[right]).astype(int)
    return left, right, labels


def make_split_frame(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_positive = len(positive_indices)
    n_negative = min(n_pairs - n_positive, len(negative_indices))
    selected = np.concatenate([positive_indices, negative_indices[:n_negative]])

    frame = pd.DataFrame(
        {
            "left_index": left[selected],
            "right_index": right[selected],
            "label": labels[selected],
        }
    )
    return frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)


def write_pair_file(path: Path, pairs: pd.DataFrame, serialized_records: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        for row in pairs.itertuples(index=False):
            writer.writerow(
                [
                    serialized_records[int(row.left_index)],
                    serialized_records[int(row.right_index)],
                    int(row.label),
                ]
            )


def write_pair_index(path: Path, pairs: pd.DataFrame) -> None:
    pairs.to_csv(path, index=False)


def prepare_task(
    benchmark: dict,
    output_root: Path,
    train_pairs: int,
    valid_pairs: int,
    test_pairs: int,
    seed: int,
) -> dict:
    task_dir = output_root / benchmark["name"]
    task_dir.mkdir(parents=True, exist_ok=True)

    serialized_records = [
        serialize_record(row) for _, row in benchmark["records"].iterrows()
    ]
    records_frame = benchmark["records"].copy()
    records_frame.insert(0, "record_index", np.arange(len(records_frame)))
    records_frame.insert(1, "entity_id", benchmark["entity_ids"])
    records_frame.insert(2, "serialized", serialized_records)
    records_frame.to_csv(task_dir / "records.csv", index=False)

    with (task_dir / "train_no_label.txt").open("w") as file:
        for record in serialized_records:
            file.write(record + "\n")

    # Sudowoodo's blocking script expects these names for two-table tasks. For
    # dirty-ER setup, both sides intentionally contain the same record universe.
    for stem in ["tableA", "tableB"]:
        with (task_dir / f"{stem}.txt").open("w") as file:
            for record in serialized_records:
                file.write(record + "\n")
        records_frame[["record_index", "serialized"]].to_csv(
            task_dir / f"{stem}.csv", index=False
        )

    left, right, labels = build_pair_indices(benchmark["entity_ids"])
    rng = np.random.default_rng(seed)
    positive_order = rng.permutation(np.flatnonzero(labels == 1))
    negative_order = rng.permutation(np.flatnonzero(labels == 0))

    negative_cursor = 0
    split_names = ["train", "valid", "test"]
    split_sizes = {
        "train": train_pairs,
        "valid": valid_pairs,
        "test": test_pairs,
    }
    positive_splits = np.array_split(positive_order, [int(0.6 * len(positive_order)), int(0.8 * len(positive_order))])
    split_frames = {}
    for split, positive_indices in zip(split_names, positive_splits):
        n_pairs = split_sizes[split]
        n_negative = min(n_pairs - len(positive_indices), len(negative_order) - negative_cursor)
        negative_indices = negative_order[negative_cursor : negative_cursor + n_negative]
        negative_cursor += n_negative
        split_frame = make_split_frame(
            left,
            right,
            labels,
            positive_indices,
            negative_indices,
            n_pairs,
            rng,
        )
        split_frames[split] = split_frame
        write_pair_file(task_dir / f"{split}.txt", split_frame, serialized_records)
        write_pair_index(task_dir / f"{split}_pairs.csv", split_frame)

    all_pairs = pd.DataFrame(
        {
            "left_index": left,
            "right_index": right,
            "label": labels,
        }
    )
    write_pair_file(task_dir / "all_pairs.txt", all_pairs, serialized_records)
    write_pair_index(task_dir / "all_pairs_pairs.csv", all_pairs)

    metadata = {
        "benchmark": benchmark["name"],
        "n_records": len(serialized_records),
        "n_entities": int(len(np.unique(benchmark["entity_ids"]))),
        "n_pairs": int(len(all_pairs)),
        "n_positive_pairs": int(labels.sum()),
        "n_negative_pairs": int((labels == 0).sum()),
        "splits": {
            split: {
                "n_pairs": int(len(frame)),
                "n_positive_pairs": int(frame["label"].sum()),
                "n_negative_pairs": int((frame["label"] == 0).sum()),
            }
            for split, frame in split_frames.items()
        },
        "serialization": "COL {field} VAL {value}",
        "notes": [
            "This is a dirty-ER single-table adaptation for Sudowoodo's pair-file interface.",
            "train_no_label.txt contains one serialized record per line for SSL pretraining.",
            "train/valid/test/all_pairs contain serialized pair rows plus labels because Sudowoodo's loaders require labels.",
            "Sudowoodo --zero still reads labels in train.txt to estimate pseudo-label count thresholds.",
        ],
    }
    with (task_dir / "identitypfn_sudowoodo_metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=Path("benchmark_data/manifest.json"), type=Path)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["amazon_google_price_numeric"],
    )
    parser.add_argument(
        "--output-root",
        default=Path("competitor_repos/sudowoodo/data/em"),
        type=Path,
    )
    parser.add_argument("--train-pairs", type=int, default=5000)
    parser.add_argument("--valid-pairs", type=int, default=2000)
    parser.add_argument("--test-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summaries = []
    for name in args.benchmarks:
        benchmark = load_benchmark(args.manifest, name)
        summaries.append(
            prepare_task(
                benchmark,
                args.output_root,
                args.train_pairs,
                args.valid_pairs,
                args.test_pairs,
                args.seed,
            )
        )

    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
