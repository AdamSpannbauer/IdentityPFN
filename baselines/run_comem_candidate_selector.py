import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

DATASETS = {
    "fodors_zagats": {
        "directory": Path("benchmark_data/deepmatcher_provided/Fodors-Zagats"),
        "left_file": "tableA.csv",
        "right_file": "tableB.csv",
        "matches_file": "matches.csv",
        "left_id_column": "id",
        "right_id_column": "id",
        "left_match_column": "fodors_id",
        "right_match_column": "zagats_id",
    },
    "dblp_acm": {
        "directory": Path("benchmark_data/deepmatcher_provided/DBLP-ACM"),
        "left_file": "tableA.csv",
        "right_file": "tableB.csv",
        "matches_file": "matches.csv",
        "left_id_column": "id",
        "right_id_column": "id",
        "left_match_column": "idDBLP",
        "right_match_column": "idACM",
        "encoding": "latin1",
    },
    "dblp_google_scholar": {
        "directory": Path("benchmark_data/deepmatcher_provided/DBLP-GoogleScholar"),
        "left_file": "tableA.csv",
        "right_file": "tableB.csv",
        "matches_file": "matches.csv",
        "left_id_column": "id",
        "right_id_column": "id",
        "left_match_column": "idDBLP",
        "right_match_column": "idScholar",
        "encoding": "utf-8-sig",
    },
    "walmart_amazon": {
        "directory": Path("benchmark_data/deepmatcher_provided/Walmart-Amazon"),
        "left_file": "tableA.csv",
        "right_file": "tableB.csv",
        "matches_file": "matches.csv",
        "left_id_column": "custom_id",
        "right_id_column": "custom_id",
        "left_match_column": "id1",
        "right_match_column": "id2",
        "short_fields": {
            "left": ["id", "upc", "brand", "groupname", "title", "price", "modelno"],
            "right": ["asin", "brand", "modelno", "category1", "title", "price"],
        },
    },
    "amazon_google": {
        "directory": Path("benchmark_data/deepmatcher_provided/Amazon-Google"),
        "left_file": "tableA.csv",
        "right_file": "tableB.csv",
        "matches_file": "matches.csv",
        "left_id_column": "id",
        "right_id_column": "id",
        "left_match_column": "idAmazon",
        "right_match_column": "idGoogleBase",
        "encoding": "latin1",
        "short_fields": {
            "left": ["id", "title", "manufacturer", "price"],
            "right": ["id", "name", "manufacturer", "price"],
        },
    },
}


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def serialize_record(row: pd.Series, fields: list[str] | None = None) -> str:
    parts = []
    items = row[fields].items() if fields is not None else row.items()
    for key, value in items:
        if key == "id" or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(f"{key}: {text}")
    return "; ".join(parts)


def make_prompt(anchor: str, candidates: list[str]) -> str:
    candidate_text = "\n".join(
        f"[{index}] {candidate}" for index, candidate in enumerate(candidates, start=1)
    )
    return (
        "Select a record from the following candidates that refers to the same "
        "real-world entity as the given record. Answer with the corresponding "
        'record number surrounded by "[]" or "[0]" if there is none.\n\n'
        f"Given entity record:\n{anchor}\n\n"
        f"Candidate records:\n{candidate_text}\n"
    )


def read_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame, set[tuple[str, str]]]:
    config = DATASETS[name]
    directory = config["directory"]
    encoding = config.get("encoding", "utf-8")
    left = pd.read_csv(
        directory / config["left_file"], dtype=str, encoding=encoding
    ).fillna("")
    right = pd.read_csv(
        directory / config["right_file"], dtype=str, encoding=encoding
    ).fillna("")
    matches = pd.read_csv(
        directory / config["matches_file"], dtype=str, encoding=encoding
    ).fillna("")
    for frame in [left, right, matches]:
        frame.columns = [
            column.lstrip("\ufeff").lstrip("ï»¿") for column in frame.columns
        ]
    match_set = {
        (normalize_id(row[config["left_match_column"]]), normalize_id(row[config["right_match_column"]]))
        for _, row in matches.iterrows()
    }
    return left, right, match_set


def generate_candidate_examples(
    dataset: str,
    top_k: int,
    limit: int | None,
    offset: int,
    field_mode: str,
) -> list[dict]:
    left, right, match_set = read_dataset(dataset)
    config = DATASETS[dataset]
    short_fields = config.get("short_fields", {})
    left_fields = short_fields.get("left") if field_mode == "short" else None
    right_fields = short_fields.get("right") if field_mode == "short" else None
    left = left.iloc[offset:]
    if limit is not None:
        left = left.head(limit)

    right_records = right.apply(
        lambda row: serialize_record(row, right_fields), axis=1
    ).tolist()
    left_records = left.apply(
        lambda row: serialize_record(row, left_fields), axis=1
    ).tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    right_matrix = vectorizer.fit_transform(right_records)
    left_matrix = vectorizer.transform(left_records)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
    neighbors.fit(right_matrix)
    _, indices = neighbors.kneighbors(
        left_matrix,
        n_neighbors=min(top_k, len(right_records)),
    )

    examples = []
    right_ids = right[config["right_id_column"]].map(normalize_id).tolist()
    for local_index, (_, left_row) in enumerate(left.iterrows()):
        row_index = offset + local_index
        candidate_indices = indices[local_index].tolist()
        candidate_ids = [right_ids[index] for index in candidate_indices]
        left_id = normalize_id(left_row[config["left_id_column"]])
        true_right_ids = sorted(
            right_id for candidate_left_id, right_id in match_set if candidate_left_id == left_id
        )
        examples.append(
            {
                "dataset": dataset,
                "field_mode": field_mode,
                "row_index": row_index,
                "id_left": left_id,
                "anchor": serialize_record(left_row, left_fields),
                "candidates": [
                    {
                        "id_right": candidate_id,
                        "record": right_records[candidate_index],
                        "label": (left_id, candidate_id) in match_set,
                    }
                    for candidate_id, candidate_index in zip(
                        candidate_ids, candidate_indices, strict=True
                    )
                ],
                "true_right_ids": true_right_ids,
                "candidate_recall_hit": any(
                    candidate_id in set(true_right_ids) for candidate_id in candidate_ids
                )
                if true_right_ids
                else None,
            }
        )
    return examples


def load_completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open() as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                completed.add((row["dataset"], row["row_index"]))
    return completed


def parse_prediction(content: str | None, n_candidates: int) -> int:
    if content is None:
        return 0
    match = re.search(r"\[(\d+)\]", content.strip())
    if not match:
        return 0
    index = int(match.group(1))
    if 0 <= index <= n_candidates:
        return index
    return 0


def request_prediction(
    example: dict,
    model: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
) -> dict:
    prompt = make_prompt(
        example["anchor"],
        [candidate["record"] for candidate in example["candidates"]],
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "seed": 42,
        "temperature": 0.0,
        "max_tokens": 4,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            OPENAI_CHAT_COMPLETIONS_URL,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            started_at = time.perf_counter()
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            elapsed_seconds = time.perf_counter() - started_at
            response_json = json.loads(raw)
            content = response_json["choices"][0]["message"]["content"]
            prediction_index = parse_prediction(content, len(example["candidates"]))
            prediction_id = (
                example["candidates"][prediction_index - 1]["id_right"]
                if prediction_index > 0
                else None
            )
            return {
                **example,
                "model": model,
                "prompt": prompt,
                "response": content,
                "prediction_index": prediction_index,
                "prediction_id": prediction_id,
                "prediction_is_match": prediction_id in set(example["true_right_ids"])
                if prediction_id is not None
                else False,
                "usage": response_json.get("usage", {}),
                "elapsed_seconds": elapsed_seconds,
            }
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            last_error = repr(error)
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))

    return {
        **example,
        "model": model,
        "prompt": prompt,
        "response": None,
        "prediction_index": 0,
        "prediction_id": None,
        "prediction_is_match": False,
        "usage": {},
        "elapsed_seconds": None,
        "error": last_error,
    }


def summarize(output_path: Path) -> None:
    rows = []
    with output_path.open() as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("No completed rows.")
        return

    pair_labels = []
    pair_predictions = []
    for row in rows:
        for index, candidate in enumerate(row["candidates"], start=1):
            pair_labels.append(bool(candidate["label"]))
            pair_predictions.append(row["prediction_index"] == index)

    print(classification_report(pair_labels, pair_predictions, digits=4, zero_division=0))
    print(confusion_matrix(pair_labels, pair_predictions))

    matched_rows = [row for row in rows if row["true_right_ids"]]
    candidate_hits = [
        row["candidate_recall_hit"]
        for row in matched_rows
        if row["candidate_recall_hit"] is not None
    ]
    candidate_recall = sum(candidate_hits) / len(candidate_hits) if candidate_hits else 0
    end_to_end_recall = (
        sum(row["prediction_is_match"] for row in matched_rows) / len(matched_rows)
        if matched_rows
        else 0
    )
    predictions = [row for row in rows if row["prediction_id"] is not None]
    end_to_end_precision = (
        sum(row["prediction_is_match"] for row in predictions) / len(predictions)
        if predictions
        else 0
    )
    end_to_end_f1 = (
        2
        * end_to_end_precision
        * end_to_end_recall
        / max(end_to_end_precision + end_to_end_recall, 1e-12)
    )
    prompt_tokens = sum(row.get("usage", {}).get("prompt_tokens", 0) for row in rows)
    completion_tokens = sum(
        row.get("usage", {}).get("completion_tokens", 0) for row in rows
    )
    errors = sum("error" in row for row in rows)
    print(
        f"rows={len(rows)} errors={errors} "
        f"matched_rows={len(matched_rows)} candidate_recall={candidate_recall:.4f} "
        f"selector_precision={end_to_end_precision:.4f} "
        f"selector_recall={end_to_end_recall:.4f} "
        f"selector_f1={end_to_end_f1:.4f} "
        f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a ComEM-style OpenAI selector on top-k two-table candidates."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="fodors_zagats")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--field-mode", choices=["all", "short"], default="all")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/comem_candidate_selector.jsonl"),
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples = generate_candidate_examples(
        dataset=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        offset=args.offset,
        field_mode=args.field_mode,
    )
    completed = load_completed_keys(args.output)
    examples = [
        example
        for example in examples
        if (example["dataset"], example["row_index"]) not in completed
    ]
    print(
        f"remaining_examples={len(examples)} dataset={args.dataset} "
        f"top_k={args.top_k} field_mode={args.field_mode} output={args.output}",
        flush=True,
    )
    if not examples:
        summarize(args.output)
        return

    started_at = time.perf_counter()
    completed_now = 0
    with args.output.open("a") as output_file:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    request_prediction,
                    example,
                    args.model,
                    api_key,
                    args.timeout_seconds,
                    args.max_retries,
                )
                for example in examples
            ]
            for future in as_completed(futures):
                result = future.result()
                output_file.write(json.dumps(result, sort_keys=True) + "\n")
                output_file.flush()
                completed_now += 1
                if completed_now % args.progress_every == 0:
                    elapsed_minutes = (time.perf_counter() - started_at) / 60
                    print(
                        f"completed={completed_now}/{len(examples)} "
                        f"elapsed={elapsed_minutes:.1f}m",
                        flush=True,
                    )

    summarize(args.output)


if __name__ == "__main__":
    main()
