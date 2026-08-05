import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


def serialize_profile(profile: dict) -> str:
    parts = []
    for key in ["fullname", "email", "phone", "addr", "dob"]:
        value = profile.get(key)
        if isinstance(value, list):
            value = " | ".join(str(item) for item in value)
        if value:
            parts.append(f"{key}: {value}")
    return "; ".join(parts)


def make_prompt(anchor: str, candidate: str) -> str:
    return (
        "Select a record from the following candidates that refers to the same "
        'real-world entity as the given record. Answer with "[1]" if the '
        'candidate is the same entity, or "[0]" if there is none.\n\n'
        f"Given entity record:\n{anchor}\n\n"
        f"Candidate records:\n[1] {candidate}\n"
    )


def load_examples(path: Path, limit: int | None, offset: int) -> list[dict]:
    examples = []
    with path.open() as file:
        for row_index, line in enumerate(file):
            if row_index < offset:
                continue
            if limit is not None and len(examples) >= limit:
                break
            row = json.loads(line)
            examples.append(
                {
                    "row_index": row_index,
                    "anchor": serialize_profile(row["profile1"]),
                    "candidate": serialize_profile(row["profile2"]),
                    "label": row["match"] == "True",
                }
            )
    return examples


def load_completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open() as file:
        for line in file:
            if line.strip():
                completed.add(json.loads(line)["row_index"])
    return completed


def parse_prediction(content: str) -> bool:
    match = re.search(r"\[(\d+)\]", content.strip())
    return bool(match and int(match.group(1)) == 1)


def request_prediction(
    example: dict,
    model: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
) -> dict:
    prompt = make_prompt(example["anchor"], example["candidate"])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "seed": 42,
        "temperature": 0.0,
        "max_tokens": 3,
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
            return {
                **example,
                "model": model,
                "prompt": prompt,
                "response": content,
                "prediction": parse_prediction(content),
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
        "prediction": False,
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

    labels = [row["label"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    print(classification_report(labels, predictions, digits=4, zero_division=0))
    print(confusion_matrix(labels, predictions))
    prompt_tokens = sum(row.get("usage", {}).get("prompt_tokens", 0) for row in rows)
    completion_tokens = sum(
        row.get("usage", {}).get("completion_tokens", 0) for row in rows
    )
    errors = sum("error" in row for row in rows)
    print(
        f"rows={len(rows)} errors={errors} "
        f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight ComEM-style OpenAI selector on BPID pairs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark_data/bpid/matching_dataset.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/comem_bpid_selector.jsonl"),
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
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
    examples = load_examples(args.input, args.limit, args.offset)
    completed = load_completed_indices(args.output)
    examples = [
        example for example in examples if example["row_index"] not in completed
    ]
    print(f"remaining_examples={len(examples)} output={args.output}")
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
                        f"elapsed={elapsed_minutes:.1f}m"
                    )

    summarize(args.output)


if __name__ == "__main__":
    main()
