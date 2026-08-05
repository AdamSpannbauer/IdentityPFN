import argparse
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "benchmark",
    "name",
    "n_records",
    "n_entities",
    "prevalence",
    "pr_auc",
    "oracle_pair_f1",
    "best_pair_f1",
    "pair_f1_at_0_5",
    "pair_f1",
    "adjusted_rand",
    "checkpoint_step",
    "step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a compact summary of ER-PFN benchmark results."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("results/paper_benchmark_tables/benchmark_table.csv"),
        help="Benchmark table CSV or run CSV to summarize.",
    )
    parser.add_argument(
        "--benchmark-run",
        action="store_true",
        help="Treat path as a checkpointed benchmark run and summarize final step rows.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to summarize for --benchmark-run. Defaults to final step.",
    )
    return parser.parse_args()


def load_rows(path: Path, benchmark_run: bool, step: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if benchmark_run:
        if "step" not in frame:
            raise ValueError(f"{path} does not contain a step column")
        selected_step = int(frame["step"].max()) if step is None else step
        frame = frame[frame["step"] == selected_step].copy()
        if frame.empty:
            raise ValueError(f"{path} has no rows for step {selected_step}")
    return frame


def compact_summary(frame: pd.DataFrame) -> pd.DataFrame:
    present_columns = [column for column in SUMMARY_COLUMNS if column in frame]
    summary = frame[present_columns].copy()

    if "name" in summary and "benchmark" not in summary:
        summary = summary.rename(columns={"name": "benchmark"})
    if "best_pair_f1" in summary and "oracle_pair_f1" not in summary:
        summary = summary.rename(columns={"best_pair_f1": "oracle_pair_f1"})
    if "pair_f1" in summary and "pair_f1_at_0_5" not in summary:
        summary = summary.rename(columns={"pair_f1": "pair_f1_at_0_5"})
    if "step" in summary and "checkpoint_step" not in summary:
        summary = summary.rename(columns={"step": "checkpoint_step"})

    ordered_columns = [
        "benchmark",
        "n_records",
        "n_entities",
        "prevalence",
        "pr_auc",
        "oracle_pair_f1",
        "pair_f1_at_0_5",
        "adjusted_rand",
        "checkpoint_step",
    ]
    return summary[[column for column in ordered_columns if column in summary]]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.path, args.benchmark_run, args.step)
    summary = compact_summary(rows)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        160,
        "display.float_format",
        "{:.6f}".format,
    ):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
