from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


LinkageColumns = Sequence[str] | Mapping[str, tuple[str, str]] | None


def score_linkage(
    model,
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    columns: LinkageColumns = None,
    field_types: Sequence[str] | str = "infer",
    k: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    if k < 1:
        raise ValueError("k must be positive")

    resolved_columns = _resolve_linkage_columns(left, right, columns)
    canonical_columns = [column for column, _left_column, _right_column in resolved_columns]
    left_records = pd.DataFrame(
        {
            canonical_column: left[left_column]
            for canonical_column, left_column, _right_column in resolved_columns
        },
        index=left.index,
    )
    right_records = pd.DataFrame(
        {
            canonical_column: right[right_column]
            for canonical_column, _left_column, right_column in resolved_columns
        },
        index=right.index,
    )

    left_internal_index = [f"left:{index}" for index in range(len(left_records))]
    right_internal_index = [f"right:{index}" for index in range(len(right_records))]
    stacked = pd.concat(
        [
            left_records.set_axis(left_internal_index),
            right_records.set_axis(right_internal_index),
        ],
        axis=0,
    )
    scores = model.predict_proba(stacked, field_types=field_types, verbose=verbose)
    cross_scores = scores.loc[left_internal_index, right_internal_index]

    values = cross_scores.to_numpy()
    top_k = min(k, values.size)
    flat_values = values.ravel()
    top_positions = np.argpartition(flat_values, -top_k)[-top_k:]
    top_positions = top_positions[np.argsort(flat_values[top_positions])[::-1]]

    rows = []
    for flat_position in top_positions:
        left_position, right_position = np.unravel_index(flat_position, values.shape)
        row: dict[str, Any] = {
            "left_index": left_records.index[left_position],
            "right_index": right_records.index[right_position],
            "score": float(values[left_position, right_position]),
        }
        for column in canonical_columns:
            row[f"left_{column}"] = left_records.iloc[left_position][column]
            row[f"right_{column}"] = right_records.iloc[right_position][column]
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def _resolve_linkage_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: LinkageColumns,
) -> list[tuple[str, str, str]]:
    if columns is None:
        common_columns = [column for column in left.columns if column in right.columns]
        if not common_columns:
            raise ValueError("left and right have no common columns")
        return [(column, column, column) for column in common_columns]

    if isinstance(columns, Mapping):
        resolved = []
        for canonical_column, source_columns in columns.items():
            if len(source_columns) != 2:
                raise ValueError("mapped columns must be (left_column, right_column)")
            left_column, right_column = source_columns
            _require_column(left, left_column, "left")
            _require_column(right, right_column, "right")
            resolved.append((canonical_column, left_column, right_column))
        if not resolved:
            raise ValueError("columns must not be empty")
        return resolved

    if isinstance(columns, str):
        raise TypeError(
            "columns must be a sequence of column names or a mapping, not a string"
        )

    resolved = []
    for column in columns:
        _require_column(left, column, "left")
        _require_column(right, column, "right")
        resolved.append((column, column, column))
    if not resolved:
        raise ValueError("columns must not be empty")
    return resolved


def _require_column(records: pd.DataFrame, column: str, side: str) -> None:
    if column not in records.columns:
        raise ValueError(f"{side} is missing column {column!r}")
