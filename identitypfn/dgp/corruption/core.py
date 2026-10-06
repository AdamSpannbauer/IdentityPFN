"""Sample field-level observation rates and apply one edit per selected cell."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .character_level import delete_character, edit_text, swap_characters
from .llm import edit_field_with_ollama
from .text_level import abbreviate, substitute_nickname


def _operations(field: str, field_type: str, allow_ollama: bool) -> list[str]:
    operations = ["delete", "swap"]
    if field_type == "text":
        operations.extend(("ocr", "keyboard", "insert", "substitute", "nickname"))
    field_name = field.lower()
    if ("name" in field_name or "company" in field_name) and "user" not in field_name:
        operations.append("abbreviate")
    if allow_ollama:
        operations.append("llm")
    return operations


def _valid_edit(value: object, edited: object, field_type: str) -> object:
    if edited is None or pd.isna(edited):
        return value
    if field_type == "numeric":
        try:
            return float(edited)
        except (TypeError, ValueError):
            return value
    if field_type == "date" and pd.isna(pd.to_datetime(edited, errors="coerce", format="mixed")):
        return value
    return str(edited)


def apply_corruptions(
    clean_records: pd.DataFrame,
    field_types: Mapping[str, str],
    rng: np.random.Generator,
    allow_ollama: bool = False,
    ollama_model: str = "llama3.2:1b",
) -> pd.DataFrame:
    """Independently observe every record, including singleton records."""
    observed = clean_records.astype(object).copy()
    for field in clean_records:
        field_type = field_types[field]
        missing_rate = float(rng.uniform(0, 0.2))
        corruption_rate = float(rng.uniform(0, 0.2))
        operations = _operations(field, field_type, allow_ollama)
        column_index = clean_records.columns.get_loc(field)
        for row_index, value in enumerate(clean_records[field]):
            if pd.isna(value):
                continue
            if rng.random() < missing_rate:
                observed.iat[row_index, column_index] = np.nan
                continue
            if rng.random() >= corruption_rate:
                continue
            operation = str(rng.choice(operations))
            if operation == "delete":
                edited = delete_character(value, rng)
            elif operation == "swap":
                edited = swap_characters(value, rng)
            elif operation in {"ocr", "keyboard", "insert", "substitute"}:
                edited = edit_text(value, operation, rng)
            elif operation == "nickname":
                edited = substitute_nickname(value, rng)
            elif operation == "abbreviate":
                edited = abbreviate(value, rng)
            else:
                edited = edit_field_with_ollama(
                    field,
                    field_type,
                    value,
                    clean_records.iloc[row_index].to_dict(),
                    ollama_model,
                )
            observed.iat[row_index, column_index] = _valid_edit(value, edited, field_type)
    return observed
