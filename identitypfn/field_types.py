from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as pandas_types


EMAIL_COLUMN_NAMES = frozenset({"email", "email_address", "e_mail"})
PHONE_COLUMN_NAMES = frozenset(
    {"phone", "phone_number", "mobile", "mobile_phone", "telephone"}
)
IDENTIFIER_COLUMN_NAMES = frozenset(
    {
        "id",
        "identifier",
        "customer_id",
        "account_id",
        "record_id",
        "external_id",
        "source_id",
        "sku",
        "upc",
        "code",
    }
)

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def infer_field_types(
    records: pd.DataFrame,
    *,
    sample_size: int = 100,
    email_threshold: float = 0.8,
    phone_threshold: float = 0.8,
) -> list[tuple[str, str]]:
    return [
        infer_field_type(
            records[column],
            sample_size=sample_size,
            email_threshold=email_threshold,
            phone_threshold=phone_threshold,
        )
        for column in records.columns
    ]


def infer_field_type(
    series: pd.Series,
    *,
    sample_size: int = 100,
    email_threshold: float = 0.8,
    phone_threshold: float = 0.8,
) -> tuple[str, str]:
    if pandas_types.is_datetime64_any_dtype(series.dtype):
        return "date", "dtype"
    if pandas_types.is_bool_dtype(series.dtype) or isinstance(
        series.dtype, pd.CategoricalDtype
    ):
        return "categorical", "dtype"
    if pandas_types.is_numeric_dtype(series.dtype):
        return "numeric", "dtype"

    column_name = _normalize_column_name(series.name)
    if column_name in EMAIL_COLUMN_NAMES:
        return "email", "column_name"
    if column_name in PHONE_COLUMN_NAMES:
        return "phone", "column_name"
    if column_name in IDENTIFIER_COLUMN_NAMES:
        return "identifier", "column_name"

    values = _sample_non_null_strings(series, sample_size)
    if values and _match_fraction(values, _is_email_value) >= email_threshold:
        return "email", "value_pattern"
    if values and _match_fraction(values, _is_phone_value) >= phone_threshold:
        return "phone", "value_pattern"
    return "text", "fallback"


def _normalize_column_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _sample_non_null_strings(series: pd.Series, sample_size: int) -> list[str]:
    values = [str(value).strip() for value in series.dropna()]
    values = [value for value in values if value]
    if sample_size > 0:
        return values[:sample_size]
    return values


def _match_fraction(values: list[str], predicate) -> float:
    return sum(predicate(value) for value in values) / len(values)


def _is_email_value(value: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(value))


def _is_phone_value(value: str) -> bool:
    value = value.strip()
    if value.startswith("+"):
        value = value[1:]
    compact = re.sub(r"[\s().\-/]", "", value)
    return compact.isdigit() and 7 <= len(compact) <= 15
