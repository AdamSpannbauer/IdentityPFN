"""Use the clean record as context for one observed-field edit."""

import json
import warnings

import pandas as pd

from ...call_ollama import OllamaTokenRepeatError, call_ollama_json


def edit_field_with_ollama(
    field: str,
    field_type: str,
    value: object,
    clean_record: dict[str, object],
    model: str,
) -> object:
    context = {
        name: None if pd.isna(item) else str(item)
        for name, item in clean_record.items()
    }
    try:
        result = call_ollama_json(
            prompt=(
                f"Field to edit: {field} (type: {field_type})\n"
                f"Original value: {value}\n"
                f"Other` record context: {json.dumps(context, ensure_ascii=False)}\n"
                "Write a plausible alternate observation of this field that keeps its"
                "type (str, date, numeric) and is consistent with the other field values."
                "Synonyms, alternate spellings, abbreviations, paraphrases, initialisms,and other realistic variations are acceptable."
                "Do not change facts, your goal is to convey a believable alternate observation of the same value."
                "If no plausible alternate observation is possible, return the original value."
                "Example 1: 'Jon Sully' could become 'John Sully' or 'Jonathan S.' or 'J.S.' or 'Jonny S.'."
                "Example 2: 'The Bronx is my home' could become 'I live in NYC'."
                'Reply only as {"value": "your edited value"}.'
            ),
            model=model,
            system_prompt='Return one edited field value as JSON: {"value": "..."}.',
            options={"temperature": 1.0},
        )
    except OllamaTokenRepeatError:
        warnings.warn("Ollama token repetition aborted a field edit; keeping original value")
        return value
    parsed = result["parsed"]
    return parsed.get("value", value) if isinstance(parsed, dict) else value
