"""Character edits for observed field values."""

from contextlib import contextmanager
from functools import lru_cache
import random

import nlpaug.augmenter.char as nac
import numpy as np


def delete_character(value: object, rng: np.random.Generator) -> str:
    text = str(value)
    if len(text) < 2:
        return text
    index = int(rng.integers(len(text)))
    return text[:index] + text[index + 1 :]


def swap_characters(value: object, rng: np.random.Generator) -> str:
    text = str(value)
    if len(text) < 2:
        return text
    index = int(rng.integers(len(text) - 1))
    characters = list(text)
    characters[index], characters[index + 1] = characters[index + 1], characters[index]
    return "".join(characters)


def _whole_value_tokenizer(value: str) -> list[str]:
    return [value]


def _whole_value_reverse_tokenizer(tokens: list[str]) -> str:
    return "".join(tokens)


@lru_cache(maxsize=1)
def _text_augmenters() -> dict[str, nac.CharAugmenter]:
    options = {
        "aug_char_min": 1,
        "aug_char_max": 2,
        "aug_char_p": 0.1,
        "aug_word_min": 1,
        "aug_word_max": 1,
        "aug_word_p": 1.0,
        "min_char": 2,
        "tokenizer": _whole_value_tokenizer,
        "reverse_tokenizer": _whole_value_reverse_tokenizer,
    }
    return {
        "ocr": nac.OcrAug(**options),
        "keyboard": nac.KeyboardAug(**options),
        "insert": nac.RandomCharAug(action="insert", **options),
        "substitute": nac.RandomCharAug(action="substitute", **options),
    }


@contextmanager
def _temporary_global_seed(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def edit_text(value: object, operation: str, rng: np.random.Generator) -> str:
    with _temporary_global_seed(int(rng.integers(0, 2**32))):
        edited = _text_augmenters()[operation].augment(str(value))
        return edited[0] if edited else str(value)
