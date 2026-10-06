"""Simple value-aware edits for text fields."""

import numpy as np

from ..people.simple import load_nickname_aliases


def substitute_nickname(value: object, rng: np.random.Generator) -> str:
    aliases = load_nickname_aliases().get(str(value).lower(), ())
    if not aliases:
        return str(value)
    return str(rng.choice(aliases)).title()


def abbreviate(value: object, rng: np.random.Generator) -> str:
    words = str(value).split()
    if not words:
        return str(value)
    index = int(rng.integers(len(words)))
    words[index] = words[index][0] + "."
    return " ".join(words)
