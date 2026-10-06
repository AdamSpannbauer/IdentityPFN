from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


PrimitiveType = Literal["text", "categorical", "numeric", "date", "identifier"]


@dataclass
class World:
    records: pd.DataFrame
    entity_ids: np.ndarray
    adjacency: np.ndarray
    field_types: list[PrimitiveType]


def sample_entity_ids(
    n_records: int, p_match: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a CRP partition with approximately the requested pair match rate."""
    alpha = (1 - p_match) / p_match
    entity_ids = [0]
    cluster_sizes = [1]

    for record in range(1, n_records):
        probabilities = np.asarray(cluster_sizes + [alpha]) / (record + alpha)
        entity = rng.choice(len(probabilities), p=probabilities)
        if entity == len(cluster_sizes):
            cluster_sizes.append(1)
        else:
            cluster_sizes[entity] += 1
        entity_ids.append(entity)

    return np.asarray(entity_ids)


def sample_uniform_slot_entity_ids(
    n_records: int, p_match: float, rng: np.random.Generator
) -> np.ndarray:
    """Seed each entity slot once, then assign remaining records uniformly."""
    pair_denominator = n_records * (n_records - 1)
    scaled_rate = p_match * pair_denominator
    n_slots = round(
        (1 - scaled_rate + np.sqrt((scaled_rate - 1) ** 2 + 4 * pair_denominator))
        / 2
    )
    return np.concatenate(
        (np.arange(n_slots), rng.integers(n_slots, size=n_records - n_slots))
    )
