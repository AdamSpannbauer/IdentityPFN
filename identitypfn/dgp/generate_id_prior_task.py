"""Generate an identity-prior task from sampled source entities."""

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from .corruption import apply_corruptions
from .people.augment_people_record_fields import (
    IMPLEMENTED_AUGMENTATION_FIELD_TYPES,
    augment_people_record_fields,
)
from .world import World, sample_uniform_slot_entity_ids


def generate_id_prior_task(
    n_records: int,
    p_match: float,
    n_fields: int,
    sample_entities: Callable[[int, np.random.Generator], pd.DataFrame],
    field_types: Mapping[str, str],
    rng: np.random.Generator,
    allow_ollama: bool = False,
    ollama_model: str = "llama3.2:1b",
    ollama_augment_rate: float | None = None,
    ollama_corrupt_rate: float | None = None,
    country_field: str | None = None,
    must_include_any: list[str] | None = None,
    augment_people: bool = True,
) -> World:
    """Sample a schema and partition, then turn source entities into records."""
    augmentation_types = {
        field: field_type
        for field, field_type in IMPLEMENTED_AUGMENTATION_FIELD_TYPES.items()
        if augment_people and field not in field_types
    }
    available_types = dict(field_types) | augmentation_types
    if must_include_any:
        anchor = str(rng.choice(must_include_any))
        remaining = [field for field in available_types if field != anchor]
        selected_fields = [anchor] + rng.choice(
            remaining, size=n_fields - 1, replace=False
        ).tolist()
        rng.shuffle(selected_fields)
    else:
        selected_fields = rng.choice(
            list(available_types), size=n_fields, replace=False
        ).tolist()
    selected_types = {
        field: (
            "text"
            if available_types[field] == "categorical" and rng.random() < 0.5
            else available_types[field]
        )
        for field in selected_fields
    }

    entity_ids = sample_uniform_slot_entity_ids(n_records, p_match, rng)
    entities = sample_entities(int(entity_ids.max()) + 1, rng)
    selected_augmentations = [
        field for field in selected_fields if field in augmentation_types
    ]
    if augment_people:
        entities = augment_people_record_fields(
            entities,
            selected_augmentations,
            rng,
            allow_ollama=allow_ollama,
            ollama_model=ollama_model,
            ollama_rate=ollama_augment_rate,
            country_field=country_field,
        )
    records = entities.iloc[entity_ids][selected_fields].reset_index(drop=True)
    records = apply_corruptions(
        records,
        selected_types,
        rng,
        allow_ollama=allow_ollama,
        ollama_model=ollama_model,
        ollama_rate=ollama_corrupt_rate,
    )

    order = rng.permutation(n_records)
    entity_ids = entity_ids[order]
    records = records.iloc[order].reset_index(drop=True)
    return World(
        records=records,
        entity_ids=entity_ids,
        adjacency=entity_ids[:, None] == entity_ids[None, :],
        field_types=[selected_types[field] for field in selected_fields],
    )


if __name__ == "__main__":
    from pathlib import Path

    from .people.sample_revelio import REVELIO_USER_FIELD_TYPES, RevelioSampler

    root = Path("local/revelio_data_sample")
    sampler = RevelioSampler(
        individual_user_dir=root / "individual_user",
        individual_position_dir=root / "individual_position",
        individual_user_education_dir=root / "individual_user_education",
        individual_user_skill_dir=root / "individual_user_skill",
        company_ref_dir=root / "company_ref",
    )
    world = generate_id_prior_task(
        n_records=20,
        p_match=0.03,
        n_fields=5,
        sample_entities=sampler.sample_entities,
        field_types=REVELIO_USER_FIELD_TYPES,
        rng=np.random.default_rng(42),
        allow_ollama=True,
        country_field="user_country",
    )

    print("schema:", list(zip(world.records.columns, world.field_types)))
    print("entities:", len(np.unique(world.entity_ids)))
    print("matched pairs:", int(np.triu(world.adjacency, k=1).sum()))
    preview = world.records.head(10).copy()
    preview.insert(0, "entity_id", world.entity_ids[:10])
    print(preview.to_string(index=False, max_colwidth=50))
