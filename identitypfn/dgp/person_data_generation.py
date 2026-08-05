import random
import csv
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import nlpaug.augmenter.char as nac
import numpy as np
import pandas as pd
from faker import Faker

from ..paths import repository_path
from .world import PrimitiveType, World, sample_entity_ids

AccountBioBackend = Literal["template", "ollama"]


@dataclass(frozen=True)
class FieldFamily:
    name: str
    primitive_type: PrimitiveType
    max_occurrences: int = 1
    repeat_penalty: float = 0.0


@dataclass(frozen=True)
class SchemaSamplerConfig:
    field_families: tuple[FieldFamily, ...]
    bos_scores: dict[str, float]
    pairwise_affinities: dict[tuple[str, str], float]
    occurrence_decay: float = 0.5


@dataclass(frozen=True)
class SchemaColumn:
    name: str
    family: FieldFamily


@dataclass(frozen=True)
class Location:
    postal_code: str
    city: str
    state: str


@dataclass(frozen=True)
class ObservationConfig:
    missing_rate: float = 0.1
    nickname_rate: float = 0.1
    corruption_rate: float = 0.2


PERSON_FIELD_FAMILIES = (
    FieldFamily("first_name", "text"),
    FieldFamily("middle_name", "text"),
    FieldFamily("last_name", "text"),
    FieldFamily("full_name", "text"),
    FieldFamily("date_of_birth", "date"),
    FieldFamily("company", "text"),
    FieldFamily("account_bio", "text"),
    FieldFamily("email", "text", max_occurrences=3, repeat_penalty=-0.8),
    FieldFamily("phone", "text", max_occurrences=3, repeat_penalty=-0.8),
    FieldFamily("street_address", "text", max_occurrences=2, repeat_penalty=-1.0),
    FieldFamily("city", "text"),
    FieldFamily("state", "categorical"),
    FieldFamily("postal_code", "text"),
    FieldFamily("identifier", "identifier", max_occurrences=3, repeat_penalty=-0.6),
)

PERSON_SCHEMA_CONFIG = SchemaSamplerConfig(
    field_families=PERSON_FIELD_FAMILIES,
    bos_scores={
        "first_name": 1.0,
        "middle_name": -0.5,
        "last_name": 1.0,
        "full_name": 0.8,
        "date_of_birth": 0.5,
        "company": 0.5,
        "account_bio": 0.3,
        "email": 1.0,
        "phone": 0.8,
        "street_address": 0.6,
        "city": 0.0,
        "state": 0.0,
        "postal_code": 0.0,
        "identifier": 1.2,
    },
    pairwise_affinities={
        ("first_name", "middle_name"): 0.5,
        ("first_name", "last_name"): 1.0,
        ("first_name", "full_name"): 0.4,
        ("first_name", "email"): 0.5,
        ("first_name", "account_bio"): 0.3,
        ("last_name", "full_name"): 0.4,
        ("last_name", "email"): 0.5,
        ("last_name", "account_bio"): 0.3,
        ("full_name", "account_bio"): 0.4,
        ("company", "account_bio"): 0.7,
        ("city", "account_bio"): 0.3,
        ("state", "account_bio"): 0.3,
        ("street_address", "city"): 1.0,
        ("street_address", "state"): 0.8,
        ("street_address", "postal_code"): 1.0,
        ("city", "state"): 0.8,
        ("city", "postal_code"): 0.6,
        ("state", "postal_code"): 0.8,
    },
)


def get_affinity(
    left: str,
    right: str,
    affinities: dict[tuple[str, str], float],
) -> float:
    """Return either ordering's affinity, defaulting to zero when unspecified."""
    # affinities is just a dict, this is a just cause keys might be ordered either way
    return affinities.get((left, right), affinities.get((right, left), 0.0))


def dampened_occurrence_weight(count: int, decay: float) -> float:
    """Return the cumulative influence of repeated selected field families."""
    return sum(decay**index for index in range(count))


def score_candidate(
    candidate: FieldFamily,
    selected_counts: Counter[str],
    config: SchemaSamplerConfig,
) -> float:
    """Score a candidate using its base, repetition, and selected-field effects."""
    score = config.bos_scores.get(candidate.name, 0.0)
    score += candidate.repeat_penalty * selected_counts[candidate.name]

    for selected_name, count in selected_counts.items():
        score += get_affinity(
            candidate.name, selected_name, config.pairwise_affinities
        ) * dampened_occurrence_weight(count, config.occurrence_decay)
    return score


def sample_schema(
    n_fields: int,
    config: SchemaSamplerConfig,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> list[FieldFamily]:
    """Sample exactly n_fields while conditioning each draw on prior selections."""
    if n_fields <= 0:
        raise ValueError("n_fields must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= config.occurrence_decay <= 1:
        raise ValueError("occurrence_decay must be between 0 and 1")
    if n_fields > sum(field.max_occurrences for field in config.field_families):
        raise ValueError("n_fields exceeds total field occurrence limits")

    selected = []
    selected_counts: Counter[str] = Counter()

    while len(selected) < n_fields:
        candidates = [
            field
            for field in config.field_families
            if selected_counts[field.name] < field.max_occurrences
        ]
        scores = np.asarray(
            [score_candidate(field, selected_counts, config) for field in candidates]
        )
        scaled_scores = scores / temperature
        probabilities = np.exp(scaled_scores - scaled_scores.max())
        probabilities /= probabilities.sum()

        selected_field = candidates[rng.choice(len(candidates), p=probabilities)]
        selected.append(selected_field)
        selected_counts[selected_field.name] += 1

    rng.shuffle(selected)
    return selected


def make_schema_columns(families: list[FieldFamily]) -> list[SchemaColumn]:
    """Assign unique column names to possibly repeated field families."""
    counts: Counter[str] = Counter()
    columns = []
    for family in families:
        counts[family.name] += 1
        suffix = f"_{counts[family.name]}" if counts[family.name] > 1 else ""
        columns.append(SchemaColumn(f"{family.name}{suffix}", family))
    return columns


@lru_cache
def load_us_locations(
    path: Path = repository_path("dgp_data", "geonames_us_zip_data", "US.txt"),
) -> tuple[Location, ...]:
    """Load GeoNames rows containing complete ZIP, city, and state values."""
    columns = [
        "country_code",
        "postal_code",
        "place_name",
        "admin_name1",
        "admin_code1",
        "admin_name2",
        "admin_code2",
        "admin_name3",
        "admin_code3",
        "latitude",
        "longitude",
        "accuracy",
    ]
    frame = pd.read_csv(path, sep="\t", names=columns, dtype=str)
    frame = frame.dropna(subset=["postal_code", "place_name", "admin_code1"])
    return tuple(
        Location(row.postal_code, row.place_name, row.admin_code1)
        for row in frame.itertuples()
    )


def _read_carlton_nickname_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open() as file:
        for row in csv.DictReader(file):
            if row["relationship"] == "has_nickname":
                pairs.append((row["name1"], row["name2"]))
    return pairs


@lru_cache
def load_nickname_aliases(
    carlton_path: Path = repository_path(
        "dgp_data", "corruption", "nicknames", "names.csv"
    ),
) -> dict[str, tuple[str, ...]]:
    """Load direct bidirectional nickname aliases without transitive closure."""
    aliases = defaultdict(set)
    pairs = _read_carlton_nickname_pairs(carlton_path)

    for canonical, nickname in pairs:
        canonical = canonical.strip().lower()
        nickname = nickname.strip().lower()
        aliases[canonical].add(nickname)
        aliases[nickname].add(canonical)
    return {name: tuple(sorted(values)) for name, values in aliases.items()}


def _generate_name_bundle(faker: Faker) -> dict[str, str]:
    first_name = faker.first_name()
    middle_name = faker.first_name()
    last_name = faker.last_name()
    return {
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "full_name": f"{first_name} {last_name}",
    }


def _generate_independent_value(family: FieldFamily, faker: Faker):
    generators = {
        "company": faker.company,
        "email": faker.email,
        "phone": faker.phone_number,
        "street_address": faker.street_address,
        "date_of_birth": faker.date_of_birth,
        "identifier": faker.uuid4,
    }
    if family.name in generators:
        return generators[family.name]()
    if family.primitive_type == "categorical":
        return faker.word()
    if family.primitive_type == "numeric":
        return faker.pyfloat()
    if family.primitive_type == "date":
        return faker.date_object()
    return faker.word()


def _generate_template_account_bio(
    name_bundle: dict[str, str],
    location_bundle: dict[str, str],
    company: str,
    rng: np.random.Generator,
) -> str:
    templates = (
        "{full_name} is based in {city}, {state} and works with {company}.",
        "{full_name} works at {company} and is located in {city}, {state}.",
        "{full_name} is a {city}, {state}-based contact associated with {company}.",
        "{full_name} represents {company} in {city}, {state}.",
    )
    template = templates[rng.integers(len(templates))]
    return template.format(
        full_name=name_bundle["full_name"],
        city=location_bundle["city"],
        state=location_bundle["state"],
        company=company,
    )


def _generate_ollama_account_bio(
    name_bundle: dict[str, str],
    location_bundle: dict[str, str],
    company: str,
    model: str,
    temperature: float | None,
    max_retries: int,
) -> str:
    """Generate one account bio with strict scalar JSON validation."""
    from ..call_ollama import call_ollama_json

    options = None if temperature is None else {"temperature": temperature}
    system_prompt = (
        "You generate concise, realistic CRM account bio fields for synthetic "
        "entity-resolution data."
    )
    base_prompt = (
        "Write one short account bio for this person using the provided facts.\n\n"
        f"Person facts:\n"
        f"- first_name: {name_bundle['first_name']}\n"
        f"- middle_name: {name_bundle['middle_name']}\n"
        f"- last_name: {name_bundle['last_name']}\n"
        f"- full_name: {name_bundle['full_name']}\n"
        f"- company: {company}\n"
        f"- city: {location_bundle['city']}\n"
        f"- state: {location_bundle['state']}\n"
        f"- postal_code: {location_bundle['postal_code']}\n\n"
        'Return exactly one JSON object with exactly one key named "value".\n'
        'The "value" must be a single string bio, not an object or array.\n'
        "Do not add metadata, labels, alternatives, explanations, or examples."
    )
    prompt = base_prompt
    last_parsed = None
    for attempt in range(max_retries + 1):
        result = call_ollama_json(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            options=options,
        )
        parsed = result["parsed"]
        last_parsed = parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("value"), str):
            return parsed["value"]

        if attempt < max_retries:
            prompt = (
                f"{base_prompt}\n\n"
                f"Your previous response violated the required schema: {parsed!r}\n"
                'Return only {"value": "<single string bio>"}.'
            )

    raise ValueError(
        "Ollama response failed account_bio schema "
        f"for model={model!r}, max_retries={max_retries}, "
        f"last_parsed={last_parsed!r}"
    )


def _generate_account_bio(
    name_bundle: dict[str, str],
    location_bundle: dict[str, str],
    company: str,
    rng: np.random.Generator,
    backend: AccountBioBackend,
    ollama_model: str,
    ollama_temperature: float | None,
    ollama_max_retries: int,
) -> str:
    if backend == "template":
        return _generate_template_account_bio(
            name_bundle=name_bundle,
            location_bundle=location_bundle,
            company=company,
            rng=rng,
        )
    if backend == "ollama":
        return _generate_ollama_account_bio(
            name_bundle=name_bundle,
            location_bundle=location_bundle,
            company=company,
            model=ollama_model,
            temperature=ollama_temperature,
            max_retries=ollama_max_retries,
        )
    raise ValueError("account_bio_backend must be 'template' or 'ollama'")


def generate_canonical_entities(
    n_entities: int,
    schema: list[SchemaColumn],
    locations: tuple[Location, ...],
    faker: Faker,
    rng: np.random.Generator,
    account_bio_backend: AccountBioBackend = "template",
    ollama_model: str = "qwen2.5:7b",
    ollama_temperature: float | None = 0.7,
    ollama_max_retries: int = 2,
) -> pd.DataFrame:
    """Generate one clean person record per latent entity."""
    name_fields = {"first_name", "middle_name", "last_name", "full_name"}
    location_fields = {"city", "state", "postal_code"}
    records = []

    for _ in range(n_entities):
        name_bundle = _generate_name_bundle(faker)
        location = locations[rng.integers(len(locations))]
        location_bundle = {
            "city": location.city,
            "state": location.state,
            "postal_code": location.postal_code,
        }
        company = faker.company()

        record = []
        for column in schema:
            family_name = column.family.name
            if family_name in name_fields:
                value = name_bundle[family_name]
            elif family_name in location_fields:
                value = location_bundle[family_name]
            elif family_name == "company":
                value = company
            elif family_name == "account_bio":
                value = _generate_account_bio(
                    name_bundle=name_bundle,
                    location_bundle=location_bundle,
                    company=company,
                    rng=rng,
                    backend=account_bio_backend,
                    ollama_model=ollama_model,
                    ollama_temperature=ollama_temperature,
                    ollama_max_retries=ollama_max_retries,
                )
            else:
                value = _generate_independent_value(column.family, faker)
            record.append(value)
        records.append(record)

    return pd.DataFrame(records, columns=[column.name for column in schema])


def _whole_value_tokenizer(value: str) -> list[str]:
    return [value]


def _whole_value_reverse_tokenizer(tokens: list[str]) -> str:
    return "".join(tokens)


@lru_cache
def get_character_augmenters():
    """Create reusable character augmenters that preserve value punctuation."""
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
    return (
        nac.OcrAug(**options),
        nac.KeyboardAug(**options),
        nac.RandomCharAug(action="insert", **options),
        nac.RandomCharAug(action="substitute", **options),
        nac.RandomCharAug(action="swap", **options),
        nac.RandomCharAug(action="delete", **options),
    )


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


def observe_value(
    value,
    column: SchemaColumn,
    config: ObservationConfig,
    rng: np.random.Generator,
):
    """Apply missingness or one character corruption to a canonical value."""
    if rng.random() < config.missing_rate:
        return np.nan

    observed = value
    if column.family.name == "first_name" and rng.random() < config.nickname_rate:
        aliases = load_nickname_aliases().get(str(observed).lower(), ())
        if aliases:
            observed = aliases[rng.integers(len(aliases))].title()

    if (
        column.family.primitive_type in {"text", "identifier"}
        and rng.random() < config.corruption_rate
    ):
        augmenters = get_character_augmenters()
        augmenter = augmenters[rng.integers(len(augmenters))]
        seed = int(rng.integers(0, 2**32))
        with _temporary_global_seed(seed):
            observed = augmenter.augment(str(observed))[0]
    return observed


def generate_observed_records(
    canonical: pd.DataFrame,
    entity_ids: np.ndarray,
    schema: list[SchemaColumn],
    config: ObservationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create one potentially missing or corrupted record per entity ID."""
    records = []
    for entity_id in entity_ids:
        record = [
            observe_value(canonical.iloc[entity_id, index], column, config, rng)
            for index, column in enumerate(schema)
        ]
        records.append(record)
    return pd.DataFrame(records, columns=[column.name for column in schema])


def generate_world(
    n_records: int = 20,
    p_match: float = 0.05,
    n_fields: int = 4,
    schema_temperature: float = 1.0,
    observation_config: ObservationConfig = ObservationConfig(),
    seed: int | None = None,
    account_bio_backend: AccountBioBackend = "template",
    ollama_model: str = "qwen2.5:7b",
    ollama_temperature: float | None = 0.7,
    ollama_max_retries: int = 2,
) -> World:
    """Generate a CRP-partitioned person world with observed duplicate records."""
    rng = np.random.default_rng(seed)
    faker = Faker("en_US")
    faker.seed_instance(seed)

    families = sample_schema(
        n_fields, PERSON_SCHEMA_CONFIG, rng, temperature=schema_temperature
    )
    schema = make_schema_columns(families)
    entity_ids = sample_entity_ids(n_records, p_match, rng)
    canonical = generate_canonical_entities(
        len(np.unique(entity_ids)),
        schema,
        load_us_locations(),
        faker,
        rng,
        account_bio_backend=account_bio_backend,
        ollama_model=ollama_model,
        ollama_temperature=ollama_temperature,
        ollama_max_retries=ollama_max_retries,
    )
    records = generate_observed_records(
        canonical, entity_ids, schema, observation_config, rng
    )

    order = rng.permutation(n_records)
    entity_ids = entity_ids[order]
    records = records.iloc[order].reset_index(drop=True)
    adjacency = entity_ids[:, None] == entity_ids[None, :]
    return World(
        records,
        entity_ids,
        adjacency,
        [column.family.primitive_type for column in schema],
    )


def generate_worlds(
    n_worlds: int,
    n_records: list[int] = [50, 100, 300],
    p_match: list[float] = [0.001, 0.005, 0.01, 0.05],
    n_fields: list[int] = [3, 5, 10],
    schema_temperature: list[float] = [0.7, 1.0, 1.5],
    missing_rate: list[float] = [0.0, 0.1, 0.2],
    nickname_rate: list[float] = [0.0, 0.1, 0.2],
    corruption_rate: list[float] = [0.0, 0.1, 0.2],
    seed: int | None = None,
    verbose: bool = False,
    account_bio_backend: AccountBioBackend = "template",
    ollama_model: str = "qwen2.5:7b",
    ollama_temperature: float | None = 0.7,
    ollama_max_retries: int = 2,
) -> list[World]:
    """Generate person worlds with independently sampled parameters."""
    rng = np.random.default_rng(seed)
    worlds = []
    for _ in range(n_worlds):
        world = generate_world(
            n_records=int(rng.choice(n_records)),
            p_match=float(rng.choice(p_match)),
            n_fields=int(rng.choice(n_fields)),
            schema_temperature=float(rng.choice(schema_temperature)),
            observation_config=ObservationConfig(
                missing_rate=float(rng.choice(missing_rate)),
                nickname_rate=float(rng.choice(nickname_rate)),
                corruption_rate=float(rng.choice(corruption_rate)),
            ),
            seed=int(rng.integers(0, 2**32)),
            account_bio_backend=account_bio_backend,
            ollama_model=ollama_model,
            ollama_temperature=ollama_temperature,
            ollama_max_retries=ollama_max_retries,
        )
        worlds.append(world)
        if verbose:
            print(
                f"Generated person world with {len(world.records)} records, "
                f"{world.records.shape[1]} fields, and "
                f"{len(np.unique(world.entity_ids))} entities."
            )
    return worlds


if __name__ == "__main__":
    world = generate_world(
        n_records=20,
        p_match=0.15,
        n_fields=8,
        observation_config=ObservationConfig(
            missing_rate=0.1, nickname_rate=0.5, corruption_rate=0.5
        ),
        seed=0,
    )
    print("Field types:", world.field_types)
    print(
        pd.concat(
            [pd.Series(world.entity_ids, name="entity_id"), world.records], axis=1
        )
    )
    pair_mask = np.triu(np.ones_like(world.adjacency, dtype=bool), k=1)
    print("Realized pair match prevalence:", world.adjacency[pair_mask].mean())
