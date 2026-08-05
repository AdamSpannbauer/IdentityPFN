import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ....paths import repository_path

AREA_CODE_DATA_PATH = repository_path(
    "dgp_data", "area_codes", "area_codes_bennetyeedogorg_ucsd_pages_area_html.csv"
)

DGP_STATE_CODES = frozenset(
    {
        "AK",
        "AL",
        "AR",
        "AZ",
        "CA",
        "CO",
        "CT",
        "DC",
        "DE",
        "FL",
        "GA",
        "HI",
        "IA",
        "ID",
        "IL",
        "IN",
        "KS",
        "KY",
        "LA",
        "MA",
        "MD",
        "ME",
        "MI",
        "MN",
        "MO",
        "MS",
        "MT",
        "NC",
        "ND",
        "NE",
        "NH",
        "NJ",
        "NM",
        "NV",
        "NY",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VA",
        "VT",
        "WA",
        "WI",
        "WV",
        "WY",
    }
)


@lru_cache
def load_state_area_codes(
    path: Path = AREA_CODE_DATA_PATH,
) -> dict[str, tuple[str, ...]]:
    area_codes_by_state: dict[str, list[str]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            state = row["Region"].strip().upper()
            area_code = row["Area Code"].strip()
            if state not in DGP_STATE_CODES:
                continue
            if not re.fullmatch(r"\d{3}", area_code):
                continue
            area_codes_by_state.setdefault(state, []).append(area_code)

    return {
        state: tuple(sorted(set(area_codes), key=int))
        for state, area_codes in area_codes_by_state.items()
    }


@lru_cache
def load_general_area_codes(path: Path = AREA_CODE_DATA_PATH) -> tuple[str, ...]:
    state_area_codes = load_state_area_codes(path)
    return tuple(
        sorted(
            {
                area_code
                for area_codes in state_area_codes.values()
                for area_code in area_codes
            },
            key=int,
        )
    )


def normalize_state(value: object) -> str:
    text = str(value).strip().upper()
    match = re.search(r"[A-Z]{2}", text)
    return match.group(0) if match else ""


def sample_area_code(state: object, rng: np.random.Generator) -> str:
    state_code = normalize_state(state)
    area_codes = load_state_area_codes().get(state_code, load_general_area_codes())
    return str(rng.choice(area_codes))


def format_phone_number(
    area_code: str,
    exchange: int,
    line_number: int,
    rng: np.random.Generator,
) -> str:
    formats = (
        f"({area_code}) {exchange:03d}-{line_number:04d}",
        f"{area_code}-{exchange:03d}-{line_number:04d}",
        f"{area_code}.{exchange:03d}.{line_number:04d}",
        f"+1 {area_code} {exchange:03d} {line_number:04d}",
    )
    return str(rng.choice(formats))


def make_phone_with_area_code(area_code: str, rng: np.random.Generator) -> str:
    exchange = int(rng.integers(200, 1000))
    line_number = int(rng.integers(0, 10000))
    return format_phone_number(area_code, exchange, line_number, rng)


def make_phone_from_state(state: object, rng: np.random.Generator) -> str:
    return make_phone_with_area_code(sample_area_code(state, rng), rng)


def make_phone_from_city_state(
    city: object,
    state: object,
    rng: np.random.Generator,
) -> str:
    del city
    return make_phone_from_state(state, rng)


def make_independent_phone(rng: np.random.Generator) -> str:
    return make_phone_with_area_code(str(rng.choice(load_general_area_codes())), rng)


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    examples = [
        ("Los Angeles", "CA"),
        ("New York", "NY"),
        ("Austin", "TX"),
        ("Columbus", "OH"),
        ("Unknown", "??"),
    ]

    for city, state in examples:
        print(f"\n{city}, {state}")
        print("state:", make_phone_from_state(state, rng))
        print("city_state:", make_phone_from_city_state(city, state, rng))
        print("independent:", make_independent_phone(rng))
