import re
import unicodedata
from datetime import date, datetime

import numpy as np


USERNAME_ADJECTIVES = (
    "bright",
    "brisk",
    "calm",
    "clear",
    "curious",
    "daring",
    "fresh",
    "golden",
    "quiet",
    "rapid",
    "silver",
    "steady",
    "urban",
)

USERNAME_NOUNS = (
    "atlas",
    "brook",
    "comet",
    "field",
    "harbor",
    "maple",
    "orbit",
    "pixel",
    "river",
    "signal",
    "summit",
    "trail",
    "vector",
)


def normalize_username_token(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"[^a-zA-Z0-9]+", "", ascii_value).lower()
    return token or "user"


def birth_year_suffix(value: object) -> str:
    if isinstance(value, datetime):
        return f"{value.year % 100:02d}"
    if isinstance(value, date):
        return f"{value.year % 100:02d}"
    text = str(value)
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return match.group(0)[-2:]
    return ""


def make_username_from_name(
    first_name: object,
    last_name: object,
    rng: np.random.Generator,
) -> str:
    first = normalize_username_token(first_name)
    last = normalize_username_token(last_name)
    patterns = (
        f"{first}{last}",
        f"{first}.{last}",
        f"{first}_{last}",
        f"{first[0]}_{last}",
        f"{first}_{last[0]}",
        f"{last}{first}",
        f"{first}{rng.integers(10, 100)}",
    )
    return str(rng.choice(patterns))


def make_username_from_name_dob(
    first_name: object,
    last_name: object,
    date_of_birth: object,
    rng: np.random.Generator,
) -> str:
    first = normalize_username_token(first_name)
    last = normalize_username_token(last_name)
    year = birth_year_suffix(date_of_birth)
    if not year:
        year = str(rng.integers(10, 100))
    patterns = (
        f"{first}{last}{year}",
        f"{first}.{last}{year}",
        f"{first}_{last}_{year}",
        f"{first[0]}{last}{year}",
    )
    return str(rng.choice(patterns))


def make_username_from_name_company(
    first_name: object,
    last_name: object,
    company: object,
    rng: np.random.Generator,
) -> str:
    first = normalize_username_token(first_name)
    last = normalize_username_token(last_name)
    company_token = normalize_username_token(company)
    company_piece = company_token[: max(4, min(len(company_token), 10))]
    patterns = (
        f"{first}.{company_piece}",
        f"{first}{last}_{company_piece}",
        f"{first[0]}{last}_{company_piece}",
        f"{company_piece}.{first}",
    )
    return str(rng.choice(patterns))


def make_independent_username(rng: np.random.Generator) -> str:
    adjective = str(rng.choice(USERNAME_ADJECTIVES))
    noun = str(rng.choice(USERNAME_NOUNS))
    number = int(rng.integers(0, 100))
    if rng.random() < 0.5:
        noun = noun.title()
    return f"{adjective}{noun}{number}"


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    examples = [
        ("Jane", "Patel", "1987-04-12", "Northstar Analytics LLC"),
        ("José", "Nowicki", "1992-11-03", "Blue River Systems Inc."),
        ("Mary Beth", "O'Neil", "1978-06-30", "Atlas & Finch Co."),
    ]

    for first_name, last_name, dob, company in examples:
        print(f"\n{first_name} {last_name} | {company}")
        print("name:", make_username_from_name(first_name, last_name, rng))
        print(
            "name_dob:",
            make_username_from_name_dob(first_name, last_name, dob, rng),
        )
        print(
            "name_company:",
            make_username_from_name_company(first_name, last_name, company, rng),
        )
        print("independent:", make_independent_username(rng))
