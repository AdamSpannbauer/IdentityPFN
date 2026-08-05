import re
import unicodedata

import numpy as np


SOURCE_SYSTEM_PREFIXES = ("CRM", "ERP", "SFDC", "MKT", "SUP")


def normalize_id_token(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"[^a-zA-Z0-9]+", "", ascii_value).upper()
    return token or "ID"


def make_customer_id(rng: np.random.Generator) -> str:
    prefix = str(rng.choice(("CUST", "CUS", "CID")))
    separator = str(rng.choice(("-", "_", "")))
    width = int(rng.choice((5, 6, 7)))
    return f"{prefix}{separator}{int(rng.integers(0, 10**width)):0{width}d}"


def make_account_id(company: object, rng: np.random.Generator) -> str:
    company_token = normalize_id_token(company)
    prefix = company_token[: max(3, min(len(company_token), 8))]
    separator = str(rng.choice(("-", "_")))
    width = int(rng.choice((4, 5, 6)))
    return f"{prefix}{separator}{int(rng.integers(0, 10**width)):0{width}d}"


def make_source_system_id(rng: np.random.Generator) -> str:
    prefix = str(rng.choice(SOURCE_SYSTEM_PREFIXES))
    separator = str(rng.choice(("-", "_", "")))
    width = int(rng.choice((5, 6)))
    return f"{prefix}{separator}{int(rng.integers(0, 10**width)):0{width}d}"


def make_external_id_from_name(
    first_name: object,
    last_name: object,
    rng: np.random.Generator,
) -> str:
    first = normalize_id_token(first_name)
    last = normalize_id_token(last_name)
    patterns = (
        f"{first[0]}{last[:7]}",
        f"{last[:8]}",
        f"{first[:4]}{last[:4]}",
    )
    stem = str(rng.choice(patterns))
    separator = str(rng.choice(("-", "_")))
    return f"{stem}{separator}{int(rng.integers(0, 100000)):05d}"


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    examples = [
        ("Jane", "Patel", "Northstar Analytics LLC"),
        ("José", "Nowicki", "Blue River Systems Inc."),
        ("Mary Beth", "O'Neil", "Atlas & Finch Co."),
    ]

    for first_name, last_name, company in examples:
        print(f"\n{first_name} {last_name} | {company}")
        print("customer:", make_customer_id(rng))
        print("account:", make_account_id(company, rng))
        print("source:", make_source_system_id(rng))
        print("external:", make_external_id_from_name(first_name, last_name, rng))
