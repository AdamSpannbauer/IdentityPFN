import re
import unicodedata
from datetime import date, datetime

import numpy as np


PERSONAL_DOMAINS = (
    "gmail.com",
    "outlook.com",
    "icloud.com",
    "yahoo.com",
    "proton.me",
    "hotmail.com",
    "aol.com",
    "comcast.net",
    "zoho.com",
    "gmx.net",
)

WORK_TLDS = ("com", "co", "net", "io")

ROLE_PREFIXES = (
    "info",
    "contact",
    "sales",
    "support",
    "hello",
    "team",
    "admin",
    "billing",
)


def normalize_email_token(value: object) -> str:
    """Convert a value into a simple ASCII email token."""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"[^a-zA-Z0-9]+", "", ascii_value).lower()
    return token or "user"


def company_to_domain(company: object, rng: np.random.Generator) -> str:
    """Create a plausible company-owned email domain."""
    normalized = unicodedata.normalize("NFKD", str(company))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    words = re.findall(r"[a-z0-9]+", ascii_value)
    words = [
        word
        for word in words
        if word not in {"and", "the", "llc", "inc", "corp", "co", "company"}
    ]
    if not words:
        words = ["example"]

    if len(words) > 1 and rng.random() < 0.4:
        stem = "-".join(words[:3])
    elif len(words) > 1 and rng.random() < 0.5:
        stem = "".join(words[:3])
    else:
        stem = words[0]

    return f"{stem}.{rng.choice(WORK_TLDS)}"


def birth_year_suffix(value: object) -> str:
    """Return a two-digit birth-year suffix when the value is date-like."""
    if isinstance(value, datetime):
        return f"{value.year % 100:02d}"
    if isinstance(value, date):
        return f"{value.year % 100:02d}"
    text = str(value)
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return match.group(0)[-2:]
    return ""


def make_personal_email(
    first_name: object,
    last_name: object,
    rng: np.random.Generator,
) -> str:
    first = normalize_email_token(first_name)
    last = normalize_email_token(last_name)
    patterns = (
        f"{first}.{last}",
        f"{first}{last}",
        f"{first[0]}.{last}",
        f"{first}.{last[0]}",
        f"{first}_{last}",
        f"{last}.{first}",
        f"{first}{rng.integers(10, 100)}",
    )
    return f"{rng.choice(patterns)}@{rng.choice(PERSONAL_DOMAINS)}"


def make_personal_email_with_dob(
    first_name: object,
    last_name: object,
    date_of_birth: object,
    rng: np.random.Generator,
) -> str:
    first = normalize_email_token(first_name)
    last = normalize_email_token(last_name)
    year = birth_year_suffix(date_of_birth)
    if not year:
        year = str(rng.integers(10, 100))
    patterns = (
        f"{first}.{last}{year}",
        f"{first}{last}{year}",
        f"{first[0]}{last}{year}",
        f"{first}_{last}_{year}",
    )
    return f"{rng.choice(patterns)}@{rng.choice(PERSONAL_DOMAINS)}"


def make_work_email(
    first_name: object,
    last_name: object,
    company: object,
    rng: np.random.Generator,
) -> str:
    first = normalize_email_token(first_name)
    last = normalize_email_token(last_name)
    domain = company_to_domain(company, rng)
    patterns = (
        f"{first}.{last}",
        f"{first}{last}",
        f"{first[0]}{last}",
        f"{first}.{last[0]}",
        first,
        last,
    )
    return f"{rng.choice(patterns)}@{domain}"


def make_role_email(company: object, rng: np.random.Generator) -> str:
    return f"{rng.choice(ROLE_PREFIXES)}@{company_to_domain(company, rng)}"


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    examples = [
        ("Jane", "Patel", "1987-04-12", "Northstar Analytics LLC"),
        ("José", "Nowicki", "1992-11-03", "Blue River Systems Inc."),
        ("Mary Beth", "O'Neil", "1978-06-30", "Atlas & Finch Co."),
    ]

    for first_name, last_name, dob, company in examples:
        print(f"\n{first_name} {last_name} | {company}")
        print("personal:", make_personal_email(first_name, last_name, rng))
        print(
            "personal_dob:",
            make_personal_email_with_dob(first_name, last_name, dob, rng),
        )
        print("work:", make_work_email(first_name, last_name, company, rng))
        print("role:", make_role_email(company, rng))
