"""Add stable synthetic fields to source people before record observation."""

from collections.abc import Collection
from functools import lru_cache
import re
import unicodedata

from faker import Faker
import numpy as np
import pandas as pd
import phonenumbers
from phonenumbers import geocoder

from ...call_ollama import call_ollama_json


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


NEW_PERSON_FIELDS = (
    "email",
    "username",
    "phone",
    "cell",
    "ssn",
    "street_address",
    "date_of_birth",
    "middle_name",
    "postal_code",
)

IMPLEMENTED_AUGMENTATION_FIELD_TYPES = {
    "ssn": "identifier",
    "email": "email",
    "phone": "phone",
    "cell": "phone",
}


def sample_ssn(rng: np.random.Generator) -> str:
    """Sample a nine-digit identifier in SSN format."""
    digits = f"{int(rng.integers(0, 1_000_000_000)):09d}"
    return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"


@lru_cache(maxsize=1)
def phone_country_regions() -> dict[str, str]:
    """Map country names to regions supported by phonenumbers."""
    regions = {}
    for region in sorted(phonenumbers.SUPPORTED_REGIONS):
        for number_type in (
            phonenumbers.PhoneNumberType.FIXED_LINE,
            phonenumbers.PhoneNumberType.MOBILE,
        ):
            example = phonenumbers.example_number_for_type(region, number_type)
            if example is not None:
                name = geocoder.country_name_for_number(example, "en")
                if name:
                    regions[name.casefold()] = region
    return regions


def sample_phone(
    country: object, kind: str, rng: np.random.Generator
) -> str:
    """Sample a valid fixed-line or mobile number for a country."""
    number_type = (
        phonenumbers.PhoneNumberType.FIXED_LINE
        if kind == "phone"
        else phonenumbers.PhoneNumberType.MOBILE
    )
    region = (
        phone_country_regions().get(str(country).strip().casefold())
        if pd.notna(country)
        else None
    )
    example = (
        phonenumbers.example_number_for_type(region, number_type)
        if region is not None
        else None
    )
    if example is None:
        eligible_regions = [
            candidate
            for candidate in sorted(phonenumbers.SUPPORTED_REGIONS)
            if phonenumbers.example_number_for_type(candidate, number_type) is not None
        ]
        region = str(rng.choice(eligible_regions))
        example = phonenumbers.example_number_for_type(region, number_type)

    national = phonenumbers.national_significant_number(example)
    for prefix_length in (min(2, len(national)), max(0, len(national) - 6)):
        for _ in range(60):
            suffix = "".join(
                str(digit)
                for digit in rng.integers(0, 10, size=len(national) - prefix_length)
            )
            candidate = phonenumbers.parse(
                f"+{example.country_code}{national[:prefix_length]}{suffix}", None
            )
            if (
                phonenumbers.is_valid_number_for_region(candidate, region)
                and phonenumbers.number_type(candidate)
                in (number_type, phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE)
            ):
                return phonenumbers.format_number(
                    candidate, phonenumbers.PhoneNumberFormat.E164
                )
    return phonenumbers.format_number(example, phonenumbers.PhoneNumberFormat.E164)


def normalize_email_token(value: object) -> str:
    """Convert a name part into an ASCII email token."""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"[^a-zA-Z0-9]+", "", ascii_value).lower()
    return token or "user"


def make_personal_email(
    first_name: object, last_name: object, rng: np.random.Generator
) -> str:
    """Generate a personal email from both name parts."""
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


def sample_single_name_email(name: object, rng: np.random.Generator) -> str:
    """Generate a personal email from one available name part."""
    token = normalize_email_token(name)
    local_part = str(rng.choice((token, f"{token}{rng.integers(10, 100)}")))
    return f"{local_part}@{rng.choice(PERSONAL_DOMAINS)}"


def sample_faker_email(rng: np.random.Generator) -> str:
    """Generate a personal email without source-person fields."""
    faker = Faker()
    faker.seed_instance(int(rng.integers(0, 2**32)))
    return faker.free_email()


def sample_ollama_email(
    first_name: object, last_name: object, model: str
) -> str:
    """Ask Ollama for a personal email using available name context."""
    name_context = []
    if pd.notna(first_name) and str(first_name).strip():
        name_context.append(f"First name: {first_name}")
    if pd.notna(last_name) and str(last_name).strip():
        name_context.append(f"Last name: {last_name}")
    result = call_ollama_json(
        prompt=(
            'Return JSON with exactly one key named "value" containing one believable '
            "personal email address. You may use the available name information or "
            "choose an unrelated address.\n"
            + ("\n".join(name_context) if name_context else "No name information available.")
        ),
        model=model,
        system_prompt="Generate a synthetic personal email address, with no explanation.",
        options={"temperature": 1.0},
    )
    return str(result["parsed"]["value"]).strip()


def sample_email(
    first_name: object,
    last_name: object,
    rng: np.random.Generator,
    allow_ollama: bool = False,
    ollama_model: str = "llama3.2:1b",
) -> str:
    """Choose uniformly among email routes supported by the available names."""
    has_first = pd.notna(first_name) and bool(str(first_name).strip())
    has_last = pd.notna(last_name) and bool(str(last_name).strip())
    routes = ["faker"]
    if has_first:
        routes.append("first")
    if has_last:
        routes.append("last")
    if has_first and has_last:
        routes.append("first_last")
    if allow_ollama:
        routes.append("ollama")

    route = str(rng.choice(routes))
    if route == "first":
        return sample_single_name_email(first_name, rng)
    if route == "last":
        return sample_single_name_email(last_name, rng)
    if route == "first_last":
        return make_personal_email(first_name, last_name, rng)
    if route == "ollama":
        return sample_ollama_email(first_name, last_name, ollama_model)
    return sample_faker_email(rng)


def augment_people_record_fields(
    people: pd.DataFrame,
    fields: Collection[str],
    rng: np.random.Generator,
    allow_ollama: bool = False,
    ollama_model: str = "llama3.2:1b",
    country_field: str | None = None,
) -> pd.DataFrame:
    """Add only the requested synthetic fields, once per person."""
    augmented = people.copy()
    if "ssn" in fields:
        augmented["ssn"] = [sample_ssn(rng) for _ in range(len(people))]
    if "email" in fields:
        augmented["email"] = [
            sample_email(
                person.get("firstname"),
                person.get("lastname"),
                rng,
                allow_ollama=allow_ollama,
                ollama_model=ollama_model,
            )
            for _, person in people.iterrows()
        ]
    for field in ("phone", "cell"):
        if field in fields:
            augmented[field] = [
                sample_phone(person.get(country_field), field, rng)
                for _, person in people.iterrows()
            ]
    return augmented
