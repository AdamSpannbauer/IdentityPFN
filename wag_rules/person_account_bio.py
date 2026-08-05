import numpy as np


CONTACT_ROLES = (
    "contact",
    "account contact",
    "profile owner",
    "representative",
    "customer contact",
    "business contact",
)

RELATION_PHRASES = (
    "associated with",
    "connected to",
    "linked to",
    "listed with",
    "recorded under",
    "attached to",
)

BUSINESS_CONTEXTS = (
    "account activity",
    "customer outreach",
    "support records",
    "contact management",
    "sales follow-up",
    "profile maintenance",
)

NAME_COMPANY_TEMPLATES = (
    "{name} works with {company}.",
    "{name} is {relation} {company}.",
    "{name} is a {role} at {company}.",
    "{name} is listed as a representative for {company}.",
    "{name} appears in records for {company}.",
    "{name} is the named {role} for {company}.",
    "{company} has {name} on file as a {role}.",
    "{name} is tied to {company} for {context}.",
    "{name} is the profile contact connected to {company}.",
    "{name} is recorded as an account contact for {company}.",
    "{name} maintains a business profile with {company}.",
    "{name} is listed in the contact system under {company}.",
    "Contact record for {name}, associated with {company}.",
    "Account profile for {name} at {company}.",
    "{company} contact profile naming {name}.",
    "The account record lists {name} as a {role} for {company}, with notes related to {context}.",
    "Contact data associates {name} with {company}, including profile details used for {context}.",
)

NAME_LOCATION_TEMPLATES = (
    "{name} is based in {place}.",
    "{name} is a {place}-based {role}.",
    "{name} maintains an account profile in {place}.",
    "{name} is listed with a location in {place}.",
    "{name} appears in regional records for {place}.",
    "{name} is attached to a contact profile in {place}.",
    "{name} has a listed service area of {place}.",
    "{name} is recorded as a local {role} in {place}.",
    "Regional profile for {name}, located in {place}.",
    "Contact record for {name} with location {place}.",
    "{name} is connected to {place} in account records.",
    "{name} is tracked as a {place} contact.",
)

COMPANY_LOCATION_TEMPLATES = (
    "{company} contact record associated with {place}.",
    "Account contact connected to {company} in {place}.",
    "{company} account profile with activity in {place}.",
    "Business contact for {company}, located in {place}.",
    "{company} has a contact profile tied to {place}.",
    "{company} record linked to {context} in {place}.",
    "{company} contact entry for the {place} region.",
    "Regional account profile for {company} in {place}.",
    "{company} appears in contact records for {place}.",
    "Customer profile for {company}, associated with {place}.",
    "{company} account activity is listed under {place}.",
    "Contact management record for {company} in {place}.",
    "{company} maintains a regional contact profile for {place}, with account notes tied to {context}.",
    "{company} has account details associated with {place}, including support context and contact follow-up.",
)

FULL_CONTEXT_TEMPLATES = (
    "{name} works with {company} and is based in {place}.",
    "{name} is a {place}-based {role} associated with {company}.",
    "{name} represents {company} for {context} in {place}.",
    "{name} is listed as a contact for {company} in {place}.",
    "{name} maintains a profile connected to {company} and {place}.",
    "{name} is recorded under {company} with a location in {place}.",
    "{company} lists {name} as a {role} for {place}.",
    "{name} is the {role} tied to {company}'s {place} account record.",
    "Contact profile for {name} at {company}, based in {place}.",
    "{name} appears in {company} records for {context} in {place}.",
    "{name} is connected to {company} through a {place} profile.",
    "{name} is listed with {company} for regional activity in {place}.",
    "{company} account note: {name} is the {place} contact.",
    "{name} maintains {company} contact details for {place}.",
    "Profile entry links {name}, {company}, and {place}.",
    "{name} is listed as the {role} for {company}, with account activity and profile updates tied to {place}.",
    "{company} records identify {name} as a {place}-based {role} for account coordination and contact maintenance.",
    "{name} appears as the primary profile contact for {company}, with business activity associated with {place}.",
    "{name} is connected to {company} as a {role} for customer communication and ongoing account work in {place}.",
    "The profile describes {name} as a {company} contact whose account details and regional notes point to {place}.",
    "Contact data ties {name} to {company} for {context}, with recent profile information centered on {place}.",
)

SPARSE_CRM_TEMPLATES = (
    "Primary contact: {name}.",
    "Account profile for {name}.",
    "{name} appears in the contact database.",
    "Contact record linked to {name}.",
    "{name} has an active account profile.",
    "Profile owner: {name}.",
    "Contact entry for {name}.",
    "{name} is listed as a customer contact.",
    "CRM profile associated with {name}.",
    "{name} appears in account records.",
    "Open contact profile for {name}.",
    "{name} is tracked in the contact system.",
)


def sample_phrase(values: tuple[str, ...], rng: np.random.Generator) -> str:
    return str(rng.choice(values))


def format_person_name(first_name: object, last_name: object) -> str:
    return " ".join(part for part in (str(first_name), str(last_name)) if part).strip()


def format_location(location: object) -> str:
    if isinstance(location, dict):
        city = location.get("city")
        state = location.get("state")
        if city and state:
            return f"{city}, {state}"
        if city:
            return str(city)
        if state:
            return str(state)
    return "the United States"


def format_company(company: object) -> str:
    return str(company).strip().rstrip(".")


def make_bio_name_company(
    first_name: object,
    last_name: object,
    company: object,
    rng: np.random.Generator,
) -> str:
    name = format_person_name(first_name, last_name)
    company = format_company(company)
    return sample_phrase(NAME_COMPANY_TEMPLATES, rng).format(
        name=name,
        company=company,
        role=sample_phrase(CONTACT_ROLES, rng),
        relation=sample_phrase(RELATION_PHRASES, rng),
        context=sample_phrase(BUSINESS_CONTEXTS, rng),
    )


def make_bio_name_location(
    first_name: object,
    last_name: object,
    location: object,
    rng: np.random.Generator,
) -> str:
    name = format_person_name(first_name, last_name)
    place = format_location(location)
    return sample_phrase(NAME_LOCATION_TEMPLATES, rng).format(
        name=name,
        place=place,
        role=sample_phrase(CONTACT_ROLES, rng),
    )


def make_bio_company_location(
    company: object,
    location: object,
    rng: np.random.Generator,
) -> str:
    place = format_location(location)
    company = format_company(company)
    return sample_phrase(COMPANY_LOCATION_TEMPLATES, rng).format(
        company=company,
        place=place,
        context=sample_phrase(BUSINESS_CONTEXTS, rng),
    )


def make_bio_full_context(
    first_name: object,
    last_name: object,
    company: object,
    location: object,
    rng: np.random.Generator,
) -> str:
    name = format_person_name(first_name, last_name)
    place = format_location(location)
    company = format_company(company)
    return sample_phrase(FULL_CONTEXT_TEMPLATES, rng).format(
        name=name,
        company=company,
        place=place,
        role=sample_phrase(CONTACT_ROLES, rng),
        context=sample_phrase(BUSINESS_CONTEXTS, rng),
    )


def make_sparse_crm_note(
    first_name: object,
    last_name: object,
    rng: np.random.Generator,
) -> str:
    name = format_person_name(first_name, last_name)
    return sample_phrase(SPARSE_CRM_TEMPLATES, rng).format(name=name)


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    location = {
        "street_address": "123 Market St",
        "city": "Columbus",
        "state": "OH",
        "postal_code": "43215",
        "country": "US",
    }
    examples = [
        ("Jane", "Patel", "Northstar Analytics LLC", location),
        ("José", "Nowicki", "Blue River Systems Inc.", location),
        ("Mary Beth", "O'Neil", "Atlas & Finch Co.", location),
    ]

    for first_name, last_name, company, place in examples:
        print(f"\n{format_person_name(first_name, last_name)} | {company}")
        print("name_company:", make_bio_name_company(first_name, last_name, company, rng))
        print("name_location:", make_bio_name_location(first_name, last_name, place, rng))
        print("company_location:", make_bio_company_location(company, place, rng))
        print(
            "full_context:",
            make_bio_full_context(first_name, last_name, company, place, rng),
        )
        print("sparse:", make_sparse_crm_note(first_name, last_name, rng))
