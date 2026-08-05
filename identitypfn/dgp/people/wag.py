from dataclasses import dataclass, field, replace
import time
from typing import Literal

from faker import Faker
import numpy as np
import pandas as pd

from ...call_ollama import OllamaModel, call_ollama_json
from .simple import (
    Location,
    ObservationConfig,
    _temporary_global_seed,
    get_character_augmenters,
    load_us_locations,
)
from .wag_rules.person_account_bio import (
    make_bio_company_location,
    make_bio_full_context,
    make_bio_name_company,
    make_bio_name_location,
    make_sparse_crm_note,
)
from .wag_rules.person_email import (
    make_personal_email,
    make_personal_email_with_dob,
    make_role_email,
    make_work_email,
)
from .wag_rules.person_id import (
    make_account_id,
    make_customer_id,
    make_external_id_from_name,
    make_source_system_id,
)
from .wag_rules.person_phone import (
    make_independent_phone,
    make_phone_from_city_state,
    make_phone_from_state,
)
from .wag_rules.person_username import (
    make_independent_username,
    make_username_from_name,
    make_username_from_name_company,
    make_username_from_name_dob,
)
from ..world import World, sample_entity_ids

PrimitiveType = Literal[
    "text", "categorical", "numeric", "date", "identifier", "email", "phone"
]


@dataclass(frozen=True)
class NodeType:
    name: str
    primitive_type: PrimitiveType | None


@dataclass(frozen=True)
class RealizedNode:
    node_type: NodeType
    value: object | None


@dataclass(frozen=True)
class Rule:
    name: str
    output_node_type: NodeType
    input_node_types: tuple[NodeType, ...]
    weight: float = field(default=1.0, kw_only=True)

    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        raise NotImplementedError


@dataclass(frozen=True)
class OllamaRule(Rule):
    system_prompt: str
    model: OllamaModel | str
    task_prompt: str | None = None
    temperature: float | None = None
    options: dict | None = None
    max_retries: int = 2
    allow_input_subset: bool = True

    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del rng
        return derive_node_with_ollama(
            input_nodes=input_nodes,
            output_node_type=self.output_node_type,
            system_prompt=self.system_prompt,
            task_prompt=self.task_prompt,
            model=self.model,
            temperature=self.temperature,
            options=self.options,
            max_retries=self.max_retries,
        )


@dataclass(frozen=True)
class FakerRule(Rule):
    faker_method: str
    locale: str = "en_US"

    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del input_nodes
        faker = Faker(self.locale)
        faker.seed_instance(int(rng.integers(0, 2**32)))
        value = getattr(faker, self.faker_method)()
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class LocationRule(Rule):
    locations: tuple[Location, ...]

    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del input_nodes
        faker = Faker("en_US")
        faker.seed_instance(int(rng.integers(0, 2**32)))
        location = self.locations[rng.integers(len(self.locations))]
        value = {
            "street_address": faker.street_address(),
            "city": location.city,
            "postal_code": location.postal_code,
            "state": location.state,
            "country": "US",
        }
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class LocationExtractRule(Rule):
    location_key: str

    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del rng
        if len(input_nodes) != 1:
            raise ValueError("LocationExtractRule expects exactly one input node")
        return RealizedNode(
            self.output_node_type,
            input_nodes[0].value[self.location_key],
        )


@dataclass(frozen=True)
class FullNameRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del rng
        values = {node.node_type: str(node.value) for node in input_nodes}
        name_parts = [
            values.get(FIRST_NAME),
            values.get(MIDDLE_NAME),
            values.get(LAST_NAME),
        ]
        value = " ".join(part for part in name_parts if part)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class DerivedEmailRule(Rule):
    domains: tuple[str, ...] = (
        "gmail.com",
        "outlook.com",
        "icloud.com",
        "yahoo.com",
        "proton.me",
    )

    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: str(node.value) for node in input_nodes}
        first = _email_token(values.get(FIRST_NAME, "user"))
        last = _email_token(values.get(LAST_NAME, "account"))
        company = _email_token(values.get(COMPANY, ""))
        domain = rng.choice(self.domains)
        if company and rng.random() < 0.5:
            domain = f"{company}.com"
        separator = rng.choice([".", "_", ""])
        suffix = "" if rng.random() < 0.7 else str(rng.integers(10, 100))
        return RealizedNode(
            self.output_node_type,
            f"{first}{separator}{last}{suffix}@{domain}".lower(),
        )


@dataclass(frozen=True)
class PersonalEmailRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_personal_email(values[FIRST_NAME], values[LAST_NAME], rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class PersonalEmailWithDobRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_personal_email_with_dob(
            values[FIRST_NAME],
            values[LAST_NAME],
            values[DATE_OF_BIRTH],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class WorkEmailRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_work_email(
            values[FIRST_NAME],
            values[LAST_NAME],
            values[COMPANY],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class RoleEmailRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_role_email(values[COMPANY], rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class UsernameRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: str(node.value) for node in input_nodes}
        first = _email_token(values.get(FIRST_NAME, "user"))
        last = _email_token(values.get(LAST_NAME, "account"))
        year = ""
        date_value = values.get(DATE_OF_BIRTH)
        if date_value:
            year = str(date_value)[:4]
        suffix = year if year and rng.random() < 0.5 else str(rng.integers(10, 1000))
        separator = rng.choice(["", "_", "."])
        return RealizedNode(
            self.output_node_type,
            f"{first}{separator}{last}{suffix}".lower(),
        )


@dataclass(frozen=True)
class UsernameFromNameRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_username_from_name(values[FIRST_NAME], values[LAST_NAME], rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class UsernameFromNameDobRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_username_from_name_dob(
            values[FIRST_NAME],
            values[LAST_NAME],
            values[DATE_OF_BIRTH],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class UsernameFromNameCompanyRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_username_from_name_company(
            values[FIRST_NAME],
            values[LAST_NAME],
            values[COMPANY],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class IndependentUsernameRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del input_nodes
        value = make_independent_username(rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class AccountBioRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del rng
        values = {node.node_type: node.value for node in input_nodes}
        first = values.get(FIRST_NAME, "Person")
        last = values.get(LAST_NAME, "")
        company = values.get(COMPANY, "an organization")
        location = values.get(LOCATION)
        if isinstance(location, dict):
            place = f"{location['city']}, {location['state']}"
        else:
            place = "the United States"
        return RealizedNode(
            self.output_node_type,
            f"{first} {last} works with {company} and is based in {place}.",
        )


@dataclass(frozen=True)
class BioFromNameCompanyRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_bio_name_company(
            values[FIRST_NAME],
            values[LAST_NAME],
            values[COMPANY],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class BioFromNameLocationRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_bio_name_location(
            values[FIRST_NAME],
            values[LAST_NAME],
            values[LOCATION],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class BioFromCompanyLocationRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_bio_company_location(
            values[COMPANY],
            values[LOCATION],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class BioFromFullContextRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_bio_full_context(
            values[FIRST_NAME],
            values[LAST_NAME],
            values[COMPANY],
            values[LOCATION],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class SparseCrmBioRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_sparse_crm_note(values[FIRST_NAME], values[LAST_NAME], rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class PhoneRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del input_nodes
        area_code = rng.integers(200, 1000)
        exchange = rng.integers(200, 1000)
        line_number = rng.integers(0, 10000)
        value = f"({area_code}) {exchange}-{line_number:04d}"
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class PhoneFromStateRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_phone_from_state(values[STATE], rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class PhoneFromCityStateRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_phone_from_city_state(values[CITY], values[STATE], rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class IndependentPhoneRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del input_nodes
        value = make_independent_phone(rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class CustomerIdRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del input_nodes
        value = make_customer_id(rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class AccountIdFromCompanyRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_account_id(values[COMPANY], rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class SourceSystemIdRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        del input_nodes
        value = make_source_system_id(rng)
        return RealizedNode(self.output_node_type, value)


@dataclass(frozen=True)
class ExternalIdFromNameRule(Rule):
    def derive(
        self, input_nodes: list[RealizedNode], rng: np.random.Generator
    ) -> RealizedNode:
        values = {node.node_type: node.value for node in input_nodes}
        value = make_external_id_from_name(
            values[FIRST_NAME],
            values[LAST_NAME],
            rng,
        )
        return RealizedNode(self.output_node_type, value)


def _email_token(value: str) -> str:
    token = "".join(character for character in value.lower() if character.isalnum())
    return token or "user"


@dataclass(frozen=True)
class RuleSet:
    output_node_type: NodeType
    rules: tuple[Rule, ...]


def serialize_realized_node(node: RealizedNode) -> dict[str, object | None]:
    return {
        "name": node.node_type.name,
        "value": node.value,
    }


def derive_node_with_ollama(
    input_nodes: list[RealizedNode],
    output_node_type: NodeType,
    system_prompt: str,
    model: OllamaModel | str,
    task_prompt: str | None = None,
    temperature: float | None = None,
    options: dict | None = None,
    max_retries: int = 2,
) -> RealizedNode:
    """Derive one entity-level node value from typed input node values."""
    if temperature is not None:
        options = dict(options or {})
        options["temperature"] = temperature

    serialized_inputs = [serialize_realized_node(node) for node in input_nodes]
    use_task_specific_prompt = task_prompt is not None
    if task_prompt is None:
        if output_node_type.name == "account_bio":
            task_prompt = (
                "Write a realistic account_bio text field for a person, CRM contact, "
                "user profile, or online account.\n"
                "Use the available facts, and when plausible add lightweight inferred "
                "context such as job title, team/function, account role, stakeholder "
                "role, support role, or sales context.\n"
                "The output should read like a field value from a profile, CRM note, "
                "directory entry, or account system.\n"
                "Do not return only a name, company, city, state, address, job title, "
                "or other single extracted field.\n"
                "Return one complete sentence."
            )
        else:
            task_prompt = (
                f"Make up a realistic {output_node_type.name} "
                "for a person/online account."
            )

    if output_node_type.name == "account_bio":
        response_contract = (
            'Return exactly one JSON object with exactly one key named "value".\n'
            'The "value" must be one string containing a complete one-sentence '
            "bio/profile/account-note field.\n"
            "Do not return only a job title, name, company, city, state, address, "
            "or other extracted field.\n"
            "Do not return an object, array, label, alternatives, explanations, "
            "examples, or nested fields.\n"
            'Good example: {"value": "Jodi Chen is listed as a Norris Group '
            'account contact for customer support and profile updates in '
            'Meadow Valley, CA."}\n'
            'Bad examples: {"value": "Account Accountant"}; '
            '{"value": "Norris Group"}; '
            '{"value": {"title": "Accountant", "company": "Norris Group"}}.'
        )
    else:
        response_contract = (
            'Return exactly one JSON object with exactly one key named "value".\n'
            'The "value" must be a single scalar field value: a string, number, '
            "boolean, or null.\n"
            "Do not return an object or array as the value.\n"
            "Do not add metadata, labels, alternatives, explanations, examples, "
            "or nested fields."
        )

    base_prompt = (
        f"{task_prompt}\n\n"
        f"Available information about the person:\n{serialized_inputs}\n\n"
        "Return the JSON object now."
        if use_task_specific_prompt
        else (
            f"{task_prompt}\n\n"
            f"Available information about the person:\n{serialized_inputs}\n\n"
            f"{response_contract}"
        )
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
        if isinstance(parsed, dict) and "value" in parsed:
            value = parsed["value"]
            if not isinstance(value, dict | list):
                return RealizedNode(output_node_type, value)

        if attempt < max_retries:
            if output_node_type.name == "account_bio":
                retry_instruction = (
                    'Return only {"value": "<complete one-sentence account bio>"}; '
                    "do not return only a title, company, location, or extracted field."
                )
            else:
                retry_instruction = (
                    'Return only {"value": <single scalar field value>}.'
                )
            prompt = (
                f"{base_prompt}\n\n"
                f"Your previous response violated the required schema: {parsed!r}\n"
                f"{retry_instruction}"
            )

    raise ValueError(
        "Ollama response failed scalar value schema "
        f"for node_type={output_node_type.name!r}, model={model!r}, "
        f"max_retries={max_retries}, last_parsed={last_parsed!r}"
    )


@dataclass(frozen=True)
class GraphEdge:
    source: NodeType
    target: NodeType


@dataclass(frozen=True)
class UniverseGraph:
    node_types: tuple[NodeType, ...]
    edges: tuple[GraphEdge, ...]
    rule_sets: dict[NodeType, RuleSet]


@dataclass(frozen=True)
class WorldGraph:
    universe: UniverseGraph
    node_types: tuple[NodeType, ...]
    edges: tuple[GraphEdge, ...]
    rule_sets: dict[NodeType, RuleSet]
    projected_node_types: tuple[NodeType, ...]


@dataclass(frozen=True)
class RealizedEntityGraph:
    world_graph: WorldGraph
    nodes: dict[NodeType, RealizedNode]
    rules: dict[NodeType, Rule]


@dataclass(frozen=True)
class GeneratedGraphWorld:
    world: World
    world_graph: WorldGraph
    entity_graphs: list[RealizedEntityGraph]
    hard_negative_events: tuple["HardNegativeEvent", ...] = ()


@dataclass(frozen=True)
class HardNegativeContract:
    name: str
    preserve: frozenset[NodeType]
    reroll: frozenset[NodeType]


@dataclass(frozen=True)
class HardNegativeEvent:
    contract_name: str
    base_entity_id: int
    target_entity_id: int
    preserved_nodes: tuple[str, ...]
    rerolled_nodes: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionSamplerConfig:
    observable_node_types: tuple[NodeType, ...]
    bos_scores: dict[NodeType, float]
    pairwise_affinities: dict[tuple[NodeType, NodeType], float]


def make_generic_ollama_rule_set(
    output_node_type: NodeType,
    edges: tuple[GraphEdge, ...],
    system_prompt: str,
    model: OllamaModel | str,
    temperature: float | None = None,
    options: dict | None = None,
    weight: float = 0.05,
    max_retries: int = 2,
) -> RuleSet:
    input_node_types = tuple(
        edge.source for edge in edges if edge.target == output_node_type
    )
    return RuleSet(
        output_node_type=output_node_type,
        rules=(
            OllamaRule(
                name=f"llm_{output_node_type.name}",
                output_node_type=output_node_type,
                input_node_types=input_node_types,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                options=options,
                max_retries=max_retries,
                weight=weight,
            ),
        ),
    )


def append_rule_to_rule_set(
    rule_sets: dict[NodeType, RuleSet],
    rule: Rule,
) -> None:
    rule_set = rule_sets[rule.output_node_type]
    rule_sets[rule.output_node_type] = RuleSet(
        output_node_type=rule_set.output_node_type,
        rules=rule_set.rules + (rule,),
    )


def sample_weighted_rule(rules: list[Rule], rng: np.random.Generator) -> Rule:
    if not rules:
        raise ValueError("Cannot sample from an empty rule list")

    weights = np.asarray([rule.weight for rule in rules], dtype=float)
    if np.any(weights < 0):
        raise ValueError("Rule weights must be non-negative")
    if weights.sum() <= 0:
        raise ValueError("At least one rule weight must be positive")

    probabilities = weights / weights.sum()
    return rules[rng.choice(len(rules), p=probabilities)]


def filter_rule_options(
    rules: tuple[Rule, ...],
    allow_ollama: bool,
    active_node_types: set[NodeType],
) -> list[Rule]:
    return [
        rule
        for rule in rules
        if (allow_ollama or not isinstance(rule, OllamaRule))
        and all(node_type in active_node_types for node_type in rule.input_node_types)
    ]


def sample_world_rule_sets(
    rule_sets: dict[NodeType, RuleSet],
    rng: np.random.Generator,
    active_node_types: set[NodeType],
    allow_ollama: bool = True,
) -> dict[NodeType, RuleSet]:
    sampled_rule_sets = {}
    for node_type, rule_set in rule_sets.items():
        options = filter_rule_options(
            rule_set.rules,
            allow_ollama=allow_ollama,
            active_node_types=active_node_types,
        )
        if not options:
            candidate_descriptions = [
                (
                    rule.name,
                    [input_node_type.name for input_node_type in rule.input_node_types],
                )
                for rule in rule_set.rules
            ]
            raise ValueError(
                f"No viable world-level rule for {node_type.name!r}; "
                f"active_nodes={sorted(node.name for node in active_node_types)}, "
                f"candidates={candidate_descriptions}"
            )
        selected_rule = sample_weighted_rule(options, rng)
        sampled_rule_sets[node_type] = RuleSet(
            output_node_type=rule_set.output_node_type,
            rules=(selected_rule,),
        )
    return sampled_rule_sets


def get_projection_affinity(
    left: NodeType,
    right: NodeType,
    affinities: dict[tuple[NodeType, NodeType], float],
) -> float:
    return affinities.get((left, right), affinities.get((right, left), 0.0))


def sample_projected_node_types(
    n_fields: int,
    config: ProjectionSamplerConfig,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> tuple[NodeType, ...]:
    if n_fields <= 0:
        raise ValueError("n_fields must be positive")
    if n_fields > len(config.observable_node_types):
        raise ValueError("n_fields exceeds observable node count")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    selected: list[NodeType] = []
    while len(selected) < n_fields:
        candidates = [
            node_type
            for node_type in config.observable_node_types
            if node_type not in selected
        ]
        scores = np.asarray(
            [
                config.bos_scores.get(candidate, 0.0)
                + sum(
                    get_projection_affinity(
                        candidate, selected_node, config.pairwise_affinities
                    )
                    for selected_node in selected
                )
                for candidate in candidates
            ]
        )
        scaled_scores = scores / temperature
        probabilities = np.exp(scaled_scores - scaled_scores.max())
        probabilities /= probabilities.sum()
        selected.append(candidates[rng.choice(len(candidates), p=probabilities)])

    rng.shuffle(selected)
    return tuple(selected)


def get_ancestor_closure(
    node_types: tuple[NodeType, ...],
    edges: tuple[GraphEdge, ...],
) -> tuple[NodeType, ...]:
    active = {ENTITY, *node_types}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.target in active and edge.source not in active:
                active.add(edge.source)
                changed = True
    return tuple(node_type for node_type in node_types_with_entity_first(active))


def node_types_with_entity_first(node_types: set[NodeType]) -> list[NodeType]:
    return sorted(
        node_types,
        key=lambda node_type: (node_type.name != "entity", node_type.name),
    )


def sample_world_graph(
    universe: UniverseGraph,
    projection_config: ProjectionSamplerConfig,
    n_fields: int,
    rng: np.random.Generator,
    projection_temperature: float = 1.0,
    allow_ollama: bool = True,
) -> WorldGraph:
    projected_node_types = sample_projected_node_types(
        n_fields=n_fields,
        config=projection_config,
        rng=rng,
        temperature=projection_temperature,
    )
    node_types = get_ancestor_closure(projected_node_types, universe.edges)
    active_node_types = set(node_types)
    edges = tuple(
        edge
        for edge in universe.edges
        if edge.source in active_node_types and edge.target in active_node_types
    )
    rule_sets = {
        node_type: universe.rule_sets[node_type]
        for node_type in node_types
        if node_type in universe.rule_sets
    }
    rule_sets = sample_world_rule_sets(
        rule_sets,
        rng=rng,
        active_node_types=active_node_types,
        allow_ollama=allow_ollama,
    )
    return WorldGraph(
        universe=universe,
        node_types=node_types,
        edges=edges,
        rule_sets=rule_sets,
        projected_node_types=projected_node_types,
    )


def topological_node_order(world_graph: WorldGraph) -> tuple[NodeType, ...]:
    ordered = []
    remaining = set(world_graph.node_types)
    active_edges = [
        edge
        for edge in world_graph.edges
        if edge.source in remaining and edge.target in remaining
    ]

    while remaining:
        ready = [
            node_type
            for node_type in world_graph.node_types
            if node_type in remaining
            and all(
                edge.source not in remaining
                for edge in active_edges
                if edge.target == node_type
            )
        ]
        if not ready:
            raise ValueError("World graph contains a dependency cycle")
        ordered.extend(ready)
        remaining.difference_update(ready)

    return tuple(ordered)


def sample_realization_rule(
    rule_set: RuleSet,
    realized_nodes: dict[NodeType, RealizedNode],
    rng: np.random.Generator,
    allow_ollama: bool = True,
) -> Rule:
    viable_rules = [
        rule
        for rule in rule_set.rules
        if all(node_type in realized_nodes for node_type in rule.input_node_types)
        and (allow_ollama or not isinstance(rule, OllamaRule))
    ]
    if not viable_rules:
        raise ValueError(f"No viable rule for {rule_set.output_node_type.name}")
    return sample_weighted_rule(viable_rules, rng)


def get_rule_input_nodes(
    rule: Rule,
    realized_nodes: dict[NodeType, RealizedNode],
    rng: np.random.Generator,
) -> list[RealizedNode]:
    input_node_types = list(rule.input_node_types)
    if (
        isinstance(rule, OllamaRule)
        and rule.allow_input_subset
        and input_node_types
    ):
        n_inputs = rng.integers(1, len(input_node_types) + 1)
        input_node_types = list(
            rng.choice(input_node_types, size=n_inputs, replace=False)
        )
    return [realized_nodes[node_type] for node_type in input_node_types]


def realize_entity_graph(
    world_graph: WorldGraph,
    rng: np.random.Generator,
    allow_ollama: bool = True,
) -> RealizedEntityGraph:
    realized_nodes = {ENTITY: RealizedNode(ENTITY, None)}
    selected_rules = {}

    for node_type in topological_node_order(world_graph):
        if node_type == ENTITY:
            continue
        rule_set = world_graph.rule_sets[node_type]
        rule = sample_realization_rule(
            rule_set,
            realized_nodes,
            rng,
            allow_ollama=allow_ollama,
        )
        input_nodes = get_rule_input_nodes(rule, realized_nodes, rng)
        realized_nodes[node_type] = rule.derive(input_nodes, rng)
        selected_rules[node_type] = rule

    return RealizedEntityGraph(
        world_graph=world_graph,
        nodes=realized_nodes,
        rules=selected_rules,
    )


def normalize_hard_negative_contract_weights(
    weights: dict[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        return DEFAULT_HARD_NEGATIVE_CONTRACT_WEIGHTS

    unknown = set(weights) - set(DEFAULT_HARD_NEGATIVE_CONTRACTS)
    if unknown:
        raise ValueError(f"Unknown hard-negative contracts: {sorted(unknown)}")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Hard-negative contract weights must be non-negative")
    return weights


def hard_negative_contract_is_eligible(
    contract: HardNegativeContract,
    active_node_types: set[NodeType],
) -> bool:
    return bool(contract.preserve & active_node_types) and bool(
        contract.reroll & active_node_types
    )


def sample_hard_negative_contract(
    active_node_types: set[NodeType],
    weights: dict[str, float],
    rng: np.random.Generator,
) -> HardNegativeContract | None:
    candidates = [
        contract
        for name, contract in DEFAULT_HARD_NEGATIVE_CONTRACTS.items()
        if weights.get(name, 0.0) > 0
        and hard_negative_contract_is_eligible(contract, active_node_types)
    ]
    if not candidates:
        return None

    candidate_weights = np.asarray([weights[contract.name] for contract in candidates])
    if candidate_weights.sum() <= 0:
        return None
    probabilities = candidate_weights / candidate_weights.sum()
    return candidates[rng.choice(len(candidates), p=probabilities)]


def dependency_descendants(
    node_types: set[NodeType],
    world_graph: WorldGraph,
    blocked_node_types: set[NodeType],
) -> set[NodeType]:
    descendants = set(node_types)
    changed = True
    while changed:
        changed = False
        for edge in world_graph.edges:
            if (
                edge.source in descendants
                and edge.target not in descendants
                and edge.target not in blocked_node_types
            ):
                descendants.add(edge.target)
                changed = True
    return descendants


def make_hard_negative_sibling_graph(
    base_graph: RealizedEntityGraph,
    contract: HardNegativeContract,
    rng: np.random.Generator,
) -> tuple[RealizedEntityGraph, tuple[str, ...], tuple[str, ...]]:
    world_graph = base_graph.world_graph
    active_node_types = set(world_graph.node_types)
    preserved_node_types = contract.preserve & active_node_types
    reroll_node_types = dependency_descendants(
        contract.reroll & active_node_types,
        world_graph,
        blocked_node_types=preserved_node_types,
    )

    realized_nodes = dict(base_graph.nodes)
    selected_rules = dict(base_graph.rules)
    for node_type in topological_node_order(world_graph):
        if node_type == ENTITY or node_type not in reroll_node_types:
            continue
        rule = selected_rules[node_type]
        input_nodes = get_rule_input_nodes(rule, realized_nodes, rng)
        realized_nodes[node_type] = rule.derive(input_nodes, rng)

    sibling_graph = RealizedEntityGraph(
        world_graph=world_graph,
        nodes=realized_nodes,
        rules=selected_rules,
    )
    preserved_names = tuple(
        node_type.name for node_type in node_types_with_entity_first(preserved_node_types)
    )
    rerolled_names = tuple(
        node_type.name for node_type in node_types_with_entity_first(reroll_node_types)
    )
    return sibling_graph, preserved_names, rerolled_names


def add_hard_negative_siblings(
    entity_graphs: list[RealizedEntityGraph],
    world_graph: WorldGraph,
    hard_negative_rate: float,
    hard_negative_contract_weights: dict[str, float] | None,
    rng: np.random.Generator,
) -> tuple[list[RealizedEntityGraph], tuple[HardNegativeEvent, ...]]:
    if not 0 <= hard_negative_rate <= 1:
        raise ValueError("hard_negative_rate must be between 0 and 1")
    if hard_negative_rate == 0 or len(entity_graphs) < 2:
        return entity_graphs, ()

    weights = normalize_hard_negative_contract_weights(hard_negative_contract_weights)
    active_node_types = set(world_graph.node_types)
    if sample_hard_negative_contract(active_node_types, weights, rng) is None:
        return entity_graphs, ()

    n_events = min(
        int(rng.binomial(len(entity_graphs), hard_negative_rate)),
        len(entity_graphs) - 1,
    )
    if n_events == 0:
        return entity_graphs, ()

    updated_graphs = list(entity_graphs)
    entity_indices = np.arange(len(entity_graphs))
    target_indices = rng.choice(entity_indices, size=n_events, replace=False)
    base_candidates = np.setdiff1d(entity_indices, target_indices)
    if not len(base_candidates):
        base_candidates = entity_indices

    events = []
    for target_index in target_indices:
        possible_bases = base_candidates[base_candidates != target_index]
        if not len(possible_bases):
            continue
        base_index = int(rng.choice(possible_bases))
        contract = sample_hard_negative_contract(active_node_types, weights, rng)
        if contract is None:
            continue
        sibling_graph, preserved_names, rerolled_names = (
            make_hard_negative_sibling_graph(
                entity_graphs[base_index],
                contract,
                rng,
            )
        )
        updated_graphs[int(target_index)] = sibling_graph
        events.append(
            HardNegativeEvent(
                contract_name=contract.name,
                base_entity_id=base_index,
                target_entity_id=int(target_index),
                preserved_nodes=preserved_names,
                rerolled_nodes=rerolled_names,
            )
        )

    return updated_graphs, tuple(events)


def generate_clean_projected_records(
    entity_graphs: list[RealizedEntityGraph],
    entity_ids: np.ndarray,
    world_graph: WorldGraph,
) -> pd.DataFrame:
    records = []
    for entity_id in entity_ids:
        entity_graph = entity_graphs[entity_id]
        record = [
            entity_graph.nodes[node_type].value
            for node_type in world_graph.projected_node_types
        ]
        records.append(record)
    return pd.DataFrame(
        records,
        columns=[node_type.name for node_type in world_graph.projected_node_types],
    )


def observe_projected_value(
    value,
    node_type: NodeType,
    config: ObservationConfig,
    rng: np.random.Generator,
):
    if rng.random() < config.missing_rate:
        return np.nan

    observed = value
    if (
        node_type.primitive_type in {"text", "identifier"}
        and rng.random() < config.corruption_rate
    ):
        augmenters = get_character_augmenters()
        augmenter = augmenters[rng.integers(len(augmenters))]
        seed = int(rng.integers(0, 2**32))
        with _temporary_global_seed(seed):
            observed = augmenter.augment(str(observed))[0]
    return observed


def observe_projected_records(
    clean_records: pd.DataFrame,
    world_graph: WorldGraph,
    config: ObservationConfig | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if config is None:
        return clean_records

    records = []
    for _, row in clean_records.iterrows():
        record = [
            observe_projected_value(row[node_type.name], node_type, config, rng)
            for node_type in world_graph.projected_node_types
        ]
        records.append(record)
    return pd.DataFrame(records, columns=clean_records.columns)


def generate_wag_world(
    n_records: int = 20,
    p_match: float = 0.05,
    n_fields: int = 4,
    projection_temperature: float = 1.0,
    observation_config: ObservationConfig | None = None,
    allow_ollama: bool = True,
    ollama_model: OllamaModel | str = "qwen2.5:7b",
    hard_negative_rate: float = 0.0,
    hard_negative_contract_weights: dict[str, float] | None = None,
    seed: int | None = None,
) -> World:
    return generate_graph_world_details(
        n_records=n_records,
        p_match=p_match,
        n_fields=n_fields,
        projection_temperature=projection_temperature,
        observation_config=observation_config,
        allow_ollama=allow_ollama,
        ollama_model=ollama_model,
        hard_negative_rate=hard_negative_rate,
        hard_negative_contract_weights=hard_negative_contract_weights,
        seed=seed,
    ).world


def generate_graph_world_details(
    n_records: int = 20,
    p_match: float = 0.05,
    n_fields: int = 4,
    projection_temperature: float = 1.0,
    observation_config: ObservationConfig | None = None,
    allow_ollama: bool = True,
    ollama_model: OllamaModel | str = "qwen2.5:7b",
    hard_negative_rate: float = 0.0,
    hard_negative_contract_weights: dict[str, float] | None = None,
    seed: int | None = None,
) -> GeneratedGraphWorld:
    rng = np.random.default_rng(seed)
    world_graph = sample_person_world_graph(
        n_fields=n_fields,
        rng=rng,
        projection_temperature=projection_temperature,
        ollama_model=ollama_model,
        allow_ollama=allow_ollama,
    )
    entity_ids = sample_entity_ids(n_records, p_match, rng)
    entity_graphs = [
        realize_entity_graph(
            world_graph,
            rng,
        )
        for _ in range(len(np.unique(entity_ids)))
    ]
    entity_graphs, hard_negative_events = add_hard_negative_siblings(
        entity_graphs,
        world_graph,
        hard_negative_rate=hard_negative_rate,
        hard_negative_contract_weights=hard_negative_contract_weights,
        rng=rng,
    )
    clean_records = generate_clean_projected_records(
        entity_graphs, entity_ids, world_graph
    )
    records = observe_projected_records(
        clean_records, world_graph, observation_config, rng
    )

    order = rng.permutation(n_records)
    entity_ids = entity_ids[order]
    records = records.iloc[order].reset_index(drop=True)
    adjacency = entity_ids[:, None] == entity_ids[None, :]
    world = World(
        records=records,
        entity_ids=entity_ids,
        adjacency=adjacency,
        field_types=[
            node_type.primitive_type for node_type in world_graph.projected_node_types
        ],
    )
    return GeneratedGraphWorld(
        world=world,
        world_graph=world_graph,
        entity_graphs=entity_graphs,
        hard_negative_events=hard_negative_events,
    )


generate_world = generate_wag_world


def generate_worlds(
    n_worlds: int,
    n_records: list[int] = [50, 100, 300],
    p_match: list[float] = [0.001, 0.005, 0.01, 0.05],
    n_fields: list[int] = [3, 5, 10],
    projection_temperature: list[float] = [0.7, 1.0, 1.5],
    missing_rate: list[float] = [0.0, 0.1, 0.2],
    nickname_rate: list[float] = [0.0, 0.1, 0.2],
    corruption_rate: list[float] = [0.0, 0.1, 0.2],
    allow_ollama: bool = True,
    ollama_model: OllamaModel | str = "qwen2.5:7b",
    hard_negative_rate: float = 0.0,
    hard_negative_contract_weights: dict[str, float] | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> list[World]:
    rng = np.random.default_rng(seed)
    worlds = []
    for _ in range(n_worlds):
        world = generate_world(
            n_records=int(rng.choice(n_records)),
            p_match=float(rng.choice(p_match)),
            n_fields=int(rng.choice(n_fields)),
            projection_temperature=float(rng.choice(projection_temperature)),
            observation_config=ObservationConfig(
                missing_rate=float(rng.choice(missing_rate)),
                nickname_rate=float(rng.choice(nickname_rate)),
                corruption_rate=float(rng.choice(corruption_rate)),
            ),
            allow_ollama=allow_ollama,
            ollama_model=ollama_model,
            hard_negative_rate=hard_negative_rate,
            hard_negative_contract_weights=hard_negative_contract_weights,
            seed=int(rng.integers(0, 2**32)),
        )
        worlds.append(world)
        if verbose:
            print(
                f"Generated WAG world with {len(world.records)} records, "
                f"{world.records.shape[1]} fields, and "
                f"{len(np.unique(world.entity_ids))} entities."
            )
    return worlds


def print_world_graph_summary(world_graph: WorldGraph) -> None:
    print("Projected nodes:", [node_type.name for node_type in world_graph.projected_node_types])
    print("Active nodes:", [node_type.name for node_type in world_graph.node_types])
    print(
        "Active edges:",
        [(edge.source.name, edge.target.name) for edge in world_graph.edges],
    )
    print(
        "Active rule counts:",
        {
            node_type.name: len(rule_set.rules)
            for node_type, rule_set in world_graph.rule_sets.items()
        },
    )


def print_entity_rule_summary(
    entity_graphs: list[RealizedEntityGraph],
    limit: int = 5,
) -> None:
    for entity_id, entity_graph in enumerate(entity_graphs[:limit]):
        print(
            f"Entity {entity_id} rules:",
            {
                node_type.name: rule.name
                for node_type, rule in entity_graph.rules.items()
            },
        )


def print_world_records(world: World) -> None:
    print("Field types:", world.field_types)
    print(
        pd.concat(
            [pd.Series(world.entity_ids, name="entity_id"), world.records],
            axis=1,
        )
    )
    pair_mask = np.triu(np.ones_like(world.adjacency, dtype=bool), k=1)
    print("Realized pair match prevalence:", world.adjacency[pair_mask].mean())


def inspect_graph_world(
    n_records: int = 12,
    p_match: float = 0.2,
    n_fields: int = 6,
    seed: int = 3,
    observed: bool = False,
) -> None:
    observation_config = None
    if observed:
        observation_config = ObservationConfig(
            missing_rate=0.1,
            nickname_rate=0.0,
            corruption_rate=0.2,
        )
    generated = generate_graph_world_details(
        n_records=n_records,
        p_match=p_match,
        n_fields=n_fields,
        observation_config=observation_config,
        allow_ollama=False,
        seed=seed,
    )
    print("\nGraph world inspection")
    print_world_graph_summary(generated.world_graph)
    print_entity_rule_summary(generated.entity_graphs)
    print_world_records(generated.world)


################################################################################################################################################################

###############################
# PERSON UNIVERSE GRAPH BUILD #
###############################

# NodeTypes
ENTITY = NodeType("entity", None)

FIRST_NAME = NodeType("first_name", "text")
MIDDLE_NAME = NodeType("middle_name", "text")
LAST_NAME = NodeType("last_name", "text")
FULL_NAME = NodeType("full_name", "text")

DATE_OF_BIRTH = NodeType("date_of_birth", "date")

COMPANY = NodeType("company", "text")
EMAIL_ADDRESS = NodeType("email_address", "email")

LOCATION = NodeType("location", None)
STREET_ADDRESS = NodeType("street_address", "text")
CITY = NodeType("city", "text")
POSTAL_CODE = NodeType("postal_code", "text")
STATE = NodeType("state", "categorical")
COUNTRY = NodeType("country", "categorical")

PHONE = NodeType("phone", "phone")
USERNAME = NodeType("username", "text")
ACCOUNT_BIO = NodeType("account_bio", "text")

ID = NodeType("id", "identifier")

DEFAULT_HARD_NEGATIVE_CONTRACTS = {
    "household": HardNegativeContract(
        name="household",
        preserve=frozenset(
            {LAST_NAME, LOCATION, STREET_ADDRESS, CITY, POSTAL_CODE, STATE, COUNTRY}
        ),
        reroll=frozenset(
            {
                FIRST_NAME,
                MIDDLE_NAME,
                FULL_NAME,
                DATE_OF_BIRTH,
                EMAIL_ADDRESS,
                PHONE,
                USERNAME,
                ACCOUNT_BIO,
                ID,
            }
        ),
    ),
    "household_shared_phone": HardNegativeContract(
        name="household_shared_phone",
        preserve=frozenset(
            {
                LAST_NAME,
                LOCATION,
                STREET_ADDRESS,
                CITY,
                POSTAL_CODE,
                STATE,
                COUNTRY,
                PHONE,
            }
        ),
        reroll=frozenset(
            {
                FIRST_NAME,
                MIDDLE_NAME,
                FULL_NAME,
                DATE_OF_BIRTH,
                EMAIL_ADDRESS,
                USERNAME,
                ACCOUNT_BIO,
                ID,
            }
        ),
    ),
    "same_name_same_area": HardNegativeContract(
        name="same_name_same_area",
        preserve=frozenset(
            {FIRST_NAME, MIDDLE_NAME, LAST_NAME, FULL_NAME, CITY, POSTAL_CODE, STATE}
        ),
        reroll=frozenset(
            {
                DATE_OF_BIRTH,
                LOCATION,
                STREET_ADDRESS,
                COUNTRY,
                EMAIL_ADDRESS,
                PHONE,
                USERNAME,
                ACCOUNT_BIO,
                ID,
            }
        ),
    ),
    "twins": HardNegativeContract(
        name="twins",
        preserve=frozenset(
            {
                LAST_NAME,
                DATE_OF_BIRTH,
                LOCATION,
                STREET_ADDRESS,
                CITY,
                POSTAL_CODE,
                STATE,
                COUNTRY,
            }
        ),
        reroll=frozenset(
            {
                FIRST_NAME,
                MIDDLE_NAME,
                FULL_NAME,
                EMAIL_ADDRESS,
                PHONE,
                USERNAME,
                ACCOUNT_BIO,
                ID,
            }
        ),
    ),
    "jr_sr": HardNegativeContract(
        name="jr_sr",
        preserve=frozenset(
            {
                FIRST_NAME,
                MIDDLE_NAME,
                LAST_NAME,
                FULL_NAME,
                LOCATION,
                STREET_ADDRESS,
                CITY,
                POSTAL_CODE,
                STATE,
                COUNTRY,
            }
        ),
        reroll=frozenset({DATE_OF_BIRTH, EMAIL_ADDRESS, PHONE, USERNAME, ID}),
    ),
}

DEFAULT_HARD_NEGATIVE_CONTRACT_WEIGHTS = {
    "household": 1.0,
    "household_shared_phone": 0.35,
    "same_name_same_area": 1.0,
    "twins": 0.25,
    "jr_sr": 0.25,
}

# GraphEdges
PERSON_UNIVERSE_NODE_TYPES = (
    ENTITY,
    FIRST_NAME,
    MIDDLE_NAME,
    LAST_NAME,
    FULL_NAME,
    DATE_OF_BIRTH,
    COMPANY,
    EMAIL_ADDRESS,
    LOCATION,
    STREET_ADDRESS,
    CITY,
    POSTAL_CODE,
    STATE,
    COUNTRY,
    PHONE,
    USERNAME,
    ACCOUNT_BIO,
    ID,
)

PERSON_UNIVERSE_EDGES = (
    #
    GraphEdge(FIRST_NAME, FULL_NAME),
    GraphEdge(MIDDLE_NAME, FULL_NAME),
    GraphEdge(LAST_NAME, FULL_NAME),
    #
    GraphEdge(FIRST_NAME, EMAIL_ADDRESS),
    GraphEdge(MIDDLE_NAME, EMAIL_ADDRESS),
    GraphEdge(LAST_NAME, EMAIL_ADDRESS),
    GraphEdge(COMPANY, EMAIL_ADDRESS),
    #
    GraphEdge(FIRST_NAME, USERNAME),
    GraphEdge(LAST_NAME, USERNAME),
    GraphEdge(DATE_OF_BIRTH, USERNAME),
    #
    GraphEdge(FIRST_NAME, ACCOUNT_BIO),
    GraphEdge(LAST_NAME, ACCOUNT_BIO),
    GraphEdge(COMPANY, ACCOUNT_BIO),
    GraphEdge(LOCATION, ACCOUNT_BIO),
    #
    GraphEdge(LOCATION, STREET_ADDRESS),
    GraphEdge(LOCATION, CITY),
    GraphEdge(LOCATION, POSTAL_CODE),
    GraphEdge(LOCATION, STATE),
    GraphEdge(LOCATION, COUNTRY),
    #
    GraphEdge(CITY, PHONE),
    GraphEdge(STATE, PHONE),
    #
    GraphEdge(FIRST_NAME, ID),
    GraphEdge(LAST_NAME, ID),
    GraphEdge(COMPANY, ID),
    #
)

PERSON_SYSTEM_PROMPT = (
    "You generate realistic synthetic person/account data as JSON. "
    "Return complete, plausible, production-like field values. "
    'Return exactly one JSON object with exactly one key named "value". '
    'The "value" must be one scalar field value only: a string, number, boolean, or null. '
    "Never return nested objects or arrays. Never add extra keys. "
    "Do not return placeholders, examples, reserved test values, generic labels, "
    "malformed fragments, or values such as example.com, @example, @email, "
    "unknown, test, lorem ipsum, or TODO unless the requested field specifically "
    "calls for them."
)

ACCOUNT_BIO_SYSTEM_PROMPT = (
    "You generate synthetic account-bio text-field values for entity-resolution data. "
    'Return exactly one JSON object with exactly one key named "value". '
    "The value must be one string containing a complete one-sentence text field. "
    "Do not return an object, array, label, alternatives, explanations, or extra keys. "
    "Do not return only a name, company, city, state, address, job title, or other "
    "extracted field. "
    "Follow the user's requested text-field style."
)

PERSON_UNIVERSE_RULE_SETS = {
    node_type: make_generic_ollama_rule_set(
        output_node_type=node_type,
        edges=PERSON_UNIVERSE_EDGES,
        system_prompt=PERSON_SYSTEM_PROMPT,
        model="qwen2.5:7b",
        temperature=1.0,
    )
    for node_type in PERSON_UNIVERSE_NODE_TYPES
    if node_type not in {ENTITY, ID}
}
PERSON_UNIVERSE_RULE_SETS[ID] = RuleSet(output_node_type=ID, rules=())

for node_type, weight in (
    (EMAIL_ADDRESS, 0.0),
    (LOCATION, 0.0),
    (STREET_ADDRESS, 0.0),
    (CITY, 0.0),
    (POSTAL_CODE, 0.0),
    (STATE, 0.0),
    (COUNTRY, 0.0),
    (PHONE, 0.0),
    (DATE_OF_BIRTH, 0.0),
    (USERNAME, 0.0),
    (ACCOUNT_BIO, 0.10),
):
    rule_set = PERSON_UNIVERSE_RULE_SETS[node_type]
    PERSON_UNIVERSE_RULE_SETS[node_type] = RuleSet(
        output_node_type=rule_set.output_node_type,
        rules=tuple(
            replace(rule, weight=weight) if isinstance(rule, OllamaRule) else rule
            for rule in rule_set.rules
        ),
    )

for node_type, faker_method in (
    (FIRST_NAME, "first_name"),
    (MIDDLE_NAME, "first_name"),
    (LAST_NAME, "last_name"),
    (DATE_OF_BIRTH, "date_of_birth"),
    (COMPANY, "company"),
    (EMAIL_ADDRESS, "email"),
    (PHONE, "phone_number"),
    (STREET_ADDRESS, "street_address"),
    (ID, "uuid4"),
):
    append_rule_to_rule_set(
        PERSON_UNIVERSE_RULE_SETS,
        FakerRule(
            name=f"faker_{node_type.name}",
            output_node_type=node_type,
            input_node_types=(),
            faker_method=faker_method,
        ),
    )

for rule in (
    FullNameRule(
        name="compose_full_name",
        output_node_type=FULL_NAME,
        input_node_types=(FIRST_NAME, MIDDLE_NAME, LAST_NAME),
    ),
    DerivedEmailRule(
        name="derive_email_from_name_company",
        output_node_type=EMAIL_ADDRESS,
        input_node_types=(FIRST_NAME, LAST_NAME, COMPANY),
    ),
    PersonalEmailRule(
        name="personal_email_from_name",
        output_node_type=EMAIL_ADDRESS,
        input_node_types=(FIRST_NAME, LAST_NAME),
    ),
    PersonalEmailWithDobRule(
        name="personal_email_from_name_dob",
        output_node_type=EMAIL_ADDRESS,
        input_node_types=(FIRST_NAME, LAST_NAME, DATE_OF_BIRTH),
    ),
    WorkEmailRule(
        name="work_email_from_name_company",
        output_node_type=EMAIL_ADDRESS,
        input_node_types=(FIRST_NAME, LAST_NAME, COMPANY),
    ),
    RoleEmailRule(
        name="role_email_from_company",
        output_node_type=EMAIL_ADDRESS,
        input_node_types=(COMPANY,),
    ),
    UsernameRule(
        name="derive_username_from_name_dob",
        output_node_type=USERNAME,
        input_node_types=(FIRST_NAME, LAST_NAME, DATE_OF_BIRTH),
    ),
    UsernameFromNameRule(
        name="username_from_name",
        output_node_type=USERNAME,
        input_node_types=(FIRST_NAME, LAST_NAME),
    ),
    UsernameFromNameDobRule(
        name="username_from_name_dob",
        output_node_type=USERNAME,
        input_node_types=(FIRST_NAME, LAST_NAME, DATE_OF_BIRTH),
    ),
    UsernameFromNameCompanyRule(
        name="username_from_name_company",
        output_node_type=USERNAME,
        input_node_types=(FIRST_NAME, LAST_NAME, COMPANY),
    ),
    IndependentUsernameRule(
        name="independent_username",
        output_node_type=USERNAME,
        input_node_types=(),
    ),
    AccountBioRule(
        name="compose_account_bio",
        output_node_type=ACCOUNT_BIO,
        input_node_types=(FIRST_NAME, LAST_NAME, COMPANY, LOCATION),
    ),
    BioFromNameCompanyRule(
        name="bio_name_company",
        output_node_type=ACCOUNT_BIO,
        input_node_types=(FIRST_NAME, LAST_NAME, COMPANY),
    ),
    BioFromNameLocationRule(
        name="bio_name_location",
        output_node_type=ACCOUNT_BIO,
        input_node_types=(FIRST_NAME, LAST_NAME, LOCATION),
    ),
    BioFromCompanyLocationRule(
        name="bio_company_location",
        output_node_type=ACCOUNT_BIO,
        input_node_types=(COMPANY, LOCATION),
    ),
    BioFromFullContextRule(
        name="bio_full_context",
        output_node_type=ACCOUNT_BIO,
        input_node_types=(FIRST_NAME, LAST_NAME, COMPANY, LOCATION),
    ),
    SparseCrmBioRule(
        name="bio_sparse_crm_note",
        output_node_type=ACCOUNT_BIO,
        input_node_types=(FIRST_NAME, LAST_NAME),
    ),
    PhoneRule(
        name="derive_phone_from_city_state",
        output_node_type=PHONE,
        input_node_types=(CITY, STATE),
    ),
    PhoneFromStateRule(
        name="phone_from_state",
        output_node_type=PHONE,
        input_node_types=(STATE,),
    ),
    PhoneFromCityStateRule(
        name="phone_from_city_state",
        output_node_type=PHONE,
        input_node_types=(CITY, STATE),
    ),
    IndependentPhoneRule(
        name="independent_phone",
        output_node_type=PHONE,
        input_node_types=(),
    ),
    CustomerIdRule(
        name="customer_id",
        output_node_type=ID,
        input_node_types=(),
    ),
    AccountIdFromCompanyRule(
        name="account_id_from_company",
        output_node_type=ID,
        input_node_types=(COMPANY,),
    ),
    SourceSystemIdRule(
        name="source_system_id",
        output_node_type=ID,
        input_node_types=(),
    ),
    ExternalIdFromNameRule(
        name="external_id_from_name",
        output_node_type=ID,
        input_node_types=(FIRST_NAME, LAST_NAME),
    ),
):
    append_rule_to_rule_set(PERSON_UNIVERSE_RULE_SETS, rule)

for name, task_prompt in (
    (
        "llm_bio_crm_note",
        "Generate a CRM contact-note field. Use the available person, company, and "
        "location facts. You may infer a plausible contact role, team/function, "
        "buying-committee role, or follow-up purpose. Do not copy the facts as a "
        "list or return only a title. Return one complete sentence.",
    ),
    (
        "llm_bio_directory_profile",
        "Generate a directory/profile field. Use the available person, company, "
        "and location facts. You may infer a plausible job title, department, "
        "role, or professional function. Keep it factual-sounding, not "
        "promotional. Return one complete sentence.",
    ),
    (
        "llm_bio_sales_context",
        "Generate a sales or marketing contact-summary field. Use the available "
        "person, company, and location facts. You may infer a plausible stakeholder "
        "function, account responsibility, outreach angle, or prospect role. Avoid "
        "dramatic personal biography. Return one complete sentence.",
    ),
    (
        "llm_bio_support_context",
        "Generate an account-support note. Use the available person, company, and "
        "location facts. You may infer a plausible support-facing role or account "
        "responsibility. The output must be a full account-note sentence, not a "
        "title or extracted field.",
    ),
    (
        "llm_bio_social_profile",
        "Generate a LinkedIn or user-account profile bio. Use the available "
        "person, company, and location facts. You may infer a plausible job title, "
        "professional focus, or role at the company. No hashtags, markdown, "
        "resume bullets, hobbies, education, or life story. Return one complete "
        "sentence.",
    ),
):
    append_rule_to_rule_set(
        PERSON_UNIVERSE_RULE_SETS,
        OllamaRule(
            name=name,
            output_node_type=ACCOUNT_BIO,
            input_node_types=(FIRST_NAME, LAST_NAME, COMPANY, LOCATION),
            system_prompt=ACCOUNT_BIO_SYSTEM_PROMPT,
            model="qwen2.5:7b",
            task_prompt=task_prompt,
            temperature=1.15,
            max_retries=2,
            allow_input_subset=False,
            weight=0.15,
        ),
    )

append_rule_to_rule_set(
    PERSON_UNIVERSE_RULE_SETS,
    LocationRule(
        name="geonames_location",
        output_node_type=LOCATION,
        input_node_types=(),
        locations=load_us_locations(),
    ),
)

for node_type, location_key in (
    (STREET_ADDRESS, "street_address"),
    (CITY, "city"),
    (POSTAL_CODE, "postal_code"),
    (STATE, "state"),
    (COUNTRY, "country"),
):
    append_rule_to_rule_set(
        PERSON_UNIVERSE_RULE_SETS,
        LocationExtractRule(
            name=f"extract_{location_key}",
            output_node_type=node_type,
            input_node_types=(LOCATION,),
            location_key=location_key,
        ),
    )

PERSON_UNIVERSE = UniverseGraph(
    node_types=PERSON_UNIVERSE_NODE_TYPES,
    edges=PERSON_UNIVERSE_EDGES,
    rule_sets=PERSON_UNIVERSE_RULE_SETS,
)

PERSON_PROJECTION_CONFIG = ProjectionSamplerConfig(
    observable_node_types=tuple(
        node_type
        for node_type in PERSON_UNIVERSE_NODE_TYPES
        if node_type not in {ENTITY, LOCATION}
    ),
    bos_scores={
        FIRST_NAME: 1.0,
        MIDDLE_NAME: -0.5,
        LAST_NAME: 1.0,
        FULL_NAME: 0.8,
        DATE_OF_BIRTH: 0.5,
        COMPANY: 0.6,
        EMAIL_ADDRESS: 1.0,
        STREET_ADDRESS: 0.4,
        CITY: 0.0,
        POSTAL_CODE: 0.0,
        STATE: 0.0,
        COUNTRY: -0.5,
        PHONE: 0.8,
        USERNAME: 0.8,
        ACCOUNT_BIO: 0.4,
        ID: 1.0,
    },
    pairwise_affinities={
        (FIRST_NAME, MIDDLE_NAME): 0.4,
        (FIRST_NAME, LAST_NAME): 1.0,
        (FIRST_NAME, FULL_NAME): 0.3,
        (LAST_NAME, FULL_NAME): 0.3,
        (FIRST_NAME, EMAIL_ADDRESS): 0.5,
        (LAST_NAME, EMAIL_ADDRESS): 0.5,
        (COMPANY, EMAIL_ADDRESS): 0.5,
        (FIRST_NAME, USERNAME): 0.4,
        (LAST_NAME, USERNAME): 0.4,
        (EMAIL_ADDRESS, USERNAME): 0.4,
        (STREET_ADDRESS, CITY): 1.0,
        (STREET_ADDRESS, POSTAL_CODE): 1.0,
        (CITY, STATE): 0.8,
        (CITY, POSTAL_CODE): 0.6,
        (STATE, POSTAL_CODE): 0.8,
        (CITY, PHONE): 0.4,
        (STATE, PHONE): 0.4,
        (COMPANY, ACCOUNT_BIO): 0.5,
    },
)


def with_ollama_model(
    universe: UniverseGraph,
    model: OllamaModel | str,
) -> UniverseGraph:
    rule_sets = {}
    for node_type, rule_set in universe.rule_sets.items():
        rule_sets[node_type] = RuleSet(
            output_node_type=rule_set.output_node_type,
            rules=tuple(
                replace(rule, model=model) if isinstance(rule, OllamaRule) else rule
                for rule in rule_set.rules
            ),
        )
    return UniverseGraph(
        node_types=universe.node_types,
        edges=universe.edges,
        rule_sets=rule_sets,
    )


def sample_person_world_graph(
    n_fields: int,
    rng: np.random.Generator,
    projection_temperature: float = 1.0,
    ollama_model: OllamaModel | str = "qwen2.5:7b",
    allow_ollama: bool = True,
) -> WorldGraph:
    return sample_world_graph(
        universe=with_ollama_model(PERSON_UNIVERSE, ollama_model),
        projection_config=PERSON_PROJECTION_CONFIG,
        n_fields=n_fields,
        rng=rng,
        projection_temperature=projection_temperature,
        allow_ollama=allow_ollama,
    )


if __name__ == "__main__":
    print("Person universe")
    print(f"Node types: {len(PERSON_UNIVERSE.node_types)}")
    print(f"Edges: {len(PERSON_UNIVERSE.edges)}")
    print(f"Rule sets: {len(PERSON_UNIVERSE.rule_sets)}")
    print(
        "Rules per node:",
        {
            node_type.name: len(rule_set.rules)
            for node_type, rule_set in PERSON_UNIVERSE.rule_sets.items()
        },
    )

    for node_type in (LOCATION, CITY, EMAIL_ADDRESS):
        print(f"\n{node_type.name}")
        print(
            "Parents:",
            [
                edge.source.name
                for edge in PERSON_UNIVERSE.edges
                if edge.target == node_type
            ],
        )
        print(
            "Rules:",
            [rule.name for rule in PERSON_UNIVERSE.rule_sets[node_type].rules],
        )

    rng = np.random.default_rng(0)
    world_graph = sample_person_world_graph(n_fields=6, rng=rng, allow_ollama=False)
    print("\nSampled world graph")
    print_world_graph_summary(world_graph)

    realized_entity_graph = realize_entity_graph(world_graph, rng, allow_ollama=False)
    print("\nRealized entity graph")
    print(
        "Rules:",
        {
            node_type.name: rule.name
            for node_type, rule in realized_entity_graph.rules.items()
        },
    )
    print(
        "Projected values:",
        {
            node_type.name: realized_entity_graph.nodes[node_type].value
            for node_type in world_graph.projected_node_types
        },
    )

    start = time.perf_counter()
    world = generate_wag_world(
        n_records=8,
        p_match=0.2,
        n_fields=4,
        observation_config=ObservationConfig(
            missing_rate=0.1, nickname_rate=0.0, corruption_rate=0.2
        ),
        allow_ollama=False,
        seed=1,
    )
    elapsed = time.perf_counter() - start
    print("\nGenerated WAG world")
    print(f"Generation time: {elapsed:.2f}s")
    print_world_records(world)

    inspect_graph_world()

    print("\nOllama model comparison")
    comparison_kwargs = dict(
        n_records=5,
        p_match=0.2,
        n_fields=5,
        observation_config=None,
        allow_ollama=True,
        seed=2,
    )
    for model in ("qwen2.5:7b", "llama3.2:1b"):
        print(f"\nModel: {model}")
        start = time.perf_counter()
        try:
            world = generate_wag_world(ollama_model=model, **comparison_kwargs)
            elapsed = time.perf_counter() - start
            print(f"Generation time: {elapsed:.2f}s")
            print_world_records(world)
        except Exception as error:
            elapsed = time.perf_counter() - start
            print(f"Generation failed after {elapsed:.2f}s: {error!r}")
