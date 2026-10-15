"""Sample clean WAG person entities for the shared identity-prior task."""

import numpy as np
import pandas as pd

from .wag import (
    DEFAULT_HARD_NEGATIVE_CONTRACT_WEIGHTS,
    generate_clean_projected_records,
    make_hard_negative_sibling_graph,
    realize_entity_graph,
    sample_hard_negative_contract,
    sample_person_world_graph,
)


class WAGEntitySampler:
    def __init__(
        self,
        n_fields: int,
        rng: np.random.Generator,
        allow_ollama: bool = False,
        ollama_model: str = "qwen2.5:7b",
    ):
        self.world_graph = sample_person_world_graph(
            n_fields=n_fields,
            rng=rng,
            projection_temperature=float(rng.choice((0.7, 1.0, 1.5))),
            allow_ollama=allow_ollama,
            ollama_model=ollama_model,
        )
        self.field_types = {
            node.name: node.primitive_type
            for node in self.world_graph.projected_node_types
        }

    def sample_entities(
        self, n_entities: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        graphs = [realize_entity_graph(self.world_graph, rng) for _ in range(n_entities)]
        return generate_clean_projected_records(
            graphs, np.arange(n_entities), self.world_graph
        )

    def sample_entities_for_schema(
        self,
        n_entities: int,
        selected_fields: list[str],
        hard_negative_rate: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        if not 0 <= hard_negative_rate <= 1:
            raise ValueError("hard_negative_rate must be between 0 and 1")
        target_pairs = int(hard_negative_rate * n_entities / 2)
        graphs = [realize_entity_graph(self.world_graph, rng) for _ in range(n_entities)]
        visible_nodes = {
            node
            for node in self.world_graph.projected_node_types
            if node.name in selected_fields
        }
        pairs = rng.choice(n_entities, size=2 * target_pairs, replace=False).reshape(-1, 2)
        realized_pairs = 0
        for base, target in pairs:
            contract = sample_hard_negative_contract(
                visible_nodes, DEFAULT_HARD_NEGATIVE_CONTRACT_WEIGHTS, rng
            )
            if contract is None:
                break
            graphs[target], _, _ = make_hard_negative_sibling_graph(
                graphs[base], contract, rng
            )
            realized_pairs += 1
        entities = generate_clean_projected_records(
            graphs, np.arange(n_entities), self.world_graph
        )
        entities.attrs["hard_negative_pairs"] = realized_pairs
        entities.attrs["hard_negative_target_pairs"] = target_pairs
        return entities
