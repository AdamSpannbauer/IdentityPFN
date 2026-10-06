from .dgp.people.simple import generate_worlds as generate_person_worlds
from .dgp.people.wag import generate_worlds as generate_wag_worlds
from .dgp.generate_id_prior_task import generate_id_prior_task
from .dgp.people.sample_revelio import REVELIO_USER_FIELD_TYPES, RevelioSampler

import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
import torch

from .utils import get_default_device


class SyntheticWorldDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data. Currently generated on the fly during prototype

    Args:
        num_steps (int): Number of generated batches (no real "epoch" with generation on the fly).
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(
        self,
        num_steps,
        batch_size,
        seed=None,
        device=None,
        verbose=False,
        generator="person",
        allow_ollama=None,
        ollama_model="qwen2.5:7b",
        missing_rate=None,
        nickname_rate=None,
        corruption_rate=None,
        prop_corruption_rate=None,
        hard_negative_rate=0.0,
        hard_negative_contract_weights=None,
        revelio_root=None,
    ):
        if generator not in {"person", "wag", "id_prior"}:
            raise ValueError("generator must be 'person', 'wag', or 'id_prior'")
        self.verbose = verbose
        self.generator = generator
        self.allow_ollama = allow_ollama
        self.ollama_model = ollama_model
        self.missing_rate = missing_rate
        self.nickname_rate = nickname_rate
        self.corruption_rate = corruption_rate
        self.prop_corruption_rate = prop_corruption_rate
        self.hard_negative_rate = hard_negative_rate
        self.hard_negative_contract_weights = hard_negative_contract_weights
        self.revelio_sampler = None
        if generator == "id_prior":
            if revelio_root is None:
                raise ValueError("revelio_root is required for id_prior")
            root = Path(revelio_root)
            self.revelio_sampler = RevelioSampler(
                individual_user_dir=root / "individual_user",
                individual_position_dir=root / "individual_position",
                individual_user_education_dir=root / "individual_user_education",
                individual_user_skill_dir=root / "individual_user_skill",
                company_ref_dir=root / "company_ref",
            )

        self.num_steps = num_steps
        self.batch_size = batch_size

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.device = device if device is not None else get_default_device()

    def __iter__(self):
        for _ in range(self.num_steps):
            world_seed = self.rng.integers(0, 2**32)

            if self.generator in {"wag", "id_prior"}:
                n_records = self.rng.integers(300, 2501)
            else:
                n_records = self.rng.integers(50, 301)
            n_fields = self.rng.integers(3, 11)
            if self.generator == "id_prior":
                world_rng = np.random.default_rng(world_seed)
                worlds = [
                    generate_id_prior_task(
                        n_records=int(n_records),
                        p_match=float(10 ** world_rng.uniform(-4, -2)),
                        n_fields=int(n_fields),
                        sample_entities=self.revelio_sampler.sample_entities,
                        field_types=REVELIO_USER_FIELD_TYPES,
                        rng=np.random.default_rng(world_rng.integers(0, 2**32)),
                        allow_ollama=bool(self.allow_ollama),
                        ollama_model=self.ollama_model,
                        country_field="user_country",
                    )
                    for _ in range(self.batch_size)
                ]
            else:
                generate_worlds = (
                    generate_wag_worlds
                    if self.generator == "wag"
                    else generate_person_worlds
                )
                generator_kwargs = {}
                if self.generator == "wag":
                    generator_kwargs["ollama_model"] = self.ollama_model
                    generator_kwargs["hard_negative_rate"] = self.hard_negative_rate
                    generator_kwargs["hard_negative_contract_weights"] = (
                        self.hard_negative_contract_weights
                    )
                    if self.allow_ollama is not None:
                        generator_kwargs["allow_ollama"] = self.allow_ollama
                    if self.missing_rate is not None:
                        generator_kwargs["missing_rate"] = self.missing_rate
                    if self.nickname_rate is not None:
                        generator_kwargs["nickname_rate"] = self.nickname_rate
                    if self.corruption_rate is not None:
                        generator_kwargs["corruption_rate"] = self.corruption_rate
                    if self.prop_corruption_rate is not None:
                        generator_kwargs["prop_corruption_rate"] = (
                            self.prop_corruption_rate
                        )
                worlds = generate_worlds(
                    n_worlds=self.batch_size,
                    n_records=[n_records],
                    n_fields=[n_fields],
                    seed=world_seed,
                    verbose=self.verbose,
                    **generator_kwargs,
                )

            batch = {
                "records": [world.records for world in worlds],
                "entity_ids": torch.stack(
                    [
                        torch.as_tensor(world.entity_ids, dtype=torch.long)
                        for world in worlds
                    ]
                ),
                "field_types": [world.field_types for world in worlds],
                "adjacency": torch.stack(
                    [
                        torch.as_tensor(world.adjacency, dtype=torch.bool)
                        for world in worlds
                    ]
                ),
            }

            yield batch

    def __len__(self):
        return self.num_steps


if __name__ == "__main__":
    from collections import Counter

    priors = SyntheticWorldDataLoader(num_steps=3, batch_size=3, verbose=True)

    for i, batch in enumerate(priors):
        print(f"\nBatch {i}:")
        print("Record shapes:", [record.shape for record in batch["records"]], "\n")

        print(
            "Entity ID lengths:",
            [len(entity_ids) for entity_ids in batch["entity_ids"]],
        )
        print(
            "Unique entity ID counts:",
            [len(set(entity_ids)) for entity_ids in batch["entity_ids"]],
            "\n",
        )

        print(
            "Field Types:",
            [Counter(field_types) for field_types in batch["field_types"]],
            "\n",
        )

        print(
            "Adjacency shapes:", [adjacency.shape for adjacency in batch["adjacency"]]
        )
        print(
            "Adjacency triu rates:",
            [adjacency.triu().float().mean() for adjacency in batch["adjacency"]],
        )
