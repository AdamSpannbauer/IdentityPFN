"""Select training sources and sample intact rows for the identity prior."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import repository_path
from .people.sample_revelio import RevelioSampler


class SourceBank:
    def __init__(self, manifest_path: str | Path, revelio_root: str | Path | None):
        with open(manifest_path) as file:
            manifest = json.load(file)
        self.sources = [
            source for source in manifest["contributors"] if source["role"] == "train"
        ]
        self.frames: dict[str, pd.DataFrame] = {}
        self.paths: dict[str, Path] = {}
        self.revelio_sampler = None

        roots = {
            "external": repository_path("training_data", "raw"),
            "magellan": repository_path("benchmark_data", "magellan"),
        }
        if revelio_root is not None:
            roots["revelio"] = Path(revelio_root)

        for source in self.sources:
            if source["kind"] == "generator":
                if source["generator"] != "wag":
                    raise ValueError(f"Unsupported generator: {source['generator']}")
                continue
            location = source["location"]
            if location["root"] not in roots:
                raise ValueError(
                    f"Missing root for {source['name']}: {location['root']}"
                )
            path = roots[location["root"]] / location["path"]
            if source["format"] == "parquet_shards":
                if not list(path.parent.glob(path.name)):
                    raise FileNotFoundError(path)
                root = roots[location["root"]]
                self.revelio_sampler = RevelioSampler(
                    individual_user_dir=root / "individual_user",
                    individual_position_dir=root / "individual_position",
                    individual_user_education_dir=root / "individual_user_education",
                    individual_user_skill_dir=root / "individual_user_skill",
                    company_ref_dir=root / "company_ref",
                )
            elif source["format"] == "csv":
                if not path.is_file():
                    raise FileNotFoundError(path)
                self.paths[source["name"]] = path
            else:
                raise ValueError(f"Unsupported format: {source['format']}")

    def sample_source(self, rng: np.random.Generator) -> dict:
        return self.sources[int(rng.integers(len(self.sources)))]

    def sample_entities(
        self, source: dict, n_entities: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        if source["format"] == "parquet_shards":
            return self.revelio_sampler.sample_entities(n_entities, rng)
        name = source["name"]
        if name not in self.frames:
            self.frames[name] = pd.read_csv(
                self.paths[name], usecols=source["field_types"], dtype=str
            )
        frame = self.frames[name]
        if n_entities > len(frame):
            raise ValueError(f"{name} has {len(frame)} rows, needs {n_entities}")
        indices = rng.choice(len(frame), size=n_entities, replace=False)
        return frame.iloc[indices].reset_index(drop=True)
