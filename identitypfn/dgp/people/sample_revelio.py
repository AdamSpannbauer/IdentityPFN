"""Sample intact people from Revelio's sharded Parquet tables."""

from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REVELIO_USER_FIELD_TYPES = {
    "user_id": "identifier",
    "firstname": "text",
    "lastname": "text",
    "fullname": "text",
    "highest_degree": "categorical",
    "sex_predicted": "categorical",
    "ethnicity_predicted": "categorical",
    "profile_linkedin_url": "identifier",
    "user_location": "text",
    "user_country": "text",
    "profile_title": "text",
    "updated_dt": "date",
    "numconnections": "numeric",
    "profile_summary": "text",
}


class RevelioSampler:
    """Sample users now; retain paths for future role, education, skill, and company joins."""

    def __init__(
        self,
        individual_user_dir: str | Path,
        individual_position_dir: str | Path,
        individual_user_education_dir: str | Path,
        individual_user_skill_dir: str | Path,
        company_ref_dir: str | Path,
    ) -> None:
        self.individual_user_dir = Path(individual_user_dir)
        self.user_files = sorted(self.individual_user_dir.glob("*.parquet"))
        self.individual_position_dir = Path(individual_position_dir)
        self.individual_user_education_dir = Path(individual_user_education_dir)
        self.individual_user_skill_dir = Path(individual_user_skill_dir)
        self.company_ref_dir = Path(company_ref_dir)

    def sample_entities(
        self, n_entities: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Sample distinct users, excluding probability and prestige columns."""
        user_file = str(self.user_files[int(rng.integers(len(self.user_files)))])
        connection = duckdb.connect()
        seed = int(rng.integers(0, 2**31))
        query = (
            "SELECT * EXCLUDE (f_prob, m_prob, white_prob, black_prob, "
            "api_prob, hispanic_prob, native_prob, multiple_prob, prestige) "
            "FROM read_parquet(?) "
            f"USING SAMPLE reservoir({n_entities} ROWS) REPEATABLE ({seed})"
        )
        people = connection.execute(query, [user_file]).df()
        connection.close()
        if not people["user_id"].is_unique:
            raise ValueError("Sampled Revelio users contain duplicate user_id values")
        return people

    def sample_entities_for_schema(
        self,
        n_entities: int,
        selected_fields: list[str],
        hard_negative_rate: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Prefer distinct users agreeing on rare values in active fields."""
        if not 0 <= hard_negative_rate <= 1:
            raise ValueError("hard_negative_rate must be between 0 and 1")
        target_pairs = int(hard_negative_rate * n_entities / 2)
        if target_pairs == 0:
            return self.sample_entities(n_entities, rng)

        candidates = self.sample_entities(3 * n_entities, rng)
        eligible_fields = [
            field
            for field in selected_fields
            if REVELIO_USER_FIELD_TYPES.get(field) in {"text", "categorical"}
            and field not in {"fullname", "profile_summary"}
        ]
        buckets_by_field = {}
        weights = {}
        anchor_weights = np.zeros(len(candidates))
        for field in eligible_fields:
            buckets = defaultdict(list)
            for row, value in enumerate(candidates[field]):
                if pd.isna(value):
                    continue
                normalized = str(value).strip().casefold()
                if normalized:
                    buckets[normalized].append(row)
            n_present = sum(len(rows) for rows in buckets.values())
            if n_present < 2:
                continue
            agreement = sum(len(rows) * (len(rows) - 1) for rows in buckets.values())
            p_u = agreement / (n_present * (n_present - 1))
            if p_u <= 0 or p_u >= 1:
                continue
            weight = -np.log(p_u)
            retained = {
                value: rows
                for value, rows in buckets.items()
                if 2 <= len(rows) <= 64
            }
            if not retained:
                continue
            buckets_by_field[field] = retained
            weights[field] = weight
            for rows in retained.values():
                anchor_weights[rows] = np.maximum(anchor_weights[rows], weight)

        available = np.ones(len(candidates), dtype=bool)
        pairs = []
        while len(pairs) < 2 * target_pairs:
            anchors = np.flatnonzero(available & (anchor_weights > 0))
            if len(anchors) == 0:
                break
            anchor = int(
                rng.choice(anchors, p=anchor_weights[anchors] / anchor_weights[anchors].sum())
            )
            scores = defaultdict(float)
            for field, buckets in buckets_by_field.items():
                value = candidates.iloc[anchor][field]
                if pd.isna(value):
                    continue
                for candidate in buckets.get(str(value).strip().casefold(), ()):
                    if candidate != anchor and available[candidate]:
                        scores[candidate] += weights[field]
            if not scores:
                anchor_weights[anchor] = 0
                continue
            matches = np.fromiter(scores, dtype=int)
            match_weights = np.fromiter(scores.values(), dtype=float)
            match = int(rng.choice(matches, p=match_weights / match_weights.sum()))
            pairs.extend((anchor, match))
            available[[anchor, match]] = False

        remaining = rng.choice(
            np.flatnonzero(available), size=n_entities - len(pairs), replace=False
        )
        selected = np.concatenate((np.asarray(pairs, dtype=int), remaining))
        people = candidates.iloc[selected].reset_index(drop=True)
        people.attrs["hard_negative_pairs"] = len(pairs) // 2
        people.attrs["hard_negative_target_pairs"] = target_pairs
        return people
