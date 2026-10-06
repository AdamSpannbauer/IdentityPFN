"""Sample intact people from Revelio's sharded Parquet tables."""

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
    "user_country": "categorical",
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
        self.individual_position_dir = Path(individual_position_dir)
        self.individual_user_education_dir = Path(individual_user_education_dir)
        self.individual_user_skill_dir = Path(individual_user_skill_dir)
        self.company_ref_dir = Path(company_ref_dir)

    def sample_entities(
        self, n_entities: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Sample distinct users, excluding probability and prestige columns."""
        user_files = str(self.individual_user_dir / "*.parquet")
        connection = duckdb.connect()
        seed = int(rng.integers(0, 2**31))
        query = (
            "SELECT * EXCLUDE (f_prob, m_prob, white_prob, black_prob, "
            "api_prob, hispanic_prob, native_prob, multiple_prob, prestige) "
            "FROM read_parquet(?) "
            f"USING SAMPLE reservoir({n_entities} ROWS) REPEATABLE ({seed})"
        )
        people = connection.execute(query, [user_files]).df()
        connection.close()
        if not people["user_id"].is_unique:
            raise ValueError("Sampled Revelio users contain duplicate user_id values")
        return people
