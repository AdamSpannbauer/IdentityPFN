from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from .field_types import infer_field_types
from .model import NanoERPFNModel
from .tokenizer import Tokenizer
from .utils import get_default_device


class IdentityPFNModel:
    def __init__(
        self,
        model: NanoERPFNModel,
        tokenizer: Tokenizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict_proba(
        self,
        records: pd.DataFrame,
        field_types: Sequence[str] | str = "infer",
        verbose: bool = True,
    ) -> pd.DataFrame:
        if isinstance(field_types, str) and field_types == "infer":
            field_type_details = infer_field_types(records)
            if verbose:
                print(
                    pd.DataFrame(
                        {
                            "column": records.columns,
                            "field_type": [
                                field_type
                                for field_type, _reason in field_type_details
                            ],
                            "reason": [
                                reason for _field_type, reason in field_type_details
                            ],
                        }
                    ).to_string(index=False)
                )
            field_types = [
                field_type for field_type, _reason in field_type_details
            ]
        elif isinstance(field_types, str):
            raise ValueError("field_types must be a sequence of field types or 'infer'")

        if len(field_types) != records.shape[1]:
            raise ValueError("field_types must have one entry per records column")

        tokenized = self.tokenizer([records], [list(field_types)])
        tokenized = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in tokenized.items()
        }

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tokenized).squeeze(0)
            scores = torch.sigmoid(logits).cpu().numpy()

        return pd.DataFrame(scores, index=records.index, columns=records.index)


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> IdentityPFNModel:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    run_config = checkpoint["run_config"]
    tokenizer = Tokenizer(
        text_backend=run_config["text_backend"],
        identifier_backend=run_config.get("identifier_backend"),
        normalize_identifiers=run_config.get("normalize_identifiers", False),
        default_phone_region=run_config.get("default_phone_region", "US"),
    )
    model = NanoERPFNModel(
        embedding_size=run_config["embedding_size"],
        text_embedding_size=tokenizer.text_embedding_dim,
        identifier_embedding_size=tokenizer.identifier_embedding_dim,
        num_attention_heads=run_config["num_attention_heads"],
        mlp_hidden_size=run_config["mlp_hidden_size"],
        num_layers=run_config["num_layers"],
        max_categories=run_config["max_categories"],
        record_representation=run_config.get("record_representation", "mean_pool"),
        adjacency_decoder=run_config.get("adjacency_decoder", "pair_mlp"),
        entity_slot_count=run_config.get("entity_slot_count", 300),
        num_slot_attention_layers=run_config.get("num_slot_attention_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return IdentityPFNModel(model, tokenizer, _resolve_device(device))


def top_pairs(scores: pd.DataFrame | np.ndarray, k: int = 10) -> pd.DataFrame:
    score_frame = _as_score_frame(scores)
    rows = []
    labels = list(score_frame.index)
    values = score_frame.to_numpy()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append(
                {
                    "left_index": labels[i],
                    "right_index": labels[j],
                    "score": float(values[i, j]),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )


def nearest_neighbors(
    scores: pd.DataFrame | np.ndarray,
    k: int = 3,
) -> pd.DataFrame:
    score_frame = _as_score_frame(scores)
    rows = []
    for query in score_frame.index:
        neighbors = score_frame.loc[query].drop(index=query)
        for rank, (neighbor, score) in enumerate(
            neighbors.sort_values(ascending=False).head(k).items(),
            start=1,
        ):
            rows.append(
                {
                    "query_index": query,
                    "neighbor_index": neighbor,
                    "rank": rank,
                    "score": float(score),
                }
            )

    return pd.DataFrame(rows)


def pair_review_table(
    scores: pd.DataFrame | np.ndarray,
    records: pd.DataFrame,
    k: int = 10,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns = list(records.columns if columns is None else columns)
    rows = []
    for pair in top_pairs(scores, k=k).to_dict("records"):
        left_index = pair["left_index"]
        right_index = pair["right_index"]
        row = {
            "left_index": left_index,
            "right_index": right_index,
            "score": pair["score"],
        }
        for column in columns:
            row[f"left_{column}"] = records.loc[left_index, column]
            row[f"right_{column}"] = records.loc[right_index, column]
        rows.append(row)

    return pd.DataFrame(rows)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return get_default_device()
    return torch.device(device)


def _as_score_frame(scores: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(scores, pd.DataFrame):
        return scores
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("scores must be a square matrix")
    return pd.DataFrame(values)
