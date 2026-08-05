from pathlib import Path
import re
from typing import Literal

import numpy as np
import pandas as pd
import torch

from .paths import repository_path

TextBackend = Literal["fasttext", "sentence_transformer"]


class Tokenizer:
    def __init__(
        self,
        text_backend: TextBackend,
        fasttext_path: str | Path = repository_path("cc.en.300.bin"),
        sentence_transformer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        identifier_backend: TextBackend | None = None,
        normalize_identifiers: bool = False,
        default_phone_region: str = "US",
    ):
        self.text_backend = text_backend
        self.identifier_backend = identifier_backend
        self.normalize_identifiers = normalize_identifiers
        self.default_phone_region = default_phone_region
        self.fasttext_path = str(fasttext_path)
        self.sentence_transformer_name = sentence_transformer_name
        self.text_model = None
        self.identifier_model = None

    def _load_text_model(self):
        if self.text_model is not None:
            return
        self.text_model = self._load_backend_model(self.text_backend)

    def _load_identifier_model(self):
        if self.identifier_backend is None or self.identifier_model is not None:
            return
        self.identifier_model = self._load_backend_model(self.identifier_backend)

    def _load_backend_model(self, backend: TextBackend):
        if backend == "fasttext":
            import fasttext

            return fasttext.load_model(self.fasttext_path)
        elif backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self.sentence_transformer_name)
        else:
            raise ValueError(f"Unknown text backend: {backend}")

    def _embed_values(self, texts: list[str], backend: TextBackend, model) -> np.ndarray:
        if backend == "fasttext":
            return np.stack([model.get_sentence_vector(text) for text in texts])
        elif backend == "sentence_transformer":
            return model.encode(texts, convert_to_numpy=True)
        raise ValueError(f"Unknown text backend: {backend}")

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        self._load_text_model()
        return self._embed_values(texts, self.text_backend, self.text_model)

    def _embed_identifiers(self, texts: list[str]) -> np.ndarray:
        if self.identifier_backend is None:
            return self._embed_texts(texts)
        self._load_identifier_model()
        return self._embed_values(
            texts, self.identifier_backend, self.identifier_model
        )

    def _embedding_dim(self, backend: TextBackend, model) -> int:
        if backend == "fasttext":
            return model.get_dimension()
        return model.get_embedding_dimension()

    @property
    def text_embedding_dim(self) -> int:
        self._load_text_model()
        return self._embedding_dim(self.text_backend, self.text_model)

    @property
    def identifier_embedding_dim(self) -> int | None:
        if self.identifier_backend is None:
            return None
        self._load_identifier_model()
        return self._embedding_dim(self.identifier_backend, self.identifier_model)

    def _normalize_identifier(self, value) -> str:
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def _normalize_email(self, value) -> str:
        return str(value).strip().lower()

    def _normalize_phone(self, value) -> str:
        text = str(value).strip()
        if not text:
            return text

        import phonenumbers

        try:
            parsed = phonenumbers.parse(text, self.default_phone_region)
            if phonenumbers.is_valid_number(parsed) or phonenumbers.is_possible_number(
                parsed
            ):
                return phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.E164
                )
        except phonenumbers.NumberParseException:
            pass

        has_leading_plus = text.startswith("+")
        digits = re.sub(r"\D", "", text)
        if digits:
            return f"+{digits}" if has_leading_plus else digits
        return text.lower()

    def _identifier_text(self, field_type: str, value) -> str:
        if not self.normalize_identifiers:
            return str(value)
        if field_type == "email":
            return self._normalize_email(value)
        if field_type == "phone":
            return self._normalize_phone(value)
        return self._normalize_identifier(value)

    def __call__(
        self, records: list[pd.DataFrame], field_types: list[list[str]]
    ) -> dict:
        batch_size = len(records)
        n_records, n_fields = records[0].shape

        # Store raw values for numeric, store cat ids for categorical, and store text embeddings for text
        numeric_values = []  # (N non-missing numeric cells in batch, )
        categorical_ids = []  # (N non-missing categorical cells in batch, )
        texts = []  # (N non-missing text cells in batch, text embedding dim)
        identifiers = []  # (N non-missing identifier-like cells in batch, identifier embedding dim)
        date_year_values = []  # (N parseable date cells in batch, )
        date_components = []  # (N parseable date cells in batch, 3)

        # Locations store [batch index, record index, field index]
        numeric_locations = []  # (N non-missing numeric cells in batch, 3)
        categorical_locations = []  # (N non-missing categorical cells in batch, 3)
        text_locations = []  # (N non-missing text cells in batch, 3)
        identifier_locations = []  # (N non-missing identifier-like cells in batch, 3)
        date_locations = []  # (N parseable date cells in batch, 3)
        missing_locations = []  # (N missing cells in batch, 3)

        for batch_index, (frame, types) in enumerate(zip(records, field_types)):
            for field_index, field_type in enumerate(types):
                values = frame.iloc[:, field_index]
                category_ids = None
                if field_type == "categorical":
                    category_ids, _ = pd.factorize(values, sort=False)

                for record_index, value in enumerate(values):
                    location = (batch_index, record_index, field_index)
                    if pd.isna(value):
                        missing_locations.append(location)
                    elif field_type == "numeric":
                        numeric_values.append(float(value))
                        numeric_locations.append(location)
                    elif field_type == "categorical":
                        categorical_ids.append(category_ids[record_index] + 1)
                        categorical_locations.append(location)
                    elif field_type == "date":
                        timestamp = pd.to_datetime(
                            value, errors="coerce", format="mixed"
                        )
                        if pd.isna(timestamp):
                            texts.append(str(value))
                            text_locations.append(location)
                        else:
                            date_year_values.append((timestamp.year - 2000) / 100)
                            date_components.append(
                                [
                                    timestamp.month,
                                    timestamp.day,
                                    timestamp.weekday() + 1,
                                ]
                            )
                            date_locations.append(location)
                    elif field_type in {"identifier", "email", "phone"}:
                        identifier_text = self._identifier_text(field_type, value)
                        if self.identifier_backend is None:
                            texts.append(identifier_text)
                            text_locations.append(location)
                        else:
                            identifiers.append(identifier_text)
                            identifier_locations.append(location)
                    else:
                        texts.append(str(value))
                        text_locations.append(location)

        if texts:
            text_embeddings = torch.from_numpy(
                self._embed_texts(texts).astype(np.float32)
            )
        else:
            text_embeddings = torch.empty(
                (0, self.text_embedding_dim), dtype=torch.float32
            )

        if identifiers:
            identifier_embeddings = torch.from_numpy(
                self._embed_identifiers(identifiers).astype(np.float32)
            )
        else:
            identifier_dim = self.identifier_embedding_dim
            identifier_embeddings = torch.empty(
                (0, 0 if identifier_dim is None else identifier_dim), dtype=torch.float32
            )

        return {
            "shape": (batch_size, n_records, n_fields),
            "numeric_values": torch.tensor(numeric_values, dtype=torch.float32),
            "numeric_locations": torch.tensor(
                numeric_locations, dtype=torch.long
            ).reshape(-1, 3),
            "categorical_ids": torch.tensor(categorical_ids, dtype=torch.long),
            "categorical_locations": torch.tensor(
                categorical_locations, dtype=torch.long
            ).reshape(-1, 3),
            "text_embeddings": text_embeddings,
            "text_locations": torch.tensor(text_locations, dtype=torch.long).reshape(
                -1, 3
            ),
            "identifier_embeddings": identifier_embeddings,
            "identifier_locations": torch.tensor(
                identifier_locations, dtype=torch.long
            ).reshape(-1, 3),
            "date_components": torch.tensor(
                date_components, dtype=torch.long
            ).reshape(-1, 3),
            "date_year_values": torch.tensor(date_year_values, dtype=torch.float32),
            "date_locations": torch.tensor(date_locations, dtype=torch.long).reshape(
                -1, 3
            ),
            "missing_locations": torch.tensor(
                missing_locations, dtype=torch.long
            ).reshape(-1, 3),
        }


if __name__ == "__main__":
    from .data_loader import SyntheticWorldDataLoader

    eg_prior = SyntheticWorldDataLoader(num_steps=1, batch_size=1)
    tokenizer = Tokenizer(text_backend="fasttext")

    for batch in eg_prior:
        tokenized = tokenizer(batch["records"], batch["field_types"])
        for key, value in tokenized.items():
            if key == "shape":
                print(f"full shape: {value}\n")
            else:
                print(
                    f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}"
                )
