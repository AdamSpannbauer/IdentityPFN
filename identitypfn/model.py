from typing import Literal
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm


class NanoERPFNModel(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        text_embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        max_categories: int = 20,
        record_representation: Literal[
            "mean_pool",
            "entity_target_column",
        ] = "mean_pool",
        adjacency_decoder: Literal[
            "pair_mlp",
            "slot_coassignment",
            "slot_pair_mlp",
        ] = "pair_mlp",
        entity_slot_count: int = 300,
        num_slot_attention_layers: int = 1,
        identifier_embedding_size: int | None = None,
    ):
        """Initializes the feature encoder, transformer stack and adjacency decoder."""
        super().__init__()
        self.feature_encoder = FeatureEncoder(
            embedding_size,
            text_embedding_size,
            max_categories,
            identifier_embedding_size=identifier_embedding_size,
        )

        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(
                    embedding_size, num_attention_heads, mlp_hidden_size
                )
            )

        self.record_representation = record_representation
        if record_representation == "mean_pool":
            self.record_representer = MeanPoolRecordRepresenter()
        elif record_representation == "entity_target_column":
            self.entity_target_column = nn.Parameter(torch.zeros(embedding_size))
            self.record_representer = EntityTargetColumnRepresenter()
        else:
            raise ValueError(f"Unknown record representation: {record_representation}")

        self.adjacency_decoder_name = adjacency_decoder
        if adjacency_decoder == "pair_mlp":
            self.adjacency_decoder = PairMLPAdjacencyDecoder(
                embedding_size, mlp_hidden_size
            )
        elif adjacency_decoder == "slot_coassignment":
            self.adjacency_decoder = SlotCoassignmentAdjacencyDecoder(
                embedding_size,
                num_attention_heads,
                entity_slot_count,
                num_slot_attention_layers,
            )
        elif adjacency_decoder == "slot_pair_mlp":
            self.adjacency_decoder = SlotPairMLPAdjacencyDecoder(
                embedding_size,
                mlp_hidden_size,
                num_attention_heads,
                entity_slot_count,
                num_slot_attention_layers,
            )
        else:
            raise ValueError(f"Unknown adjacency decoder: {adjacency_decoder}")

    def forward(self, tokenized_cells: dict) -> torch.Tensor:
        # from here on T=Tables, R=Records, C=Columns, E=embedding size
        # converts cell values to embeddings, so (T,R,C) -> (T,R,C,E)

        cell_embeddings = self.feature_encoder(tokenized_cells)

        if self.record_representation == "entity_target_column":
            num_tables, num_records, _, embedding_size = cell_embeddings.shape
            entity_column = self.entity_target_column.view(1, 1, 1, embedding_size)
            entity_column = entity_column.expand(num_tables, num_records, 1, -1)
            cell_embeddings = torch.cat([cell_embeddings, entity_column], dim=2)

        # repeatedly applies the transformer block on (T,R,C,E)
        for block in self.transformer_blocks:
            cell_embeddings = block(cell_embeddings)

        record_embeddings = self.record_representer(cell_embeddings)
        return self.adjacency_decoder(record_embeddings)


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        text_embedding_size: int,
        max_categories: int,
        identifier_embedding_size: int | None = None,
    ):
        """Creates modality-specific layers that embed cells."""
        super().__init__()
        self.embedding_size = embedding_size
        self.has_identifier_projection = identifier_embedding_size is not None
        self.numeric_projection = nn.Linear(1, embedding_size)
        self.categorical_embedding = nn.Embedding(max_categories + 1, embedding_size)
        self.text_projection = nn.Linear(text_embedding_size, embedding_size)
        if identifier_embedding_size is not None:
            self.identifier_projection = nn.Linear(
                identifier_embedding_size, embedding_size
            )
        self.date_encoder = DateEncoder(embedding_size)
        self.missing_embedding = nn.Parameter(torch.zeros(embedding_size))

    def forward(self, src: dict) -> torch.Tensor:
        """Projects compact modality values into a dense cell embedding tensor."""
        num_tables, num_records, num_columns = src["shape"]
        cell_embeddings = self.missing_embedding.new_zeros(
            num_tables, num_records, num_columns, self.embedding_size
        )

        num_table_columns = num_tables * num_columns

        # Assign each numeric cell to its unique (table, column) normalization group
        numeric_cell_locations = src["numeric_locations"]
        numeric_cell_values = src["numeric_values"]
        if numeric_cell_values.numel():
            # table_column_id = table_index * num_columns + column_index
            table_column_ids = (
                numeric_cell_locations[:, 0] * num_columns
                + numeric_cell_locations[:, 2]
            )

            table_column_counts = numeric_cell_values.new_zeros(num_table_columns)
            table_column_sums = numeric_cell_values.new_zeros(num_table_columns)
            table_column_counts.scatter_add_(
                0, table_column_ids, torch.ones_like(numeric_cell_values)
            )
            table_column_sums.scatter_add_(0, table_column_ids, numeric_cell_values)
            table_column_means = table_column_sums / table_column_counts.clamp_min(1)

            centered_numeric_values = (
                numeric_cell_values - table_column_means[table_column_ids]
            )
            table_column_squared_sums = numeric_cell_values.new_zeros(num_table_columns)
            table_column_squared_sums.scatter_add_(
                0, table_column_ids, centered_numeric_values.square()
            )
            table_column_stds = (
                torch.sqrt(table_column_squared_sums / table_column_counts.clamp_min(1))
                + 1e-20
            )
            numeric_cell_values = torch.clip(
                centered_numeric_values / table_column_stds[table_column_ids],
                -100,
                100,
            )

            cell_embeddings[
                numeric_cell_locations[:, 0],
                numeric_cell_locations[:, 1],
                numeric_cell_locations[:, 2],
            ] = self.numeric_projection(numeric_cell_values.unsqueeze(-1))

        categorical_cell_locations = src["categorical_locations"]
        if src["categorical_ids"].numel():
            cell_embeddings[
                categorical_cell_locations[:, 0],
                categorical_cell_locations[:, 1],
                categorical_cell_locations[:, 2],
            ] = self.categorical_embedding(src["categorical_ids"])

        text_cell_locations = src["text_locations"]
        if src["text_embeddings"].numel():
            cell_embeddings[
                text_cell_locations[:, 0],
                text_cell_locations[:, 1],
                text_cell_locations[:, 2],
            ] = self.text_projection(src["text_embeddings"])

        identifier_cell_locations = src.get("identifier_locations")
        identifier_embeddings = src.get("identifier_embeddings")
        if (
            not self.has_identifier_projection
            and identifier_embeddings is not None
            and identifier_embeddings.numel()
        ):
            raise ValueError(
                "Received identifier embeddings but model has no identifier projection"
            )
        if (
            self.has_identifier_projection
            and identifier_embeddings is not None
            and identifier_embeddings.numel()
        ):
            cell_embeddings[
                identifier_cell_locations[:, 0],
                identifier_cell_locations[:, 1],
                identifier_cell_locations[:, 2],
            ] = self.identifier_projection(identifier_embeddings)

        date_cell_locations = src["date_locations"]
        if src["date_components"].numel():
            cell_embeddings[
                date_cell_locations[:, 0],
                date_cell_locations[:, 1],
                date_cell_locations[:, 2],
            ] = self.date_encoder(src["date_year_values"], src["date_components"])

        missing_cell_locations = src["missing_locations"]
        cell_embeddings[
            missing_cell_locations[:, 0],
            missing_cell_locations[:, 1],
            missing_cell_locations[:, 2],
        ] = self.missing_embedding

        return cell_embeddings


class DateEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """Creates numeric year and categorical calendar component embeddings."""
        super().__init__()
        self.year_projection = nn.Linear(1, embedding_size)
        self.month_embedding = nn.Embedding(13, embedding_size)
        self.day_embedding = nn.Embedding(32, embedding_size)
        self.weekday_embedding = nn.Embedding(8, embedding_size)

    def forward(
        self, year_values: torch.Tensor, components: torch.Tensor
    ) -> torch.Tensor:
        """Project year and embed month, day, and weekday into one date vector."""
        return (
            self.year_projection(year_values.unsqueeze(-1))
            + self.month_embedding(components[:, 0])
            + self.day_embedding(components[:, 1])
            + self.weekday_embedding(components[:, 2])
        )


class TransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    """

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.self_attention_between_records = MultiheadAttention(
            embedding_size,
            num_attention_heads,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.self_attention_between_columns = MultiheadAttention(
            embedding_size,
            num_attention_heads,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        self.mlp_in = Linear(
            embedding_size, mlp_hidden_size, device=device, dtype=dtype
        )
        self.mlp_out = Linear(
            mlp_hidden_size, embedding_size, device=device, dtype=dtype
        )

        self.column_attention_norm = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.record_attention_norm = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.mlp_norm = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )

    def forward(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Takes the embeddings of the tables as input and applies self-attention between columns and self-attention between records
        followed by a simple 2 layer MLP.

        Args:
            cell_embeddings: (torch.Tensor) a tensor of shape (num_tables, num_records, num_columns, embedding_size)
                                             that contains all the cell embeddings
        Returns
            (torch.Tensor) a tensor of shape (num_tables, num_records, num_columns, embedding_size)
        """
        num_tables, num_records, num_columns, embedding_size = cell_embeddings.shape

        # attention between columns
        cell_embeddings = cell_embeddings.reshape(
            num_tables * num_records, num_columns, embedding_size
        )
        cell_embeddings = (
            self.self_attention_between_columns(
                cell_embeddings, cell_embeddings, cell_embeddings
            )[0]
            + cell_embeddings
        )
        cell_embeddings = cell_embeddings.reshape(
            num_tables, num_records, num_columns, embedding_size
        )
        cell_embeddings = self.column_attention_norm(cell_embeddings)

        # attention between records
        cell_embeddings = cell_embeddings.transpose(1, 2)
        cell_embeddings = cell_embeddings.reshape(
            num_tables * num_columns, num_records, embedding_size
        )
        cell_embeddings = (
            self.self_attention_between_records(
                cell_embeddings, cell_embeddings, cell_embeddings
            )[0]
            + cell_embeddings
        )
        cell_embeddings = cell_embeddings.reshape(
            num_tables, num_columns, num_records, embedding_size
        )
        cell_embeddings = cell_embeddings.transpose(2, 1)
        cell_embeddings = self.record_attention_norm(cell_embeddings)

        # MLP after attention
        cell_embeddings = (
            self.mlp_out(F.gelu(self.mlp_in(cell_embeddings))) + cell_embeddings
        )
        cell_embeddings = self.mlp_norm(cell_embeddings)
        return cell_embeddings


class MeanPoolRecordRepresenter(nn.Module):
    def forward(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        return cell_embeddings.mean(dim=2)


class EntityTargetColumnRepresenter(nn.Module):
    def forward(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        return cell_embeddings[:, :, -1, :]


class PairMLPAdjacencyDecoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int):
        super().__init__()
        self.pair_mlp = PairDifferenceMLP(embedding_size, mlp_hidden_size)

    def forward(self, record_embeddings: torch.Tensor) -> torch.Tensor:
        pairwise_difference = torch.abs(
            record_embeddings.unsqueeze(2) - record_embeddings.unsqueeze(1)
        )
        return self.pair_mlp(pairwise_difference).squeeze(-1)


class SlotCoassignmentAdjacencyDecoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        entity_slot_count: int,
        num_slot_attention_layers: int,
    ):
        super().__init__()
        self.slot_mixer = EntitySlotMixer(
            embedding_size,
            num_attention_heads,
            entity_slot_count,
            num_slot_attention_layers,
        )

    def forward(self, record_embeddings: torch.Tensor) -> torch.Tensor:
        slot_weights, _ = self.slot_mixer(record_embeddings)
        adjacency_probabilities = torch.matmul(
            slot_weights, slot_weights.transpose(1, 2)
        )
        adjacency_probabilities = adjacency_probabilities.clamp(1e-6, 1 - 1e-6)
        return torch.logit(adjacency_probabilities)


class SlotPairMLPAdjacencyDecoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int,
        num_attention_heads: int,
        entity_slot_count: int,
        num_slot_attention_layers: int,
    ):
        super().__init__()
        self.slot_mixer = EntitySlotMixer(
            embedding_size,
            num_attention_heads,
            entity_slot_count,
            num_slot_attention_layers,
        )
        self.pair_mlp = PairDifferenceMLP(embedding_size, mlp_hidden_size)

    def forward(self, record_embeddings: torch.Tensor) -> torch.Tensor:
        _, slot_context = self.slot_mixer(record_embeddings)
        slot_conditioned_records = record_embeddings + slot_context
        pairwise_difference = torch.abs(
            slot_conditioned_records.unsqueeze(2)
            - slot_conditioned_records.unsqueeze(1)
        )
        return self.pair_mlp(pairwise_difference).squeeze(-1)


class EntitySlotMixer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        entity_slot_count: int,
        num_slot_attention_layers: int,
    ):
        super().__init__()
        self.initial_slots = nn.Parameter(torch.randn(entity_slot_count, embedding_size))
        self.slot_attention_layers = nn.ModuleList(
            [
                MultiheadAttention(
                    embedding_size,
                    num_attention_heads,
                    batch_first=True,
                )
                for _ in range(num_slot_attention_layers)
            ]
        )
        self.slot_norms = nn.ModuleList(
            [LayerNorm(embedding_size) for _ in range(num_slot_attention_layers)]
        )

    def forward(
        self, record_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tables = record_embeddings.shape[0]
        slots = self.initial_slots.unsqueeze(0).expand(num_tables, -1, -1)
        for attention, norm in zip(self.slot_attention_layers, self.slot_norms):
            updated_slots = attention(slots, record_embeddings, record_embeddings)[0]
            slots = norm(slots + updated_slots)

        slot_logits = torch.matmul(record_embeddings, slots.transpose(1, 2))
        slot_weights = torch.softmax(slot_logits, dim=-1)
        slot_context = torch.matmul(slot_weights, slots)
        return slot_weights, slot_context


class PairDifferenceMLP(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int):
        """Initializes the MLP that converts pair embeddings to adjacency logits."""
        super().__init__()
        self.mlp_in = nn.Linear(embedding_size, mlp_hidden_size)
        self.mlp_out = nn.Linear(mlp_hidden_size, 1)

    def forward(self, pairwise_difference: torch.Tensor) -> torch.Tensor:
        """
        Applies an MLP to pairwise record differences to get adjacency logits.

        Args:
            pairwise_difference: (torch.Tensor) a tensor of shape
                                 (num_tables, num_records, num_records, embedding_size)
        Returns:
            (torch.Tensor) a tensor of shape (num_tables, num_records, num_records, 1)
        """
        return self.mlp_out(F.gelu(self.mlp_in(pairwise_difference)))


class NanoERPFNLinker:
    """Transductive sklearn-like interface for entity linkage."""

    def __init__(self, model, tokenizer, threshold: float = 0.5, device=None):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.threshold = threshold

    def _move_to_device(self, tokenized_cells: dict) -> dict:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in tokenized_cells.items()
        }

    def decision_function(self, X: pd.DataFrame, field_types: list[str]) -> np.ndarray:
        """Returns raw pairwise adjacency logits."""
        tokenized_cells = self.tokenizer([X], [field_types])
        tokenized_cells = self._move_to_device(tokenized_cells)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tokenized_cells).squeeze(0)
        return logits.cpu().numpy()

    def fit(self, X: pd.DataFrame, field_types: list[str]):
        """Links all records in X and stores pairwise and cluster outputs."""
        from scipy.sparse.csgraph import connected_components

        self.adjacency_logits_ = self.decision_function(X, field_types)
        self.adjacency_proba_ = torch.sigmoid(
            torch.from_numpy(self.adjacency_logits_)
        ).numpy()
        adjacency = self.adjacency_proba_ >= self.threshold
        _, self.labels_ = connected_components(adjacency, directed=False)
        return self

    def fit_predict(self, X: pd.DataFrame, field_types: list[str]) -> np.ndarray:
        """Links records and returns connected-component entity labels."""
        return self.fit(X, field_types).labels_

    def fit_predict_adjacency(
        self, X: pd.DataFrame, field_types: list[str]
    ) -> np.ndarray:
        """Links records and returns pairwise adjacency probabilities."""
        return self.fit(X, field_types).adjacency_proba_


if __name__ == "__main__":
    from .dgp.person_data_generation import generate_world
    from .tokenizer import Tokenizer

    world = generate_world(n_records=20, n_fields=4, seed=0)
    for text_backend in ["fasttext", "sentence_transformer"]:
        tokenizer = Tokenizer(text_backend=text_backend)
        for record_representation in ["mean_pool", "entity_target_column"]:
            for adjacency_decoder in ["pair_mlp", "slot_coassignment", "slot_pair_mlp"]:
                print(
                    f"\ntext_backend={text_backend}, "
                    f"record_representation={record_representation}, "
                    f"adjacency_decoder={adjacency_decoder}"
                )
                model = NanoERPFNModel(
                    embedding_size=32,
                    text_embedding_size=tokenizer.text_embedding_dim,
                    num_attention_heads=4,
                    mlp_hidden_size=64,
                    num_layers=2,
                    record_representation=record_representation,
                    adjacency_decoder=adjacency_decoder,
                    entity_slot_count=20,
                )
                linker = NanoERPFNLinker(model, tokenizer)

                adjacency_proba = linker.fit_predict_adjacency(
                    world.records, world.field_types
                )

                print("\tAdjacency probabilities:", adjacency_proba.shape)
                print("\tEntity labels:", linker.labels_.shape)
                print("\tSymmetric:", np.allclose(adjacency_proba, adjacency_proba.T))
