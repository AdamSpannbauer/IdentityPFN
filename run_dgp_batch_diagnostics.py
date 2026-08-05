from argparse import ArgumentParser
from collections import Counter

import numpy as np

from identitypfn.data_loader import SyntheticWorldDataLoader


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def pair_prevalence(adjacency: np.ndarray) -> float:
    pair_mask = np.triu(np.ones_like(adjacency, dtype=bool), k=1)
    return float(adjacency[pair_mask].mean())


def max_categorical_cardinality(records, field_types: list[str]) -> int:
    cardinalities = [
        records.iloc[:, field_index].nunique(dropna=True)
        for field_index, field_type in enumerate(field_types)
        if field_type == "categorical"
    ]
    return max(cardinalities, default=0)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--generator", choices=["person", "wag"], default="wag")
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    args = parser.parse_args()

    loader = SyntheticWorldDataLoader(
        num_steps=args.num_batches,
        batch_size=args.batch_size,
        generator=args.generator,
        seed=args.seed,
        allow_ollama=False if args.no_ollama else None,
        ollama_model=args.ollama_model,
    )

    record_counts = []
    entity_counts = []
    pair_prevalences = []
    max_cat_cardinalities = []
    field_type_counts = Counter()
    zero_positive_worlds = 0

    for batch in loader:
        for records, entity_ids, field_types, adjacency in zip(
            batch["records"],
            batch["entity_ids"],
            batch["field_types"],
            batch["adjacency"],
        ):
            adjacency_array = adjacency.numpy()
            pair_mask = np.triu(np.ones_like(adjacency_array, dtype=bool), k=1)
            prevalence = pair_prevalence(adjacency_array)

            record_counts.append(len(records))
            entity_counts.append(int(entity_ids.unique().numel()))
            pair_prevalences.append(prevalence)
            max_cat_cardinalities.append(
                max_categorical_cardinality(records, field_types)
            )
            field_type_counts.update(field_types)
            if not adjacency_array[pair_mask].any():
                zero_positive_worlds += 1

    n_worlds = args.num_batches * args.batch_size
    print(
        f"DGP batch diagnostics: generator={args.generator}, "
        f"worlds={n_worlds}, seed={args.seed}, "
        f"allow_ollama={not args.no_ollama}, ollama_model={args.ollama_model}"
    )
    print("records:", summarize(record_counts))
    print("entities:", summarize(entity_counts))
    print("pair_prevalence:", summarize(pair_prevalences))
    print("zero_positive_worlds:", zero_positive_worlds)
    print("field_type_counts:", dict(field_type_counts))
    print("max_categorical_cardinality:", summarize(max_cat_cardinalities))


if __name__ == "__main__":
    main()
