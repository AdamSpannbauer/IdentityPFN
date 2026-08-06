from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import average_precision_score

from identitypfn.data_loader import SyntheticWorldDataLoader
from identitypfn.model import NanoERPFNLinker, NanoERPFNModel
from identitypfn.dgp.people.simple import generate_worlds as generate_person_worlds
from identitypfn.dgp.people.wag import generate_worlds as generate_wag_worlds
from identitypfn.tokenizer import Tokenizer
from identitypfn.train import train
from identitypfn.utils import get_default_device, set_randomness_seed


def evaluate_synthetic_worlds(
    linker: NanoERPFNLinker,
    generator: str,
    n_worlds: int,
    n_records: int,
    n_fields: int,
    seed: int,
    allow_ollama: bool | None,
    ollama_model: str,
) -> tuple[float, float, list[float]]:
    generate_worlds = (
        generate_wag_worlds if generator == "wag" else generate_person_worlds
    )
    generator_kwargs = {}
    if generator == "wag":
        generator_kwargs["ollama_model"] = ollama_model
        if allow_ollama is not None:
            generator_kwargs["allow_ollama"] = allow_ollama
    worlds = generate_worlds(
        n_worlds=n_worlds,
        n_records=[n_records],
        n_fields=[n_fields],
        seed=seed,
        **generator_kwargs,
    )
    pr_aucs = []
    prevalences = []
    for world in worlds:
        pair_mask = np.triu(np.ones_like(world.adjacency, dtype=bool), k=1)
        probabilities = linker.fit_predict_adjacency(
            world.records, world.field_types
        )[pair_mask]
        targets = world.adjacency[pair_mask]
        pr_aucs.append(average_precision_score(targets, probabilities))
        prevalences.append(float(targets.mean()))
    return float(np.mean(pr_aucs)), float(np.mean(prevalences)), pr_aucs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--generator", choices=["person", "wag"], default="wag")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps-per-print", type=int, default=5)
    parser.add_argument("--max-categories", type=int, default=512)
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--eval-worlds", type=int, default=0)
    parser.add_argument("--eval-records", type=int, default=80)
    parser.add_argument("--eval-fields", type=int, default=6)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--eval-generators",
        nargs="+",
        choices=["person", "wag"],
        default=["wag", "person"],
    )
    args = parser.parse_args()
    allow_ollama = False if args.no_ollama else None

    set_randomness_seed(args.seed)
    device = get_default_device()
    tokenizer = Tokenizer(text_backend="fasttext")
    model = NanoERPFNModel(
        embedding_size=32,
        text_embedding_size=tokenizer.text_embedding_dim,
        num_attention_heads=4,
        mlp_hidden_size=64,
        num_layers=1,
        max_categories=args.max_categories,
        entity_slot_count=64,
    )
    prior = SyntheticWorldDataLoader(
        num_steps=args.steps,
        batch_size=args.batch_size,
        generator=args.generator,
        seed=args.seed,
        allow_ollama=allow_ollama,
        ollama_model=args.ollama_model,
    )

    print(
        f"Running DGP smoke: generator={args.generator}, "
        f"steps={args.steps}, batch_size={args.batch_size}, seed={args.seed}, "
        f"allow_ollama={not args.no_ollama}, ollama_model={args.ollama_model}"
    )
    train(
        model,
        tokenizer,
        prior,
        lr=args.lr,
        steps_per_print=args.steps_per_print,
        eval_func=None,
    )

    if args.eval_worlds > 0:
        linker = NanoERPFNLinker(model, tokenizer, device=device)
        for eval_generator in args.eval_generators:
            mean_pr_auc, mean_prevalence, pr_aucs = evaluate_synthetic_worlds(
                linker=linker,
                generator=eval_generator,
                n_worlds=args.eval_worlds,
                n_records=args.eval_records,
                n_fields=args.eval_fields,
                seed=args.eval_seed,
                allow_ollama=allow_ollama,
                ollama_model=args.ollama_model,
            )
            formatted_pr_aucs = ", ".join(f"{value:.4f}" for value in pr_aucs)
            print(
                f"eval_generator={eval_generator} | "
                f"mean_pr_auc={mean_pr_auc:.4f} | "
                f"mean_prevalence={mean_prevalence:.4f} | "
                f"pr_aucs=[{formatted_pr_aucs}]"
            )


if __name__ == "__main__":
    main()
