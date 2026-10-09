import argparse
import csv
import hashlib
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np
import schedulefree
import torch

from identitypfn.data_loader import SyntheticWorldDataLoader
from identitypfn.model import NanoERPFNLinker, NanoERPFNModel
from identitypfn.dgp.people.simple import ObservationConfig, generate_world
from identitypfn.tokenizer import Tokenizer
from identitypfn.train import eval_frame, load_benchmarks
from identitypfn.utils import get_default_device, set_randomness_seed


@dataclass
class FrozenDGPConfig:
    missing_rate: float = 0.1
    nickname_rate: float = 0.05
    corruption_rate: float = 0.1
    seed: int = 502


@dataclass
class ExperimentConfig:
    random_seed: int = 1337
    num_steps: int = 500
    eval_every: int = 50
    batch_size: int = 16
    learning_rate: float = 4e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    pos_weight: float | None = None
    progress_every: int = 25
    train_generator: Literal["person", "wag", "id_prior", "bank"] = "person"
    revelio_root: str | None = None
    train_manifest: str = "training_data/manifest.json"
    id_prior_allow_ollama: bool = False
    id_prior_ollama_model: str = "llama3.2:1b"
    id_prior_ollama_augment_rate: float | None = None
    id_prior_ollama_corrupt_rate: float | None = None
    wag_allow_ollama: bool = False
    wag_ollama_model: str = "qwen2.5:7b"
    wag_missing_rate: list[float] | None = None
    wag_nickname_rate: list[float] | None = None
    wag_corruption_rate: list[float] | None = None
    wag_prop_corruption_rate: list[float] | None = None
    hard_negative_rate: float = 0.0
    hard_negative_contract_weights: dict[str, float] | None = None

    embedding_size: int = 96
    num_attention_heads: int = 4
    mlp_hidden_size: int = 192
    num_layers: int = 3
    max_categories: int = 128
    record_representation: Literal["mean_pool", "entity_target_column"] = (
        "entity_target_column"
    )
    adjacency_decoder: Literal[
        "pair_mlp", "slot_coassignment", "slot_pair_mlp"
    ] = "pair_mlp"
    entity_slot_count: int = 300
    num_slot_attention_layers: int = 1

    text_backend: Literal["fasttext", "sentence_transformer"] = "sentence_transformer"
    identifier_backend: Literal["fasttext", "sentence_transformer"] | None = None
    normalize_identifiers: bool = False
    default_phone_region: str = "US"
    benchmark_names: list[str] = field(
        default_factory=lambda: [
            "nc_voters_shared",
            "nc_voters_changed",
            "fodors_zagats",
            "dblp_acm",
            "amazon_google_price_numeric",
        ]
    )
    frozen_dgp_n_records: int = 250
    frozen_dgp_p_match: float = 0.05
    frozen_dgp_n_fields: int = 6
    frozen_dgp_schema_temperature: float = 1.0
    frozen_dgp_configs: dict[str, FrozenDGPConfig] = field(
        default_factory=lambda: {"person_dgp": FrozenDGPConfig()}
    )

    results_directory: Path = Path("results/benchmark_runs")
    checkpoint_directory: Path = Path("results/model_checkpoints")
    save_models: bool = False
    save_best_model: bool = True
    save_last_model: bool = True
    resume_checkpoint: Path | None = None


def build_run_config(config: ExperimentConfig) -> dict:
    run_config = asdict(config)
    run_config.pop("resume_checkpoint")
    if config.train_generator != "bank":
        run_config.pop("train_manifest")
    if config.id_prior_ollama_augment_rate is None:
        run_config.pop("id_prior_ollama_augment_rate")
    if config.id_prior_ollama_corrupt_rate is None:
        run_config.pop("id_prior_ollama_corrupt_rate")
    run_config["results_directory"] = str(config.results_directory)
    run_config["checkpoint_directory"] = str(config.checkpoint_directory)
    return run_config


def make_run_id(started_at: datetime, run_config: dict) -> tuple[str, str]:
    serialized_config = json.dumps(run_config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(serialized_config.encode()).hexdigest()[:8]
    run_id = (
        f"{started_at:%Y%m%d_%H%M%S}_seed{run_config['random_seed']}_{config_hash}"
    )
    return run_id, config_hash


def world_to_benchmark(name: str, world) -> dict:
    return {
        "name": name,
        "description": "Frozen synthetic person-DGP sniff-test world.",
        "kind": "single_table",
        "entity_sample": None,
        "records": world.records,
        "field_types": world.field_types,
        "entity_ids": world.entity_ids,
    }


def load_experiment_benchmarks(config: ExperimentConfig) -> list[dict]:
    benchmarks = load_benchmarks(
        "benchmark_data/manifest.json", names=config.benchmark_names
    )

    for name, frozen_config in config.frozen_dgp_configs.items():
        observation_config = ObservationConfig(
            missing_rate=frozen_config.missing_rate,
            nickname_rate=frozen_config.nickname_rate,
            corruption_rate=frozen_config.corruption_rate,
        )
        world = generate_world(
            n_records=config.frozen_dgp_n_records,
            p_match=config.frozen_dgp_p_match,
            n_fields=config.frozen_dgp_n_fields,
            schema_temperature=config.frozen_dgp_schema_temperature,
            observation_config=observation_config,
            seed=frozen_config.seed,
        )
        benchmarks.append(world_to_benchmark(name, world))

    return benchmarks


def make_model(
    config: ExperimentConfig, tokenizer: Tokenizer, device: torch.device
) -> NanoERPFNModel:
    return NanoERPFNModel(
        embedding_size=config.embedding_size,
        text_embedding_size=tokenizer.text_embedding_dim,
        identifier_embedding_size=tokenizer.identifier_embedding_dim,
        num_attention_heads=config.num_attention_heads,
        mlp_hidden_size=config.mlp_hidden_size,
        num_layers=config.num_layers,
        max_categories=config.max_categories,
        record_representation=config.record_representation,
        adjacency_decoder=config.adjacency_decoder,
        entity_slot_count=config.entity_slot_count,
        num_slot_attention_layers=config.num_slot_attention_layers,
    ).to(device)


def move_to_device(values: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def clone_state_dict(model: NanoERPFNModel) -> dict:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def checkpoint_frame(
    model: NanoERPFNModel,
    tokenizer: Tokenizer,
    benchmarks: list[dict],
    device: torch.device,
    step: int,
) -> pd.DataFrame:
    linker = NanoERPFNLinker(model, tokenizer, device=device)
    frame = eval_frame(linker, benchmarks).reset_index()
    frame.insert(0, "step", step)
    return frame


def save_model_checkpoint(
    config: ExperimentConfig,
    run_id: str,
    run_config: dict,
    checkpoint_states: dict[int, dict],
    step: int,
    label: str,
) -> Path:
    config.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_directory / f"{run_id}_{label}_step{step:04d}.pt"
    torch.save(
        {
            "run_id": run_id,
            "step": step,
            "label": label,
            "run_config": run_config,
            "model_state_dict": checkpoint_states[step],
        },
        checkpoint_path,
    )
    print(f"saved {label} model checkpoint: {checkpoint_path}")
    return checkpoint_path


def save_live_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def write_loss_log(path: Path, losses: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "weighted_training_loss"])
        writer.writeheader()
        writer.writerows(losses)


def random_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_experiment(config: ExperimentConfig) -> Path:
    set_randomness_seed(config.random_seed)
    device = get_default_device()
    run_config = build_run_config(config)
    resume_state = None
    if config.resume_checkpoint is not None:
        resume_state = torch.load(
            config.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if resume_state["label"] != "latest":
            raise ValueError("Resume requires a latest checkpoint")
        if resume_state["run_config"] != run_config:
            raise ValueError("Resume configuration differs from the checkpoint")
        started_at = datetime.fromisoformat(resume_state["run_started_at"])
        run_id = resume_state["run_id"]
        config_hash = resume_state["config_hash"]
    else:
        started_at = datetime.now().astimezone()
        run_id, config_hash = make_run_id(started_at, run_config)
    print(f"device: {device}")
    print(f"run_id: {run_id}")

    tokenizer = Tokenizer(
        text_backend=config.text_backend,
        identifier_backend=config.identifier_backend,
        normalize_identifiers=config.normalize_identifiers,
        default_phone_region=config.default_phone_region,
    )
    benchmarks = load_experiment_benchmarks(config)
    model = make_model(config, tokenizer, device)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    prior = SyntheticWorldDataLoader(
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        seed=config.random_seed,
        generator=config.train_generator,
        allow_ollama=(
            config.id_prior_allow_ollama
            if config.train_generator in {"id_prior", "bank"}
            else config.wag_allow_ollama
        ),
        ollama_model=(
            config.id_prior_ollama_model
            if config.train_generator in {"id_prior", "bank"}
            else config.wag_ollama_model
        ),
        id_prior_ollama_augment_rate=config.id_prior_ollama_augment_rate,
        id_prior_ollama_corrupt_rate=config.id_prior_ollama_corrupt_rate,
        revelio_root=config.revelio_root,
        train_manifest=config.train_manifest,
        bank_wag_allow_ollama=config.wag_allow_ollama,
        bank_wag_ollama_model=config.wag_ollama_model,
        missing_rate=config.wag_missing_rate,
        nickname_rate=config.wag_nickname_rate,
        corruption_rate=config.wag_corruption_rate,
        prop_corruption_rate=config.wag_prop_corruption_rate,
        hard_negative_rate=config.hard_negative_rate,
        hard_negative_contract_weights=config.hard_negative_contract_weights,
    )

    config.results_directory.mkdir(parents=True, exist_ok=True)
    run_results_path = config.results_directory / f"{run_id}.csv"
    loss_log_path = config.results_directory / f"{run_id}_loss.csv"
    if run_results_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_results_path}")
    train_started_at = time.perf_counter()

    if resume_state is None:
        checkpoint_frames = []
        training_losses = []
        completed_step = 0
        optimizer.eval()
        initial_frame = checkpoint_frame(model, tokenizer, benchmarks, device, 0)
        checkpoint_frames.append(initial_frame)
        best_step = 0
        best_score = float(initial_frame["pr_auc"].mean())
        best_state = clone_state_dict(model)
        optimizer.train()
        write_loss_log(loss_log_path, training_losses)
        if config.save_models and config.save_best_model:
            save_live_checkpoint(
                config.checkpoint_directory / f"{run_id}_best.pt",
                {
                    "run_id": run_id,
                    "step": best_step,
                    "label": "best",
                    "run_config": run_config,
                    "model_state_dict": best_state,
                },
            )
    else:
        completed_step = int(resume_state["step"])
        if completed_step >= config.num_steps:
            raise ValueError("Checkpoint has already reached the configured num_steps")
        model.load_state_dict(resume_state["training_model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        prior.rng.bit_generator.state = resume_state["loader_rng_state"]
        prior.num_steps = config.num_steps - completed_step
        checkpoint_frames = resume_state["checkpoint_frames"]
        training_losses = resume_state["training_losses"]
        best_step = int(resume_state["best_step"])
        best_score = float(resume_state["best_score"])
        best_state = resume_state["best_model_state_dict"]
        write_loss_log(loss_log_path, training_losses)
        restore_random_state(resume_state["random_state"])
        print(f"resuming after step {completed_step}", flush=True)

    for step, full_data in enumerate(prior, start=completed_step + 1):
        model.train()
        optimizer.train()
        optimizer.zero_grad()

        tokenized_cells = move_to_device(
            tokenizer(full_data["records"], full_data["field_types"]), device
        )
        targets = full_data["adjacency"].to(device).float()
        logits = model(tokenized_cells)
        mask = torch.triu(torch.ones_like(targets, dtype=torch.bool), diagonal=1)
        pair_logits = logits[mask]
        pair_targets = targets[mask]
        num_positive = pair_targets.sum()
        num_negative = pair_targets.numel() - num_positive
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pair_logits,
            pair_targets,
            pos_weight=num_negative / num_positive
            if config.pos_weight is None
            else torch.tensor(config.pos_weight, device=pair_logits.device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        training_losses.append(
            {"step": step, "weighted_training_loss": float(loss.detach().cpu())}
        )
        with loss_log_path.open("a", newline="") as file:
            csv.writer(file).writerow(
                [step, training_losses[-1]["weighted_training_loss"]]
            )

        if config.progress_every > 0 and step % config.progress_every == 0:
            elapsed_seconds = time.perf_counter() - train_started_at
            seconds_per_step = elapsed_seconds / (step - completed_step)
            sources = (
                f" sources={','.join(full_data['source_names'])}"
                if config.train_generator == "bank"
                else ""
            )
            print(
                "step "
                f"{step}/{config.num_steps} "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"pos={int(num_positive.detach().cpu())} "
                f"neg={int(num_negative.detach().cpu())} "
                f"elapsed={elapsed_seconds / 60:.1f}m "
                f"rate={seconds_per_step:.2f}s/step{sources}",
                flush=True,
            )

        if step % config.eval_every == 0 or step == config.num_steps:
            model.eval()
            optimizer.eval()
            inference_state = clone_state_dict(model)
            frame = checkpoint_frame(model, tokenizer, benchmarks, device, step)
            checkpoint_frames.append(frame)
            score = float(frame["pr_auc"].mean())
            if score > best_score:
                best_step = step
                best_score = score
                best_state = inference_state
                if config.save_models and config.save_best_model:
                    save_live_checkpoint(
                        config.checkpoint_directory / f"{run_id}_best.pt",
                        {
                            "run_id": run_id,
                            "step": best_step,
                            "label": "best",
                            "run_config": run_config,
                            "model_state_dict": best_state,
                        },
                    )
            optimizer.train()
            if config.save_models and config.save_last_model:
                save_live_checkpoint(
                    config.checkpoint_directory / f"{run_id}_latest.pt",
                    {
                        "run_id": run_id,
                        "step": step,
                        "label": "latest",
                        "run_config": run_config,
                        "run_started_at": started_at.isoformat(),
                        "config_hash": config_hash,
                        "model_state_dict": inference_state,
                        "training_model_state_dict": clone_state_dict(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loader_rng_state": deepcopy(prior.rng.bit_generator.state),
                        "random_state": random_state(),
                        "checkpoint_frames": checkpoint_frames,
                        "training_losses": training_losses,
                        "best_step": best_step,
                        "best_score": best_score,
                        "best_model_state_dict": best_state,
                    },
                )
            print(f"evaluated step {step}", flush=True)

    history = pd.concat(checkpoint_frames, ignore_index=True)
    loss_history = pd.DataFrame(training_losses)
    run_results = history.merge(loss_history, on="step", how="left")
    run_results.insert(0, "run_id", run_id)
    run_results.insert(1, "run_started_at", started_at.isoformat())
    run_results.insert(2, "config_hash", config_hash)
    for key, value in reversed(run_config.items()):
        serialized_value = (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else value
        )
        run_results.insert(3, key, serialized_value)

    run_results.to_csv(run_results_path, index=False)
    print(f"saved run results: {run_results_path}")

    mean_history = history.groupby("step").mean(numeric_only=True)
    best_mean_step = best_step
    final_step = int(history["step"].max())
    print(f"best mean PR-AUC checkpoint: {best_mean_step}")
    print(f"final checkpoint: {final_step}")
    summary_steps = [best_mean_step]
    if final_step != best_mean_step:
        summary_steps.append(final_step)
    print(mean_history.loc[summary_steps])

    if config.save_models:
        if config.save_best_model:
            save_model_checkpoint(
                config,
                run_id,
                run_config,
                {best_step: best_state},
                best_mean_step,
                "best",
            )
        if config.save_last_model and final_step != best_mean_step:
            save_model_checkpoint(
                config, run_id, run_config, {final_step: inference_state}, final_step, "last"
            )
        elif config.save_last_model:
            print("best and last checkpoint are the same step; saved one checkpoint")

    return run_results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a checkpointed benchmark experiment outside the notebook."
    )
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=4e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--train-generator",
        choices=["person", "wag", "id_prior", "bank"],
        default="person",
    )
    parser.add_argument("--revelio-root", type=str)
    parser.add_argument("--train-manifest", default="training_data/manifest.json")
    parser.add_argument("--id-prior-allow-ollama", action="store_true")
    parser.add_argument("--id-prior-ollama-model", default="llama3.2:1b")
    parser.add_argument("--id-prior-ollama-augment-rate", type=float)
    parser.add_argument("--id-prior-ollama-corrupt-rate", type=float)
    parser.add_argument("--wag-allow-ollama", action="store_true")
    parser.add_argument("--wag-ollama-model", default="qwen2.5:7b")
    parser.add_argument(
        "--wag-missing-rate",
        type=float,
        action="append",
        help="Allowed WAG missingness rates. Repeat for a list.",
    )
    parser.add_argument(
        "--wag-nickname-rate",
        type=float,
        action="append",
        help="Allowed WAG nickname-substitution rates. Repeat for a list.",
    )
    parser.add_argument(
        "--wag-corruption-rate",
        type=float,
        action="append",
        help="Allowed WAG character-corruption rates. Repeat for a list.",
    )
    parser.add_argument(
        "--wag-prop-corruption-rate",
        type=float,
        action="append",
        help="Allowed WAG propagated-corruption rates. Repeat for a list.",
    )
    parser.add_argument("--hard-negative-rate", type=float, default=0.0)
    parser.add_argument(
        "--text-backend",
        choices=["fasttext", "sentence_transformer"],
        default="sentence_transformer",
    )
    parser.add_argument(
        "--identifier-backend",
        choices=["fasttext", "sentence_transformer"],
        default=None,
    )
    parser.add_argument("--normalize-identifiers", action="store_true")
    parser.add_argument("--default-phone-region", default="US")
    parser.add_argument(
        "--record-representation",
        choices=["mean_pool", "entity_target_column"],
        default="entity_target_column",
    )
    parser.add_argument(
        "--adjacency-decoder",
        choices=["pair_mlp", "slot_coassignment", "slot_pair_mlp"],
        default="pair_mlp",
    )
    parser.add_argument("--entity-slot-count", type=int, default=300)
    parser.add_argument("--num-slot-attention-layers", type=int, default=1)
    parser.add_argument("--embedding-size", type=int, default=96)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-categories", type=int, default=128)
    parser.add_argument(
        "--benchmark-name",
        action="append",
        dest="benchmark_names",
        help="Benchmark to evaluate. Repeat to evaluate multiple benchmarks.",
    )
    parser.add_argument("--frozen-dgp-n-records", type=int, default=250)
    parser.add_argument("--frozen-dgp-p-match", type=float, default=0.05)
    parser.add_argument("--frozen-dgp-n-fields", type=int, default=6)
    parser.add_argument("--frozen-dgp-schema-temperature", type=float, default=1.0)
    parser.add_argument(
        "--no-frozen-dgp",
        action="store_true",
        help="Skip the frozen synthetic person-DGP benchmark.",
    )
    parser.add_argument(
        "--results-directory", type=Path, default=Path("results/benchmark_runs")
    )
    parser.add_argument(
        "--checkpoint-directory", type=Path, default=Path("results/model_checkpoints")
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-save-best-model", action="store_true")
    parser.add_argument("--no-save-last-model", action="store_true")
    parser.add_argument(
        "--resume", type=Path, help="Resume a latest checkpoint using its saved run settings."
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["label"] != "latest":
            raise ValueError("Resume requires a latest checkpoint")
        saved = checkpoint["run_config"].copy()
        saved["results_directory"] = Path(saved["results_directory"])
        saved["checkpoint_directory"] = Path(saved["checkpoint_directory"])
        saved["frozen_dgp_configs"] = {
            name: FrozenDGPConfig(**value)
            for name, value in saved["frozen_dgp_configs"].items()
        }
        return ExperimentConfig(**saved, resume_checkpoint=args.resume)
    for rate in (args.id_prior_ollama_augment_rate, args.id_prior_ollama_corrupt_rate):
        if rate is not None and not 0 <= rate <= 1:
            raise ValueError("id-prior Ollama rates must be between 0 and 1")
    if not args.id_prior_allow_ollama and any(
        rate is not None and rate > 0
        for rate in (args.id_prior_ollama_augment_rate, args.id_prior_ollama_corrupt_rate)
    ):
        raise ValueError("Positive id-prior Ollama rates require --id-prior-allow-ollama")
    frozen_dgp_configs = {} if args.no_frozen_dgp else {"person_dgp": FrozenDGPConfig()}
    return ExperimentConfig(
        random_seed=args.random_seed,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        pos_weight=args.pos_weight,
        progress_every=args.progress_every,
        train_generator=args.train_generator,
        revelio_root=args.revelio_root,
        train_manifest=args.train_manifest,
        id_prior_allow_ollama=args.id_prior_allow_ollama,
        id_prior_ollama_model=args.id_prior_ollama_model,
        id_prior_ollama_augment_rate=args.id_prior_ollama_augment_rate,
        id_prior_ollama_corrupt_rate=args.id_prior_ollama_corrupt_rate,
        wag_allow_ollama=args.wag_allow_ollama,
        wag_ollama_model=args.wag_ollama_model,
        wag_missing_rate=args.wag_missing_rate,
        wag_nickname_rate=args.wag_nickname_rate,
        wag_corruption_rate=args.wag_corruption_rate,
        wag_prop_corruption_rate=args.wag_prop_corruption_rate,
        hard_negative_rate=args.hard_negative_rate,
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        num_layers=args.num_layers,
        max_categories=args.max_categories,
        record_representation=args.record_representation,
        adjacency_decoder=args.adjacency_decoder,
        entity_slot_count=args.entity_slot_count,
        num_slot_attention_layers=args.num_slot_attention_layers,
        text_backend=args.text_backend,
        identifier_backend=args.identifier_backend,
        normalize_identifiers=args.normalize_identifiers,
        default_phone_region=args.default_phone_region,
        benchmark_names=args.benchmark_names
        if args.benchmark_names is not None
        else ExperimentConfig().benchmark_names,
        frozen_dgp_n_records=args.frozen_dgp_n_records,
        frozen_dgp_p_match=args.frozen_dgp_p_match,
        frozen_dgp_n_fields=args.frozen_dgp_n_fields,
        frozen_dgp_schema_temperature=args.frozen_dgp_schema_temperature,
        frozen_dgp_configs=frozen_dgp_configs,
        results_directory=args.results_directory,
        checkpoint_directory=args.checkpoint_directory,
        save_models=args.save_models,
        save_best_model=not args.no_save_best_model,
        save_last_model=not args.no_save_last_model,
    )


if __name__ == "__main__":
    run_experiment(config_from_args(parse_args()))
