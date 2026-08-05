import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "identitypfn_mpl_config"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from identitypfn.model import NanoERPFNLinker
from run_field_evidence_diagnostics import (
    load_model_bundle,
    pair_indices_and_targets,
    prepare_benchmark_tasks,
)
from identitypfn.utils import get_default_device


CHECKPOINT_PATH = Path(
    "results/model_checkpoints/20260715_195803_seed1337_20005214_best_step1500.pt"
)
OUTPUT_PATH = Path("overleaf_git/figs/nc_pairwise_diagnostics_default_wag_1500.png")


def agreement_counts(records, left_indices, right_indices):
    left = records.iloc[left_indices].reset_index(drop=True)
    right = records.iloc[right_indices].reset_index(drop=True)
    return (left.eq(right) | (left.isna() & right.isna())).sum(axis=1).to_numpy()


def main():
    device = get_default_device()
    benchmark = prepare_benchmark_tasks(
        Path("benchmark_data/manifest.json"), ["nc_voters_changed"]
    )[0]

    model, tokenizer, _ = load_model_bundle(CHECKPOINT_PATH, device)
    linker = NanoERPFNLinker(model, tokenizer, device=device)
    linker.fit(benchmark["records"], benchmark["field_types"])

    left_indices, right_indices, targets = pair_indices_and_targets(benchmark)
    probabilities = linker.adjacency_proba_[left_indices, right_indices]
    counts = agreement_counts(benchmark["records"], left_indices, right_indices)
    pr_auc = average_precision_score(targets, probabilities)
    pairs = pd.DataFrame(
        {
            "target": targets,
            "probability": probabilities,
            "matching_fields": counts,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4), dpi=300)
    fig.patch.set_facecolor("white")

    bins = np.linspace(0, 1, 31)
    axes[0].hist(
        pairs.loc[~pairs["target"], "probability"],
        bins=bins,
        alpha=0.65,
        density=True,
        label="negative",
    )
    axes[0].hist(
        pairs.loc[pairs["target"], "probability"],
        bins=bins,
        alpha=0.65,
        density=True,
        label="positive",
    )
    axes[0].set(
        title=f"Pair-probability distributions (PR-AUC={pr_auc:.3f})",
        xlabel="probability",
        ylabel="density",
    )
    axes[0].legend()

    plot_rng = np.random.default_rng(0)
    max_negative_points_per_group = 500
    plot_groups = []
    group_counts = pairs.groupby(["matching_fields", "target"]).size()
    for (matching_fields, target), group in pairs.groupby(["matching_fields", "target"]):
        if not target and len(group) > max_negative_points_per_group:
            group = group.sample(max_negative_points_per_group, random_state=0)
        group = group[["probability"]].copy()
        group["target"] = target
        group["x"] = matching_fields + (-0.12 if not target else 0.12)
        group["x"] += plot_rng.uniform(-0.08, 0.08, len(group))
        plot_groups.append(group)
    plot_pairs = pd.concat(plot_groups, ignore_index=True)
    for target, group in plot_pairs.groupby("target"):
        axes[1].scatter(
            group["x"],
            group["probability"],
            s=10,
            alpha=0.25,
            label="positive" if target else "negative",
        )
    for (matching_fields, target), count in group_counts.items():
        axes[1].annotate(
            f"n={count}",
            (matching_fields + (-0.12 if not target else 0.12), 1.01),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    axes[1].set(
        title="Probability by target and matching fields",
        xlabel="number of exactly matching fields",
        ylabel="probability",
        ylim=(-0.02, 1.15),
    )
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
