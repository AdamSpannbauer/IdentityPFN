"""Auxiliary losses for record embeddings in generated identity tasks.

Inputs are record embeddings of shape (tasks, records, features) and Boolean
identity adjacency of shape (tasks, records, records). Identities are local to
each task, so no loss compares records across tasks.
"""

import torch
import torch.nn.functional as F


def _mean_or_zero(losses: list[torch.Tensor], embeddings: torch.Tensor) -> torch.Tensor:
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0.0


def _pair_masks(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    off_diagonal = ~torch.eye(adjacency.shape[0], dtype=torch.bool, device=adjacency.device)
    positives = adjacency.bool() & off_diagonal
    return positives, ~adjacency.bool() & off_diagonal


def _identity_labels(adjacency: torch.Tensor) -> torch.Tensor:
    diagonal = torch.eye(adjacency.shape[0], dtype=torch.bool, device=adjacency.device)
    return (adjacency.bool() | diagonal).long().argmax(dim=1)


def _paired_views(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the first two records of each identity with duplicate views."""
    n = adjacency.shape[0]
    indices = torch.arange(n, device=adjacency.device)
    labels = _identity_labels(adjacency)
    later_matches = adjacency.bool() & (indices[None, :] > indices[:, None])
    first = indices[(labels == indices) & later_matches.any(dim=1)]
    second = later_matches[first].long().argmax(dim=1)
    return first, second


def supervised_contrastive_loss(
    embeddings: torch.Tensor, adjacency: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    losses = []
    for records, matches in zip(embeddings, adjacency):
        positives, _ = _pair_masks(matches)
        anchors = positives.any(dim=1)
        if not anchors.any():
            continue
        z = F.normalize(records, dim=-1)
        similarity = z @ z.T / temperature
        similarity.fill_diagonal_(float("-inf"))
        log_denominator = torch.logsumexp(similarity, dim=1)
        log_probability = similarity - log_denominator[:, None]
        positive_log_probability = log_probability.masked_fill(~positives, 0.0)
        losses.append(
            -(positive_log_probability.sum(dim=1) / positives.sum(dim=1).clamp_min(1))[anchors].mean()
        )
    return _mean_or_zero(losses, embeddings)


def circle_loss(
    embeddings: torch.Tensor, adjacency: torch.Tensor,
    margin: float = 0.25, scale: float = 64.0,
) -> torch.Tensor:
    losses = []
    for records, matches in zip(embeddings, adjacency):
        positives, negatives = _pair_masks(matches)
        anchors = positives.any(dim=1) & negatives.any(dim=1)
        if not anchors.any():
            continue
        z = F.normalize(records, dim=-1)
        similarity = z @ z.T
        positive_weight = F.relu(1 + margin - similarity.detach())
        negative_weight = F.relu(similarity.detach() + margin)
        positive_logits = (-scale * positive_weight * (similarity - (1 - margin))).masked_fill(
            ~positives, float("-inf")
        )
        negative_logits = (scale * negative_weight * (similarity - margin)).masked_fill(
            ~negatives, float("-inf")
        )
        losses.append(F.softplus(
            torch.logsumexp(positive_logits[anchors], dim=1)
            + torch.logsumexp(negative_logits[anchors], dim=1)
        ).mean())
    return _mean_or_zero(losses, embeddings)


def triplet_loss(
    embeddings: torch.Tensor, adjacency: torch.Tensor, margin: float = 0.2
) -> torch.Tensor:
    losses = []
    for records, matches in zip(embeddings, adjacency):
        positives, negatives = _pair_masks(matches)
        anchors = positives.any(dim=1) & negatives.any(dim=1)
        if not anchors.any():
            continue
        z = F.normalize(records, dim=-1)
        distances = torch.cdist(z, z)
        hardest_positive = distances.masked_fill(~positives, float("-inf")).max(dim=1).values
        hardest_negative = distances.masked_fill(~negatives, float("inf")).min(dim=1).values
        losses.append(F.relu(hardest_positive[anchors] - hardest_negative[anchors] + margin).mean())
    return _mean_or_zero(losses, embeddings)


def contrastive_margin_loss(
    embeddings: torch.Tensor, adjacency: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    losses = []
    for records, matches in zip(embeddings, adjacency):
        upper = torch.triu(torch.ones_like(matches, dtype=torch.bool), diagonal=1)
        if not upper.any():
            continue
        z = F.normalize(records, dim=-1)
        distances = torch.cdist(z, z)[upper]
        positive = matches.bool()[upper]
        positive_loss = distances[positive].square().mean() if positive.any() else distances.sum() * 0.0
        negative_loss = (
            F.relu(margin - distances[~positive]).square().mean()
            if (~positive).any() else distances.sum() * 0.0
        )
        losses.append(positive_loss + negative_loss)
    return _mean_or_zero(losses, embeddings)


def _identity_means(records: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    _, inverse = torch.unique(_identity_labels(adjacency), return_inverse=True)
    n_identities = int(inverse.max()) + 1
    sums = records.new_zeros((n_identities, records.shape[-1])).index_add(0, inverse, records)
    counts = torch.bincount(inverse, minlength=n_identities).to(records.dtype)
    return sums / counts[:, None]


def vicreg_loss(
    embeddings: torch.Tensor, adjacency: torch.Tensor,
    sim_weight: float = 25.0, var_weight: float = 25.0, cov_weight: float = 1.0,
) -> torch.Tensor:
    """Paired-view invariance with variance and covariance across identity means."""
    losses = []
    for records, matches in zip(embeddings, adjacency):
        first, second = _paired_views(matches)
        if first.numel() == 0:
            continue
        invariance = F.mse_loss(records[first], records[second])
        identities = _identity_means(records, matches)
        if len(identities) < 2:
            losses.append(sim_weight * invariance)
            continue
        centered = identities - identities.mean(dim=0)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(1 - std).mean()
        covariance = centered.T @ centered / (len(identities) - 1)
        covariance = covariance - torch.diag_embed(covariance.diagonal())
        decorrelation = covariance.square().sum() / records.shape[-1]
        losses.append(sim_weight * invariance + var_weight * variance + cov_weight * decorrelation)
    return _mean_or_zero(losses, embeddings)


def barlow_twins_loss(
    embeddings: torch.Tensor, adjacency: torch.Tensor, off_diagonal_weight: float = 0.005
) -> torch.Tensor:
    losses = []
    for records, matches in zip(embeddings, adjacency):
        first, second = _paired_views(matches)
        if first.numel() < 2:
            continue
        left, right = records[first], records[second]
        left = (left - left.mean(dim=0)) / (left.std(dim=0, unbiased=False) + 1e-4)
        right = (right - right.mean(dim=0)) / (right.std(dim=0, unbiased=False) + 1e-4)
        cross_correlation = left.T @ right / first.numel()
        diagonal = cross_correlation.diagonal().sub(1).square().sum()
        off_diagonal = cross_correlation - torch.diag_embed(cross_correlation.diagonal())
        losses.append(diagonal + off_diagonal_weight * off_diagonal.square().sum())
    return _mean_or_zero(losses, embeddings)


def koleo_loss(
    embeddings: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Spread records using each record's nearest non-self neighbor."""
    losses = []
    for records in embeddings:
        if len(records) < 2:
            continue
        z = F.normalize(records, dim=-1)
        similarity = z @ z.T
        similarity.fill_diagonal_(float("-inf"))
        nearest_indices = similarity.argmax(dim=1)
        nearest = F.pairwise_distance(z, z[nearest_indices], eps=eps)
        losses.append(-torch.log(nearest + eps).mean())
    return _mean_or_zero(losses, embeddings)
