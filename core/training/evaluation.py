from __future__ import annotations


def compute_fitness(
    loss_sum: float,
    flops: float = 0.0,
    flops_penalty_lambda: float = 0.0,
    eps: float = 1e-8,
) -> float:
    if loss_sum <= 0:
        return 0.0
    base = 1.0 / (loss_sum + eps)
    return base - (flops_penalty_lambda * flops)


def is_breakthrough(
    candidate_loss_sum: float,
    baseline_loss_sum: float,
    candidate_time_ms: float,
    baseline_time_ms: float,
    improvement_ratio: float = 0.05,
) -> bool:
    target_loss = (1.0 - improvement_ratio) * baseline_loss_sum
    return candidate_loss_sum <= target_loss and candidate_time_ms <= baseline_time_ms
