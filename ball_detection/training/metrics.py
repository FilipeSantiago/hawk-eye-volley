from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int
    threshold: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_binary_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    logits_cpu = logits.detach().reshape(-1).cpu()
    targets_cpu = targets.detach().reshape(-1).cpu().to(dtype=torch.int64)

    probabilities = torch.sigmoid(logits_cpu)
    predictions = (probabilities >= threshold).to(dtype=torch.int64)

    tp = int(((predictions == 1) & (targets_cpu == 1)).sum().item())
    tn = int(((predictions == 0) & (targets_cpu == 0)).sum().item())
    fp = int(((predictions == 1) & (targets_cpu == 0)).sum().item())
    fn = int(((predictions == 0) & (targets_cpu == 1)).sum().item())

    total = tp + tn + fp + fn
    accuracy = _safe_divide(tp + tn, total)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        threshold=threshold,
    )
