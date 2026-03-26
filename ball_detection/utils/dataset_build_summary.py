from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DatasetBuildSummary:
    processed_ball: int = 0
    processed_not_ball: int = 0
    skipped: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)

