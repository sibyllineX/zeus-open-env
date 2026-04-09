from __future__ import annotations

from dataclasses import dataclass

try:
    from ..models import LeakHunterState
except ImportError:
    from models import LeakHunterState


@dataclass(frozen=True)
class DifficultyBand:
    expert: tuple[float, float]
    good: tuple[float, float]
    decent: tuple[float, float]
    bad: tuple[float, float]
    random: tuple[float, float]


# Calibrated 2026-04-07 against WNTR-backed scores across seeds 11-25.
# Expert = exact clamp (~0.95), Good = correct section (~0.65),
# Decent = nearby pipe (~0.35-0.50), Bad = wrong area (~0.20-0.35).
BANDS = {
    "easy": DifficultyBand(
        expert=(0.90, 1.00),
        good=(0.60, 0.90),
        decent=(0.35, 0.60),
        bad=(0.15, 0.35),
        random=(0.00, 0.15),
    ),
    "medium": DifficultyBand(
        expert=(0.88, 1.00),
        good=(0.58, 0.88),
        decent=(0.33, 0.58),
        bad=(0.13, 0.33),
        random=(0.00, 0.13),
    ),
    "hard": DifficultyBand(
        expert=(0.85, 1.00),
        good=(0.55, 0.85),
        decent=(0.30, 0.55),
        bad=(0.10, 0.30),
        random=(0.00, 0.10),
    ),
}


def grade_state(state: LeakHunterState) -> float:
    raw = float(state.final_score or 0.0)
    return max(0.01, min(0.99, raw))


def classify_score(difficulty: str, score: float) -> str:
    bands = BANDS[difficulty]
    for label in ("expert", "good", "decent", "bad", "random"):
        lo, hi = getattr(bands, label)
        if lo <= score <= hi:
            return label
    if score > bands.expert[1]:
        return "above_expert"
    return "below_random"
