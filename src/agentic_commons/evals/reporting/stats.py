"""Small-sample honesty for pass rates."""

import math


def wilson_interval(passed: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a pass rate.

    Chosen over the normal approximation because golden sets are small:
    at 50 cases and a 90% pass rate the normal interval runs past 100%,
    while Wilson stays inside [0, 1] and is honest about the width.
    """
    if total <= 0:
        return 0.0, 0.0
    rate = passed / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)
