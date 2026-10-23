"""Target decay schedule (Section 3.2): tau = 1 - (1 - tau_base) * (cos(pi k / K) + 1) / 2, tau_base = 0.996."""
import math


def tau_schedule(k: int, K: int, tau_base: float = 0.996) -> float:
    return 1 - (1 - tau_base) * (math.cos(math.pi * k / K) + 1) / 2
