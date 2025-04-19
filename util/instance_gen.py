from __future__ import annotations

"""knapsack_instance_generator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility for generating synthetic 0-1 Knapsack problem instances.

Implements the three instance families described in Refaei-Afshar et-al. (2020):

* **RI** – Random Instances
* **FI** – Fixed‑capacity, fixed‑size Instances
* **HI** – Hard (strongly‑correlated) Instances

A fourth family, **SS** (Subset‑Sum‑like), is included as a creative extension.

Example
-------
>>> from knapsack_instance_generator import KnapsackInstanceGenerator
>>> gen = KnapsackInstanceGenerator(seed=42)
>>> ri = gen.generate('RI', M=3, N=10, R=100)
>>> fi = gen.generate('FI', M=2, N=50)  # capacity defaults to paper value
>>> hi = gen.generate('HI', M=2, N=20, R=100)
>>> ss = gen.generate('SS', M=1, N=8, R=50)
>>> print(ri[0].to_dict())
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class KnapsackInstance:
    """Container for a single knapsack instance."""

    values: List[float]
    weights: List[float]
    capacity: float

    # convenient helper -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain Python dict (useful for JSON serialisation)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------


class KnapsackInstanceGenerator:
    """API class that creates families of 0-1 knapsack instances."""

    def __init__(self, seed: int | None = None):
        """Create a new generator.

        Parameters
        ----------
        seed : int | None, optional
            Seed for NumPy's random number generator; ``None`` → nondeterministic.
        """

        self._rng = np.random.default_rng(seed)

    # internal helper -------------------------------------------------------

    def _get_rng(self, seed: int | None):
        """Return a deterministic RNG when *seed* is supplied, otherwise reuse the
        instance RNG to keep the stream continuous."""
        return np.random.default_rng(seed) if seed is not None else self._rng

    # -------------------------------------------------------------------
    # Instance family: RI – Random Instances (paper §5.2)
    # -------------------------------------------------------------------

    def generate_random_instances(
        self,
        M: int,
        N: int,
        R: int,
        *,
        seed: int | None = None,
    ) -> List[KnapsackInstance]:
        """Random Instances (RI).

        Each instance *p* has a random number of items *nₚ ∈ [1, N]*. For every item
        ``i`` we sample an integer weight ``wᵢ`` and value ``vᵢ`` uniformly from
        ``[1, R]``. The knapsack capacity ``Wₚ`` is an integer drawn uniformly from
        ``[R/10, 3R]``.
        """

        rng = self._get_rng(seed)
        inst: List[Dict[str, Any]] = []
        for _ in range(M):
            n_items = int(rng.integers(1, N + 1))
            weights = rng.integers(1, R + 1, size=n_items)
            values = rng.integers(1, R + 1, size=n_items)
            capacity = int(rng.integers(max(1, R // 10), 3 * R + 1))
            inst.append(
                KnapsackInstance(values.tolist(), weights.tolist(), capacity).to_dict()
            )
        return inst

    # -------------------------------------------------------------------
    # Instance family: FI – Fixed capacity & size (paper §5.2)
    # -------------------------------------------------------------------

    def generate_fixed_instances(
        self,
        M: int,
        N: int,
        *,
        capacity: float | None = None,
        seed: int | None = None,
    ) -> List[KnapsackInstance]:
        """Fixed-capacity, fixed-size Instances (FI).

        * Exactly *N* items per instance.
        * ``vᵢ, wᵢ`` ∼ *U(0, 1)* (continuous).
        * All instances share the same capacity *W*.
          If *capacity* is *None*, defaults follow the paper:
            * ``N = 50``  → 12.5
            * ``N ∈ {300, 500}`` → 37.5
            * otherwise → ``0.25 × N``.
        """

        if capacity is None:
            if N == 50:
                capacity = 12.5
            elif N in (300, 500):
                capacity = 37.5
            else:
                capacity = 0.25 * N

        rng = self._get_rng(seed)
        inst: List[Dict[str, Any]] = []
        for _ in range(M):
            weights = rng.random(N)
            values = rng.random(N)
            inst.append(
                KnapsackInstance(values.tolist(), weights.tolist(), float(capacity)).to_dict()
            )
        return inst

    # -------------------------------------------------------------------
    # Instance family: HI – Hard Instances (paper §5.2 & Pisinger 2005)
    # -------------------------------------------------------------------

    def generate_hard_instances(
        self,
        M: int,
        N: int,
        R: int,
        *,
        seed: int | None = None,
    ) -> List[KnapsackInstance]:
        """Hard (strongly-correlated) Instances (HI).

        For instance index *p* (1-based):
        * ``wᵢ`` ∼ *U{1, R}*
        * ``vᵢ = wᵢ + R/10`` (strong correlation)
        * ``Wₚ = (p / (M + 1)) · Σwᵢ``
        """

        rng = self._get_rng(seed)
        inst: List[Dict[str, Any]] = []
        for p in range(1, M + 1):
            weights = rng.integers(1, R + 1, size=N)
            values = weights + R / 10.0
            capacity = (p / (M + 1)) * weights.sum()
            inst.append(
                KnapsackInstance(values.tolist(), weights.tolist(), float(capacity)).to_dict()
            )
        return inst

    # -------------------------------------------------------------------
    # Creative extension: SS – Subset-Sum-like instances
    # -------------------------------------------------------------------

    def generate_subset_sum_instances(
        self,
        M: int,
        N: int,
        R: int,
        *,
        seed: int | None = None,
    ) -> List[KnapsackInstance]:
        """Subset-Sum-like (SS) instances.

        Designed to sit close to the NP-complete subset-sum boundary:
        * ``wᵢ`` ~ *U{1, R}*
        * ``vᵢ = wᵢ`` (perfect correlation)
        * ``W = ½ · Σwᵢ``
        """

        rng = self._get_rng(seed)
        inst: List[Dict[str, Any]] = []
        for _ in range(M):
            weights = rng.integers(1, R + 1, size=N)
            values = weights.copy()
            capacity = 0.5 * weights.sum()
            inst.append(
                KnapsackInstance(values.tolist(), weights.tolist(), float(capacity)).to_dict()
            )
        return inst

    # -------------------------------------------------------------------
    # Unified public interface
    # -------------------------------------------------------------------

    def generate(
        self,
        instance_type: Literal["RI", "FI", "HI", "SS"],
        **kwargs,
    ) -> List[KnapsackInstance]:
        """Factory method.

        Parameters
        ----------
        instance_type : {'RI', 'FI', 'HI', 'SS'}
            Family of instances to create.
        **kwargs
            Forwarded to the dedicated *generate_* method.
        """

        if instance_type == "RI":
            return self.generate_random_instances(**kwargs)
        if instance_type == "FI":
            return self.generate_fixed_instances(**kwargs)
        if instance_type == "HI":
            return self.generate_hard_instances(**kwargs)
        if instance_type == "SS":
            return self.generate_subset_sum_instances(**kwargs)
        raise ValueError(f"Unknown instance_type: {instance_type!r}")

    # -------------------------------------------------------------------
    # Convenience I/O helpers
    # -------------------------------------------------------------------

    @staticmethod
    def save_json(instances: Sequence[KnapsackInstance], path: str, *, indent: int = 2):
        """Save *instances* to *path* as JSON."""
        with open(path, "w", encoding="utf-8") as fp:
            json.dump([inst.to_dict() for inst in instances], fp, indent=indent)

    @staticmethod
    def load_json(path: str) -> List[KnapsackInstance]:
        """Load a list of instances previously saved with :py:meth:`save_json`."""
        with open(path, "r", encoding="utf-8") as fp:
            raw = json.load(fp)
        return [KnapsackInstance(**item) for item in raw]
