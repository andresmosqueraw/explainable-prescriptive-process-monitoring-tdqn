from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBuffer:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        """Initialize RNG if not provided (for deterministic sampling)."""
        if self.rng is None:
            self.rng = np.random.default_rng()

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample batch with controlled RNG for reproducibility."""
        # Ensure RNG is initialized (mypy safety check)
        rng = self.rng if self.rng is not None else np.random.default_rng()
        idx = rng.integers(0, len(self.states), size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


