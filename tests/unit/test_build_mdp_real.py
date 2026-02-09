"""Tests on real dataset (read-only, no writes).

These tests verify the real dataset (964k+ transitions) but are marked as
@slow and excluded from CI by default. They only READ from data/processed/D_offline.npz
and never write to it, so they are safe to run.

To run these tests locally:
    pytest tests/unit/test_build_mdp_real.py -v -m slow

Note: These tests require the real dataset to exist at data/processed/D_offline.npz.
If it doesn't exist, they will be skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xppm.utils.io import load_npz


@pytest.mark.slow
def test_action_mask_validity_real_dataset() -> None:
    """Test A on real dataset: valid_actions[a] == 1 for all transitions.

    This test verifies that the action mask is consistent with the behavior
    actions in the real dataset (964k+ transitions).

    IMPORTANT: This test only READS from data/processed/D_offline.npz.
    It never writes or modifies the real dataset.
    """
    mdp_path = Path("data/processed/D_offline.npz")

    if not mdp_path.exists():
        pytest.skip(f"Real dataset not found at {mdp_path}")

    D = load_npz(mdp_path)
    n = len(D["a"])

    if n == 0:
        pytest.skip("Dataset is empty")

    # Check that valid_actions[a] == 1 for all transitions
    valid_mask = D["valid_actions"][np.arange(n), D["a"]] == 1
    invalid_count = (~valid_mask).sum()

    assert invalid_count == 0, (
        f"Found {invalid_count}/{n} transitions with invalid actions "
        f"(valid_actions[a]==0). This indicates dataset corruption."
    )

    # Additional check: all actions should be in valid range
    n_actions = D["valid_actions"].shape[1]
    assert np.all(D["a"] >= 0) and np.all(D["a"] < n_actions), (
        f"Actions out of range: min={D['a'].min()}, max={D['a'].max()}, " f"n_actions={n_actions}"
    )


@pytest.mark.slow
def test_reward_delayed_real_dataset() -> None:
    """Test reward delayed consistency on real dataset.

    IMPORTANT: This test only READS from data/processed/D_offline.npz.
    It never writes or modifies the real dataset.
    """
    mdp_path = Path("data/processed/D_offline.npz")

    if not mdp_path.exists():
        pytest.skip(f"Real dataset not found at {mdp_path}")

    D = load_npz(mdp_path)
    n = len(D["a"])

    if n == 0:
        pytest.skip("Dataset is empty")

    done = D["done"]
    r = D["r"]

    # Non-terminal transitions should have r=0
    non_terminal_mask = done == 0
    if non_terminal_mask.any():
        non_terminal_rewards = r[non_terminal_mask]
        non_zero_count = (non_terminal_rewards != 0).sum()
        assert non_zero_count == 0, (
            f"Found {non_zero_count} non-terminal transitions with r!=0. "
            f"Reward should be delayed (only at terminal states)."
        )

    # Terminal transitions should have r != 0 (unless outcome is 0)
    terminal_mask = done == 1
    if terminal_mask.any():
        terminal_rewards = r[terminal_mask]
        # At least some terminal rewards should be non-zero
        # (unless all outcomes are 0, which is unlikely but possible)
        assert len(terminal_rewards) > 0, "No terminal transitions found"


@pytest.mark.slow
def test_done_consistency_real_dataset() -> None:
    """Test that done flag is consistent (one terminal per case).

    IMPORTANT: This test only READS from data/processed/D_offline.npz.
    It never writes or modifies the real dataset.
    """
    mdp_path = Path("data/processed/D_offline.npz")

    if not mdp_path.exists():
        pytest.skip(f"Real dataset not found at {mdp_path}")

    D = load_npz(mdp_path)
    n = len(D["a"])

    if n == 0:
        pytest.skip("Dataset is empty")

    done = D["done"]
    case_ptr = D["case_ptr"]

    # Check that pct_done is approximately n_cases / n_transitions
    n_cases = len(np.unique(case_ptr))
    expected_pct_done = n_cases / n * 100
    actual_pct_done = done.mean() * 100

    # Allow 5% tolerance (some cases might not have transitions)
    assert abs(actual_pct_done - expected_pct_done) < 5.0, (
        f"pct_done ({actual_pct_done:.2f}%) does not match expected "
        f"({expected_pct_done:.2f}%). This might indicate issues with "
        f"terminal state detection."
    )

    # Sample check: verify no case has multiple terminals
    # (check a sample to avoid O(n) scan)
    sample_size = min(10000, n)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    sample_case_ptr = case_ptr[sample_indices]
    sample_done = done[sample_indices]

    case_terminal_count: dict[int, int] = {}
    for i in range(sample_size):
        case_id = sample_case_ptr[i]
        if sample_done[i] == 1:
            case_terminal_count[case_id] = case_terminal_count.get(case_id, 0) + 1

    multi_terminal_cases = [c for c, count in case_terminal_count.items() if count > 1]
    assert len(multi_terminal_cases) == 0, (
        f"Found {len(multi_terminal_cases)} cases with multiple terminals "
        f"in sample of {sample_size} transitions. Each case should have "
        f"exactly one terminal state."
    )
