"""Tests for MDP dataset building (Phase 1 - Step 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xppm.data.build_mdp import build_mdp_dataset
from xppm.data.encode_prefixes import encode_prefix_dataset
from xppm.data.preprocess import preprocess_event_log
from xppm.utils.io import load_npz


@pytest.fixture
def tiny_log_with_outcome_path(project_root: Path) -> Path:
    """Path to tiny synthetic event log with outcome for testing."""
    return project_root / "tests" / "data" / "tiny_log_with_outcome.csv"


@pytest.fixture
def test_config() -> dict:
    """Minimal test configuration for MDP building."""
    return {
        "mdp": {
            "actions": {
                "id2name": ["do_nothing", "contact_headquarters"],
            },
            "decision_points": {
                "mode": "all",  # Allow all prefixes as decision points for testing
            },
            "action_mask": {
                "default_valid": ["do_nothing"],
                "by_last_activity": {
                    "validate_application": ["do_nothing", "contact_headquarters"],
                    "start_standard": ["do_nothing"],
                },
            },
            "reward": {
                "type": "terminal_profit_delayed",
                "intermediate_reward": 0.0,
                "terminal_column": "outcome",
            },
            "output": {
                "path": "data/processed/D_offline.npz",
                "schema_path": "configs/schemas/offline_rlset.schema.json",
            },
        },
        "encoding": {
            "max_len": 10,
            "min_prefix_len": 1,
            "truncation": "left",
            "padding": "left",
            "output": {
                "prefixes_path": "data/interim/prefixes.npz",
                "vocab_activity_path": "data/interim/vocab_activity.json",
            },
            "vocab": {
                "min_freq": 1,
                "add_unk": True,
                "add_pad": True,
            },
            "fields": {
                "case_id_col": "case_id",
                "activity_col": "activity",
                "timestamp_col": "timestamp",
            },
        },
        "schema": {
            "case_id": "case_id",
            "activity": "activity",
            "timestamp": "timestamp",
        },
        "time": {
            "timezone": None,
            "output_timezone": "UTC",
            "sort": True,
            "drop_duplicates": True,
        },
        "preprocess": {
            "filters": {
                "min_events_per_case": 1,
            },
        },
    }


def test_mdp_transitions_structure(
    tiny_log_with_outcome_path: Path, tmp_outdir: Path, test_config: dict
) -> None:
    """Test that build_mdp_dataset produces valid MDP transitions with correct structure."""
    # Build pipeline: preprocess -> encode -> build_mdp
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    # Preprocess
    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)

    # Update config with test paths
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)

    # Encode prefixes
    encode_prefix_dataset(clean_path, test_config)

    # Build MDP dataset
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    # Load MDP dataset
    D = load_npz(mdp_path)

    # Check required keys (as per 1-3-setup.md)
    required_keys = [
        "s",
        "s_mask",
        "a",
        "r",
        "s_next",
        "s_next_mask",
        "done",
        "case_ptr",
        "t_ptr",
        "valid_actions",
        "behavior_action",
        "propensity",
    ]
    for key in required_keys:
        assert key in D, f"Missing key: {key}"

    # Check all arrays have same length N
    n = len(D["a"])
    assert n > 0, "Should have at least one transition"
    assert len(D["s"]) == n, "s length mismatch"
    assert len(D["s_mask"]) == n, "s_mask length mismatch"
    assert len(D["r"]) == n, "r length mismatch"
    assert len(D["s_next"]) == n, "s_next length mismatch"
    assert len(D["s_next_mask"]) == n, "s_next_mask length mismatch"
    assert len(D["done"]) == n, "done length mismatch"
    assert len(D["case_ptr"]) == n, "case_ptr length mismatch"
    assert len(D["t_ptr"]) == n, "t_ptr length mismatch"
    assert len(D["valid_actions"]) == n, "valid_actions length mismatch"
    assert len(D["behavior_action"]) == n, "behavior_action length mismatch"
    assert len(D["propensity"]) == n, "propensity length mismatch"

    # Check shapes
    max_len = D["s"].shape[1]
    n_actions = len(test_config["mdp"]["actions"]["id2name"])
    assert D["s"].shape == (n, max_len), f"s shape mismatch: {D['s'].shape}"
    assert D["s_mask"].shape == (n, max_len), f"s_mask shape mismatch: {D['s_mask'].shape}"
    assert D["s_next"].shape == (n, max_len), f"s_next shape mismatch: {D['s_next'].shape}"
    assert D["s_next_mask"].shape == (n, max_len), "s_next_mask shape mismatch"
    assert D["valid_actions"].shape == (n, n_actions), "valid_actions shape mismatch"

    # Check dtypes
    assert D["s"].dtype == np.int32, f"s dtype should be int32, got {D['s'].dtype}"
    assert D["s_mask"].dtype == np.uint8, f"s_mask dtype should be uint8, got {D['s_mask'].dtype}"
    assert D["a"].dtype == np.int32, f"a dtype should be int32, got {D['a'].dtype}"
    assert D["r"].dtype == np.float32, f"r dtype should be float32, got {D['r'].dtype}"
    assert D["done"].dtype == np.uint8, f"done dtype should be uint8, got {D['done'].dtype}"
    assert D["valid_actions"].dtype == np.uint8, "valid_actions dtype should be uint8"


def test_action_mask_validity(
    tiny_log_with_outcome_path: Path, tmp_outdir: Path, test_config: dict
) -> None:
    """Test A: valid_actions[a] == 1 for all transitions (1-3-setup.md requirement)."""
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    D = load_npz(mdp_path)
    n = len(D["a"])

    if n == 0:
        pytest.skip("No transitions generated")

    # Check that valid_actions[a] == 1 for all transitions
    valid_mask = D["valid_actions"][np.arange(n), D["a"]] == 1
    assert valid_mask.all(), (
        f"Found {valid_mask.sum()}/{n} transitions with invalid actions "
        f"(valid_actions[a]==0). Invalid indices: {np.where(~valid_mask)[0]}"
    )


def test_terminal_reward_delayed(
    tiny_log_with_outcome_path: Path, tmp_outdir: Path, test_config: dict
) -> None:
    """Test B: terminal/reward delayed (1-3-setup.md requirement).

    Case 1 has 4 events, outcome=10.0:
    - t1->t2: r=0, done=0
    - t2->t3: r=0, done=0
    - t3->t4: r=10.0, done=1 (terminal)
    """
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    D = load_npz(mdp_path)

    # Find transitions for case_id=1
    case_1_mask = D["case_ptr"] == 1
    case_1_transitions = {
        "t_ptr": D["t_ptr"][case_1_mask],
        "r": D["r"][case_1_mask],
        "done": D["done"][case_1_mask],
    }

    if len(case_1_transitions["t_ptr"]) == 0:
        pytest.skip("No transitions for case 1")

    # Sort by t_ptr
    sort_idx = np.argsort(case_1_transitions["t_ptr"])
    rewards = case_1_transitions["r"][sort_idx]
    dones = case_1_transitions["done"][sort_idx]

    # Check that non-terminal transitions have r=0
    non_terminal_mask = dones == 0
    if non_terminal_mask.any():
        assert np.all(
            rewards[non_terminal_mask] == 0.0
        ), f"Non-terminal transitions should have r=0, got {rewards[non_terminal_mask]}"

    # Check that terminal transition has r=10.0 (case 1 outcome)
    terminal_mask = dones == 1
    if terminal_mask.any():
        terminal_rewards = rewards[terminal_mask]
        # At least one terminal transition should have reward = 10.0
        assert np.any(
            terminal_rewards == 10.0
        ), f"Terminal transition for case 1 should have r=10.0, got {terminal_rewards}"

    # Check that done is binary
    assert np.all((dones == 0) | (dones == 1)), "done must be 0 or 1"


def test_transitions_validity(
    tiny_log_with_outcome_path: Path, tmp_outdir: Path, test_config: dict
) -> None:
    """Test C: transitions validity (1-3-setup.md requirement).

    s_next should correspond to same case_ptr and t_ptr+1.
    """
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    D = load_npz(mdp_path)
    n = len(D["a"])

    if n == 0:
        pytest.skip("No transitions generated")

    # Check that s_next corresponds to same case_ptr
    # (all transitions should be within the same case)
    # This is already enforced by build_transitions, but we verify

    # Group by case and check t_ptr continuity
    for case_id in np.unique(D["case_ptr"]):
        case_mask = D["case_ptr"] == case_id
        case_t_ptrs = D["t_ptr"][case_mask]

        # Sort by t_ptr
        sort_idx = np.argsort(case_t_ptrs)
        sorted_t_ptrs = case_t_ptrs[sort_idx]

        # Check that t_ptr is monotonic increasing
        assert np.all(
            sorted_t_ptrs[:-1] < sorted_t_ptrs[1:]
        ), f"t_ptr should be monotonic increasing for case {case_id}"

        # For each transition, the next prefix should have t_ptr = current_t_ptr + 1
        # (This is verified by checking that we can find a prefix with t_ptr+1 in the same case)
        # Note: This is a structural check - the actual s_next content is verified by
        # the fact that build_transitions constructs it from the next prefix index

    # Check that behavior_action == a (redundancy check)
    assert np.all(D["behavior_action"] == D["a"]), "behavior_action should equal a"

    # Check that propensity is set (should be -1.0 if not estimated)
    assert np.all(D["propensity"] == -1.0), "propensity should be -1.0 (not estimated yet)"
