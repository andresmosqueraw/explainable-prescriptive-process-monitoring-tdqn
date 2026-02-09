"""Validation and splitting of MDP dataset (Phase 1 - Step 01b)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from xppm.utils.io import load_npz, save_json
from xppm.utils.logging import ensure_dir, get_logger

logger = get_logger(__name__)

# Required arrays in D_offline.npz
REQUIRED_ARRAYS = [
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
]


def validate_npz_structure(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Validate structural integrity of D_offline.npz (hard-fail).

    Args:
        data: Dictionary loaded from D_offline.npz

    Returns:
        Dictionary with validation results and metadata

    Raises:
        ValueError: If validation fails
    """
    # Check required arrays exist
    missing = [key for key in REQUIRED_ARRAYS if key not in data]
    if missing:
        raise ValueError(f"Missing required arrays in D_offline.npz: {missing}")

    # Get dimensions
    n = len(data["a"])
    if n == 0:
        raise ValueError("Dataset is empty (N=0)")

    # Check shapes are consistent
    max_len = data["s"].shape[1]
    n_actions = data["valid_actions"].shape[1]

    if data["s"].shape != (n, max_len):
        raise ValueError(f"s shape mismatch: expected {(n, max_len)}, got {data['s'].shape}")
    if data["s_mask"].shape != (n, max_len):
        raise ValueError(
            f"s_mask shape mismatch: expected {(n, max_len)}, got {data['s_mask'].shape}"
        )
    if data["s_next"].shape != (n, max_len):
        raise ValueError(
            f"s_next shape mismatch: expected {(n, max_len)}, got {data['s_next'].shape}"
        )
    if data["s_next_mask"].shape != (n, max_len):
        raise ValueError(
            f"s_next_mask shape mismatch: expected {(n, max_len)}, "
            f"got {data['s_next_mask'].shape}"
        )
    if data["valid_actions"].shape != (n, n_actions):
        raise ValueError(
            f"valid_actions shape mismatch: expected {(n, n_actions)}, "
            f"got {data['valid_actions'].shape}"
        )

    # Check all arrays have same length N
    for key in ["r", "done", "case_ptr", "t_ptr"]:
        if len(data[key]) != n:
            raise ValueError(f"{key} length mismatch: expected {n}, got {len(data[key])}")

    # Check dtypes
    if not np.issubdtype(data["a"].dtype, np.integer):
        raise ValueError(f"a dtype should be integer, got {data['a'].dtype}")
    if data["done"].dtype not in (np.uint8, np.bool_):
        raise ValueError(f"done dtype should be uint8 or bool, got {data['done'].dtype}")
    if data["valid_actions"].dtype not in (np.uint8, np.bool_):
        raise ValueError(
            f"valid_actions dtype should be uint8 or bool, got {data['valid_actions'].dtype}"
        )

    # Check mask validity: valid_actions[i, a[i]] == 1 for all i
    invalid_mask = data["valid_actions"][np.arange(n), data["a"]] == 0
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        raise ValueError(
            f"Found {n_invalid} transitions with invalid actions "
            f"(valid_actions[a]==0). Dataset is corrupted."
        )

    # Check done is binary (0 or 1)
    done_values = np.unique(data["done"])
    if not np.all((done_values == 0) | (done_values == 1)):
        raise ValueError(f"done must be 0 or 1, found values: {done_values}")

    # Check t_ptr starts at 0 per case and increments by 1
    # Group by case and verify t_ptr continuity
    case_groups: dict[int, list[int]] = defaultdict(list)
    for i, case_id in enumerate(data["case_ptr"]):
        case_groups[case_id].append(i)

    for case_id, indices in case_groups.items():
        case_t_ptrs = data["t_ptr"][indices]
        sorted_indices = np.argsort(case_t_ptrs)
        sorted_t_ptrs = case_t_ptrs[sorted_indices]

        # Check starts at 0 (or min_prefix_len, but typically 0)
        if sorted_t_ptrs[0] < 0:
            raise ValueError(f"Case {case_id} has negative t_ptr: {sorted_t_ptrs[0]}")

        # Check increments by 1 (allow some gaps if min_prefix_len > 1)
        # At minimum, should be monotonic
        if not np.all(sorted_t_ptrs[:-1] < sorted_t_ptrs[1:]):
            raise ValueError(f"Case {case_id} has non-monotonic t_ptr: {sorted_t_ptrs[:10]}")

    logger.info("✅ Structural validation passed: %d transitions, %d cases", n, len(case_groups))

    return {
        "n_transitions": n,
        "n_cases": len(case_groups),
        "max_len": max_len,
        "n_actions": n_actions,
        "pct_invalid_action": 0.0,  # Verified to be 0
    }


def validate_mdp_logic(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Validate MDP logic (transitions continuity and reward sanity).

    Args:
        data: Dictionary loaded from D_offline.npz

    Returns:
        Dictionary with validation results

    Raises:
        ValueError: If critical validation fails
    """
    n = len(data["a"])
    case_ptr = data["case_ptr"]
    t_ptr = data["t_ptr"]
    done = data["done"]
    r = data["r"]

    # Build index: (case_ptr, t_ptr) -> row_index
    transition_index: dict[tuple[int, int], int] = {}
    for i in range(n):
        key = (int(case_ptr[i]), int(t_ptr[i]))
        transition_index[key] = i

    # Check: for each non-terminal transition, next state should exist
    n_missing_next = 0
    for i in range(n):
        if done[i] == 0:  # Non-terminal
            case_id = int(case_ptr[i])
            current_t = int(t_ptr[i])
            next_key = (case_id, current_t + 1)
            if next_key not in transition_index:
                n_missing_next += 1

    if n_missing_next > 0:
        logger.warning(
            "Found %d non-terminal transitions without next state (t_ptr+1 missing). "
            "This may indicate incomplete cases or data issues.",
            n_missing_next,
        )

    # Reward sanity check
    non_terminal_mask = done == 0
    terminal_mask = done == 1

    non_terminal_rewards = r[non_terminal_mask]
    terminal_rewards = r[terminal_mask]

    n_non_terminal_nonzero = (non_terminal_rewards != 0).sum()
    if n_non_terminal_nonzero > 0:
        raise ValueError(
            f"Found {n_non_terminal_nonzero} non-terminal transitions with r!=0. "
            f"Reward should be delayed (only at terminal states)."
        )

    logger.info("✅ MDP logic validation passed")
    logger.info(
        "   Non-terminal rewards: %d transitions, all r=0",
        len(non_terminal_rewards),
    )
    logger.info(
        "   Terminal rewards: %d transitions, mean=%.2f, std=%.2f",
        len(terminal_rewards),
        terminal_rewards.mean() if len(terminal_rewards) > 0 else 0.0,
        terminal_rewards.std() if len(terminal_rewards) > 0 else 0.0,
    )

    return {
        "n_missing_next_states": n_missing_next,
        "non_terminal_reward_zero_pct": 100.0,  # Verified to be 100%
        "terminal_reward_mean": (
            float(terminal_rewards.mean()) if len(terminal_rewards) > 0 else 0.0
        ),
        "terminal_reward_std": (
            float(terminal_rewards.std()) if len(terminal_rewards) > 0 else 0.0
        ),
    }


def split_by_case(
    case_ptr: np.ndarray,
    method: str = "random_case",
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    case_start_ts: dict[int, float] | None = None,
) -> dict[str, np.ndarray]:
    """Split dataset by case_id (case_ptr) without leakage.

    Args:
        case_ptr: Array of case IDs for each transition
        method: Split method ("random_case" or "temporal_case")
        seed: Random seed for reproducibility
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        case_start_ts: Optional mapping from case_id to start timestamp (for temporal split)

    Returns:
        Dictionary with train/val/test masks (boolean arrays)

    Raises:
        ValueError: If ratios don't sum to 1.0 or method is invalid
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    unique_cases = np.unique(case_ptr)
    n_cases = len(unique_cases)

    if method == "random_case":
        # Shuffle cases with seed
        rng = np.random.RandomState(seed)
        shuffled_cases = rng.permutation(unique_cases)

        # Split by proportions (handle rounding to ensure all cases are assigned)
        n_train = int(n_cases * train_ratio)
        n_val = int(n_cases * val_ratio)
        # n_test = n_cases - n_train - n_val (to handle rounding)
        # Ensure at least 1 case per split if possible
        if n_cases >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0 and n_cases - n_train >= 2:
                n_val = 1

        train_cases = set(shuffled_cases[:n_train])
        val_cases = set(shuffled_cases[n_train : n_train + n_val])
        test_cases = set(shuffled_cases[n_train + n_val :])

    elif method == "temporal_case":
        if case_start_ts is None:
            raise ValueError("temporal_case requires case_start_ts mapping")

        # Sort cases by start timestamp
        case_timestamps = np.array([case_start_ts.get(int(c), 0.0) for c in unique_cases])
        sorted_indices = np.argsort(case_timestamps)
        sorted_cases = unique_cases[sorted_indices]

        # Split by temporal order
        n_train = int(n_cases * train_ratio)
        n_val = int(n_cases * val_ratio)

        train_cases = set(sorted_cases[:n_train])
        val_cases = set(sorted_cases[n_train : n_train + n_val])
        test_cases = set(sorted_cases[n_train + n_val :])

    else:
        raise ValueError(f"Unknown split method: {method}")

    # Verify no overlap
    if train_cases & val_cases:
        raise ValueError("Train and val cases overlap!")
    if train_cases & test_cases:
        raise ValueError("Train and test cases overlap!")
    if val_cases & test_cases:
        raise ValueError("Val and test cases overlap!")

    # Create masks for transitions
    train_mask = np.isin(case_ptr, list(train_cases))
    val_mask = np.isin(case_ptr, list(val_cases))
    test_mask = np.isin(case_ptr, list(test_cases))

    # Verify all transitions are covered
    total_covered = train_mask.sum() + val_mask.sum() + test_mask.sum()
    if total_covered != len(case_ptr):
        raise ValueError(f"Split doesn't cover all transitions: {total_covered}/{len(case_ptr)}")

    logger.info(
        "Split by case: train=%d cases (%d trans), val=%d cases (%d trans), "
        "test=%d cases (%d trans)",
        len(train_cases),
        train_mask.sum(),
        len(val_cases),
        val_mask.sum(),
        len(test_cases),
        test_mask.sum(),
    )

    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "train_cases": sorted(train_cases),  # type: ignore[dict-item]
        "val_cases": sorted(val_cases),  # type: ignore[dict-item]
        "test_cases": sorted(test_cases),  # type: ignore[dict-item]
    }


def compute_drift_stats(
    data: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute basic drift statistics across splits.

    Args:
        data: Dictionary loaded from D_offline.npz
        splits: Dictionary with train/val/test masks

    Returns:
        Dictionary with drift statistics
    """
    case_ptr = data["case_ptr"]
    t_ptr = data["t_ptr"]
    done = data["done"]
    r = data["r"]
    a = data["a"]

    stats: dict[str, Any] = {}

    for split_name in ["train", "val", "test"]:
        mask = splits[f"{split_name}_mask"]

        # Episode length (max t_ptr per case in this split)
        # t_ptr is 1-indexed (prefix length), so episode_length = max(t_ptr)
        split_cases = np.unique(case_ptr[mask])
        episode_lengths = []
        for case_id in split_cases:
            case_mask = (case_ptr == case_id) & mask
            if case_mask.any():
                max_t = t_ptr[case_mask].max()
                episode_lengths.append(int(max_t))  # t_ptr is already 1-indexed (prefix length)

        # Action rate (percentage of each action)
        split_actions = a[mask]
        n_actions = data["valid_actions"].shape[1]
        if len(split_actions) > 0:
            action_counts = np.bincount(split_actions, minlength=n_actions)
            action_rates = action_counts / len(split_actions)
        else:
            action_rates = np.zeros(n_actions)

        # Terminal reward stats
        split_terminal_mask = mask & (done == 1)
        terminal_rewards = r[split_terminal_mask]

        stats[f"{split_name}_episode_len_mean"] = (
            float(np.mean(episode_lengths)) if episode_lengths else 0.0
        )
        stats[f"{split_name}_episode_len_std"] = (
            float(np.std(episode_lengths)) if episode_lengths else 0.0
        )
        stats[f"{split_name}_action_rates"] = [float(x) for x in action_rates.tolist()]
        # Action rate for specific actions (if we know action names, but for now just log rates)
        # For time_contact_HQ: action 0 = do_nothing, action 1 = contact_headquarters
        if len(action_rates) >= 2:
            stats[f"{split_name}_action_rate_do_nothing"] = float(action_rates[0])
            stats[f"{split_name}_action_rate_contact_hq"] = (
                float(action_rates[1]) if len(action_rates) > 1 else 0.0
            )
        stats[f"{split_name}_reward_mean_terminal"] = (
            float(terminal_rewards.mean()) if len(terminal_rewards) > 0 else 0.0
        )
        stats[f"{split_name}_reward_std_terminal"] = (
            float(terminal_rewards.std()) if len(terminal_rewards) > 0 else 0.0
        )

    # Simple drift flag: compare train vs test episode lengths
    train_ep_mean = stats.get("train_episode_len_mean", 0.0)
    test_ep_mean = stats.get("test_episode_len_mean", 0.0)
    if train_ep_mean > 0:
        episode_len_drift = abs(test_ep_mean - train_ep_mean) / train_ep_mean
        stats["episode_len_drift_pct"] = float(episode_len_drift * 100)
        stats["drift_flag_episode_len"] = episode_len_drift > 0.2  # 20% threshold
    else:
        stats["episode_len_drift_pct"] = 0.0
        stats["drift_flag_episode_len"] = False

    return stats


def validate_and_split_dataset(
    npz_path: str | Path,
    splits_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate D_offline.npz and create train/val/test splits.

    Args:
        npz_path: Path to D_offline.npz
        splits_path: Path to output splits.json
        config: Configuration dictionary

    Returns:
        Dictionary with validation and split statistics
    """
    # Load dataset
    logger.info("Loading MDP dataset from %s", npz_path)
    data = load_npz(npz_path)

    # 1. Validate structure
    logger.info("Validating NPZ structure...")
    structure_stats = validate_npz_structure(data)

    # 2. Validate MDP logic
    logger.info("Validating MDP logic...")
    mdp_stats = validate_mdp_logic(data)

    # 3. Get split configuration
    split_cfg = config.get("validation_split", {})
    method = split_cfg.get("split_strategy", "case_id")
    if method == "case_id":
        method = "random_case"  # Default to random_case
    elif method == "temporal":
        method = "temporal_case"

    seed = config.get("repro", {}).get("seed", 42)
    ratios = split_cfg.get("ratios", {})
    train_ratio = ratios.get("train", 0.7)
    val_ratio = ratios.get("val", 0.1)
    test_ratio = ratios.get("test", 0.2)

    # For temporal split, we need case_start_ts (not available in NPZ currently)
    # Fail with clear error instead of silent fallback
    if method == "temporal_case":
        raise ValueError(
            "temporal_case split method requires case_start_ts mapping, "
            "but this is not available in D_offline.npz. "
            "Either: (1) use random_case method, or (2) provide case_start_ts "
            "by loading case timestamps from clean.parquet and passing to split_by_case()."
        )

    # 4. Split by case
    logger.info("Splitting dataset by case (method: %s)...", method)
    splits = split_by_case(
        data["case_ptr"],
        method=method,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # 5. Compute drift stats
    logger.info("Computing drift statistics...")
    drift_stats = compute_drift_stats(data, splits)

    # 6. Prepare splits.json
    n_cases = {
        "train": len(splits["train_cases"]),
        "val": len(splits["val_cases"]),
        "test": len(splits["test_cases"]),
    }
    n_transitions = {
        "train": int(splits["train_mask"].sum()),
        "val": int(splits["val_mask"].sum()),
        "test": int(splits["test_mask"].sum()),
    }

    splits_json = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "split_method": method,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "cases": {
            "train": [int(c) for c in splits["train_cases"]],
            "val": [int(c) for c in splits["val_cases"]],
            "test": [int(c) for c in splits["test_cases"]],
        },
        "n_cases": n_cases,
        "n_transitions": n_transitions,
        "sanity": {
            "pct_invalid_action_global": structure_stats["pct_invalid_action"],
            "non_terminal_reward_zero_pct_global": mdp_stats["non_terminal_reward_zero_pct"],
            "pct_done_global": float((data["done"] == 1).mean() * 100),
        },
        "range_checks": {
            "status": "skipped",
            "reason": "No tabular state_features in NPZ (only prefix tokens). "
            "Range checks apply only when features like amount, quality, cum_cost are stored.",
        },
    }

    # 7. Save splits.json
    ensure_dir(Path(splits_path).parent)
    save_json(splits_json, splits_path)
    logger.info("Saved splits to %s", splits_path)

    # 8. Compute pct_done per split
    for split_name in ["train", "val", "test"]:
        mask = splits[f"{split_name}_mask"]
        split_done = data["done"][mask]
        report_key = f"{split_name}_pct_done"
        drift_stats[report_key] = float(split_done.mean() * 100) if len(split_done) > 0 else 0.0

    # 9. Document range checks status (paper-proof: explicit about what was skipped)
    # Check if we have tabular state features in NPZ
    has_tabular_features = any(
        key in data
        for key in ["amount", "est_quality", "cum_cost", "mu_features", "state_features"]
    )
    if has_tabular_features:
        range_checks_status = {
            "status": "not_implemented",
            "reason": "Tabular features found but range checks not yet implemented",
        }
    else:
        range_checks_status = {
            "status": "skipped",
            "reason": "No tabular state_features in NPZ (only prefix tokens). "
            "Range checks apply only when features like amount, quality, cum_cost are stored.",
        }

    # 10. Prepare report
    report = {
        **structure_stats,
        **mdp_stats,
        **drift_stats,
        "n_cases": n_cases,
        "n_transitions": n_transitions,
        "range_checks": range_checks_status,
    }

    return report
