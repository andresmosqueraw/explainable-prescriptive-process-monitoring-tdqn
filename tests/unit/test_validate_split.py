"""Tests for validate_and_split (Phase 1 - Step 01b)."""

from __future__ import annotations

import numpy as np
import pytest

from xppm.data.build_mdp import build_mdp_dataset
from xppm.data.encode_prefixes import encode_prefix_dataset
from xppm.data.preprocess import preprocess_event_log
from xppm.data.validate_split import (
    compute_drift_stats,
    split_by_case,
    validate_and_split_dataset,
    validate_mdp_logic,
    validate_npz_structure,
)
from xppm.utils.io import load_json, load_npz


@pytest.fixture
def test_config() -> dict:
    """Minimal test configuration for validation and splitting."""
    return {
        "mdp": {
            "actions": {
                "id2name": ["do_nothing", "contact_headquarters"],
            },
            "decision_points": {
                "mode": "all",
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
        "repro": {
            "seed": 42,
            "deterministic": False,
        },
    }


def test_validate_npz_structure_valid(tiny_log_with_outcome_path, tmp_outdir, test_config):
    """Test that validate_npz_structure passes for valid dataset."""
    # Build pipeline: preprocess -> encode -> build_mdp
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    # Load and validate
    data = load_npz(mdp_path)
    stats = validate_npz_structure(data)

    assert stats["n_transitions"] > 0
    assert stats["n_cases"] > 0
    assert stats["pct_invalid_action"] == 0.0


def test_validate_npz_structure_missing_array(tmp_outdir):
    """Test that validate_npz_structure fails for missing arrays."""
    # Create invalid dataset (missing 'a')
    data = {
        "s": np.array([[1, 2, 3]], dtype=np.int32),
        "s_mask": np.array([[1, 1, 1]], dtype=np.uint8),
        # Missing 'a'
        "r": np.array([0.0], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="Missing required arrays"):
        validate_npz_structure(data)


def test_validate_npz_structure_invalid_action(tmp_outdir):
    """Test that validate_npz_structure fails for invalid actions."""
    # Create dataset with invalid action
    data = {
        "s": np.array([[1, 2, 3]], dtype=np.int32),
        "s_mask": np.array([[1, 1, 1]], dtype=np.uint8),
        "a": np.array([1], dtype=np.int32),  # Action 1
        "r": np.array([0.0], dtype=np.float32),
        "s_next": np.array([[1, 2, 3]], dtype=np.int32),
        "s_next_mask": np.array([[1, 1, 1]], dtype=np.uint8),
        "done": np.array([0], dtype=np.uint8),
        "case_ptr": np.array([1], dtype=np.int32),
        "t_ptr": np.array([1], dtype=np.int32),
        "valid_actions": np.array([[1, 0]], dtype=np.uint8),  # Only action 0 is valid
    }

    with pytest.raises(ValueError, match="Found.*transitions with invalid actions"):
        validate_npz_structure(data)


def test_validate_mdp_logic_reward_sanity(tiny_log_with_outcome_path, tmp_outdir, test_config):
    """Test that validate_mdp_logic checks reward sanity."""
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    data = load_npz(mdp_path)
    stats = validate_mdp_logic(data)

    assert stats["non_terminal_reward_zero_pct"] == 100.0


def test_split_by_case_disjoint(tiny_log_with_outcome_path, tmp_outdir, test_config):
    """Test that split_by_case produces disjoint splits."""
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    data = load_npz(mdp_path)
    splits = split_by_case(
        data["case_ptr"],
        method="random_case",
        seed=42,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )

    # Check no overlap
    train_cases = set(splits["train_cases"])
    val_cases = set(splits["val_cases"])
    test_cases = set(splits["test_cases"])

    assert not (train_cases & val_cases), "Train and val cases overlap"
    assert not (train_cases & test_cases), "Train and test cases overlap"
    assert not (val_cases & test_cases), "Val and test cases overlap"

    # Check all transitions covered
    total = splits["train_mask"].sum() + splits["val_mask"].sum() + splits["test_mask"].sum()
    assert total == len(data["case_ptr"]), "Split doesn't cover all transitions"


def test_split_by_case_no_leakage(tiny_log_with_outcome_path, tmp_outdir, test_config):
    """Test that all transitions of a case fall in the same split (no leakage)."""
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    data = load_npz(mdp_path)
    splits = split_by_case(
        data["case_ptr"],
        method="random_case",
        seed=42,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )

    # Check 5 random cases: all their transitions should be in same split
    unique_cases = np.unique(data["case_ptr"])
    sample_cases = unique_cases[: min(5, len(unique_cases))]

    for case_id in sample_cases:
        case_mask = data["case_ptr"] == case_id

        in_train = splits["train_mask"][case_mask].all()
        in_val = splits["val_mask"][case_mask].all()
        in_test = splits["test_mask"][case_mask].all()

        # Each case should be in exactly one split
        assert (
            (in_train and not in_val and not in_test)
            or (not in_train and in_val and not in_test)
            or (not in_train and not in_val and in_test)
        ), f"Case {case_id} has transitions in multiple splits (leakage)"


def test_compute_drift_stats(tiny_log_with_outcome_path, tmp_outdir, test_config):
    """Test that compute_drift_stats produces numbers without errors."""
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    data = load_npz(mdp_path)
    splits = split_by_case(
        data["case_ptr"],
        method="random_case",
        seed=42,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )

    drift_stats = compute_drift_stats(data, splits)

    # Check that drift stats are computed
    assert "train_episode_len_mean" in drift_stats
    assert "test_episode_len_mean" in drift_stats
    assert "drift_flag_episode_len" in drift_stats
    assert isinstance(drift_stats["drift_flag_episode_len"], bool)


def test_validate_and_split_dataset_end_to_end(tiny_log_with_outcome_path, tmp_outdir, test_config):
    """Test end-to-end validate_and_split_dataset."""
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    vocab_path = tmp_outdir / "vocab_activity.json"
    mdp_path = tmp_outdir / "D_offline.npz"
    splits_path = tmp_outdir / "splits.json"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path, test_config)
    test_config["encoding"]["output"]["prefixes_path"] = str(prefixes_path)
    test_config["encoding"]["output"]["vocab_activity_path"] = str(vocab_path)
    encode_prefix_dataset(clean_path, test_config)
    build_mdp_dataset(prefixes_path, clean_path, vocab_path, mdp_path, test_config["mdp"])

    # Add split config
    test_config["validation_split"] = {
        "split_strategy": "case_id",
        "ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
    }

    # Run validation and splitting
    report = validate_and_split_dataset(mdp_path, splits_path, test_config)

    # Check splits.json exists and is valid
    assert splits_path.exists()
    splits = load_json(splits_path)

    assert "version" in splits
    assert "cases" in splits
    assert "n_cases" in splits
    assert "n_transitions" in splits
    assert "sanity" in splits

    # Check report has expected keys
    assert "n_cases" in report
    assert "n_transitions" in report
    assert "train_episode_len_mean" in report
