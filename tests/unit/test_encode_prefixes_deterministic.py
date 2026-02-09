from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xppm.data.encode_prefixes import encode_prefix_dataset
from xppm.utils.io import load_npz


def test_encode_deterministic(tmp_path: Path) -> None:
    """Test that encoding is deterministic: same input → same output."""
    # Create minimal synthetic log
    df = pd.DataFrame(
        {
            "case_id": [0, 0, 0, 1, 1],
            "activity": ["A", "B", "C", "A", "D"],
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"]
            ),
        }
    )

    clean_path = tmp_path / "clean.parquet"
    df.to_parquet(clean_path, index=False)

    config = {
        "encoding": {
            "max_len": 10,
            "min_prefix_len": 1,
            "truncation": "left",
            "padding": "left",
            "vocab": {"min_freq": 1, "add_unk": True, "add_pad": True},
            "fields": {"activity_col": "activity"},
            "output": {
                "prefixes_path": str(tmp_path / "prefixes1.npz"),
                "vocab_activity_path": str(tmp_path / "vocab1.json"),
            },
        }
    }

    # Run encoding twice
    encode_prefix_dataset(clean_path, config)

    config["encoding"]["output"]["prefixes_path"] = str(tmp_path / "prefixes2.npz")
    config["encoding"]["output"]["vocab_activity_path"] = str(tmp_path / "vocab2.json")
    encode_prefix_dataset(clean_path, config)

    # Load both outputs
    data1 = load_npz(tmp_path / "prefixes1.npz")
    data2 = load_npz(tmp_path / "prefixes2.npz")

    # Compare
    assert np.array_equal(data1["X"], data2["X"]), "X should be identical"
    assert np.array_equal(data1["mask"], data2["mask"]), "mask should be identical"
    assert np.array_equal(data1["case_ptr"], data2["case_ptr"]), "case_ptr should be identical"
    assert np.array_equal(data1["t_ptr"], data2["t_ptr"]), "t_ptr should be identical"

    # Compare vocabularies
    vocab1 = json.load(open(tmp_path / "vocab1.json"))
    vocab2 = json.load(open(tmp_path / "vocab2.json"))
    assert vocab1["token2id"] == vocab2["token2id"], "Vocabulary should be identical"


def test_padding_truncation(tmp_path: Path) -> None:
    """Test that padding and truncation work correctly."""
    # Create case with 5 events, max_len=3
    df = pd.DataFrame(
        {
            "case_id": [0, 0, 0, 0, 0],
            "activity": ["A", "B", "C", "D", "E"],
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )

    clean_path = tmp_path / "clean.parquet"
    df.to_parquet(clean_path, index=False)

    config = {
        "encoding": {
            "max_len": 3,
            "min_prefix_len": 1,
            "truncation": "left",  # Keep last 3 events
            "padding": "left",
            "vocab": {"min_freq": 1, "add_unk": True, "add_pad": True},
            "fields": {"activity_col": "activity"},
            "output": {
                "prefixes_path": str(tmp_path / "prefixes.npz"),
                "vocab_activity_path": str(tmp_path / "vocab.json"),
            },
        }
    }

    encode_prefix_dataset(clean_path, config)
    data = load_npz(tmp_path / "prefixes.npz")

    # Check final prefix (t=5) should have last 3 activities: C, D, E
    # Find prefix with t_ptr=5 (or the last one)
    final_idx = len(data["t_ptr"]) - 1
    final_X = data["X"][final_idx]
    final_mask = data["mask"][final_idx]

    # With left truncation, last 3 should be C, D, E
    # Check that mask has exactly 3 ones (true_len=3, max_len=3)
    assert final_mask.sum() == 3, f"Mask should have 3 ones, got {final_mask.sum()}"

    # Check that the sequence contains the expected tokens (order depends on padding)
    # With left padding, tokens should be on the right
    non_pad_tokens = final_X[final_mask == 1]
    assert len(non_pad_tokens) == 3, "Should have 3 non-padding tokens"


def test_vocab_stable(tmp_path: Path) -> None:
    """Test that vocabulary is stable (deterministic ordering)."""
    df = pd.DataFrame(
        {
            "case_id": [0, 0, 1, 1, 2],
            "activity": ["A", "B", "A", "C", "B"],  # A:2, B:2, C:1
            "timestamp": pd.to_datetime(["2024-01-01"] * 5),
        }
    )

    clean_path = tmp_path / "clean.parquet"
    df.to_parquet(clean_path, index=False)

    config = {
        "encoding": {
            "max_len": 10,
            "min_prefix_len": 1,
            "truncation": "left",
            "padding": "left",
            "vocab": {"min_freq": 1, "add_unk": True, "add_pad": True},
            "fields": {"activity_col": "activity"},
            "output": {
                "prefixes_path": str(tmp_path / "prefixes.npz"),
                "vocab_activity_path": str(tmp_path / "vocab.json"),
            },
        }
    }

    encode_prefix_dataset(clean_path, config)
    vocab = json.load(open(tmp_path / "vocab.json"))

    # Check that vocabulary is deterministic (sorted by frequency, then alphabetically)
    # A and B both have freq 2, so should be sorted alphabetically: A, B, then C
    id2token = vocab["id2token"]
    # Skip special tokens (PAD, UNK)
    real_tokens = [t for t in id2token if t not in ["<PAD>", "<UNK>"]]
    assert real_tokens == ["A", "B", "C"], f"Expected ['A', 'B', 'C'], got {real_tokens}"
