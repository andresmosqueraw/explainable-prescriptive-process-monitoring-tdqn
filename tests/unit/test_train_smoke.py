from __future__ import annotations

import numpy as np
import pytest
import torch

from xppm.data.build_mdp import build_mdp_dataset
from xppm.data.encode_prefixes import encode_prefixes
from xppm.data.preprocess import preprocess_event_log
from xppm.rl.train_tdqn import TDQNConfig, train_tdqn
from xppm.utils.io import load_npz, save_npz


def test_training_smoke(tiny_log_path, tmp_outdir):
    """Smoke test: training should run end-to-end without errors."""
    # Build minimal dataset
    clean_path = tmp_outdir / "clean.parquet"
    prefixes_path = tmp_outdir / "prefixes.npz"
    mdp_path = tmp_outdir / "D_offline.npz"
    splits_path = tmp_outdir / "splits.json"

    preprocess_event_log(tiny_log_path, clean_path)
    encode_prefixes(clean_path, prefixes_path)
    build_mdp_dataset(prefixes_path, mdp_path, splits_path)

    # Load dataset
    D = load_npz(mdp_path)

    # Subsample to small size for smoke test (max 100 transitions)
    n = min(100, len(D["actions"]))
    if n == 0:
        pytest.skip("No transitions in dataset")

    # Create small dataset
    D_small_path = tmp_outdir / "D_small.npz"
    save_npz(
        D_small_path,
        states=D["states"][:n],
        actions=D["actions"][:n],
        rewards=D["rewards"][:n],
        next_states=D["next_states"][:n],
        dones=D["dones"][:n],
    )

    # Train with minimal config
    config = TDQNConfig(
        state_dim=int(D["states"].shape[-1]) if D["states"].ndim > 1 else 1,
        n_actions=int(D["actions"].max() + 1) if len(D["actions"]) > 0 else 2,
        hidden_dim=32,  # Small for smoke test
        gamma=0.99,
        learning_rate=1e-3,
        batch_size=min(32, n),  # Don't exceed dataset size
        max_epochs=1,  # Just 1 epoch for smoke test
    )

    checkpoint_path = tmp_outdir / "checkpoint.ckpt"
    metadata_path = tmp_outdir / "metadata.json"

    # Run training (should not crash)
    metrics = train_tdqn(
        config=config,
        dataset_path=D_small_path,
        checkpoint_path=checkpoint_path,
        seed=42,
        deterministic=False,  # Faster for smoke test
        config_hash="test",
        metadata_output=metadata_path,
        tracking_config=None,  # No tracking for smoke test
    )

    # Check metrics returned
    assert metrics is not None, "Training should return metrics"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"

    # Check loss is finite
    if "final_loss" in metrics:
        assert np.isfinite(metrics["final_loss"]), "Final loss should be finite"

    # Check checkpoint was created
    assert checkpoint_path.exists(), "Checkpoint should be created"

    # Verify checkpoint can be loaded
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    assert isinstance(state_dict, dict), "Checkpoint should be a state dict"
