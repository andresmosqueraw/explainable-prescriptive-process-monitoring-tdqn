import json
from pathlib import Path

import numpy as np
import torch

from xppm.ope.behavior_model import BehaviorPolicy
from xppm.ope.doubly_robust import doubly_robust_estimate
from xppm.ope.report import save_ope_report
from xppm.rl.train_tdqn import TransformerQNetwork


def test_ope_dr_smoke(tmp_path: Path) -> None:
    N = 4  # total transitions
    max_len = 3
    n_actions = 2
    # vocab_size derived from s.max()+1 when token2id is empty
    vocab_size = 1  # s will be all-zeros

    d_model = 16
    n_heads = 2
    n_layers = 1

    rng = np.random.default_rng(0)

    # --- D_offline.npz (keys expected by doubly_robust_estimate) ---
    s = np.zeros((N, max_len), dtype=np.int32)  # all-zero tokens
    s_mask = np.ones((N, max_len), dtype=np.uint8)
    a = rng.integers(0, n_actions, size=N, dtype=np.int32)
    r = rng.standard_normal(N).astype(np.float32)
    valid_actions = np.ones((N, n_actions), dtype=np.uint8)
    case_ptr = np.array([0, 0, 1, 1], dtype=np.int32)  # 2 cases × 2 transitions

    data_path = tmp_path / "D_offline.npz"
    np.savez_compressed(
        data_path,
        s=s,
        s_mask=s_mask,
        a=a,
        r=r,
        valid_actions=valid_actions,
        case_ptr=case_ptr,
    )

    # --- splits.json — both cases in test ---
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"cases": {"test": [0, 1]}}))

    # --- vocab_activity.json — empty token2id so loader uses s.max()+1 ---
    vocab_path = tmp_path / "vocab_activity.json"
    vocab_path.write_text(json.dumps({"token2id": {}, "id2token": []}))

    # --- TransformerQNetwork checkpoint ---
    q_net = TransformerQNetwork(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,
        n_actions=n_actions,
    )
    ckpt = tmp_path / "Q_theta.ckpt"
    torch.save({"model_state_dict": q_net.state_dict()}, ckpt)

    # --- BehaviorPolicy — uniform probs over all N transitions ---
    probs = np.full((N, n_actions), 1.0 / n_actions, dtype=np.float32)
    behavior = BehaviorPolicy(probs=probs, n_actions=n_actions, metrics={})

    # --- Minimal config ---
    config = {
        "repro": {"seed": 42},
        "training": {
            "transformer": {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "max_len": max_len,
                "dropout": 0.0,
            },
            "tdqn": {"gamma": 0.99},
        },
        "ope": {"pi_e_temperature": 1.0, "evaluate_heuristic": False},
        "mdp": {
            "actions": {
                "id2name": ["do_nothing", "act"],
                "noop_action": "do_nothing",
            }
        },
    }

    metrics = doubly_robust_estimate(
        ckpt, data_path, splits_path, vocab_path, config, behavior, n_bootstrap=5
    )
    out = tmp_path / "ope_dr.json"
    save_ope_report(metrics, out)

    assert "results" in metrics
    assert "tdqn_dr_mean" in metrics["results"]
    assert out.exists()
