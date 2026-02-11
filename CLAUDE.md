# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xPPM-TDQN: Explainable Prescriptive Process Monitoring using Transformer-based Double Q-Network with offline reinforcement learning. An academic research pipeline for learning intervention policies from event logs, evaluating them off-policy, and generating explainable recommendations.

The codebase is in Spanish/English mixed documentation. Config and code are English; setup docs and comments are often in Spanish.

## Commands

### Installation
```bash
pip install -e .[dev]          # basic install with dev tools
pip install -e .[dev,tracking] # also install mlflow/wandb
```

### Pipeline (sequential, each step depends on previous)
```bash
make preprocess    # 01: event log -> clean.parquet
make build_rlset   # 02+03: encode prefixes + build MDP dataset (D_offline.npz)
# Then manually: python scripts/01b_validate_and_split.py --config configs/config.yaml --overwrite
make train         # 04: train TDQN offline -> Q_theta.ckpt
make ope           # 05: off-policy evaluation (doubly robust)
make xai           # 06: generate explanations
make distill       # 08: distill to decision tree
make serve         # run FastAPI policy server
```

All scripts accept `--config configs/config.yaml`. Some require `--overwrite` to regenerate existing outputs, and support `--dry-run` for validation without execution.

### Testing & Code Quality
```bash
pytest                         # run tests (slow tests excluded by default via addopts)
pytest -m slow                 # run only slow tests
pytest -m "not slow" -v        # verbose non-slow tests
pytest tests/unit/test_X.py    # run a single test file
ruff check .                   # lint
ruff check --fix .             # lint with auto-fix
mypy src                       # type check
pre-commit run --all-files     # all pre-commit hooks (ruff, formatting, yaml)
```

### DVC Pipeline
```bash
dvc repro    # rerun full pipeline based on dependency graph
dvc pull     # fetch versioned data from remote
dvc push     # upload artifacts to remote
```

## Architecture

### Two-layer structure: library (`src/xppm/`) vs CLI (`scripts/`)

- `src/xppm/` — importable library with all logic, organized by domain
- `scripts/01-08` — thin CLI entrypoints numbered to match the paper's phases, each calling into `src/xppm/`

### Pipeline phases (data flows top-to-bottom)

**Phase 1 — Data to Offline RL Dataset:**
`Event Log` → `preprocess` → `clean.parquet` → `encode_prefixes` → `prefixes.npz` + `vocab_activity.json` → `build_mdp` → `D_offline.npz` → `validate_and_split` → `splits.json`

**Phase 2 — Training & Evaluation:**
`D_offline.npz` + `splits.json` → `train_tdqn` → `Q_theta.ckpt` → `ope_dr` → `ope_dr.json`

**Phase 3 — Explainability & Deployment:**
`Q_theta.ckpt` → `explain_policy` (risk attributions, delta-Q, policy summary) → `fidelity_tests` → `distill_policy` (VIPER → decision tree) → `policy_server` (FastAPI + guard)

### Key modules in `src/xppm/`

- **`data/`** — preprocessing, prefix encoding (tokenization + padding), MDP dataset construction (states, actions, rewards, action masks), schema validation
- **`rl/`** — TDQN training loop (`train_tdqn.py`), offline replay buffer, evaluation metrics
- **`rl/models/`** — `transformer.py` (SimpleTransformerEncoder), `q_network.py` (MLP Q-head), `masking.py` (action mask → -inf on invalid actions)
- **`ope/`** — behavior policy estimation (multiclass logistic regression), doubly robust estimator with bootstrap CIs
- **`xai/`** — integrated gradients attributions, delta-Q contrastive explanations, fidelity tests (Q-drop, action-flip, rank-consistency), policy summary clustering
- **`distill/`** — VIPER-style distillation to decision tree, rule export to SQL
- **`serving/`** — FastAPI server, policy guard (uncertainty + OOD detection via Mahalanobis), fallback to no-op
- **`utils/`** — config loading with SHA-256 hashing, I/O helpers (parquet/npz/json/yaml), seed management, experiment tracking (W&B/MLflow)

### Configuration

All behavior is driven by `configs/config.yaml` (~295 lines). Key sections: `schema` (column mapping), `encoding` (max_len=50, vocab), `mdp` (actions: do_nothing/contact_headquarters, action masks by activity, terminal delayed reward), `training` (Double-DQN, cosine LR scheduler, batch_size=128), `ope`, `xai`, `fidelity`, `distill`, `serving`.

Hyperparameter defaults for DVC are in `params.yaml`. Sweep templates in `configs/sweeps/`.

### MDP Dataset Format (`D_offline.npz`)

- `s`: state prefix token IDs `[N, max_len=50]`
- `a`: action taken (0=do_nothing, 1=contact_headquarters)
- `r`: delayed terminal reward (0.0 intermediate, outcome at final state)
- `s_next`, `valid_actions` (binary mask), `propensity` (-1.0 placeholder until OPE step estimates it)

### Model Architecture

Embedding → TransformerEncoder (d_model=128, 4 heads, 3 layers) → MLP Q-head → Q-values per action. Action masking sets invalid actions to -inf before argmax. Stabilized with Double-DQN, target network (updated every 2000 steps), gradient clipping (norm=10), Huber loss.

## Code Style

- **Ruff**: line-length=100, target Python 3.10, select rules E/F/I
- **MyPy**: Python 3.10, ignore_missing_imports=True
- **Pre-commit**: ruff (lint+format), end-of-file-fixer, trailing-whitespace, check-yaml, max file 10MB
- Slow tests are marked with `@pytest.mark.slow` and excluded by default
