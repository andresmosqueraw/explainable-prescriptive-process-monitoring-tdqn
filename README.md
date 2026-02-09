## xPPM-TDQN

Pipeline and library for explainable prescriptive process monitoring with TDQN and offline RL.

**Design principles**

- `src/xppm/`: reusable library code (data, RL, OPE, XAI, distillation, serving).
- `scripts/`: thin CLI entrypoints (01–08 + `policy_server.py`) matching the paper figures.
- Config-driven: `configs/config.yaml` + `params.yaml`.
- Reproducible pipelines: `dvc.yaml` (data → RL set → training → OPE).

**Quick start**

```bash
pip install -e .

python scripts/01_preprocess_log.py --config configs/config.yaml
python scripts/02_encode_prefixes.py --config configs/config.yaml
python scripts/03_build_mdp_dataset.py --config configs/config.yaml
python scripts/04_train_tdqn_offline.py --config configs/config.yaml
python scripts/05_run_ope_dr.py --config configs/config.yaml
python scripts/06_explain_policy.py --config configs/config.yaml
python scripts/07_fidelity_tests.py --config configs/config.yaml
python scripts/08_distill_policy.py --config configs/config.yaml
```


# Explainable Prescriptive Process Monitoring (Offline TDQN + OPE + XAI)

Repo para:
1) Construir dataset offline RL desde event logs (Phase 1)
2) Entrenar TDQN offline + evaluar con OPE DR (Phase 2)
3) Generar explicaciones + tests de fidelidad + distillation + deploy (Phase 3)

## Requisitos
- Python 3.10+
- (Opcional) CUDA para entrenamiento
- Recomendado: `uv` o `poetry` o `conda`
- (Opcional) W&B o MLflow para tracking
- (Opcional) DVC para versionar data/artifacts

## Instalación reproducible

### Opción 1: uv (recomendado)
```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
# Lock file: uv.lock (generado automáticamente)
```

### Opción 2: poetry
```bash
poetry install
# Lock file: poetry.lock
```

### Opción 3: pip (básico)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
# Para reproducibilidad: pip freeze > requirements-lock.txt
```

**Nota:** Para reproducibilidad completa, usa `uv.lock` o `poetry.lock` en lugar de instalar desde rangos sueltos.

## Data versioning (DVC)

Large data files are versioned with DVC (not in Git). Set up remote storage:

### Set remote (example local):
```bash
mkdir -p /mnt/dvc-store
dvc remote add -d storage /mnt/dvc-store
dvc remote modify storage type local
```

### Get exact data for this commit:
```bash
dvc pull
```

### Recompute pipeline:
```bash
dvc repro
dvc push
```

### Tracked files:
- `data/interim/clean.parquet`
- `data/processed/D_offline.npz`
- `data/processed/splits.json`

See `dvc.yaml` for the full pipeline definition.

**Note on `propensity` field in `D_offline.npz`:**
The `propensity` field (behavior policy μ(a|s)) is set to `-1.0` as a placeholder
during dataset building. It will be estimated in `05_run_ope_dr.py` using
`behavior_model.py`. See `configs/schemas/offline_rlset.schema.json` for details.

## Building MDP Dataset (Step 3)

### Regenerate real dataset (official command)

To rebuild the MDP dataset from scratch (requires `--overwrite` flag):

```bash
python scripts/03_build_mdp_dataset.py --config configs/config.yaml --overwrite
```

**Important:** The `--overwrite` flag is required to prevent accidental overwrites
of real datasets. This ensures you explicitly confirm when regenerating.

### Validate inputs without building (dry-run)

To validate that all input files exist and config is correct without building:

```bash
python scripts/03_build_mdp_dataset.py --config configs/config.yaml --dry-run
```

This checks:
- Prefixes file exists (`data/interim/prefixes.npz`)
- Clean log exists (`data/interim/clean.parquet`)
- Vocabulary exists (`data/interim/vocab_activity.json`)
- Config is valid

### First-time build (no overwrite needed)

If the output file doesn't exist, you can build without `--overwrite`:

```bash
python scripts/03_build_mdp_dataset.py --config configs/config.yaml
```

## Experiment tracking

Configure tracking in `configs/config.yaml`:

```yaml
tracking:
  enabled: true
  backend: wandb  # or mlflow
  wandb:
    project: "xppm-tdqn"
  mlflow:
    experiment_name: "xppm-tdqn"
```

Each run logs: git commit, config hash, DVC data hashes, metrics, and artifacts.
