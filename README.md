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

## Instalación rápida (ejemplo con uv)
```bash
uv venv
source .venv/bin/activate
uv pip install -e .