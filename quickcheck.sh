#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .[dev]

pytest

python -m src.cli --help || true
python scripts/04_train_tdqn_offline.py --help || true

echo "Quick check completed."


