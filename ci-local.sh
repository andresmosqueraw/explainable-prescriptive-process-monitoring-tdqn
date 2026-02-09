#!/usr/bin/env bash
set -euo pipefail

# Local CI script that replicates GitHub Actions CI workflow
# Run this before pushing to verify everything passes

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "🔍 Running local CI checks..."
echo ""

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
python -m pip install --upgrade pip --quiet
pip install -e .[dev] --quiet

# Lint with ruff
echo "🔍 Running ruff linting..."
ruff check .

# Type check with mypy
echo "🔍 Running mypy type checking..."
mypy src

# Run tests
echo "🧪 Running tests..."
pytest

echo ""
echo "✅ All CI checks passed!"

