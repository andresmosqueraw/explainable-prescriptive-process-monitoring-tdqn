from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_ope_report(metrics: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


