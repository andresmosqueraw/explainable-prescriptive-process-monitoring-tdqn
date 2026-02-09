from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import load_yaml


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path | None = None
    config_hash: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML and compute hash for reproducibility."""
        path_obj = Path(path)
        raw = load_yaml(path_obj)
        
        # Normalize config (sort keys) and compute hash
        normalized = yaml.safe_dump(raw, sort_keys=True, default_flow_style=False)
        config_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        
        return cls(raw=raw, path=path_obj, config_hash=config_hash)

    def hash_config(self) -> str:
        """Compute hash of resolved config (incorporates params.yaml if referenced)."""
        if self.config_hash:
            return self.config_hash
        # Recompute if needed
        normalized = yaml.safe_dump(self.raw, sort_keys=True, default_flow_style=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]



