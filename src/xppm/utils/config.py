from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import load_yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dict. override values take precedence."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _resolve_strings(obj: Any, dataset_name: str) -> Any:
    """Recursively substitute {dataset_name} in all string values."""
    if isinstance(obj, str):
        return obj.replace("{dataset_name}", dataset_name)
    elif isinstance(obj, dict):
        return {k: _resolve_strings(v, dataset_name) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_strings(v, dataset_name) for v in obj]
    return obj


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

    @classmethod
    def for_dataset(cls, base_yaml: str | Path, dataset_name: str | None) -> "Config":
        """Load base config and optionally deep-merge a dataset-specific overlay.

        The overlay file is looked up at:
            <base_yaml_dir>/datasets/<dataset_name>.yaml

        After merging, ``{dataset_name}`` placeholders in all string values are
        resolved so path templates like ``data/{dataset_name}/interim`` expand
        automatically.

        If *dataset_name* is ``None`` the base config is returned unchanged
        (backward-compatible with the old ``Config.from_yaml`` call-site).
        """
        base_path = Path(base_yaml)
        raw = load_yaml(base_path)

        if dataset_name:
            # Attempt to load dataset-specific overlay
            dataset_yaml = base_path.parent / "datasets" / f"{dataset_name}.yaml"
            if dataset_yaml.exists():
                overlay = load_yaml(dataset_yaml)
                raw = deep_merge(raw, overlay)

            # Stamp dataset_name into the config dict
            raw["dataset_name"] = dataset_name

            # Resolve {dataset_name} placeholders in all string values
            raw = _resolve_strings(raw, dataset_name)

        normalized = yaml.safe_dump(raw, sort_keys=True, default_flow_style=False)
        config_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

        return cls(raw=raw, path=base_path, config_hash=config_hash)

    def resolve_paths(self, dataset_name: str) -> "Config":
        """Return a new Config with ``{dataset_name}`` substituted in all string values."""
        new_raw = _resolve_strings(self.raw, dataset_name)
        normalized = yaml.safe_dump(new_raw, sort_keys=True, default_flow_style=False)
        config_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return Config(raw=new_raw, path=self.path, config_hash=config_hash)

    def hash_config(self) -> str:
        """Compute hash of resolved config (incorporates params.yaml if referenced)."""
        if self.config_hash:
            return self.config_hash
        # Recompute if needed
        normalized = yaml.safe_dump(self.raw, sort_keys=True, default_flow_style=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
