from __future__ import annotations

import json
import logging
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def get_logger(name: str = "xppm") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_env_fingerprint() -> dict[str, Any]:
    """Capture environment fingerprint (Python, packages, GPU info)."""
    env = {
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    
    # PyTorch info
    env["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        env["cuda_available"] = True
        env["cuda_version"] = torch.version.cuda
        env["cudnn_version"] = torch.backends.cudnn.version()
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        env["cuda_available"] = False
    
    return env


def start_run_metadata(
    stage: str,
    config_path: str | Path,
    config_hash: str,
    seed: int,
    deterministic: bool,
    data_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Initialize run metadata at start of a stage.
    
    Args:
        stage: Stage name (train, ope, xai, etc.)
        config_path: Path to config file
        config_hash: Hash of config
        seed: Random seed used
        deterministic: Whether deterministic mode was enabled
        data_fingerprint: Hash/fingerprint of data files
    
    Returns:
        Metadata dict (to be finalized with finalize_run_metadata)
    """
    from .io import get_git_commit
    
    git_info = get_git_commit()
    run_id = f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    metadata = {
        "run_id": run_id,
        "stage": stage,
        "started_at": datetime.now().isoformat(),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "data_fingerprint": data_fingerprint,
        "seed": seed,
        "deterministic": deterministic,
        "env": get_env_fingerprint(),
    }
    
    return metadata


def finalize_run_metadata(
    metadata: dict[str, Any],
    outputs: list[str | Path] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize run metadata with completion time and outputs.
    
    Args:
        metadata: Metadata dict from start_run_metadata
        outputs: List of output file paths created
        metrics: Optional metrics dict to include
    
    Returns:
        Finalized metadata dict
    """
    metadata["finished_at"] = datetime.now().isoformat()
    if outputs:
        metadata["outputs"] = [str(p) for p in outputs]
    if metrics:
        metadata["metrics"] = metrics
    
    return metadata


def save_run_metadata(metadata: dict[str, Any], output_path: str | Path) -> None:
    """Save run metadata to JSON file."""
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


