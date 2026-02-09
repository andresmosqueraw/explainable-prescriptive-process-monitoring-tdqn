from __future__ import annotations

import argparse
import json
from pathlib import Path

from xppm.data.encode_prefixes import encode_prefix_dataset
from xppm.utils.config import Config
from xppm.utils.io import fingerprint_data, get_git_commit
from xppm.utils.logging import (
    ensure_dir,
    finalize_run_metadata,
    get_logger,
    init_tracker,
    save_run_metadata,
    start_run_metadata,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode prefixes from clean log (Phase 1 - Step 2)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    # Load config
    config_obj = Config.from_yaml(args.config)
    cfg = config_obj.raw

    # Setup reproducibility
    seed = cfg.get("repro", {}).get("seed", 42)
    deterministic = cfg.get("repro", {}).get("deterministic", False)

    # Get paths from config
    paths_cfg = cfg.get("paths", {})
    encoding_cfg = cfg.get("encoding", {})
    output_cfg = encoding_cfg.get("output", {})

    clean_log_path = cfg.get("data", {}).get("output_clean_path", "data/interim/clean.parquet")
    prefixes_path = Path(output_cfg.get("prefixes_path", "data/interim/prefixes.npz"))
    vocab_path = Path(output_cfg.get("vocab_activity_path", "data/interim/vocab_activity.json"))

    # Initialize tracking
    tracking_cfg = cfg.get("tracking", {})
    tracker = None
    if tracking_cfg.get("enabled", False):
        tracker = init_tracker(tracking_cfg)
        if tracker.enabled:
            git_info = get_git_commit()
            run_name = f"encode_prefixes_{Path(clean_log_path).stem}"
            commit_hash = git_info["commit"]
            commit_short = commit_hash[:8] if isinstance(commit_hash, str) else "unknown"
            tags = {
                "stage": "phase1_encode",
                "dataset": Path(clean_log_path).stem,
                "git_commit": commit_short,
                "git_dirty": str(git_info["dirty"]),
                "config_hash": config_obj.hash_config()[:8],
            }
            # Add clean log hash
            try:
                clean_log_hash = fingerprint_data([clean_log_path])[:16]
                tags["clean_log_hash"] = clean_log_hash
            except Exception as e:
                logger.warning("Could not compute clean log hash: %s", e)
                tags["clean_log_hash"] = "unknown"

            params = {
                "max_len": encoding_cfg.get("max_len", 50),
                "min_prefix_len": encoding_cfg.get("min_prefix_len", 1),
                "padding_side": encoding_cfg.get("padding", "left"),
                "truncation_side": encoding_cfg.get("truncation", "left"),
                "vocab_min_freq": encoding_cfg.get("vocab", {}).get("min_freq", 1),
            }
            tracker.init_run(run_name=run_name, stage="phase1_encode", tags=tags, params=params)

    # Start run metadata
    artifacts_dir = Path(paths_cfg.get("artifacts_dir", "artifacts"))
    reports_dir = artifacts_dir / "reports"
    ensure_dir(reports_dir)

    metadata = start_run_metadata(
        stage="encode",
        config_path=args.config,
        config_hash=config_obj.hash_config(),
        seed=seed,
        deterministic=deterministic,
        data_fingerprint=(
            fingerprint_data([clean_log_path]) if Path(clean_log_path).exists() else None
        ),
    )

    try:
        # Run encoding
        logger.info("Starting prefix encoding: %s -> %s", clean_log_path, prefixes_path)
        stats = encode_prefix_dataset(clean_log_path, cfg)

        # Log metrics to tracker
        if tracker and tracker.enabled:
            tracker.log_metrics(
                {
                    "n_cases": stats.get("n_cases", 0),
                    "n_prefixes": stats.get("n_prefixes", 0),
                    "avg_prefix_len": stats.get("prefix_length_mean", 0),
                    "p95_prefix_len": stats.get("prefix_length_p95", 0),
                    "unk_tokens_pct": stats.get("unk_tokens_pct", 0),
                    "vocab_size": stats.get("vocab_size", 0),
                }
            )

        # Save encoding report
        report_path = reports_dir / "prefix_encoding_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info("Saved encoding report to %s", report_path)

        # Log artifacts to tracker
        if tracker and tracker.enabled and tracking_cfg.get("log_artifacts", True):
            tracker.log_artifact(str(prefixes_path), artifact_path="data")
            tracker.log_artifact(str(vocab_path), artifact_path="vocab")
            tracker.log_artifact(str(report_path), artifact_path="reports")
            if tracking_cfg.get("log_config", True):
                tracker.log_artifact(args.config, artifact_path="config")

        # Finalize metadata
        metadata = finalize_run_metadata(
            metadata,
            outputs=[str(prefixes_path), str(vocab_path), str(report_path)],
            metrics=stats,
        )
        metadata["status"] = "completed"

        # Save run metadata
        run_metadata_path = reports_dir / "run_metadata_encode.json"
        save_run_metadata(metadata, run_metadata_path)

        logger.info("✅ Prefix encoding completed successfully")
        logger.info("   Output: %s", prefixes_path)
        logger.info(
            "   Stats: %d prefixes, vocab_size=%d, unk_pct=%.2f%%",
            stats.get("n_prefixes", 0),
            stats.get("vocab_size", 0),
            stats.get("unk_tokens_pct", 0),
        )

    except Exception as e:
        logger.error("❌ Prefix encoding failed: %s", e, exc_info=True)
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        run_metadata_path = reports_dir / "run_metadata_encode.json"
        save_run_metadata(metadata, run_metadata_path)
        if tracker and tracker.enabled:
            tracker.finish()
        raise

    finally:
        if tracker and tracker.enabled:
            tracker.finish()


if __name__ == "__main__":
    main()
