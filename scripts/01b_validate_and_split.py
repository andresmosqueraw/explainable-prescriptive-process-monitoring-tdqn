"""Validate and split MDP dataset (Phase 1 - Step 01b)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xppm.data.validate_split import validate_and_split_dataset
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
        description="Validate D_offline.npz and create train/val/test splits (Phase 1 - Step 01b)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. Loads configs/datasets/{name}.yaml on top of --config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if it exists (use with caution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and config without creating splits (no writes)",
    )
    args = parser.parse_args()

    # Load config
    config_obj = Config.for_dataset(args.config, args.dataset)
    cfg = config_obj.raw

    # Setup reproducibility
    seed = cfg.get("repro", {}).get("seed", 42)
    deterministic = cfg.get("repro", {}).get("deterministic", False)

    # Get paths from config
    paths_cfg = cfg.get("paths", {})
    split_cfg = cfg.get("validation_split", {})
    mdp_cfg = cfg.get("mdp", {})

    npz_path = mdp_cfg.get("output", {}).get("path", "data/processed/D_offline.npz")
    splits_path = Path(paths_cfg.get("data_processed_dir", "data/processed")) / split_cfg.get(
        "out_splits_json", "splits.json"
    )

    # Dry-run mode: validate inputs without creating splits
    if args.dry_run:
        logger.info("DRY-RUN mode: Validating inputs without creating splits")
        # Validate paths exist (no file creation, no tracking, no artifacts)
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"MDP dataset not found: {npz_path}")

        # Compute hashes for validation (read-only, no writes)
        try:
            config_hash = config_obj.hash_config()
            npz_hash = fingerprint_data([npz_path])[:16]
            logger.info("✅ All input files exist and are valid")
            logger.info("   MDP dataset: %s (hash: %s)", npz_path, npz_hash)
            logger.info("   Config hash: %s", config_hash[:16])
            logger.info("   Splits would be: %s", splits_path)
            logger.info("   (No files created, no tracking, no artifacts)")
        except Exception as e:
            logger.warning("Could not compute hashes: %s", e)
            logger.info("✅ MDP dataset exists: %s", npz_path)
            logger.info("   Splits would be: %s", splits_path)
        return

    # Protection: don't overwrite real splits accidentally
    splits_path_obj = Path(splits_path).resolve()
    if splits_path_obj.exists():
        # Strict check: allow overwrite only if in safe test directories
        output_str = str(splits_path_obj)
        is_safe_test_path = (
            "/tmp/pytest-" in output_str
            or "/tmp/tmp" in output_str
            or "/.tmp/" in output_str
            or output_str.endswith("/.tmp")
            or "/artifacts/tests/" in output_str
            or output_str.endswith("/artifacts/tests")
        )

        if not is_safe_test_path:
            # For real runs, require explicit --overwrite flag
            if not args.overwrite:
                raise FileExistsError(
                    f"Output file {splits_path} already exists. "
                    f"To overwrite, use --overwrite flag. "
                    f"This prevents accidental overwrites of real splits."
                )
            else:
                logger.warning(
                    "Overwriting existing splits: %s (--overwrite flag provided)",
                    splits_path,
                )

    # Initialize tracking
    tracking_cfg = cfg.get("tracking", {})
    tracker = None
    if tracking_cfg.get("enabled", False):
        tracker = init_tracker(tracking_cfg)
        if tracker and tracker.enabled:
            git_info = get_git_commit()
            run_name = f"validate_split_{Path(npz_path).stem}"
            commit_hash = git_info["commit"]
            commit_short = commit_hash[:8] if isinstance(commit_hash, str) else "unknown"
            tags = {
                "stage": "phase1_validate_split",
                "dataset": Path(npz_path).stem,
                "git_commit": commit_short,
                "git_dirty": str(git_info["dirty"]),
                "config_hash": config_obj.hash_config()[:8],
            }
            # Add data hash
            try:
                npz_hash = fingerprint_data([npz_path])[:16]
                tags["npz_hash"] = npz_hash
            except Exception as e:
                logger.warning("Could not compute data hash: %s", e)

            params = {
                "split_method": split_cfg.get("split_strategy", "case_id"),
                "train_ratio": split_cfg.get("ratios", {}).get("train", 0.7),
                "val_ratio": split_cfg.get("ratios", {}).get("val", 0.1),
                "test_ratio": split_cfg.get("ratios", {}).get("test", 0.2),
                "seed": seed,
            }
            tracker.init_run(
                run_name=run_name, stage="phase1_validate_split", tags=tags, params=params
            )

    # Start run metadata
    artifacts_dir = Path(paths_cfg.get("artifacts_dir", "artifacts"))
    reports_dir = artifacts_dir / "reports"
    ensure_dir(reports_dir)

    metadata = start_run_metadata(
        stage="validate_split",
        config_path=args.config,
        config_hash=config_obj.hash_config(),
        seed=seed,
        deterministic=deterministic,
        data_fingerprint=(fingerprint_data([npz_path]) if Path(npz_path).exists() else None),
    )

    try:
        # Run validation and splitting
        logger.info("Starting validation and splitting: %s -> %s", npz_path, splits_path)
        report = validate_and_split_dataset(npz_path, splits_path, cfg)

        # Save split report
        report_path = reports_dir / "split_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Saved split report to %s", report_path)

        # Log metrics to tracker
        if tracker and tracker.enabled:
            # Split metrics
            metrics = {
                "n_cases_train": report.get("n_cases", {}).get("train", 0),
                "n_cases_val": report.get("n_cases", {}).get("val", 0),
                "n_cases_test": report.get("n_cases", {}).get("test", 0),
                "n_transitions_train": report.get("n_transitions", {}).get("train", 0),
                "n_transitions_val": report.get("n_transitions", {}).get("val", 0),
                "n_transitions_test": report.get("n_transitions", {}).get("test", 0),
                "pct_done_train": (
                    report.get("train_pct_done", 0.0) if "train_pct_done" in report else 0.0
                ),
                "pct_done_val": (
                    report.get("val_pct_done", 0.0) if "val_pct_done" in report else 0.0
                ),
                "pct_done_test": (
                    report.get("test_pct_done", 0.0) if "test_pct_done" in report else 0.0
                ),
                "reward_mean_terminal_train": report.get("train_reward_mean_terminal", 0.0),
                "reward_mean_terminal_val": report.get("val_reward_mean_terminal", 0.0),
                "reward_mean_terminal_test": report.get("test_reward_mean_terminal", 0.0),
                "episode_len_mean_train": report.get("train_episode_len_mean", 0.0),
                "episode_len_mean_val": report.get("val_episode_len_mean", 0.0),
                "episode_len_mean_test": report.get("test_episode_len_mean", 0.0),
                "action_rate_do_nothing_train": report.get("train_action_rate_do_nothing", 0.0),
                "action_rate_do_nothing_val": report.get("val_action_rate_do_nothing", 0.0),
                "action_rate_do_nothing_test": report.get("test_action_rate_do_nothing", 0.0),
                "action_rate_contact_hq_train": report.get("train_action_rate_contact_hq", 0.0),
                "action_rate_contact_hq_val": report.get("val_action_rate_contact_hq", 0.0),
                "action_rate_contact_hq_test": report.get("test_action_rate_contact_hq", 0.0),
            }
            # Add drift flags if available
            if "drift_flag_episode_len" in report:
                metrics["drift_flag_episode_len"] = float(report["drift_flag_episode_len"])
            if "episode_len_drift_pct" in report:
                metrics["episode_len_drift_pct"] = report["episode_len_drift_pct"]

            tracker.log_metrics(metrics)

        # Log artifacts to tracker
        if tracker and tracker.enabled and tracking_cfg.get("log_artifacts", True):
            tracker.log_artifact(str(splits_path), artifact_path="data")
            tracker.log_artifact(str(report_path), artifact_path="reports")
            if tracking_cfg.get("log_config", True):
                tracker.log_artifact(args.config, artifact_path="metadata")
            # Ensure all reproducibility tags are set
            git_info = get_git_commit()
            config_hash = config_obj.hash_config()
            try:
                npz_hash = fingerprint_data([npz_path])[:16]
                tracker.set_tags(
                    {
                        "git_commit": git_info["commit"][:8]
                        if isinstance(git_info["commit"], str)
                        else "unknown",
                        "git_dirty": str(git_info["dirty"]),
                        "config_hash": config_hash[:16],
                        "npz_hash": npz_hash,
                    }
                )
            except Exception as e:
                logger.warning("Could not set all reproducibility tags: %s", e)

        # Finalize metadata
        metadata = finalize_run_metadata(
            metadata,
            outputs=[str(splits_path), str(report_path)],
            metrics=report,
        )
        metadata["status"] = "completed"

        # Save run metadata
        run_metadata_path = reports_dir / "run_metadata_validate_split.json"
        save_run_metadata(metadata, run_metadata_path)

        logger.info("✅ Validation and splitting completed successfully")
        logger.info("   Output: %s", splits_path)
        logger.info(
            "   Stats: train=%d cases (%d trans), val=%d cases (%d trans), "
            "test=%d cases (%d trans)",
            report.get("n_cases", {}).get("train", 0),
            report.get("n_transitions", {}).get("train", 0),
            report.get("n_cases", {}).get("val", 0),
            report.get("n_transitions", {}).get("val", 0),
            report.get("n_cases", {}).get("test", 0),
            report.get("n_transitions", {}).get("test", 0),
        )

    except Exception as e:
        logger.error("❌ Validation and splitting failed: %s", e, exc_info=True)
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        run_metadata_path = reports_dir / "run_metadata_validate_split.json"
        save_run_metadata(metadata, run_metadata_path)
        if tracker and tracker.enabled:
            tracker.finish()
        raise

    finally:
        if tracker and tracker.enabled:
            tracker.finish()


if __name__ == "__main__":
    main()
