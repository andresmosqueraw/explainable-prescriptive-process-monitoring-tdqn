from __future__ import annotations

import argparse
import json
from pathlib import Path

from xppm.data.build_mdp import build_mdp_dataset
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
        description="Build MDP dataset from prefixes (Phase 1 - Step 3)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if it exists (use with caution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and config without building dataset (no writes)",
    )
    args = parser.parse_args()

    # Load config
    config_obj = Config.from_yaml(args.config)
    cfg = config_obj.raw

    # Setup reproducibility
    seed = cfg.get("repro", {}).get("seed", 42)
    deterministic = cfg.get("repro", {}).get("deterministic", False)

    # Get paths from config
    paths_cfg = cfg.get("paths", {})
    mdp_cfg = cfg.get("mdp", {})
    output_cfg = mdp_cfg.get("output", {})

    prefixes_path = (
        cfg.get("encoding", {}).get("output", {}).get("prefixes_path", "data/interim/prefixes.npz")
    )
    clean_log_path = cfg.get("data", {}).get("output_clean_path", "data/interim/clean.parquet")
    vocab_path = (
        cfg.get("encoding", {})
        .get("output", {})
        .get("vocab_activity_path", "data/interim/vocab_activity.json")
    )
    output_path = output_cfg.get("path", "data/processed/D_offline.npz")

    # Dry-run mode: validate inputs without building (skip everything else)
    if args.dry_run:
        logger.info("DRY-RUN mode: Validating inputs without building dataset")
        # Validate paths exist (no file creation, no tracking, no artifacts)
        if not Path(prefixes_path).exists():
            raise FileNotFoundError(f"Prefixes file not found: {prefixes_path}")
        if not Path(clean_log_path).exists():
            raise FileNotFoundError(f"Clean log file not found: {clean_log_path}")
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        # Compute hashes for validation (read-only, no writes)
        try:
            config_hash = config_obj.hash_config()
            prefixes_hash = fingerprint_data([prefixes_path])[:16]
            clean_log_hash = fingerprint_data([clean_log_path])[:16]
            logger.info("✅ All input files exist and are valid")
            logger.info("   Prefixes: %s (hash: %s)", prefixes_path, prefixes_hash)
            logger.info("   Clean log: %s (hash: %s)", clean_log_path, clean_log_hash)
            logger.info("   Vocabulary: %s", vocab_path)
            logger.info("   Config hash: %s", config_hash[:16])
            logger.info("   Output would be: %s", output_path)
            logger.info("   (No files created, no tracking, no artifacts)")
        except Exception as e:
            logger.warning("Could not compute hashes: %s", e)
            logger.info("✅ All input files exist")
            logger.info("   Prefixes: %s", prefixes_path)
            logger.info("   Clean log: %s", clean_log_path)
            logger.info("   Vocabulary: %s", vocab_path)
            logger.info("   Output would be: %s", output_path)
        return

    # Initialize tracking (skip in dry-run)
    tracking_cfg = cfg.get("tracking", {})
    tracker = None
    if tracking_cfg.get("enabled", False):
        tracker = init_tracker(tracking_cfg)
        if tracker.enabled:
            git_info = get_git_commit()
            run_name = f"build_mdp_{Path(prefixes_path).stem}"
            commit_hash = git_info["commit"]
            commit_short = commit_hash[:8] if isinstance(commit_hash, str) else "unknown"
            tags = {
                "stage": "phase1_build_mdp",
                "dataset": Path(clean_log_path).stem,
                "git_commit": commit_short,
                "git_dirty": str(git_info["dirty"]),
                "config_hash": config_obj.hash_config()[:8],
            }
            # Add data hashes
            try:
                prefixes_hash = fingerprint_data([prefixes_path])[:16]
                clean_log_hash = fingerprint_data([clean_log_path])[:16]
                tags["prefixes_hash"] = prefixes_hash
                tags["clean_log_hash"] = clean_log_hash
            except Exception as e:
                logger.warning("Could not compute data hashes: %s", e)

            params = {
                "max_len": cfg.get("encoding", {}).get("max_len", 50),
                "decision_point_mode": mdp_cfg.get("decision_points", {}).get("mode"),
                "n_actions": len(mdp_cfg.get("actions", {}).get("id2name", [])),
                "reward_type": mdp_cfg.get("reward", {}).get("type"),
            }
            tracker.init_run(run_name=run_name, stage="phase1_build_mdp", tags=tags, params=params)

    # Start run metadata (skip in dry-run)
    artifacts_dir = Path(paths_cfg.get("artifacts_dir", "artifacts"))
    reports_dir = artifacts_dir / "reports"
    ensure_dir(reports_dir)

    metadata = start_run_metadata(
        stage="build_mdp",
        config_path=args.config,
        config_hash=config_obj.hash_config(),
        seed=seed,
        deterministic=deterministic,
        data_fingerprint=(
            fingerprint_data([prefixes_path, clean_log_path])
            if Path(prefixes_path).exists() and Path(clean_log_path).exists()
            else None
        ),
    )

    # Protection: don't overwrite real dataset accidentally (skip for dry-run)
    output_path_obj = Path(output_path).resolve()
    if output_path_obj.exists():
        # Strict check: allow overwrite only if in safe test directories
        # - pytest tmp_path (contains /tmp/pytest- or /tmp/tmp)
        # - .tmp/ directory
        # - artifacts/tests/ directory
        output_str = str(output_path_obj)
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
                    f"Output file {output_path} already exists. "
                    f"To overwrite, use --overwrite flag. "
                    f"This prevents accidental overwrites of real datasets."
                )
            else:
                logger.warning(
                    "Overwriting existing dataset: %s (--overwrite flag provided)",
                    output_path,
                )

    try:
        # Run MDP building
        logger.info("Starting MDP dataset building: %s -> %s", prefixes_path, output_path)
        stats = build_mdp_dataset(prefixes_path, clean_log_path, vocab_path, output_path, mdp_cfg)

        # Save schema
        default_schema_path = "configs/schemas/offline_rlset.schema.json"
        schema_path = Path(output_cfg.get("schema_path", default_schema_path))
        ensure_dir(schema_path.parent)
        schema = {
            "dataset_name": "D_offline",
            "version": "1.0",
            "n_transitions": stats["n_transitions"],
            "max_len": stats["max_len"],
            "n_actions": stats["n_actions"],
            "arrays": {
                "s": {
                    "dtype": "int32",
                    "shape": ["N", stats["max_len"]],
                    "desc": "prefix token ids",
                },
                "s_mask": {
                    "dtype": "uint8",
                    "shape": ["N", stats["max_len"]],
                    "desc": "1 real, 0 pad",
                },
                "a": {"dtype": "int32", "shape": ["N"], "desc": "behavior action id"},
                "r": {"dtype": "float32", "shape": ["N"], "desc": "reward (delayed terminal)"},
                "s_next": {"dtype": "int32", "shape": ["N", stats["max_len"]]},
                "s_next_mask": {"dtype": "uint8", "shape": ["N", stats["max_len"]]},
                "done": {"dtype": "uint8", "shape": ["N"]},
                "valid_actions": {
                    "dtype": "uint8",
                    "shape": ["N", stats["n_actions"]],
                    "desc": "action mask",
                },
                "behavior_action": {
                    "dtype": "int32",
                    "shape": ["N"],
                    "desc": "behavior action id (redundant with a)",
                },
                "propensity": {
                    "dtype": "float32",
                    "shape": ["N"],
                    "desc": "behavior propensity (estimated later, -1.0 if not estimated)",
                },
                "case_ptr": {"dtype": "int32", "shape": ["N"]},
                "t_ptr": {"dtype": "int32", "shape": ["N"]},
            },
            "reward_definition": "0 for non-terminal, outcome(case) at terminal",
            "decision_points": mdp_cfg.get("decision_points", {}).get("mode", "unknown"),
        }
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        logger.info("Saved schema to %s", schema_path)

        # Save build report
        report_path = reports_dir / "mdp_build_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info("Saved build report to %s", report_path)

        # Log metrics to tracker (5 golden metrics + extras)
        if tracker and tracker.enabled:
            # 5 golden metrics for quick comparison across runs
            golden_metrics = {
                "n_transitions": stats.get("n_transitions", 0),
                "n_cases_used": stats.get("n_cases_used", 0),
                "pct_done": stats.get("pct_done", 0),
                "reward_mean": stats.get("reward_mean", 0),
                "reward_p95": stats.get("reward_p95", 0),
                "pct_invalid_action": stats.get("pct_invalid_action", 0),
            }
            # Additional metrics for detailed analysis
            extra_metrics = {
                "reward_std": stats.get("reward_std", 0),
                "reward_p50": stats.get("reward_p50", 0),
                "non_terminal_reward_zero_pct": stats.get("non_terminal_reward_zero_pct", 100.0),
                "terminal_reward_nonzero_pct": stats.get("terminal_reward_nonzero_pct", 0.0),
            }
            tracker.log_metrics({**golden_metrics, **extra_metrics})

        # Log artifacts to tracker (metadata bundle for reproducibility)
        if tracker and tracker.enabled and tracking_cfg.get("log_artifacts", True):
            # Core data artifacts
            tracker.log_artifact(str(output_path), artifact_path="data")
            # Metadata bundle: config, schema, report (for full reproducibility)
            tracker.log_artifact(str(schema_path), artifact_path="metadata")
            tracker.log_artifact(str(report_path), artifact_path="metadata")
            if tracking_cfg.get("log_config", True):
                tracker.log_artifact(args.config, artifact_path="metadata")
            # Ensure all reproducibility tags are set (git_commit, config_hash, data_hash)
            git_info = get_git_commit()
            config_hash = config_obj.hash_config()
            try:
                prefixes_hash = fingerprint_data([prefixes_path])[:16]
                clean_log_hash = fingerprint_data([clean_log_path])[:16]
                tracker.set_tags(
                    {
                        "git_commit": git_info["commit"][:8]
                        if isinstance(git_info["commit"], str)
                        else "unknown",
                        "git_dirty": str(git_info["dirty"]),
                        "config_hash": config_hash[:16],
                        "prefixes_hash": prefixes_hash,
                        "clean_log_hash": clean_log_hash,
                    }
                )
            except Exception as e:
                logger.warning("Could not set all reproducibility tags: %s", e)

        # Finalize metadata
        metadata = finalize_run_metadata(
            metadata,
            outputs=[str(output_path), str(schema_path), str(report_path)],
            metrics=stats,
        )
        metadata["status"] = "completed"

        # Save run metadata
        run_metadata_path = reports_dir / "run_metadata_build_mdp.json"
        save_run_metadata(metadata, run_metadata_path)

        logger.info("✅ MDP dataset building completed successfully")
        logger.info("   Output: %s", output_path)
        logger.info(
            "   Stats: %d transitions, %d cases, reward_mean=%.2f",
            stats.get("n_transitions", 0),
            stats.get("n_cases_used", 0),
            stats.get("reward_mean", 0),
        )

    except Exception as e:
        logger.error("❌ MDP dataset building failed: %s", e, exc_info=True)
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        run_metadata_path = reports_dir / "run_metadata_build_mdp.json"
        save_run_metadata(metadata, run_metadata_path)
        if tracker and tracker.enabled:
            tracker.finish()
        raise

    finally:
        if tracker and tracker.enabled:
            tracker.finish()


if __name__ == "__main__":
    main()
