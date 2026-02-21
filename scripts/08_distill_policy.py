"""Script to distill TDQN policy into decision tree surrogate."""

import argparse
from pathlib import Path

from xppm.distill.distill_policy import distill_policy
from xppm.distill.export_rules import export_rules
from xppm.utils.config import Config
from xppm.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill TDQN policy to decision tree")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. Loads configs/datasets/{name}.yaml on top of --config.",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (overrides config)")
    parser.add_argument(
        "--n-states", type=int, default=None, help="Number of states (overrides config)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, help="Max tree depth (overrides config)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--export-only", action="store_true", help="Only export rules from existing tree.pkl"
    )
    parser.add_argument(
        "--tree-pkl", type=str, default=None, help="Path to tree.pkl (for export-only)"
    )

    args = parser.parse_args()

    cfg_obj = Config.for_dataset(args.config, args.dataset)
    cfg = cfg_obj.raw

    # Override config with CLI args
    if args.ckpt:
        if "distill" not in cfg:
            cfg["distill"] = {}
        cfg["distill"]["teacher_checkpoint"] = args.ckpt
    if args.n_states:
        if "distill" not in cfg:
            cfg["distill"] = {}
        if "sample" not in cfg["distill"]:
            cfg["distill"]["sample"] = {}
        cfg["distill"]["sample"]["n_states"] = args.n_states
    if args.max_depth:
        if "distill" not in cfg:
            cfg["distill"] = {}
        if "surrogate" not in cfg["distill"]:
            cfg["distill"]["surrogate"] = {}
        cfg["distill"]["surrogate"]["max_depth"] = args.max_depth
    if args.output_dir:
        if "distill" not in cfg:
            cfg["distill"] = {}
        cfg["distill"]["out_dir"] = args.output_dir

    if args.export_only:
        # Only export rules from existing tree.pkl
        tree_pkl_path = args.tree_pkl or Path(
            cfg.get("paths", {}).get("artifacts_dir", "artifacts")
        ) / cfg.get("distill", {}).get("out_dir", "distill") / cfg.get("distill", {}).get(
            "outputs", {}
        ).get("tree_pkl", "tree.pkl")

        if not Path(tree_pkl_path).exists():
            raise FileNotFoundError(f"Tree pickle not found: {tree_pkl_path}")

        output_dir = args.output_dir or Path(
            cfg.get("paths", {}).get("artifacts_dir", "artifacts")
        ) / cfg.get("distill", {}).get("out_dir", "distill")

        logger.info("Exporting rules from %s to %s", tree_pkl_path, output_dir)
        export_rules(tree_pkl_path, output_dir, cfg)
        logger.info("Export completed")
    else:
        # Full distillation pipeline
        logger.info("Starting policy distillation...")
        artifacts = distill_policy(cfg)

        # Export rules
        tree_pkl_path = artifacts.get("tree_pkl")
        if tree_pkl_path:
            output_dir = Path(tree_pkl_path).parent
            logger.info("Exporting rules...")
            export_rules(tree_pkl_path, output_dir, cfg)
            logger.info("Rules exported to %s", output_dir)

        logger.info("Distillation completed. Artifacts: %s", artifacts)


if __name__ == "__main__":
    main()
