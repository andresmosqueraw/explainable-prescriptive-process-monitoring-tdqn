"""Phase 3 - Step 7: Fidelity Tests (Q-drop, Action-flip, Rank-consistency)."""

import argparse

from xppm.utils.config import Config
from xppm.utils.logging import get_logger
from xppm.xai.fidelity_tests import run_fidelity_tests

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fidelity tests: Q-drop, action-flip, rank-consistency",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to Q_theta checkpoint (overrides config)",
    )
    parser.add_argument(
        "--xai-dir",
        type=str,
        default=None,
        help="Directory with XAI outputs (overrides config, looks for final/ subdir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (overrides config)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Limit number of items for smoke test (default: all)",
    )
    parser.add_argument(
        "--p-remove",
        type=float,
        nargs="+",
        default=None,
        help="Removal percentages (default: [0.1, 0.2, 0.3, 0.5])",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=None,
        help="Number of random repetitions (default: 20, use 5 for smoke test)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (verbose logging of perturbations, Q-values, actions)",
    )
    args = parser.parse_args()

    config_obj = Config.from_yaml(args.config)
    cfg = config_obj.raw

    # CLI overrides
    if args.ckpt:
        cfg.setdefault("xai", {})["checkpoint_path"] = args.ckpt
    if args.xai_dir:
        # Update xai out_dir to point to provided directory
        cfg.setdefault("xai", {})["out_dir"] = args.xai_dir
    if args.output:
        cfg.setdefault("fidelity", {})["out_csv"] = args.output
    if args.n_items is not None:
        cfg.setdefault("fidelity", {})["n_items"] = args.n_items
    if args.p_remove is not None:
        cfg.setdefault("fidelity", {})["p_remove"] = args.p_remove
    if args.n_random is not None:
        cfg.setdefault("fidelity", {})["n_random"] = args.n_random
    if args.debug:
        cfg.setdefault("fidelity", {})["debug"] = True

    run_fidelity_tests(cfg, config_obj=config_obj)
    logger.info("07_fidelity_tests completed.")


if __name__ == "__main__":
    main()
