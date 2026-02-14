"""Export decision tree to SQL rules and human-readable text."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree import _tree, export_text

from xppm.utils.io import save_json
from xppm.utils.logging import get_logger

logger = get_logger(__name__)


def export_tree_to_text(
    tree: Any,
    feature_names: list[str],
    output_path: str | Path,
) -> None:
    """Export decision tree to human-readable text.

    Args:
        tree: Trained DecisionTreeClassifier
        feature_names: List of feature names
        output_path: Path to save text file
    """
    tree_rules_text = export_text(
        tree,
        feature_names=feature_names,
        max_depth=10,  # full tree
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tree_rules_text)

    logger.info("Exported tree rules text to %s", output_path)


def export_tree_to_sql(
    tree: Any,
    feature_names: list[str],
    action_names: list[str],
    fidelity_metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Export decision tree to SQL CASE WHEN rules.

    Args:
        tree: Trained DecisionTreeClassifier
        feature_names: List of feature names
        action_names: List of action names
        fidelity_metrics: Fidelity metrics dict
        output_path: Path to save SQL file
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature
    ]

    def recurse(node: int, depth: int, conditions: list[str]) -> list[str]:
        """Recursively build CASE WHEN clauses."""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = float(tree_.threshold[node])

            # Left child (<=)
            left_conditions = conditions + [f"{name} <= {threshold:.4f}"]
            left_sql = recurse(tree_.children_left[node], depth + 1, left_conditions)

            # Right child (>)
            right_conditions = conditions + [f"{name} > {threshold:.4f}"]
            right_sql = recurse(tree_.children_right[node], depth + 1, right_conditions)

            return left_sql + right_sql
        else:
            # Leaf node
            class_id = int(np.argmax(tree_.value[node]))
            action = action_names[class_id] if class_id < len(action_names) else "do_nothing"
            confidence = float(tree_.value[node][0][class_id] / tree_.value[node][0].sum())

            condition_str = " AND ".join(conditions) if conditions else "TRUE"
            return [f"WHEN {condition_str} THEN '{action}'  -- confidence: {confidence:.2f}"]

    sql_cases = recurse(0, 0, [])

    sql_template = f"""-- Policy Decision Tree (exported from sklearn)
-- Generated: {datetime.now().isoformat()}
-- Action agreement: {fidelity_metrics.get('action_agreement_global', 0.0):.3f}
-- Tree depth: {fidelity_metrics.get('tree_depth', 0)}
-- Tree leaves: {fidelity_metrics.get('tree_n_leaves', 0)}

SELECT
  CASE
    {chr(10).join('    ' + case for case in sql_cases)}
    ELSE 'do_nothing'  -- default fallback
  END AS recommended_action
FROM case_features;
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sql_template)

    logger.info("Exported SQL rules to %s", output_path)


def export_rules(
    tree_pkl_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Export decision tree to SQL and text rules.

    Args:
        tree_pkl_path: Path to tree.pkl file
        output_dir: Directory to save exported rules
        config: Optional config dict (for metadata)

    Returns:
        Dictionary mapping artifact name to file path
    """
    import pickle

    # Load tree
    with open(tree_pkl_path, "rb") as f:
        tree_data = pickle.load(f)

    tree = tree_data["model"]
    feature_names = tree_data["feature_names"]
    action_names = tree_data["action_names"]
    fidelity_metrics = tree_data.get("fidelity_metrics", {})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export text rules
    text_path = output_dir / "tree_rules.txt"
    export_tree_to_text(tree, feature_names, text_path)

    # Export SQL rules
    sql_path = output_dir / "rules.sql"
    export_tree_to_sql(tree, feature_names, action_names, fidelity_metrics, sql_path)

    # Export metadata
    metadata_path = output_dir / "rules_metadata.json"
    metadata = {
        "fidelity_metrics": fidelity_metrics,
        "tree_metadata": tree_data.get("metadata", {}),
        "feature_names": feature_names,
        "action_names": action_names,
        "timestamp": datetime.now().isoformat(),
    }
    if config:
        metadata["config"] = config

    save_json(metadata, metadata_path)

    return {
        "tree_rules_txt": text_path,
        "rules_sql": sql_path,
        "rules_metadata": metadata_path,
    }
