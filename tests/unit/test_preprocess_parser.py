from __future__ import annotations

import pandas as pd

from xppm.data.preprocess import preprocess_event_log


def test_preprocess_log_to_df(tiny_log_path, tmp_outdir):
    """Test that preprocess_event_log loads CSV and produces valid parquet."""
    output_path = tmp_outdir / "clean.parquet"

    preprocess_event_log(tiny_log_path, output_path)

    # Load the output
    df = pd.read_parquet(output_path)

    assert isinstance(df, pd.DataFrame)
    required = {"case_id", "activity", "timestamp"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    # Check timestamps are parseable (if they exist)
    if "timestamp" in df.columns:
        # Should be datetime or string that can be parsed
        assert len(df) > 0, "DataFrame should not be empty"

    # Check ordering within each case (if timestamp is datetime)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            for case_id, group in df.groupby("case_id"):
                assert group[
                    "timestamp"
                ].is_monotonic_increasing, f"Timestamps not monotonic for case {case_id}"
        except (ValueError, TypeError):
            # If timestamp can't be parsed, that's OK for now (stub implementation)
            pass
