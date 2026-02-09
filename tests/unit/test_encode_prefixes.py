from __future__ import annotations

import numpy as np

from xppm.data.encode_prefixes import encode_prefixes
from xppm.data.preprocess import preprocess_event_log
from xppm.utils.io import load_npz


def test_encode_prefixes_shapes(tiny_log_path, tmp_outdir):
    """Test that encode_prefixes produces valid output shapes."""
    # First preprocess
    clean_path = tmp_outdir / "clean.parquet"
    preprocess_event_log(tiny_log_path, clean_path)

    # Then encode (saves as .npz based on save_npz)
    prefixes_path = tmp_outdir / "prefixes.npz"
    encode_prefixes(clean_path, prefixes_path)

    # Load output
    data = load_npz(prefixes_path)

    assert "prefixes" in data, "Output should contain 'prefixes' key"
    prefixes = data["prefixes"]

    assert isinstance(prefixes, np.ndarray), "Prefixes should be numpy array"
    assert prefixes.ndim >= 1, "Prefixes should have at least 1 dimension"
    assert len(prefixes) > 0, "Should have at least one prefix"

    # Check dtype is reasonable
    assert np.issubdtype(prefixes.dtype, np.integer) or np.issubdtype(
        prefixes.dtype, np.floating
    ), f"Prefixes dtype {prefixes.dtype} should be numeric"
