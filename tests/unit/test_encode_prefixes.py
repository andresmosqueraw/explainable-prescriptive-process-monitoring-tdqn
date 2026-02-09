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

    # Check required keys
    required_keys = ["X", "mask", "case_ptr", "t_ptr"]
    for key in required_keys:
        assert key in data, f"Output should contain '{key}' key"

    X = data["X"]
    mask = data["mask"]

    assert isinstance(X, np.ndarray), "X should be numpy array"
    assert X.ndim == 2, "X should be 2D [N_prefixes, max_len]"
    assert len(X) > 0, "Should have at least one prefix"

    assert isinstance(mask, np.ndarray), "mask should be numpy array"
    assert mask.shape == X.shape, "mask should have same shape as X"

    # Check dtypes
    assert np.issubdtype(X.dtype, np.integer), f"X dtype {X.dtype} should be integer"
    assert (
        mask.dtype == np.uint8 or mask.dtype == np.bool_
    ), f"mask dtype {mask.dtype} should be uint8 or bool"
