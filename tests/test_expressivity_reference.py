"""Independent algebraic validation across all requested embeddings and readouts."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("width", [2, 3])
def test_full_coefficient_reference_matrix(width, tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/validate_expressivity.py"),
            "--source",
            str(root),
            "--output",
            str(tmp_path),
            "--widths",
            str(width),
            "--seeds",
            "17",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["models"] == 144
    assert result["output_spectra"] == 336
    assert result["max_coefficient_error"] < 2e-10
    assert result["max_direct_error"] < 2e-10
    assert result["max_closed_form_error"] < 2e-11
