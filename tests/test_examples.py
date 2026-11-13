"""Check that every standalone example runs without deprecation warnings."""
from pathlib import Path
import runpy
import warnings

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("script", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_example(script):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        runpy.run_path(str(script), run_name="__main__")
