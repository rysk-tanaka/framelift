from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic_ai import models

from framelift._internal import synth

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

# Tests must never reach a real model provider.
models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def samples(tmp_path: Path) -> tuple[Path, list[synth.SampleTruth]]:
    """Three synthetic shop screenshots and their ground truth."""
    out = tmp_path / "samples"
    truths = synth.generate(out, n=3, seed=42)
    return out, truths
