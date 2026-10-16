from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from framelift._internal import synth


def test_generate_is_deterministic(tmp_path: Path) -> None:
    a = synth.generate(tmp_path / "a", n=4, seed=7)
    b = synth.generate(tmp_path / "b", n=4, seed=7)
    assert a == b
    assert (tmp_path / "a" / a[0].file).read_bytes() == (tmp_path / "b" / b[0].file).read_bytes()


def test_truth_file_matches_images(tmp_path: Path) -> None:
    truths = synth.generate(tmp_path, n=4, seed=1)
    lines = (tmp_path / "truth.jsonl").read_text().splitlines()
    assert [json.loads(line)["file"] for line in lines] == [t.file for t in truths]
    for truth in truths:
        with Image.open(tmp_path / truth.file) as image:
            assert image.size == synth._SIZES[truth.orientation]


def test_items_fit_on_screen(tmp_path: Path) -> None:
    for truth in synth.generate(tmp_path, n=40, seed=3):
        assert synth._MIN_ITEMS <= len(truth.items) <= synth._max_rows(truth.orientation)
        assert len({item.name for item in truth.items}) == len(truth.items)
