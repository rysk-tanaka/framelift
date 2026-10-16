from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from framelift._internal import synth
from framelift._internal.dataset import load_dataset


def test_generate_is_deterministic(tmp_path: Path) -> None:
    a = synth.generate(tmp_path / "a", n=4, seed=7)
    b = synth.generate(tmp_path / "b", n=4, seed=7)
    assert a == b
    assert (tmp_path / "a" / a[0].file).read_bytes() == (tmp_path / "b" / b[0].file).read_bytes()


def test_dataset_file_matches_images(tmp_path: Path) -> None:
    truths = synth.generate(tmp_path, n=4, seed=1)
    dataset, _ = load_dataset(tmp_path / synth.DATASET_FILE)  # also checks every sha256
    assert [case.image for case in dataset.cases] == [t.file for t in truths]
    for truth, case in zip(truths, dataset.cases, strict=True):
        with Image.open(tmp_path / truth.file) as image:
            assert image.size == synth._SIZES[truth.orientation]
            assert (case.metadata["width"], case.metadata["height"]) == image.size


def test_items_fit_on_screen(tmp_path: Path) -> None:
    for truth in synth.generate(tmp_path, n=40, seed=3):
        assert synth._MIN_ITEMS <= len(truth.items) <= synth._max_rows(truth.orientation)
        assert len({item.name for item in truth.items}) == len(truth.items)


def test_a_video_needs_enough_items_to_fill_the_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_items"):
        synth.generate_video(tmp_path / "short.mp4", n_items=4)
