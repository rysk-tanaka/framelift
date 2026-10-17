from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
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


def test_the_defaults_write_the_screens_as_drawn(tmp_path: Path) -> None:
    for truth in synth.generate(tmp_path, n=2, seed=5):
        drawn = io.BytesIO()
        synth.render(truth).save(drawn, format="PNG")
        assert (tmp_path / truth.file).read_bytes() == drawn.getvalue()
    dataset, _ = load_dataset(tmp_path / synth.DATASET_FILE)
    assert all(set(case.metadata) == {"orientation", "width", "height"} for case in dataset.cases)


def _content(truths: list[synth.SampleTruth]) -> list[tuple[str, str, list[synth.SampleItem]]]:
    return [(t.orientation, t.shop_name, t.items) for t in truths]


def test_degrading_keeps_the_ground_truth(tmp_path: Path) -> None:
    plain = synth.generate(tmp_path / "plain", n=4, seed=9)
    degraded = synth.generate(
        tmp_path / "degraded", n=4, seed=9, scale=0.5, noise=8, jpeg_quality=70
    )
    assert _content(degraded) == _content(plain)


@pytest.mark.parametrize("scale", [0.5, 0.1])
def test_scale_resizes_the_screens(tmp_path: Path, scale: float) -> None:
    truths = synth.generate(tmp_path, n=2, seed=1, scale=scale)
    dataset, _ = load_dataset(tmp_path / synth.DATASET_FILE)
    for truth, case in zip(truths, dataset.cases, strict=True):
        width, height = synth._SIZES[truth.orientation]
        with Image.open(tmp_path / truth.file) as image:
            assert image.size == (round(width * scale), round(height * scale))
            assert (case.metadata["width"], case.metadata["height"]) == image.size
        assert case.metadata["scale"] == scale


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float32)


def test_noise_has_the_given_spread(tmp_path: Path) -> None:
    synth.generate(tmp_path / "plain", n=1, seed=2)
    synth.generate(tmp_path / "noisy", n=1, seed=2, noise=8)
    spread = np.std(
        _pixels(tmp_path / "noisy/shop-000.png") - _pixels(tmp_path / "plain/shop-000.png")
    )
    assert 7 < float(spread) < 8.5  # a little under 8 where 0-255 cuts the noise off


@pytest.mark.parametrize("seed", [2, -1], ids=["seed", "negative-seed"])
def test_noise_is_the_same_for_the_same_seed(tmp_path: Path, seed: int) -> None:
    synth.generate(tmp_path / "a", n=2, seed=seed, noise=8)
    synth.generate(tmp_path / "b", n=2, seed=seed, noise=8)
    for name in ("shop-000.png", "shop-001.png"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def _noise(out: Path, seed: int) -> np.ndarray:
    synth.generate(out / "plain", n=1, seed=seed)
    synth.generate(out / "noisy", n=1, seed=seed, noise=8)
    return _pixels(out / "noisy/shop-000.png") - _pixels(out / "plain/shop-000.png")


def test_noise_differs_between_seeds(tmp_path: Path) -> None:
    # The same noise would differ only where 0-255 cuts it off, which the two screens do in
    # different places; independent noise differs by about 9 on average.
    difference = np.abs(_noise(tmp_path / "a", seed=1) - _noise(tmp_path / "b", seed=2))
    assert float(difference.mean()) > 5


def test_jpeg_quality_writes_jpeg(tmp_path: Path) -> None:
    truths = synth.generate(tmp_path, n=2, seed=3, jpeg_quality=70)
    assert [t.file for t in truths] == ["shop-000.jpg", "shop-001.jpg"]
    dataset, _ = load_dataset(tmp_path / synth.DATASET_FILE)  # also checks every sha256
    assert [case.name for case in dataset.cases] == ["shop-000", "shop-001"]
    assert dataset.cases[0].metadata["jpeg_quality"] == 70
    with Image.open(tmp_path / "shop-000.jpg") as image:
        assert image.format == "JPEG"


@pytest.mark.parametrize(
    "degradation",
    [
        {"scale": 0.0},
        {"scale": 0.09},
        {"scale": 4.5},
        {"scale": math.nan},
        {"scale": math.inf},
        {"noise": -1.0},
        {"noise": 65.0},
        {"noise": math.nan},
        {"jpeg_quality": 0},
        {"jpeg_quality": 96},
        {"jpeg_quality": True},
    ],
)
def test_out_of_range_degradation_is_rejected(tmp_path: Path, degradation: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(degradation))):
        synth.generate(tmp_path / "out", n=1, **degradation)
    assert not (tmp_path / "out").exists()
