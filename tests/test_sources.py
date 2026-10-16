from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from framelift._internal.sources import expand_inputs, load_image


def test_expand_directory_glob_and_file_without_duplicates(samples: tuple[Path, list]) -> None:
    root, truths = samples
    first = root / truths[0].file
    paths = expand_inputs([root, str(root / "*.png"), first])
    assert [p.name for p in paths] == [t.file for t in truths]


def test_expand_ignores_non_images_in_directories(samples: tuple[Path, list]) -> None:
    root, truths = samples
    assert (root / "truth.jsonl").exists()
    assert len(expand_inputs([root])) == len(truths)


def test_expand_raises_when_nothing_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        expand_inputs([tmp_path / "missing.png"])
    with pytest.raises(FileNotFoundError):
        expand_inputs([str(tmp_path / "*.png")])


def test_load_image_hashes(samples: tuple[Path, list]) -> None:
    root, truths = samples
    path = root / truths[0].file
    image = load_image(path)
    assert image.media_type == "image/png"
    assert image.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(image.phash) == 16


def test_load_image_rejects_non_images(tmp_path: Path) -> None:
    fake = tmp_path / "fake.png"
    fake.write_text("not an image")
    with pytest.raises(ValueError, match="Not a readable image"):
        load_image(fake)
