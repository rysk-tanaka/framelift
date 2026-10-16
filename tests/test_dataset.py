from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from framelift._internal.dataset import DatasetCase, load_dataset


def _move_images(src: Path, dst: Path) -> None:
    dst.mkdir()
    for image in src.glob("*.png"):
        image.rename(dst / image.name)


def test_synth_dataset_loads_and_matches(samples) -> None:
    root, truths = samples
    dataset, resolved = load_dataset(root / "dataset.yaml")
    assert resolved == root
    assert [case.image for case in dataset.cases] == [t.file for t in truths]
    assert dataset.cases[0].expected["shop_name"] == truths[0].shop_name


def test_root_argument_points_at_images_elsewhere(samples, tmp_path: Path) -> None:
    root, _ = samples
    moved = tmp_path / "moved"
    _move_images(root, moved)
    _, resolved = load_dataset(root / "dataset.yaml", root=moved)
    assert resolved == moved


def test_all_mismatches_are_reported_together(samples) -> None:
    root, truths = samples
    (root / truths[0].file).unlink()
    (root / truths[1].file).write_bytes((root / truths[2].file).read_bytes())
    (root / truths[2].file).write_bytes(b"not an image")
    with pytest.raises(ValueError) as excinfo:
        load_dataset(root / "dataset.yaml")
    message = str(excinfo.value)
    assert "shop-000: image not found" in message
    assert "shop-001: sha256 mismatch" in message
    assert "shop-002: Not a readable image" in message


def test_unknown_keys_are_rejected(samples) -> None:
    root, _ = samples
    path = root / "dataset.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["cases"][0]["expect"] = raw["cases"][0].pop("expected")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="framelift/dataset/v1"):
        load_dataset(path)


def test_duplicate_case_names_are_rejected(samples) -> None:
    root, _ = samples
    path = root / "dataset.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["cases"][1]["name"] = raw["cases"][0]["name"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)


def test_expected_values_are_compared_in_json_form() -> None:
    raw = yaml.safe_load("name: c\nimage: c.png\nsha256: x\nexpected: {date: 2026-10-16}\n")
    assert DatasetCase.model_validate(raw).expected == {"date": "2026-10-16"}


def test_sha256_is_compared_case_insensitively(samples) -> None:
    root, _ = samples
    path = root / "dataset.yaml"
    raw = yaml.safe_load(path.read_text())
    for case in raw["cases"]:
        case["sha256"] = f" {case['sha256'].upper()} "  # as some tools print it
    path.write_text(yaml.safe_dump(raw))
    load_dataset(path)  # does not raise


def test_a_truncated_image_is_reported_with_the_others(samples) -> None:
    root, truths = samples
    image = root / truths[0].file
    image.write_bytes(image.read_bytes()[:200])
    path = root / "dataset.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["cases"][0]["sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="shop-000: Not a readable image"):
        load_dataset(path)


@pytest.mark.parametrize("image", ["../outside.png", "sub/../../outside.png", "/etc/outside.png"])
def test_images_outside_the_root_are_rejected(samples, image: str) -> None:
    root, _ = samples
    path = root / "dataset.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["cases"][0]["image"] = image
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="inside the dataset root"):
        load_dataset(path)
