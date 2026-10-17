from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from framelift._internal import synth
from framelift._internal.dataset import DatasetCase, load_dataset, write_dataset
from framelift._internal.sources import expand_inputs, hash_file


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


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


@pytest.mark.parametrize("jpeg_quality", [None, 70], ids=["same-format", "other-format"])
def test_regenerating_removes_what_the_earlier_run_wrote(
    tmp_path: Path, jpeg_quality: int | None
) -> None:
    synth.generate(tmp_path, n=3)
    truths = synth.generate(tmp_path, n=2, jpeg_quality=jpeg_quality)
    assert [p.name for p in expand_inputs([tmp_path])] == [t.file for t in truths]
    record = (tmp_path / synth.RECORD_FILE).read_text()
    assert record == "".join(f"{hash_file(tmp_path / t.file)}  {t.file}\n" for t in truths)


def test_files_the_record_does_not_list_are_kept(tmp_path: Path) -> None:
    synth.generate(tmp_path, n=2)
    for name in ("other.png", "shop-150.png", "shop-scroll-005.mp4"):
        (tmp_path / name).write_bytes(b"the user's own")
    synth.generate(tmp_path, n=1, jpeg_quality=70)
    assert _names(tmp_path) == [
        "dataset.yaml",
        "other.png",
        "shop-000.jpg",
        "shop-150.png",
        "shop-scroll-005.mp4",
        "synth.sha256",
    ]


def test_a_dataset_the_user_wrote_does_not_remove_their_images(tmp_path: Path) -> None:
    # The user's own captures under synth-like names, listed with their actual SHA-256 in a
    # dataset.yaml of their own: without synth.sha256, nothing of theirs is synth's to remove.
    names = [f"shop-{i:03d}.png" for i in range(3)]
    for name in names:
        Image.new("RGB", (8, 8), (len(name), 0, 0)).save(tmp_path / name)
    cases = [
        DatasetCase(name=name, image=name, sha256=hash_file(tmp_path / name), expected={})
        for name in names
    ]
    write_dataset(tmp_path / synth.DATASET_FILE, cases)
    synth.generate(tmp_path, n=1, jpeg_quality=70)
    assert all((tmp_path / name).is_file() for name in names)


def test_a_file_the_user_replaced_is_kept(tmp_path: Path) -> None:
    synth.generate(tmp_path, n=3)
    Image.new("RGB", (8, 8)).save(tmp_path / "shop-002.png")  # a real screenshot, say
    synth.generate(tmp_path, n=1)
    assert sorted(p.name for p in tmp_path.glob("shop-*")) == ["shop-000.png", "shop-002.png"]


@pytest.mark.parametrize(
    "record",
    [b"\xff\xfe not utf-8", b"../outside.png\n", f"{'0' * 64}  ../outside.png\n".encode()],
    ids=["not-utf8", "not-a-record", "outside"],
)
def test_a_record_that_cannot_be_used_removes_nothing(tmp_path: Path, record: bytes) -> None:
    out = tmp_path / "out"
    (tmp_path / "outside.png").write_bytes(b"")
    synth.generate(out, n=3)
    (out / synth.RECORD_FILE).write_bytes(record)
    synth.generate(out, n=1)
    assert sorted(p.name for p in out.glob("shop-*")) == [
        "shop-000.png",
        "shop-001.png",
        "shop-002.png",
    ]
    assert (tmp_path / "outside.png").exists()


def _contents(directory: Path) -> dict[str, bytes | None]:
    return {p.name: p.read_bytes() if p.is_file() else None for p in directory.iterdir()}


@pytest.mark.parametrize("jpeg_quality", [None, 70], ids=["same-format", "other-format"])
@pytest.mark.parametrize("stage", ["drawing", "dataset"])
def test_a_failed_run_leaves_the_directory_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, jpeg_quality: int | None
) -> None:
    synth.generate(tmp_path, n=3)
    before = _contents(tmp_path)

    def fail(*args: object) -> None:
        raise OSError("disk full")

    if stage == "dataset":
        monkeypatch.setattr(synth, "write_dataset", fail)
    else:
        render, drawn = synth.render, []

        def fail_on_the_second(truth: synth.SampleTruth) -> Image.Image:
            drawn.append(truth)
            if len(drawn) == 2:
                fail()
            return render(truth)

        monkeypatch.setattr(synth, "render", fail_on_the_second)
    with pytest.raises(OSError, match="disk full"):
        synth.generate(tmp_path, n=3, noise=8, jpeg_quality=jpeg_quality)
    assert _contents(tmp_path) == before  # nothing replaced, no staging directory left


def _shops(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("shop-*"))


def test_a_file_that_cannot_be_removed_stays_listed_for_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synth.generate(tmp_path, n=3)
    unlink = Path.unlink

    def locked(path: Path, missing_ok: bool = False) -> None:
        if path.name == "shop-001.png":
            raise PermissionError("read-only")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", locked)
        with pytest.warns(synth.StaleFilesWarning, match="shop-001.png"):
            synth.generate(tmp_path, n=1)
    load_dataset(tmp_path / synth.DATASET_FILE)  # the new dataset is in place
    assert _shops(tmp_path) == ["shop-000.png", "shop-001.png"]
    assert "shop-002.png" not in (tmp_path / synth.RECORD_FILE).read_text()

    synth.generate(tmp_path, n=1)  # no longer locked
    assert _shops(tmp_path) == ["shop-000.png"]


def test_a_run_stopped_before_removing_leaves_them_to_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synth.generate(tmp_path, n=3)

    def stopped(*args: object) -> dict[str, str]:
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(synth, "_remove_stale", stopped)
        with pytest.raises(KeyboardInterrupt):
            synth.generate(tmp_path, n=1, jpeg_quality=70)
    assert _shops(tmp_path) == ["shop-000.jpg", "shop-000.png", "shop-001.png", "shop-002.png"]

    synth.generate(tmp_path, n=1, jpeg_quality=70)
    assert _shops(tmp_path) == ["shop-000.jpg"]


def test_fewer_videos_remove_the_rest_and_none_keep_them(tmp_path: Path) -> None:
    pytest.importorskip("av")
    videos = ["shop-scroll-000.mp4", "shop-scroll-001.mp4"]
    synth.generate(tmp_path, n=1, videos=2)

    synth.generate(tmp_path, n=2)  # screenshots only: the videos stay, and so do their lines
    assert sorted(p.name for p in tmp_path.glob("shop-scroll-*")) == videos
    record = (tmp_path / synth.RECORD_FILE).read_text()
    assert all(f"{hash_file(tmp_path / name)}  {name}\n" in record for name in videos)

    synth.generate(tmp_path, n=1, videos=1)  # the carried line lets this remove the second
    assert sorted(p.name for p in tmp_path.glob("shop-scroll-*")) == videos[:1]


def test_a_run_stopped_while_moving_leaves_the_earlier_files_removable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synth.generate(tmp_path, n=3)
    replace = Path.replace

    def stopped(path: Path, target: Path) -> Path:
        if path.name == "shop-001.png":  # shop-000.png and the record are moved by then
            raise KeyboardInterrupt
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", stopped)
        with pytest.raises(KeyboardInterrupt):
            synth.generate(tmp_path, n=3, noise=8)
    # The earlier shop-001.png and shop-002.png are still there, and still listed as synth's.
    synth.generate(tmp_path, n=1)
    assert _shops(tmp_path) == ["shop-000.png"]


def test_lines_of_videos_the_user_removed_are_dropped(tmp_path: Path) -> None:
    synth.generate(tmp_path, n=1)
    record = tmp_path / synth.RECORD_FILE
    record.write_text(record.read_text() + f"{'0' * 64}  shop-scroll-000.mp4\n")
    synth.generate(tmp_path, n=1)
    assert "shop-scroll-000.mp4" not in record.read_text()


def test_a_failed_final_record_does_not_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synth.generate(tmp_path, n=3)
    write_record, calls = synth._write_record, []

    def second_fails(*args: Any) -> None:
        calls.append(args)
        if len(calls) == 2:
            raise OSError("disk full")
        write_record(*args)

    with monkeypatch.context() as patch:
        patch.setattr(synth, "_write_record", second_fails)
        synth.generate(tmp_path, n=1, jpeg_quality=70)
    assert _shops(tmp_path) == ["shop-000.jpg"]
    load_dataset(tmp_path / synth.DATASET_FILE)
