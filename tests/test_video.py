from __future__ import annotations

import itertools
import random
import sys
import warnings
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imagehash
import numpy as np
import pytest
from PIL import Image, ImageDraw

from framelift._internal import synth, video
from framelift._internal.layout import Layout

pytest.importorskip("av")

FPS = 10


@pytest.fixture(scope="module")
def scrolling(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, synth.SampleVideo]:
    path = tmp_path_factory.mktemp("video") / "shop-scroll.mp4"
    return path, synth.generate_video(path, seed=1, fps=FPS)


def _view(seed: int = 3) -> synth.ListView:
    return synth.ListView(synth._random_truth(random.Random(seed), "x", "landscape", count=12))


def _hold_of(sample: synth.SampleVideo, timestamp_ms: int) -> int | None:
    for index, hold in enumerate(sample.holds):
        if hold.start_ms <= timestamp_ms < hold.end_ms:
            return index
    return None


def test_the_synthetic_video_scrolls_through_overlapping_screens(scrolling) -> None:
    _, sample = scrolling
    shown = [{item.name for item in hold.items} for hold in sample.holds]

    assert len(sample.holds) >= 2
    assert any(a & b for a, b in itertools.pairwise(shown))
    assert set().union(*shown) == {item.name for item in sample.truth.items}


@pytest.mark.parametrize("frame_rate", [4.0, 2.0])
def test_one_sharp_frame_per_still_screen(scrolling, frame_rate: float) -> None:
    path, sample = scrolling
    frames = list(video.sample_video(path, frame_rate=frame_rate))

    assert [frame.index for frame in frames] == list(range(len(sample.holds)))
    # Every picked frame lies in a still part, a different one each time, in order.
    assert [_hold_of(sample, frame.timestamp_ms) for frame in frames] == list(
        range(len(sample.holds))
    )


def test_a_slow_scroll_does_not_merge_the_screens_around_it(tmp_path: Path) -> None:
    view = _view()
    hold = [view.render(0)] * 15
    # 0.75 px per frame: about 2 px between samples at 4 fps.
    slow = [view.render(i * 0.75) for i in range(1, 337)]
    end = [view.render(252)] * 15
    path = tmp_path / "slow.mp4"
    synth.write_video(path, hold + slow + end, fps=FPS)

    frames = list(video.sample_video(path))

    def difference(a: Image.Image, b: Image.Image) -> float:
        return video._difference(video._Sample.of(a, 0), video._Sample.of(b, 0))

    threshold = video.DEFAULT_MOTION_THRESHOLD
    # The first and last frames show the screens before and after the scroll; one segment
    # spanning the scroll would have picked a single frame for both.
    assert difference(frames[0].image, hold[0]) <= threshold
    assert difference(frames[-1].image, end[0]) <= threshold
    assert difference(frames[0].image, frames[-1].image) > threshold
    # Even a few pixels move the text across tiles, so the slow scroll is movement, not a still
    # screen of its own.
    assert len(frames) == 2


def test_the_same_screen_after_scrolling_away_and_back_is_read_again(tmp_path: Path) -> None:
    view = _view()
    frames = [view.render(0)] * 15
    frames += [view.render(42 * (i + 0.5), blur=42) for i in range(6)] + [view.render(252)] * 15
    frames += [view.render(252 - 42 * (i + 0.5), blur=42) for i in range(6)]
    frames += [view.render(0)] * 15
    path = tmp_path / "back.mp4"
    synth.write_video(path, frames, fps=FPS)

    picked = list(video.sample_video(path))

    assert len(picked) == 3
    first, last = imagehash.hex_to_hash(picked[0].phash), imagehash.hex_to_hash(picked[2].phash)
    assert first - last <= 6  # alike enough that a hash-based split would have merged them


def test_a_video_that_never_stops_yields_its_sharpest_frame(tmp_path: Path) -> None:
    view = _view()
    blurs = [40, 30, 20, 0, 20, 30, 40]  # the sharpest frame is in the middle
    frames = [view.render(60 * i, blur=blur) for i, blur in enumerate(blurs)]
    path = tmp_path / "moving.mp4"
    synth.write_video(path, frames, fps=FPS)

    picked = list(video.sample_video(path, frame_rate=FPS))

    assert [frame.timestamp_ms for frame in picked] == [3 * 1000 // FPS]


def test_a_low_frame_rate_still_reads_the_video_once(scrolling) -> None:
    path, _ = scrolling
    assert len(list(video.sample_video(path, frame_rate=1.0))) >= 1


def _marked() -> Image.Image:
    image = Image.new("RGB", (320, 160), (0, 0, 0))
    ImageDraw.Draw(image).rectangle((0, 0, 60, 40), fill=(255, 255, 255))
    return image


def _white_corner(image: Image.Image) -> str:
    w, h = image.size
    corners = {"top-left": (5, 5), "top-right": (w - 5, 5)}
    corners |= {"bottom-left": (5, h - 5), "bottom-right": (w - 5, h - 5)}
    pixels = np.asarray(image.convert("L"))
    (corner,) = [name for name, (x, y) in corners.items() if pixels[y, x] > 200]
    return corner


@pytest.mark.parametrize(
    ("rotation", "size", "corner"),
    [
        (90, (160, 320), "bottom-left"),
        (270, (160, 320), "top-right"),  # PyAV reports it as -90
        (180, (320, 160), "bottom-right"),  # and this one as -180
    ],
)
def test_display_rotation_is_applied(
    tmp_path: Path, rotation: int, size: tuple[int, int], corner: str
) -> None:
    path = tmp_path / f"rotated-{rotation}.mp4"
    synth.write_video(path, [_marked()] * 10, fps=FPS, rotation=rotation)

    (frame,) = video.sample_video(path)

    assert frame.image.size == size
    assert _white_corner(frame.image) == corner


def test_frame_times_fall_back_to_the_first_usable_rate() -> None:
    rates = [Fraction(0, 1), None, Fraction(10, 1)]
    assert video.frame_time(1.5, 7, rates) == 1.5
    assert video.frame_time(None, 7, rates) == 0.7
    assert video.frame_time(None, 7, [Fraction(0, 1), None]) is None


def test_unreadable_videos_are_value_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Not a readable video"):
        list(video.sample_video(path))


def test_a_missing_video_extra_explains_how_to_install_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(ImportError, match=r"framelift\[video\]"):
        video.import_av()


def test_a_still_screen_encoded_as_one_long_frame_is_kept(tmp_path: Path) -> None:
    # Variable-frame-rate screen recordings write a frame only when the screen changes, so a
    # screen shown for 3 s can be a single frame.
    view = _view()
    screens = [view.render(0), view.render(252), view.render(504), view.render(504)]
    path = tmp_path / "vfr.mp4"
    synth.write_video(path, screens, fps=FPS, timestamps_ms=[0, 3000, 6000, 6500])

    picked = list(video.sample_video(path))

    assert [frame.timestamp_ms for frame in picked] == [0, 3000, 6000]


def test_a_final_still_screen_written_as_one_frame_is_kept(tmp_path: Path) -> None:
    # The recording ends on a screen that was written once and stayed for 3 s; the frame's
    # duration says how long it was shown.
    view = _view()
    path = tmp_path / "vfr-end.mp4"
    screens = [view.render(0), view.render(252)]
    synth.write_video(path, screens, fps=FPS, timestamps_ms=[0, 5000], end_ms=8000)

    picked = list(video.sample_video(path))

    assert [frame.timestamp_ms for frame in picked] == [0, 5000]


@pytest.mark.parametrize("offset", [168, 336, 504])
def test_a_list_moved_by_an_even_number_of_rows_is_a_new_screen(
    tmp_path: Path, offset: int
) -> None:
    # Moving by an even number of rows keeps the row stripes in place, so only the text changes.
    view = _view()
    path = tmp_path / "rows.mp4"
    synth.write_video(path, [view.render(0)] * 15 + [view.render(offset)] * 15, fps=FPS)

    picked = [frame.timestamp_ms for frame in video.sample_video(path)]
    assert len(picked) == 2 and picked[0] < 1500 <= picked[1]


def test_a_page_with_the_same_layout_and_other_text_is_a_new_screen(tmp_path: Path) -> None:
    path = tmp_path / "pages.mp4"
    pages = [_view(seed=3).render(0)] * 15 + [_view(seed=4).render(0)] * 15
    synth.write_video(path, pages, fps=FPS)

    assert len(list(video.sample_video(path))) == 2


def test_a_small_animation_does_not_split_a_still_screen(tmp_path: Path) -> None:
    screen = _view().render(0)
    frames = []
    for i in range(30):
        frame = screen.copy()
        x = 1100 + (i % 4) * 20  # a sprite moving in the header's corner
        ImageDraw.Draw(frame).ellipse((x, 10, x + 80, 90), fill=(255, 0, 0))
        frames.append(frame)
    path = tmp_path / "animated.mp4"
    synth.write_video(path, frames, fps=FPS)

    assert len(list(video.sample_video(path))) == 1


def _beside_a_figure(screen: Image.Image, box: tuple[float, float, float, float]) -> list:
    """The screen with a large block in ``box`` (fractions) that jumps every 0.8 s, like a
    character's idle animation next to the list being read."""
    width, height = screen.size
    left, top, right, bottom = box
    frames = []
    for i in range(32):
        frame = screen.copy()
        x = left * width + (i // 8) * (right - left) * width / 5
        block = (x, top * height, x + (right - left) * width / 5 * 2, bottom * height)
        ImageDraw.Draw(frame).rectangle(block, fill=(240, 200, 40))
        frames.append(frame)
    return frames


def test_movement_outside_the_crop_does_not_split_a_still_screen(tmp_path: Path) -> None:
    path = tmp_path / "figure.mp4"
    synth.write_video(path, _beside_a_figure(_view().render(0), (0.0, 0.0, 0.4, 1.0)), fps=FPS)
    beside = Layout(locate={"landscape": (0.4, 0.0, 1.0, 1.0)}, redact={})

    assert len(list(video.sample_video(path))) == 4
    (frame,) = video.sample_video(path, layout=beside)
    assert frame.image.size == (1280, 720)  # whole screens are yielded, cropped later


def test_movement_under_a_redaction_does_not_split_a_still_screen(tmp_path: Path) -> None:
    path = tmp_path / "figure.mp4"
    synth.write_video(path, _beside_a_figure(_view().render(0), (0.0, 0.0, 0.4, 1.0)), fps=FPS)
    covered = Layout(locate="full", redact={"landscape": ((0.0, 0.0, 0.4, 1.0),)})

    assert len(list(video.sample_video(path, layout=covered))) == 1


def test_a_layout_without_a_box_for_the_screen_compares_whole_screens(tmp_path: Path) -> None:
    path = tmp_path / "figure.mp4"
    synth.write_video(path, _beside_a_figure(_view().render(0), (0.0, 0.0, 0.4, 1.0)), fps=FPS)
    portrait_only = Layout(locate={"portrait": (0.4, 0.0, 1.0, 1.0)}, redact={})

    assert len(list(video.sample_video(path, layout=portrait_only))) == 4


@pytest.mark.parametrize(
    ("rotation", "box", "count"),
    [
        (90, (0.0, 0.5, 1.0, 1.0), 1),
        (90, (0.0, 0.0, 1.0, 0.5), 4),
        (270, (0.0, 0.5, 1.0, 1.0), 1),
        (270, (0.0, 0.0, 1.0, 0.5), 4),
        (180, (0.0, 0.5, 1.0, 1.0), 1),
        (180, (0.0, 0.0, 1.0, 0.5), 4),
    ],
)
def test_the_crop_is_found_on_the_upright_screen(
    tmp_path: Path, rotation: int, box: tuple[float, float, float, float], count: int
) -> None:
    # The figure moves in the top half of the upright screen.
    size = (640, 360) if rotation == 180 else (360, 640)
    upright = _view().render(0).resize(size)
    path = tmp_path / "rotated.mp4"
    # Stored turned clockwise by the rotation, and shown upright by turning it back.
    turn_back = {
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    stored = [
        frame.transpose(turn_back[rotation])
        for frame in _beside_a_figure(upright, (0.0, 0.0, 1.0, 0.45))
    ]
    synth.write_video(path, stored, fps=FPS, rotation=rotation)
    orientation = "landscape" if rotation == 180 else "portrait"
    layout = Layout(locate={orientation: box}, redact={})

    assert len(list(video.sample_video(path, layout=layout))) == count


def _corner_pages() -> list[Image.Image]:
    """A still screen whose bottom-right corner shows one page of text, then another."""
    frames = []
    for seed in (5, 6):
        page = _view().render(0)
        draw = ImageDraw.Draw(page)
        draw.rectangle((960, 648, 1280, 720), fill=(255, 255, 255))
        pick = random.Random(seed)
        for row in range(3):
            for column in range(12):
                if pick.random() < 0.5:
                    x, y = 970 + column * 25, 655 + row * 20
                    draw.rectangle((x, y, x + 18, y + 12), fill=(0, 0, 0))
        frames += [page] * 15
    return frames


def test_a_change_beside_a_wide_redaction_is_still_a_new_screen(tmp_path: Path) -> None:
    path = tmp_path / "corner.mp4"
    synth.write_video(path, _corner_pages(), fps=FPS)
    # All but the corner is redacted: its tiles alone tell whether the screen moved.
    corner_only = Layout(
        locate="full",
        redact={"landscape": ((0.0, 0.0, 1.0, 0.9), (0.0, 0.9, 0.75, 1.0))},
    )

    assert len(list(video.sample_video(path, layout=corner_only))) == 2


def _column_pages() -> list[Image.Image]:
    """A still screen whose column at x 600-630 shows one page of text, then another."""
    frames = []
    for seed in (5, 6):
        page = _view().render(0)
        draw = ImageDraw.Draw(page)
        draw.rectangle((600, 0, 630, 720), fill=(255, 255, 255))
        pick = random.Random(seed)
        for row in range(36):
            if pick.random() < 0.5:
                draw.rectangle((604, row * 20 + 4, 626, row * 20 + 16), fill=(0, 0, 0))
        frames += [page] * 15
    return frames


def test_a_change_in_a_narrow_strip_left_unredacted_is_a_new_screen(tmp_path: Path) -> None:
    path = tmp_path / "column.mp4"
    synth.write_video(path, _column_pages(), fps=FPS)
    # Only the column is shown, under half of every tile it crosses.
    column_only = Layout(
        locate="full",
        redact={"landscape": ((0.0, 0.0, 600 / 1280, 1.0), (630 / 1280, 0.0, 1.0, 1.0))},
    )

    assert len(list(video.sample_video(path, layout=column_only))) == 2


def test_a_change_in_the_row_beside_a_redaction_counts() -> None:
    before = np.full((256, 256), 128, dtype=np.uint8)
    after = before.copy()
    # Rows 137-143 are the shown part (7 of 16 rows) of the tiles cut by the redaction.
    after[137:144] = 255
    shown = np.ones(before.shape, dtype=np.bool_)
    shown[:137] = False
    a, b = (
        video._Sample.of_gray(np.where(shown, gray, 0).astype(np.uint8), 0, _render, shown)
        for gray in (before, after)
    )

    assert video._difference(a, b) > video.DEFAULT_MOTION_THRESHOLD


def _render() -> Image.Image:
    return Image.new("L", (1, 1))


def test_a_small_crop_is_not_enlarged_for_the_movement_check() -> None:
    band = video._Sample.of_gray(np.zeros((72, 1280), dtype=np.uint8), 0, _render)
    screen = video._Sample.of_gray(np.zeros((720, 1280), dtype=np.uint8), 0, _render)

    assert band.small.shape == (64, 256)
    assert screen.small.shape == (256, 256)


def test_a_crop_too_thin_for_the_laplacian_has_no_sharpness() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample = video._Sample.of_gray(np.full((1, 100), 128, dtype=np.uint8), 0, _render)

    assert sample.sharpness == 0


def test_the_edges_of_a_redaction_do_not_count_as_sharpness() -> None:
    gray = np.full((100, 100), 128, dtype=np.uint8)
    gray[20:60, 20:60] = 0
    shown = np.ones(gray.shape, dtype=np.bool_)
    shown[20:60, 20:60] = False

    assert video._Sample.of_gray(gray, 0, _render).sharpness > 0
    assert video._Sample.of_gray(gray, 0, _render, shown).sharpness == 0


def test_a_cropped_sample_and_a_whole_screen_are_different_screens() -> None:
    gray = np.full((256, 256), 128, dtype=np.uint8)
    cropped = video._Sample.of_gray(gray, 0, _render)
    whole = video._Sample.of_gray(gray, 0, _render, sendable=False)

    assert video._difference(cropped, whole) == 1.0
    assert video._difference(whole, cropped) == 1.0


def test_a_redacted_sample_and_one_without_redactions_are_different_screens() -> None:
    gray = np.full((256, 256), 128, dtype=np.uint8)
    shown = np.ones(gray.shape, dtype=np.bool_)
    shown[:, :128] = False
    redacted = video._Sample.of_gray(np.where(shown, gray, 0).astype(np.uint8), 0, _render, shown)
    plain = video._Sample.of_gray(gray, 0, _render)

    assert video._difference(redacted, plain) == 1.0
    assert video._difference(plain, redacted) == 1.0


def test_a_video_that_never_stands_still_falls_back_to_a_sample_it_can_crop() -> None:
    rng = np.random.default_rng(0)
    segmenter = video._Segmenter(video.DEFAULT_MOTION_THRESHOLD, video.MIN_STILL_SAMPLES)
    # Noise is far sharper than the smooth gradient, but the layout has no crop for it.
    smooth = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    samples = [
        video._Sample.of_gray(
            rng.integers(0, 256, (256, 256), dtype=np.uint8), 0, _render, sendable=False
        ),
        video._Sample.of_gray(smooth, 1, _render),
        video._Sample.of_gray(
            rng.integers(0, 256, (256, 256), dtype=np.uint8), 2, _render, sendable=False
        ),
    ]
    frames = [frame for sample in samples for frame in segmenter.add(sample)]
    frames += segmenter.finish()

    assert [frame.timestamp_ms for frame in frames] == [1000]


def _run(segmenter: video._Segmenter, samples: list[video._Sample]) -> list[int]:
    frames = [frame for sample in samples for frame in segmenter.add(sample)]
    frames += segmenter.finish()
    return [frame.timestamp_ms for frame in frames]


def test_a_view_the_layout_cannot_crop_splits_even_at_a_threshold_of_1() -> None:
    smooth = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    noise = np.random.default_rng(0).integers(0, 256, (256, 256), dtype=np.uint8)
    cropped = [video._Sample.of_gray(smooth, t, _render) for t in (0, 1)]
    whole = [video._Sample.of_gray(noise, t, _render, sendable=False) for t in (2, 3)]
    segmenter = video._Segmenter(1.0, video.MIN_STILL_SAMPLES)

    # One segment would have sent only the sharper sample, which cannot be cropped.
    assert _run(segmenter, cropped + whole) == [0, 2000]


def test_a_video_still_only_where_the_layout_cannot_crop_also_sends_one_it_can() -> None:
    rng = np.random.default_rng(0)
    still = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    whole = [video._Sample.of_gray(still, t, _render, sendable=False) for t in (0, 1)]
    # Then only scrolling where the layout can crop: no still segment of its own.
    moving = [
        video._Sample.of_gray(rng.integers(0, 256, (256, 256), dtype=np.uint8), t, _render)
        for t in (2, 3, 4)
    ]
    segmenter = video._Segmenter(video.DEFAULT_MOTION_THRESHOLD, video.MIN_STILL_SAMPLES)

    picked = _run(segmenter, whole + moving)
    assert len(picked) == 2 and picked[0] < 2000 <= picked[1]


@pytest.mark.parametrize(
    ("frame_rate", "motion_threshold"),
    [(0, 0.05), (float("nan"), 0.05), (float("inf"), 0.05), (4, -0.1), (4, 1.5), (4, float("nan"))],
)
def test_sampling_settings_are_checked(frame_rate: float, motion_threshold: float) -> None:
    with pytest.raises(ValueError, match=r"frame_rate|motion_threshold"):
        video.check_sampling(frame_rate, motion_threshold)


def test_sampling_instants_are_counted_without_stepping() -> None:
    assert video.ticks_before(0.0, 1.0, 0.25) == 4  # 0, 0.25, 0.5 and 0.75
    assert video.ticks_before(0.0, 0.3, 0.1) == 3  # 0.3 itself belongs to the frame at 0.3
    assert video.ticks_before(1.0, 0.5, 0.25) == 0
    # A huge jump in timestamps is one division, not millions of steps.
    assert video.ticks_before(0.0, 1e7, 0.25) == 40_000_000


def test_the_end_of_a_video_counts_from_its_start_time() -> None:
    frame = SimpleNamespace(duration=None, time_base=None)
    stream = SimpleNamespace(duration=None, time_base=None, start_time=None)
    container = SimpleNamespace(start_time=1_400_000, duration=5_000_000)  # microseconds

    assert video._end_time(frame, 6.0, stream, container) == 6.4


@pytest.mark.parametrize("stream_start", [Fraction(7, 5) * 1000, None])
def test_a_stream_duration_counts_from_the_start_of_the_stream(
    stream_start: Fraction | None,
) -> None:
    # The stream lasts 5 s from its start at 1.4 s; with no stream start, the origin of 1.4 s
    # comes from the container instead. Either way the last frame is shown until 5 s.
    frame = SimpleNamespace(duration=None, time_base=None)
    stream = SimpleNamespace(duration=5000, time_base=Fraction(1, 1000), start_time=stream_start)
    container = SimpleNamespace(start_time=1_400_000, duration=None)
    origin = video._start_time(stream, container)

    assert origin == pytest.approx(1.4)
    assert video._end_time(frame, 4.0, stream, container, origin) == 5.0


def test_timestamps_count_from_the_start_of_the_stream(tmp_path: Path) -> None:
    # The MP4 records that its stream starts at 1.4 s, so its first frame is at 0 in the video.
    path = tmp_path / "late.mp4"
    synth.write_video(path, [_view().render(0)] * 20, fps=FPS, timestamps_ms=range(1400, 3400, 100))

    (frame,) = video.sample_video(path)

    assert frame.timestamp_ms < 1400  # 1400 or later on the container's own clock


class _StubContainer:
    """A container FFmpeg will not write: its video stream decodes to ``frames``, and neither the
    frames, the stream nor the container has a duration."""

    def __init__(self, frames: list[object]) -> None:
        stream = SimpleNamespace(
            thread_type=None,
            average_rate=None,
            guessed_rate=None,
            start_time=None,
            time_base=None,
            duration=None,
        )
        self.streams = SimpleNamespace(video=[stream])
        self.start_time = None
        self.duration = None
        self._frames = frames

    def __enter__(self) -> _StubContainer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def decode(self, stream: object) -> Iterator[object]:
        return iter(self._frames)


def _stub_av(monkeypatch, frames: list[object]) -> None:
    av = video.import_av()
    fake_av = SimpleNamespace(open=lambda path: _StubContainer(frames), error=av.error)
    monkeypatch.setattr(video, "import_av", lambda *args, **kwargs: fake_av)


def test_a_video_without_frames_is_unreadable(tmp_path: Path, monkeypatch) -> None:
    _stub_av(monkeypatch, [])

    with pytest.raises(ValueError, match="no frames"):
        list(video.sample_video(tmp_path / "empty.mkv"))


def test_a_one_frame_video_without_any_duration_is_still_read(tmp_path: Path, monkeypatch) -> None:
    image = _view().render(0)
    frame = SimpleNamespace(
        time=0.0,
        duration=None,
        time_base=None,
        rotation=0,
        to_ndarray=lambda format: np.asarray(image.convert("L")),
        to_image=lambda: image,
    )
    _stub_av(monkeypatch, [frame])

    (picked,) = video.sample_video(tmp_path / "one-frame.mkv")

    assert picked.timestamp_ms == 0


def test_an_error_while_picking_the_last_frame_is_an_unreadable_video(
    scrolling, monkeypatch
) -> None:
    path, _ = scrolling
    av = video.import_av()

    def fails(self: Any) -> Iterator[Any]:
        raise av.error.FFmpegError(1, "decoder failed")
        yield

    monkeypatch.setattr(video._Segmenter, "finish", fails)

    with pytest.raises(ValueError, match="decoder failed"):
        list(video.sample_video(path))
