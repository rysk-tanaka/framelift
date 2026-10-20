"""Pick the frames of a screen recording worth reading: one sharp frame per still screen.

Frames are sampled at a fixed rate and split into still segments by how much each sample moved.
Movement is the share of tiles of a small grayscale copy that changed, rather than a perceptual
hash, which is built to ignore small shifts and blur and so cannot tell a scrolling frame from a
still one. Counting changed tiles, rather than averaging the whole screen, catches a screen whose
layout stays the same while its text changes (a list scrolled by whole rows, a page or tab
switch), and ignores a small animation such as a clock or an effect in one corner.
Each still segment contributes its sharpest sample, so samples taken while the screen moves are
not sent; only a video that never stands still is read once, from its sharpest sample.

Given a `Layout`, movement and sharpness are measured on the part of the screen that is sent, so
that movement elsewhere (a character's idle animation beside the list being read) does not split
a still screen. Each tile counts toward the share that changed by how much of it is not
redacted, so a wide redaction does not hide the change in the rest, nor a narrow strip left
unredacted its own; the sharpness leaves out the edges of redactions. A
small crop is shrunk less than a whole screen, but always a few times, so that codec noise is
averaged away before the threshold sees it.

Each sample is the frame on screen at a sampling instant, so a variable-frame-rate recording that
encodes a still screen as one long frame counts as one sample per instant it was shown.

Needs the ``video`` extra (PyAV).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import imagehash
import numpy as np
from PIL import Image

from framelift._internal.layout import NoCropError, cut, encode_png, orientation_of
from framelift._internal.sources import ImageSource, UnreadableInputError, error_reason

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from framelift._internal.layout import Layout
    from framelift.profile import Orientation

DEFAULT_FRAME_RATE = 4.0
DEFAULT_MOTION_THRESHOLD = 0.05
MIN_STILL_SAMPLES = 2

# A sample is shrunk to at most _DIFF_SIZE pixels a side and split into _TILES x _TILES tiles; a
# tile changed when its mean absolute difference (0-255) exceeds _TILE_CHANGE. Measured on the
# synthetic shop screen, scrolling by whole rows or swapping the page changes 10% of the tiles or
# more, while codec noise, a small animation and one changed row stay below 2%.
_DIFF_SIZE = 256
_TILES = 16
_TILE_CHANGE = 3.0
# A sample is shrunk at least this many times a side, so that codec noise, which the threshold
# would take for change pixel by pixel, is averaged away as it is on a whole screen. Not more:
# the few pixels a slow scroll moves between samples stay visible when shrunk up to about as
# many times as a whole 720p screen is (2.8 down to 256 rows), and vanish beyond.
_MIN_SHRINK = 2.5
# Sampling instants are computed in floating point; without a tolerance, 0.1 + 0.1 + 0.1 lands
# just after a frame at 0.3 s and the instant would be credited to the frame before it.
_TIME_TOLERANCE = 1e-6
# Indexed by counterclockwise quarter turns (see _quarter_turns).
_TRANSPOSE = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
)


@dataclass(frozen=True)
class VideoFrame:
    """The sharpest sample of one still segment."""

    index: int
    """Position among the frames picked from this video, from 0."""
    timestamp_ms: int
    image: Image.Image
    """Upright, with the video's display rotation applied."""
    sharpness: float


class MissingVideoExtraError(ImportError):
    """PyAV is not installed. A subclass of its own, so callers can tell it from an
    `ImportError` raised by other code, such as a profile's validator."""


def import_av(purpose: str = "Video input", hint: str = "") -> Any:
    """Import PyAV, or raise `MissingVideoExtraError` saying how to install the ``video`` extra
    (followed by ``hint``, if any)."""
    try:
        import av
    except ImportError as exc:
        message = f'{purpose} needs the video extra: pip install "framelift[video]"{hint}'
        raise MissingVideoExtraError(message) from exc
    return av


def check_frame_rate(frame_rate: float) -> float:
    """Return ``frame_rate``, or raise `ValueError` unless it is finite and > 0 (NaN and infinity
    would otherwise fail later, far from the cause)."""
    if not (math.isfinite(frame_rate) and frame_rate > 0):
        raise ValueError(f"frame_rate must be a finite number > 0, got {frame_rate}")
    return frame_rate


def check_motion_threshold(motion_threshold: float) -> float:
    """Return ``motion_threshold``, or raise `ValueError` unless it is between 0 and 1."""
    if not 0 <= motion_threshold <= 1:
        raise ValueError(f"motion_threshold must be between 0 and 1, got {motion_threshold}")
    return motion_threshold


def check_sampling(frame_rate: float, motion_threshold: float) -> None:
    """Raise `ValueError` for an invalid ``frame_rate`` or ``motion_threshold``."""
    check_frame_rate(frame_rate)
    check_motion_threshold(motion_threshold)


def _valid_rate(rate: Any) -> float | None:
    # PyAV reports an unknown rate as None, 0/1 or 0/0; only a finite, positive one is usable.
    try:
        value = float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def frame_time(time: float | None, number: int, rates: Iterable[Any]) -> float | None:
    """A decoded frame's time in seconds, or None when it cannot be determined.

    Uses the frame's own time (PyAV's ``pts * time_base``), then the frame number divided by the
    first usable stream rate (average, then guessed).
    """
    if time is not None:
        return time
    for rate in rates:
        value = _valid_rate(rate)
        if value is not None:
            return number / value
    return None


def _quarter_turns(rotation: float) -> int:
    """A display rotation (counterclockwise degrees, as PyAV reports it) as counterclockwise
    quarter turns, 0-3; 0 for an angle that is not a quarter turn."""
    degrees = round(rotation) % 360
    return degrees // 90 if degrees % 90 == 0 else 0


def upright(image: Image.Image, rotation: float) -> Image.Image:
    """Apply a video's display rotation (counterclockwise degrees, as PyAV reports it)."""
    transpose = _TRANSPOSE[_quarter_turns(rotation)]
    return image.transpose(transpose) if transpose is not None else image


def ticks_before(next_time: float, time: float, step: float) -> int:
    """How many sampling instants, starting at ``next_time`` and ``step`` apart, come before a
    frame shown at ``time``; the frame before it was on screen at each of them.

    Computed directly rather than by stepping, so a large jump in timestamps stays cheap.
    """
    span = time - _TIME_TOLERANCE - next_time
    if span <= 0:
        return 0
    # The small slack keeps an instant that lands exactly on ``time`` from counting twice.
    return math.ceil(span / step - 1e-9)


def _diff_size(width: int, height: int) -> tuple[int, int]:
    """The size a sample is shrunk to: at most _DIFF_SIZE a side, shrunk at least _MIN_SHRINK
    times, and a multiple of _TILES. A side that would be under _TILES is _TILES, the only case
    where a sample is enlarged, so that it still has _TILES x _TILES tiles."""

    def side(length: int) -> int:
        return max(_TILES, min(_DIFF_SIZE, int(length / _MIN_SHRINK) // _TILES * _TILES))

    return side(width), side(height)


def _tiles(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """The sum of ``values`` in each of the _TILES x _TILES tiles."""
    height, width = values.shape
    return values.reshape(_TILES, height // _TILES, _TILES, width // _TILES).sum(axis=(1, 3))


def _same_view(a: _Measured, b: _Measured) -> bool:
    """Whether two samples measured the same view: screens of one orientation, both cropped by
    the layout or both measured whole, both redacted or neither. The layout's crop and
    redactions depend only on the orientation, so such samples measured the same part of the
    screen, whatever its resolution. Samples that fail this always count as different screens,
    whatever the motion threshold."""
    return (
        a.orientation == b.orientation
        and a.sendable == b.sendable
        and (a.shown is None) == (b.shown is None)
    )


def _resized(values: NDArray[np.float32], shape: tuple[int, int]) -> NDArray[np.float32]:
    if values.shape == shape:
        return values
    height, width = shape
    image = Image.fromarray(values).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def _difference(a: _Measured, b: _Measured) -> float:
    """The share of tiles (0-1) that changed between two samples, each tile counted by how much
    of it is not redacted; 1 for samples that did not measure the same view (see `_same_view`).

    Samples of different sizes, as after the resolution changed, are compared at the smaller
    size: each covers the whole of what it measured, so shrinking both to one size lines them
    up, as every sample was shrunk to 256x256 before 0.5.
    """
    if not _same_view(a, b):
        return 1.0
    shape = (min(a.small.shape[0], b.small.shape[0]), min(a.small.shape[1], b.small.shape[1]))
    small_a, shown_a = a.at(shape)
    small_b, shown_b = b.at(shape)
    if shown_a is None or shown_b is None:
        height, width = shape
        means = _tiles(np.abs(small_a - small_b)) / (height * width / _TILES**2)
        return float((means > _TILE_CHANGE).mean())
    # The redacted parts are black in both, and must not dilute a tile's change. Samples of one
    # size share their redactions, and usually their mask (see _View); otherwise the smaller
    # mask keeps the comparison the same both ways.
    shown = shown_a if shown_a is shown_b else np.minimum(shown_a, shown_b)
    weight = _tiles(shown)
    total = weight.sum()
    if total <= 0:
        return 0.0  # all redacted: nothing to compare
    means = _tiles(np.abs(small_a - small_b) * shown) / np.maximum(weight, 1e-9)
    # Each tile counts by how much of it is shown, so a thin strip left unredacted, or a row
    # beside a redaction, still counts; a tile barely shown, whose mean is noisy, barely does.
    return float(weight[means > _TILE_CHANGE].sum() / total)


def _clear(shown: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The pixels inside the border whose Laplacian sees no redacted pixel."""
    return (
        shown[:-2, 1:-1] & shown[2:, 1:-1] & shown[1:-1, :-2] & shown[1:-1, 2:] & shown[1:-1, 1:-1]
    )


def _laplacian_variance(gray: NDArray[np.float32], clear: NDArray[np.bool_] | None) -> float:
    """Variance of the Laplacian: higher for sharper images, low for motion blur.

    Only the ``clear`` pixels (see `_clear`) count, when given, so that the edges of a redaction
    do not swamp the difference between a sharp and a blurred sample; 0 when no pixel counts,
    as in a crop less than 3 pixels high or wide.
    """
    # The Laplacian needs a pixel on each side.
    if min(gray.shape) < 3:
        return 0.0
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    if clear is None:
        return float(laplacian.var())
    return float(laplacian[clear].var()) if clear.any() else 0.0


@dataclass(frozen=True, eq=False)
class _View:
    """What is measured of every screen of one size and rotation in a video: the layout's crop
    and redactions, which depend on nothing else. Worked out once, so that the masks are not
    built again for every sample."""

    size: tuple[int, int]
    """The size samples are shrunk to (see `_diff_size`)."""
    shown: NDArray[np.bool_] | None
    """Which measured pixels are not redacted; ``None`` when none are."""
    small_shown: NDArray[np.float32] | None
    """``shown`` shrunk to ``size``, shared by every sample of the view."""
    clear: NDArray[np.bool_] | None
    """The pixels the sharpness counts (see `_clear`)."""
    sendable: bool
    orientation: Orientation
    turns: int = 0
    """Counterclockwise quarter turns applied to the decoded plane before it is measured: those
    that make it upright with a layout, 0 without one."""
    box: tuple[int, int, int, int] | None = None
    """The pixels of the turned plane measured (left, top, right, bottom); ``None`` for all. In a
    view `_Views` makes, only a view with a box has ``shown``."""

    @classmethod
    def of(
        cls,
        width: int,
        height: int,
        shown: NDArray[np.bool_] | None = None,
        *,
        sendable: bool = True,
        orientation: Orientation | None = None,
        turns: int = 0,
        box: tuple[int, int, int, int] | None = None,
    ) -> _View:
        """The view of a measured part ``width`` x ``height``; ``orientation`` is the whole
        screen's, by default the part's."""
        size = _diff_size(width, height)
        small_shown = clear = None
        if shown is not None:
            mask = Image.fromarray(shown.astype(np.float32))
            small_shown = np.asarray(mask.resize(size, Image.Resampling.BILINEAR))
            clear = _clear(shown)
        if orientation is None:
            orientation = orientation_of((width, height))
        return cls(size, shown, small_shown, clear, sendable, orientation, turns, box)


class _Views:
    """The `_View` of one video's decoded size and rotation, worked out when it changes.

    Only the last is kept: samples come in order and the size seldom changes, while a recording
    resized again and again would otherwise keep a full-size mask for every size it had.
    """

    def __init__(self, layout: Layout | None) -> None:
        self._layout = None if layout is None or layout.is_identity else layout
        self._key: tuple[tuple[int, int], int] | None = None
        self._view: _View | None = None

    def get(self, shape: tuple[int, int], rotation: float) -> _View:
        """The view of a decoded luma plane of ``shape`` (height, width)."""
        key = (shape, _quarter_turns(rotation))
        if self._view is None or key != self._key:
            self._key, self._view = key, self._make(*key)
        return self._view

    def _make(self, shape: tuple[int, int], turns: int) -> _View:
        height, width = shape
        if self._layout is None:
            # Not even turned upright: every sample of a stream has the same rotation, and the
            # Laplacian's variance does not change under a quarter turn.
            return _View.of(width, height)
        # The layout's boxes are of the upright screen.
        if turns % 2:
            width, height = height, width
        orientation = orientation_of((width, height))
        try:
            box, shown = self._layout.mask((width, height))
        except NoCropError:
            # The whole screen, then; frame_to_source reports the picked frame as uncropped.
            return _View.of(width, height, sendable=False, orientation=orientation, turns=turns)
        left, top, right, bottom = box
        return _View.of(
            right - left, bottom - top, shown, orientation=orientation, turns=turns, box=box
        )


@dataclass(frozen=True, eq=False)
class _Measured:
    """What a sample is compared by. Holds no decoded frame, so it can be kept for comparing
    while the frame is let go. Compared by identity: its fields are arrays."""

    small: NDArray[np.float32]
    shown: NDArray[np.float32] | None
    """How much of each pixel of ``small`` is not redacted (0-1); ``None`` when none is."""
    sendable: bool
    """False when the layout had no crop for the screen, so the whole screen was measured and
    `frame_to_source` cannot send the sample."""
    orientation: Orientation
    """Of the whole screen, which decides the layout's crop and redactions: upright with a
    layout; without one, as decoded, which is the same for every sample of a stream."""
    _shrunk: dict[tuple[int, int], tuple[NDArray[np.float32], NDArray[np.float32] | None]] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def at(self, shape: tuple[int, int]) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
        """``small`` and ``shown`` shrunk to ``shape``, kept for the last shape asked, so that
        a segment's first sample is not shrunk again for every sample after the resolution
        changed."""
        if self.small.shape == shape:
            return self.small, self.shown
        shrunk = self._shrunk.get(shape)
        if shrunk is None:
            self._shrunk.clear()
            shown = None if self.shown is None else _resized(self.shown, shape)
            shrunk = self._shrunk[shape] = (_resized(self.small, shape), shown)
        return shrunk


@dataclass(frozen=True)
class _Sample:
    timestamp_ms: int
    measured: _Measured
    sharpness: float
    render: Callable[[], Image.Image]
    """Builds the full, upright color image; called only for a sample that is picked. Holds the
    decoded frame."""
    pixels: int
    """How many pixels were measured, before they were shrunk to ``measured.small``."""

    @classmethod
    def of_gray(
        cls,
        gray: NDArray[np.uint8],
        time: float,
        render: Callable[[], Image.Image],
        shown: NDArray[np.bool_] | None = None,
        *,
        sendable: bool = True,
        orientation: Orientation | None = None,
    ) -> _Sample:
        """``orientation`` is the whole screen's; by default, ``gray``'s."""
        height, width = gray.shape
        view = _View.of(width, height, shown, sendable=sendable, orientation=orientation)
        return cls.of_view(gray, time, render, view)

    @classmethod
    def of_view(
        cls, gray: NDArray[np.uint8], time: float, render: Callable[[], Image.Image], view: _View
    ) -> _Sample:
        """A sample of the part ``view`` measures, ``gray``, already redacted."""
        # One grayscale array serves both the movement check and the sharpness.
        height, width = gray.shape
        small_image = Image.fromarray(gray).resize(view.size, Image.Resampling.BILINEAR)
        small = np.asarray(small_image, dtype=np.float32)
        sharp = _laplacian_variance(gray.astype(np.float32), view.clear)
        measured = _Measured(small, view.small_shown, view.sendable, view.orientation)
        return cls(round(time * 1000), measured, sharp, render, height * width)

    @classmethod
    def of(cls, image: Image.Image, time: float) -> _Sample:
        return cls.of_gray(np.asarray(image.convert("L")), time, lambda: image)


class _Segmenter:
    """Splits samples into still segments and yields each segment's sharpest sample.

    Keeps a constant amount of state, so long videos do not grow memory. Of the decoded frames,
    it holds only the current segment's best sample's and (until a sample the layout can crop is
    yielded) the fallback's; the segment's first and the previous sample keep only what they are
    compared by. `sample_video` holds the frames it is decoding besides.
    """

    def __init__(self, motion_threshold: float, min_still_samples: int) -> None:
        self._threshold = motion_threshold
        self._min_samples = min_still_samples
        self._first: _Measured | None = None
        self._previous: _Measured | None = None
        self._best: _Sample | None = None
        self._count = 0
        self._fallback: _Sample | None = None
        self._emitted = 0
        self._sent = False  # whether a sample the layout can crop was yielded

    def add(self, sample: _Sample, repeats: int = 1) -> Iterator[VideoFrame]:
        """Add a sample that was on screen at ``repeats`` sampling instants in a row."""
        if self._previous is not None and self._first is not None:
            # Comparing with the segment's first sample too catches slow scrolling, which moves
            # little between samples but replaces the whole screen over time.
            # Another view splits even at a threshold of 1, which a share of 1 does not exceed.
            measured = sample.measured
            moved = (
                not _same_view(measured, self._previous)
                or _difference(measured, self._previous) > self._threshold
                or _difference(measured, self._first) > self._threshold
            )
            if moved:
                yield from self._close()
        if self._count == 0:
            self._first = sample.measured
        self._count += repeats
        self._previous = sample.measured
        # More pixels first: after the resolution changed, a segment holds both, and a frame
        # shrunk to fewer pixels has a higher variance of the Laplacian without being sharper.
        if self._best is None or (sample.pixels, sample.sharpness) > (
            self._best.pixels,
            self._best.sharpness,
        ):
            self._best = sample
        # The fallback is sent, so a sample the layout can crop beats a sharper one it cannot.
        if not self._sent and (
            self._fallback is None
            or (sample.measured.sendable, sample.pixels, sample.sharpness)
            > (self._fallback.measured.sendable, self._fallback.pixels, self._fallback.sharpness)
        ):
            self._fallback = sample

    def finish(self) -> Iterator[VideoFrame]:
        yield from self._close()
        # A video that never stood still still gets read once, from its sharpest sample; so does
        # one that stood still only where the layout cannot crop, from a sample it can.
        fallback = self._fallback
        if (
            not self._sent
            and fallback is not None
            and (self._emitted == 0 or fallback.measured.sendable)
        ):
            yield self._emit(fallback)

    def _close(self) -> Iterator[VideoFrame]:
        best, count = self._best, self._count
        self._best, self._count, self._first = None, 0, None
        if best is not None and count >= self._min_samples:
            yield self._emit(best)

    def _emit(self, sample: _Sample) -> VideoFrame:
        image = sample.render()
        frame = VideoFrame(
            index=self._emitted,
            timestamp_ms=sample.timestamp_ms,
            image=image,
            sharpness=sample.sharpness,
        )
        self._emitted += 1
        if sample.measured.sendable:
            self._sent = True
            self._fallback = None
        return frame


def sample_video(
    path: Path,
    *,
    frame_rate: float = DEFAULT_FRAME_RATE,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    min_still_samples: int = MIN_STILL_SAMPLES,
    layout: Layout | None = None,
) -> Generator[VideoFrame, None, None]:
    """Yield the sharpest frame of every still segment of a video, in order; a fallback (below)
    comes last, and can be earlier than the frames before it.

    Frames are sampled ``frame_rate`` times per second. A sample joins the current segment when,
    compared with both the previous sample and the segment's first sample, at most
    ``motion_threshold`` of the screen (the share of 16x16 tiles, 0-1) changed. Segments
    with fewer than ``min_still_samples`` samples are movement and are dropped, so a screen shown
    for less than about ``min_still_samples / frame_rate`` seconds can be missed. A video with no
    still segment yields its sharpest sample, as does one still only where ``layout`` has no
    crop, from a sample it can crop. With a ``layout``, both are measured on the part of
    the screen it sends: each tile counts toward the share by how much of it is not redacted, and
    the sharpness only pixels clear of redactions. The frames yielded are still whole screens.

    Raises `MissingVideoExtraError` without PyAV and `UnreadableInputError` (a `ValueError`) for
    unreadable videos.
    """
    check_sampling(frame_rate, motion_threshold)
    av = import_av()
    try:
        container = av.open(str(path))
    except av.error.FFmpegError as exc:
        raise UnreadableInputError("video", path, error_reason(exc)) from exc
    with container:
        if not container.streams.video:
            raise UnreadableInputError("video", path, "no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # decode with several threads; long recordings are common
        rates = (stream.average_rate, stream.guessed_rate)
        # Frame times count from the container's clock; timestamps are reported from the start
        # of the stream, which is not 0 in every .mov, .mkv or .webm.
        origin = _start_time(stream, container)
        segmenter = _Segmenter(motion_threshold, min_still_samples)
        views = _Views(layout)
        step = 1 / frame_rate
        next_time: float | None = None
        # The frame on screen until the next one arrives; it is converted only if a sampling
        # instant falls in that time.
        shown: tuple[Any, float] | None = None
        try:
            for number, frame in enumerate(container.decode(stream)):
                time = frame_time(frame.time, number, rates)
                if time is None:
                    raise UnreadableInputError("video", path, "cannot determine frame times")
                if frame.time is not None:  # a time from the frame number already starts at 0
                    time -= origin
                if next_time is None:
                    next_time = time
                if shown is not None:
                    repeats = ticks_before(next_time, time, step)
                    if repeats:
                        next_time += repeats * step
                        yield from segmenter.add(_to_sample(*shown, views), repeats)
                shown = (frame, time)
            if shown is None or next_time is None:
                raise UnreadableInputError("video", path, "no frames")
            # The last frame stays on screen until the video ends, and a variable-frame-rate
            # recording may write its final still screen as that one frame. When the end is
            # unknown, the frame covers only the instant it lands on, so a one-frame video is
            # still read.
            last_frame, last_time = shown
            end = _end_time(last_frame, last_time, stream, container, origin)
            if end is not None and end > last_time + _TIME_TOLERANCE:
                repeats = ticks_before(next_time, end, step)
            else:
                repeats = 1 if next_time <= last_time + _TIME_TOLERANCE else 0
            if repeats:
                yield from segmenter.add(_to_sample(last_frame, last_time, views), repeats)
            # Inside the try: the last segment's frame is converted to color here.
            yield from segmenter.finish()
        except av.error.FFmpegError as exc:
            raise UnreadableInputError("video", path, error_reason(exc)) from exc


def _start_time(stream: Any, container: Any) -> float:
    """When the stream starts on the container's clock, in seconds (0 when unknown)."""
    if stream.start_time is not None and stream.time_base is not None:
        return float(stream.start_time * stream.time_base)
    if container.start_time is not None:
        return container.start_time / 1_000_000  # av.time_base: microseconds
    return 0.0


def _end_time(
    frame: Any, time: float, stream: Any, container: Any, origin: float = 0.0
) -> float | None:
    """When the last frame (shown at ``time``) stops being shown, in seconds from ``origin``
    (the stream's start): from its own duration, else the stream's or the container's. None when
    none of them is known."""
    if frame.duration and frame.time_base is not None:
        return time + float(frame.duration * frame.time_base)
    if stream.duration and stream.time_base is not None:
        # Measured from the stream's start, which is the origin, even when the stream does not
        # say where it starts and the origin came from the container.
        return float(stream.duration * stream.time_base)
    if container.duration:
        # Both in av.time_base units (microseconds); the duration counts from the start time.
        return ((container.start_time or 0) + container.duration) / 1_000_000 - origin
    return None


def _to_sample(frame: Any, time: float, views: _Views) -> _Sample:
    # Movement and sharpness need only the luma plane, at full size but without converting to
    # RGB: the color image is built only for the samples that are sent.
    gray = frame.to_ndarray(format="gray")

    def render() -> Image.Image:
        return upright(frame.to_image(), frame.rotation)

    view = views.get(gray.shape, frame.rotation)
    # Counterclockwise quarter turns, as upright turns the color image.
    if view.turns:
        gray = np.rot90(gray, k=view.turns)
    if view.box is not None:
        gray = cut(gray, view.box, view.shown)
    return _Sample.of_view(np.ascontiguousarray(gray), time, render, view)


def frame_to_source(
    path: Path, source_sha256: str, frame: VideoFrame, layout: Layout
) -> ImageSource:
    """Redact and crop a picked frame with ``layout`` and encode it as PNG, ready to send.

    Raises `NoCropError` when ``layout`` has no crop box for the frame's orientation.
    """
    view, crop = layout.apply(frame.image)
    data = encode_png(view)
    return ImageSource(
        path=path,
        data=data,
        media_type="image/png",
        sha256=hashlib.sha256(data).hexdigest(),
        # Of what is sent, as for a still image.
        phash=str(imagehash.phash(view)),
        source_sha256=source_sha256,
        crop=crop,
        view=view,
        frame_index=frame.index,
        timestamp_ms=frame.timestamp_ms,
    )
