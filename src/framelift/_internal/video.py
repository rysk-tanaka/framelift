"""Pick the frames of a screen recording worth reading: one sharp frame per still screen.

Frames are sampled at a fixed rate and split into still segments by how much each sample moved.
Movement is the share of tiles of a small grayscale copy that changed, rather than a perceptual
hash, which is built to ignore small shifts and blur and so cannot tell a scrolling frame from a
still one. Counting changed tiles, rather than averaging the whole screen, catches a screen whose
layout stays the same while its text changes (a list scrolled by whole rows, a page or tab
switch), and ignores a small animation such as a clock or an effect in one corner.
Each still segment contributes its sharpest sample, so samples taken while the screen moves are
not sent; only a video that never stands still is read once, from its sharpest sample.

Each sample is the frame on screen at a sampling instant, so a variable-frame-rate recording that
encodes a still screen as one long frame counts as one sample per instant it was shown.

Needs the ``video`` extra (PyAV).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import imagehash
import numpy as np
from PIL import Image

from framelift._internal.layout import encode_png
from framelift._internal.sources import ImageSource, UnreadableInputError, error_reason

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from framelift._internal.layout import Layout

DEFAULT_FRAME_RATE = 4.0
DEFAULT_MOTION_THRESHOLD = 0.05
MIN_STILL_SAMPLES = 2

# A sample is shrunk to _DIFF_SIZE and split into _TILES x _TILES tiles; a tile changed when its
# mean absolute difference (0-255) exceeds _TILE_CHANGE. Measured on the synthetic shop screen,
# scrolling by whole rows or swapping the page changes 10% of the tiles or more, while codec
# noise, a small animation and one changed row stay below 2%.
_DIFF_SIZE = (256, 256)
_TILES = 16
_TILE_CHANGE = 3.0
# Sampling instants are computed in floating point; without a tolerance, 0.1 + 0.1 + 0.1 lands
# just after a frame at 0.3 s and the instant would be credited to the frame before it.
_TIME_TOLERANCE = 1e-6
_TRANSPOSE = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


@dataclass(frozen=True)
class VideoFrame:
    """The sharpest sample of one still segment."""

    index: int
    """Position among the frames picked from this video, from 0."""
    timestamp_ms: int
    image: Image.Image
    """Upright, with the video's display rotation applied."""
    phash: str
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


def upright(image: Image.Image, rotation: float) -> Image.Image:
    """Apply a video's display rotation (counterclockwise degrees, as PyAV reports it)."""
    transpose = _TRANSPOSE.get(round(rotation) % 360)
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


def _difference(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """The share of tiles (0-1) that changed between two small grayscale copies."""
    size = _DIFF_SIZE[0] // _TILES
    tiles = np.abs(a - b).reshape(_TILES, size, _TILES, size).mean(axis=(1, 3))
    return float((tiles > _TILE_CHANGE).mean())


def _laplacian_variance(gray: NDArray[np.float32]) -> float:
    """Variance of the Laplacian: higher for sharper images, low for motion blur."""
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


@dataclass(frozen=True)
class _Sample:
    timestamp_ms: int
    small: NDArray[np.float32]
    sharpness: float
    render: Callable[[], Image.Image]
    """Builds the full, upright color image; called only for a sample that is picked."""

    @classmethod
    def of_gray(
        cls, gray: NDArray[np.uint8], time: float, render: Callable[[], Image.Image]
    ) -> _Sample:
        # One grayscale array serves both the movement check and the sharpness.
        small_image = Image.fromarray(gray).resize(_DIFF_SIZE, Image.Resampling.BILINEAR)
        small = np.asarray(small_image, dtype=np.float32)
        sharp = _laplacian_variance(gray.astype(np.float32))
        return cls(round(time * 1000), small, sharp, render)

    @classmethod
    def of(cls, image: Image.Image, time: float) -> _Sample:
        return cls.of_gray(np.asarray(image.convert("L")), time, lambda: image)


class _Segmenter:
    """Splits samples into still segments and yields each segment's sharpest sample.

    Keeps a constant amount of state, so long videos do not grow memory: the current segment's
    best sample, the small copies of its first and of the previous sample, and (until the first
    segment is kept) the sharpest sample so far as a fallback.
    """

    def __init__(self, motion_threshold: float, min_still_samples: int) -> None:
        self._threshold = motion_threshold
        self._min_samples = min_still_samples
        self._first: NDArray[np.float32] | None = None
        self._previous: NDArray[np.float32] | None = None
        self._best: _Sample | None = None
        self._count = 0
        self._fallback: _Sample | None = None
        self._emitted = 0

    def add(self, sample: _Sample, repeats: int = 1) -> Iterator[VideoFrame]:
        """Add a sample that was on screen at ``repeats`` sampling instants in a row."""
        if self._previous is not None and self._first is not None:
            # Comparing with the segment's first sample too catches slow scrolling, which moves
            # little between samples but replaces the whole screen over time.
            moved = (
                _difference(sample.small, self._previous) > self._threshold
                or _difference(sample.small, self._first) > self._threshold
            )
            if moved:
                yield from self._close()
        if self._count == 0:
            self._first = sample.small
        self._count += repeats
        self._previous = sample.small
        if self._best is None or sample.sharpness > self._best.sharpness:
            self._best = sample
        if self._emitted == 0 and (
            self._fallback is None or sample.sharpness > self._fallback.sharpness
        ):
            self._fallback = sample

    def finish(self) -> Iterator[VideoFrame]:
        yield from self._close()
        # A video that never stood still still gets read once, from its sharpest sample.
        if self._emitted == 0 and self._fallback is not None:
            yield self._emit(self._fallback)

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
            phash=str(imagehash.phash(image)),
            sharpness=sample.sharpness,
        )
        self._emitted += 1
        self._fallback = None
        return frame


def sample_video(
    path: Path,
    *,
    frame_rate: float = DEFAULT_FRAME_RATE,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    min_still_samples: int = MIN_STILL_SAMPLES,
) -> Generator[VideoFrame, None, None]:
    """Yield the sharpest frame of every still segment of a video, in order.

    Frames are sampled ``frame_rate`` times per second. A sample joins the current segment when,
    compared with both the previous sample and the segment's first sample, at most
    ``motion_threshold`` of the screen (the share of 16x16 tiles, 0-1) changed. Segments
    with fewer than ``min_still_samples`` samples are movement and are dropped, so a screen shown
    for less than about ``min_still_samples / frame_rate`` seconds can be missed. A video with no
    still segment yields its sharpest sample.

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
                        yield from segmenter.add(_to_sample(*shown), repeats)
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
                yield from segmenter.add(_to_sample(last_frame, last_time), repeats)
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


def _to_sample(frame: Any, time: float) -> _Sample:
    # Movement and sharpness need only the luma plane, at full size but without converting to
    # RGB or rotating: every sample of a stream has the same rotation, and the Laplacian's
    # variance does not change under a quarter turn. The color image is built only for the
    # samples that are sent.
    gray = frame.to_ndarray(format="gray")
    return _Sample.of_gray(gray, time, lambda: upright(frame.to_image(), frame.rotation))


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
        # The picked frame's hash describes the whole screen; the record is of what was sent.
        phash=frame.phash if layout.is_identity else str(imagehash.phash(view)),
        source_sha256=source_sha256,
        crop=crop,
        view=view,
        frame_index=frame.index,
        timestamp_ms=frame.timestamp_ms,
    )
