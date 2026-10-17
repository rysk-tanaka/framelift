"""Synthetic sample generator: screenshots of a fictional shop with known ground truth.

The shop, item names and prices are invented, so the images can be shared, used in tests and
used as a public benchmark without any real game or app content.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import random
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from framelift._internal.dataset import DatasetCase, write_dataset
from framelift._internal.sources import hash_file
from framelift._internal.video import import_av

if TYPE_CHECKING:
    from collections.abc import Iterable

Orientation = Literal["portrait", "landscape"]

_SHOP_PREFIXES = ["Moonlit", "Copper", "Wandering", "Silverleaf", "Lantern", "Hollow", "Amber"]
_SHOP_SUFFIXES = ["Emporium", "Bazaar", "Trading Post", "Market", "Workshop", "Pantry"]
_ITEM_ADJECTIVES = ["Moon", "Ember", "Frost", "Dew", "Iron", "Velvet", "Starlit", "Mossy", "Sun"]
_ITEM_NOUNS = ["Herb", "Ingot", "Thread", "Lantern", "Tonic", "Feather", "Pebble", "Bread", "Rope"]

_SIZES: dict[Orientation, tuple[int, int]] = {"portrait": (720, 1280), "landscape": (1280, 720)}
_HEADER_HEIGHT, _LIST_TOP, _ROW_HEIGHT = 110, 150, 84
_MIN_ITEMS, _MAX_ITEMS = 4, 7


def _max_rows(orientation: Orientation) -> int:
    height = _SIZES[orientation][1]
    return min(_MAX_ITEMS, (height - _LIST_TOP) // _ROW_HEIGHT)


@dataclass(frozen=True)
class SampleItem:
    name: str
    price: int


@dataclass(frozen=True)
class SampleTruth:
    file: str
    orientation: Orientation
    shop_name: str
    items: list[SampleItem]


def _random_truth(
    rng: random.Random, file: str, orientation: Orientation, count: int | None = None
) -> SampleTruth:
    shop = f"{rng.choice(_SHOP_PREFIXES)} {rng.choice(_SHOP_SUFFIXES)}"
    names: set[str] = set()
    if count is None:
        count = rng.randint(_MIN_ITEMS, _max_rows(orientation))
    if count > len(_ITEM_ADJECTIVES) * len(_ITEM_NOUNS):
        raise ValueError("too many items for the fictional item names")
    while len(names) < count:
        names.add(f"{rng.choice(_ITEM_ADJECTIVES)} {rng.choice(_ITEM_NOUNS)}")
    items = [SampleItem(name=n, price=rng.randrange(10, 2000, 5)) for n in sorted(names)]
    return SampleTruth(file=file, orientation=orientation, shop_name=shop, items=items)


_BACKGROUND = (34, 30, 46)


def _draw_header(draw: ImageDraw.ImageDraw, width: int, shop_name: str) -> None:
    title = ImageFont.load_default(size=40)
    draw.rectangle((0, 0, width, _HEADER_HEIGHT), fill=(70, 52, 96))
    draw.text((32, 34), shop_name, font=title, fill=(250, 236, 200))


def _draw_rows(draw: ImageDraw.ImageDraw, width: int, top: int, items: list[SampleItem]) -> None:
    body = ImageFont.load_default(size=30)
    for index, item in enumerate(items):
        y = top + index * _ROW_HEIGHT
        fill = (52, 46, 70) if index % 2 == 0 else (44, 39, 60)
        draw.rounded_rectangle((24, y, width - 24, y + _ROW_HEIGHT - 12), radius=12, fill=fill)
        draw.text((48, y + 20), item.name, font=body, fill=(235, 235, 245))
        price = f"{item.price} G"
        price_width = draw.textlength(price, font=body)
        draw.text((width - 48 - price_width, y + 20), price, font=body, fill=(255, 214, 102))


def render(truth: SampleTruth) -> Image.Image:
    """Draw a simple shop screen for ``truth``."""
    width, height = _SIZES[truth.orientation]
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_header(draw, width, truth.shop_name)
    _draw_rows(draw, width, _LIST_TOP, truth.items)
    return image


DATASET_FILE = "dataset.yaml"
RECORD_FILE = "synth.sha256"
"""Every screenshot and video `generate` wrote, with its SHA-256, in lines like those of
``sha256sum`` (while a run moves its files in, a name can have two lines, the earlier and the new
SHA-256). A later run removes only files listed there, and only while they still hold those bytes,
so a user's own files in the directory are never removed (files under the names `generate` writes
are overwritten, though)."""

MIN_SCALE, MAX_SCALE = 0.1, 4.0
MAX_NOISE = 64.0
MIN_JPEG_QUALITY, MAX_JPEG_QUALITY = 1, 95
# The defaults draw the screens as they are; generate() records only settings that differ.
DEFAULT_SCALE, DEFAULT_NOISE = 1.0, 0.0


def check_scale(scale: float) -> float:
    """Return ``scale``, or raise `ValueError` unless it is from 0.1 to 4 (NaN is rejected too).

    Below 0.1 the text is too small to read, and much smaller factors would round a side down to
    no pixels at all."""
    if not MIN_SCALE <= scale <= MAX_SCALE:
        raise ValueError(f"scale must be from {MIN_SCALE} to {MAX_SCALE}, got {scale}")
    return scale


def check_noise(noise: float) -> float:
    """Return ``noise``, or raise `ValueError` unless it is from 0 to 64 (NaN is rejected too)."""
    if not 0 <= noise <= MAX_NOISE:
        raise ValueError(f"noise must be from 0 to {MAX_NOISE}, got {noise}")
    return noise


def check_jpeg_quality(jpeg_quality: int | None) -> int | None:
    """Return ``jpeg_quality``, or raise `ValueError` unless it is None or an integer from 1 to
    95; Pillow itself takes any number."""
    is_quality = (
        isinstance(jpeg_quality, int)
        and not isinstance(jpeg_quality, bool)
        and MIN_JPEG_QUALITY <= jpeg_quality <= MAX_JPEG_QUALITY
    )
    if jpeg_quality is not None and not is_quality:
        raise ValueError(
            f"jpeg_quality must be an integer from {MIN_JPEG_QUALITY} to {MAX_JPEG_QUALITY}, "
            f"got {jpeg_quality!r}"
        )
    return jpeg_quality


def _degrade(
    image: Image.Image, *, scale: float, noise: float, rng: np.random.Generator
) -> Image.Image:
    if scale != 1:
        width, height = image.size
        size = (round(width * scale), round(height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
    if noise:
        # One float32 array, worked on in place: at scale 4 a screen has 44 million values, and
        # float64 temporaries would take about a gigabyte. The image is added as uint8, unconverted.
        pixels = rng.standard_normal((*image.size[::-1], 3), dtype=np.float32)
        pixels *= noise
        np.add(pixels, np.asarray(image), out=pixels)
        np.rint(pixels, out=pixels)
        np.clip(pixels, 0, 255, out=pixels)
        image = Image.fromarray(pixels.astype(np.uint8))
    return image


def _to_case(
    truth: SampleTruth, sha256: str, size: tuple[int, int], degradation: dict[str, Any]
) -> DatasetCase:
    width, height = size
    return DatasetCase(
        name=Path(truth.file).stem,
        image=truth.file,
        sha256=sha256,
        expected={"shop_name": truth.shop_name, "items": [asdict(i) for i in truth.items]},
        metadata={"orientation": truth.orientation, "width": width, "height": height} | degradation,
    )


def generate(
    out_dir: Path,
    *,
    n: int,
    seed: int = 0,
    scale: float = DEFAULT_SCALE,
    noise: float = DEFAULT_NOISE,
    jpeg_quality: int | None = None,
    videos: int = 0,
) -> list[SampleTruth]:
    """Write ``n`` screenshots, a ``dataset.yaml`` with their ground truth and ``videos``
    scrolling videos (``shop-scroll-000.mp4`` and on, seeded ``seed``, ``seed + 1``...).

    Each screen is resized by ``scale`` (0.1-4), gets Gaussian noise with a standard deviation of
    ``noise`` (in 0-255 pixel values, up to 64) and is saved as a JPEG of ``jpeg_quality`` (1-95)
    instead of a PNG, in that order, like a screen captured on a device and then compressed. The
    defaults write the screens as drawn. The ground truth depends only on ``seed``.

    Files that an earlier run listed in ``synth.sha256``, that still hold the bytes listed and
    that this run did not write again (in the other format, or past ``n`` or ``videos``) are
    removed; left there, a run over the directory would read them too. With ``videos`` 0 the
    earlier run's videos are kept. Files that cannot be removed stay listed, so that the next run
    tries again, and are reported as a `StaleFilesWarning`.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if videos < 0:
        raise ValueError("videos must be >= 0")
    check_scale(scale)
    check_noise(noise)
    check_jpeg_quality(jpeg_quality)
    if videos:
        import_av("Synthetic videos")  # before anything is drawn
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # A stream of its own, so that noise does not change the items drawn. Seeded through a string,
    # as numpy takes no negative seed.
    noise_rng = np.random.default_rng(random.Random(f"{seed}:noise").getrandbits(64))
    suffix = ".png" if jpeg_quality is None else ".jpg"
    # Only what differs from the defaults, so that the default dataset.yaml stays as it was.
    settings = {"scale": scale, "noise": noise, "jpeg_quality": jpeg_quality}
    defaults = {"scale": DEFAULT_SCALE, "noise": DEFAULT_NOISE, "jpeg_quality": None}
    degradation = {key: value for key, value in settings.items() if value != defaults[key]}
    earlier = _read_record(out_dir)
    truths: list[SampleTruth] = []
    cases: list[DatasetCase] = []
    written: dict[str, set[str]] = {}
    # Everything is drawn aside and moved in at the end, so a run that fails or is stopped while
    # drawing leaves the directory as it was. Inside out_dir, so that the moves are renames on one
    # file system.
    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=".synth-", ignore_cleanup_errors=True
    ) as staging_dir:
        staging = Path(staging_dir)
        for index in range(n):
            orientation: Orientation = "portrait" if index % 2 == 0 else "landscape"
            truth = _random_truth(rng, f"shop-{index:03d}{suffix}", orientation)
            image = _degrade(render(truth), scale=scale, noise=noise, rng=noise_rng)
            buffer = io.BytesIO()
            if jpeg_quality is None:
                image.save(buffer, format="PNG")
            else:
                image.save(buffer, format="JPEG", quality=jpeg_quality)
            data = buffer.getvalue()
            (staging / truth.file).write_bytes(data)
            sha256 = hashlib.sha256(data).hexdigest()
            written[truth.file] = {sha256}
            truths.append(truth)
            cases.append(_to_case(truth, sha256, image.size, degradation))
        for index in range(videos):
            name = f"shop-scroll-{index:03d}.mp4"
            generate_video(staging / name, seed=seed + index)
            written[name] = {hash_file(staging / name)}
        # Screenshots alone are being redrawn: the earlier videos stay, and so do their lines
        # (but not those of videos the user has removed since).
        kept = {
            name: hashes
            for name, hashes in earlier.items()
            if not videos and _VIDEO.fullmatch(name) and (out_dir / name).is_file()
        }
        stale = {
            name: hashes
            for name, hashes in earlier.items()
            if name not in written and name not in kept
        }
        write_dataset(staging / DATASET_FILE, cases)
        # The record goes first and lists the stale files too, and for a name written again both
        # the earlier and the new SHA-256, so that whatever happens next every file synth wrote
        # stays listed: a run stopped among the renames below, or before the stale files are
        # removed, leaves nothing that a later run cannot remove. Same-named files that are not
        # moved yet may no longer match the earlier dataset.yaml, though.
        _write_record(
            staging,
            out_dir,
            {name: hashes | earlier.get(name, set()) for name, hashes in written.items()}
            | kept
            | stale,
        )
        for name in [*written, DATASET_FILE]:
            (staging / name).replace(out_dir / name)
        failed = _remove_stale(out_dir, stale)
        # Then only this run's files, the kept videos, and those that could not be removed for the
        # next run to try again. Everything is in place by now and the first record lists it all,
        # so a failure here only leaves more lines than needed: not a reason to fail the run.
        with contextlib.suppress(OSError):
            _write_record(staging, out_dir, written | kept | failed)
    if failed:
        warnings.warn(
            f"framelift synth could not remove {', '.join(sorted(failed))} from {out_dir}, written "
            "by an earlier run; a run over the directory reads them too until a later run "
            "removes them",
            StaleFilesWarning,
            stacklevel=2,
        )
    return truths


def _write_record(staging: Path, out_dir: Path, entries: dict[str, set[str]]) -> None:
    """Replace ``synth.sha256`` in ``out_dir`` with ``entries``, written in ``staging`` and moved
    in, so that it is never left half-written. A name may have more than one line."""
    record = "".join(
        f"{sha256}  {name}\n" for name, hashes in entries.items() for sha256 in sorted(hashes)
    )
    (staging / RECORD_FILE).write_text(record, encoding="utf-8")
    (staging / RECORD_FILE).replace(out_dir / RECORD_FILE)


class StaleFilesWarning(UserWarning):
    """Files an earlier run of synth wrote could not be removed; a run over the directory reads
    them too."""


_SCREENSHOT = re.compile(r"shop-\d{3,}\.(?:png|jpg)")
_VIDEO = re.compile(r"shop-scroll-\d{3,}\.mp4")
_RECORD_LINE = re.compile(r"([0-9a-f]{64})  (\S+)")


def _read_record(out_dir: Path) -> dict[str, set[str]]:
    """The files that ``synth.sha256`` in ``out_dir`` lists, with the SHA-256 each may have; none
    when there is none or it cannot be read. Only names `generate` writes count, which also keeps
    them inside ``out_dir``."""
    try:
        lines = (out_dir / RECORD_FILE).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):  # ValueError covers a file that is not UTF-8
        return {}
    record: dict[str, set[str]] = {}
    for line in lines:
        match = _RECORD_LINE.fullmatch(line)
        if match and (_SCREENSHOT.fullmatch(match[2]) or _VIDEO.fullmatch(match[2])):
            record.setdefault(match[2], set()).add(match[1])
    return record


def _remove_stale(out_dir: Path, stale: dict[str, set[str]]) -> dict[str, set[str]]:
    """Remove the ``stale`` files that still hold bytes listed for them, and return those that
    could not be removed. Files that are gone or were changed since are no longer synth's."""
    failed: dict[str, set[str]] = {}
    for name, hashes in stale.items():
        path = out_dir / name
        try:
            if path.is_file() and hash_file(path) in hashes:
                path.unlink()
        except OSError:
            failed[name] = hashes
    return failed


# Scrolling video: a landscape screen whose list area shows exactly _VIEW_ROWS rows, so no row is
# ever cut off at a still position. Rows are scrolled _SCROLL_ROWS at a time.
_VIDEO_SIZE = _SIZES["landscape"]
_VIEW_ROWS = 6
_VIEW_HEIGHT = _VIEW_ROWS * _ROW_HEIGHT
_SCROLL_ROWS = 3
_HOLD_SECONDS = 1.5
_SCROLL_SECONDS = 0.6
_BLUR_STEPS = 8


@dataclass(frozen=True)
class Hold:
    """A still part of a synthetic video."""

    start_ms: int
    end_ms: int
    items: list[SampleItem]
    """Items whose whole row is visible."""


@dataclass(frozen=True)
class SampleVideo:
    """A synthetic video's ground truth: every item, and what each still part shows."""

    truth: SampleTruth
    holds: list[Hold]


class ListView:
    """A shop screen whose item list can be shown at any scroll offset (in pixels)."""

    def __init__(self, truth: SampleTruth) -> None:
        width, _ = _VIDEO_SIZE
        self.truth = truth
        canvas = Image.new("RGB", (width, len(truth.items) * _ROW_HEIGHT), _BACKGROUND)
        _draw_rows(ImageDraw.Draw(canvas), width, 0, truth.items)
        self._rows = np.asarray(canvas, dtype=np.float32)
        frame = Image.new("RGB", _VIDEO_SIZE, _BACKGROUND)
        draw = ImageDraw.Draw(frame)
        _draw_header(draw, width, truth.shop_name)
        # The footer hides the rows below the list area.
        draw.rectangle((0, _LIST_TOP + _VIEW_HEIGHT, width, _VIDEO_SIZE[1]), fill=(70, 52, 96))
        self._frame = frame

    @property
    def max_offset(self) -> int:
        return max(0, (len(self.truth.items) - _VIEW_ROWS) * _ROW_HEIGHT)

    def visible_items(self, offset: int) -> list[SampleItem]:
        """Items whose whole row box is inside the list area at ``offset``."""
        return [
            item
            for row, item in enumerate(self.truth.items)
            if row * _ROW_HEIGHT >= offset
            and row * _ROW_HEIGHT + _ROW_HEIGHT - 12 <= offset + _VIEW_HEIGHT
        ]

    def render(self, offset: float, blur: float = 0.0) -> Image.Image:
        """The screen at ``offset``; ``blur`` averages offsets over that many pixels, like the
        motion blur of a frame captured while scrolling."""
        steps = _BLUR_STEPS if blur > 0 else 1
        shifts = np.linspace(-blur / 2, blur / 2, steps) if blur > 0 else [0.0]
        view = np.zeros((_VIEW_HEIGHT, self._rows.shape[1], 3), dtype=np.float32)
        for shift in shifts:
            top = round(min(max(offset + shift, 0), self.max_offset))
            view += self._rows[top : top + _VIEW_HEIGHT]
        image = self._frame.copy()
        list_area = Image.fromarray((view / len(shifts)).astype(np.uint8))
        image.paste(list_area, (0, _LIST_TOP))
        return image


def write_video(
    path: Path,
    frames: Iterable[Image.Image],
    *,
    fps: int,
    rotation: int = 0,
    timestamps_ms: Iterable[int] | None = None,
    end_ms: int | None = None,
) -> None:
    """Encode frames as an H.264 MP4.

    ``rotation`` sets the display rotation metadata. ``timestamps_ms`` gives each frame its own
    time, for a variable-frame-rate video like a screen recording that only writes a frame when
    the screen changes; without it, frames are ``1 / fps`` apart. ``end_ms`` (with
    ``timestamps_ms``) keeps the last frame on screen until then, as a recorder does when the
    screen stays still until it stops; otherwise the last frame lasts ``1 / fps``.
    """
    av = import_av("Synthetic videos")
    times = iter(timestamps_ms) if timestamps_ms is not None else None
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.pix_fmt = "yuv420p"
        if times is not None:
            stream.codec_context.time_base = Fraction(1, 1000)
        if rotation:
            stream.set_display_rotation(rotation)
        packets = []
        last_ms = 0
        for index, image in enumerate(frames):
            if index == 0:  # the size can only be set before the first frame opens the codec
                stream.width, stream.height = image.size
            frame = av.VideoFrame.from_image(image)
            if times is not None:
                last_ms = next(times)
                frame.pts, frame.time_base = last_ms, Fraction(1, 1000)
            packets += stream.encode(frame)
        packets += stream.encode()
        if end_ms is not None and times is not None and packets:
            # The encoder ignores a frame's duration, so it is set on the last packet instead.
            last = max(packets, key=lambda packet: packet.pts)
            last.duration = round((end_ms - last_ms) / 1000 / last.time_base)
        for packet in packets:
            container.mux(packet)


def generate_video(path: Path, *, seed: int = 0, n_items: int = 12, fps: int = 10) -> SampleVideo:
    """Write an MP4 of a shop list that stays still, scrolls a few rows, stays still...

    Returns the full ground truth and, for every still part, the items fully shown in it.
    """
    if n_items < _VIEW_ROWS:
        # The list area always shows _VIEW_ROWS whole rows; fewer items would not fill it.
        raise ValueError(f"n_items must be >= {_VIEW_ROWS}")
    rng = random.Random(seed)
    truth = _random_truth(rng, path.name, "landscape", count=n_items)
    view = ListView(truth)
    step = _SCROLL_ROWS * _ROW_HEIGHT
    stops = sorted({*range(0, view.max_offset, step), view.max_offset})

    frames: list[Image.Image] = []
    holds: list[Hold] = []
    hold_frames = round(_HOLD_SECONDS * fps)
    scroll_frames = round(_SCROLL_SECONDS * fps)
    for stop_index, offset in enumerate(stops):
        start_ms = round(len(frames) * 1000 / fps)
        frames += [view.render(offset)] * hold_frames
        end_ms = round(len(frames) * 1000 / fps)
        holds.append(Hold(start_ms, end_ms, view.visible_items(offset)))
        if stop_index + 1 < len(stops):
            distance = stops[stop_index + 1] - offset
            speed = distance / scroll_frames
            frames += [
                view.render(offset + speed * (i + 0.5), blur=speed) for i in range(scroll_frames)
            ]
    write_video(path, frames, fps=fps)
    return SampleVideo(truth=truth, holds=holds)
