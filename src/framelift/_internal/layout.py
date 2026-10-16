"""Make the image that is sent: redact, then crop, following `Profile.locate` and
`Profile.redact`.

Boxes are fractions of the upright screen, so an image is turned upright from its EXIF
orientation first; otherwise a JPEG shown in portrait would be read as landscape and the wrong
area painted over.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image, ImageDraw, ImageOps

from framelift.observation import Crop

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from framelift.profile import Box, Orientation, Profile

_FULL: Box = (0.0, 0.0, 1.0, 1.0)


class NoCropError(Exception):
    """The screen's orientation has no box in `Profile.locate`, so nothing may be sent.

    Not a `ValueError`: callers treat those as unreadable input, and this is a profile that does
    not cover the screen, not a broken file.
    """

    def __init__(self, orientation: Orientation, size: tuple[int, int]) -> None:
        super().__init__(f"Profile.locate has no crop box for {orientation} screens")
        self.orientation = orientation
        self.size = size
        self.sha256: str | None = None
        """SHA-256 of the input file, when the caller knows it."""

    @property
    def crop(self) -> Crop:
        return Crop(orientation=self.orientation, size=self.size, bbox=None, redact=[])


def orientation_of(size: tuple[int, int]) -> Orientation:
    width, height = size
    return "portrait" if height > width else "landscape"


def _has_alpha(image: Image.Image) -> bool:
    # A palette or RGB/L PNG can mark one color transparent (tRNS); converting it to RGB keeps
    # that, so a black mask over a black transparent color would come out transparent.
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def normalize(image: Image.Image) -> Image.Image:
    """The image upright (EXIF orientation applied) as RGB, or RGBA when it has transparency.

    Returns a new image; ``image`` is left as it is.
    """
    upright = ImageOps.exif_transpose(image)
    if upright.mode == "I" or upright.mode.startswith("I;16"):
        # A 16-bit grayscale PNG. Pillow clips, not scales, its samples to 8 bits, which would
        # turn the screen white.
        upright = upright.convert("I").point(lambda value: value * (1 / 256)).convert("L")
    converted = upright.convert("RGBA" if _has_alpha(upright) else "RGB")
    converted.info.pop("transparency", None)
    return converted


def _edge(fraction: float, length: int, rounding: Callable[[float], int]) -> int:
    position = fraction * length
    # An edge that falls on a pixel boundary stays there: 0.07 * 100 is 7.000000000000001, and
    # rounding that outward would take in a row the box does not cover.
    nearest = round(position)
    return nearest if math.isclose(position, nearest, abs_tol=1e-6) else rounding(position)


def _span(start: float, end: float, length: int) -> tuple[int, int]:
    first = max(0, _edge(start, length, math.floor))
    last = min(length, _edge(end, length, math.ceil))
    # A box narrower than the snapping tolerance would put both edges on one boundary; it still
    # covers part of a pixel, so it keeps that pixel.
    if last <= first:
        first, last = (first, first + 1) if first < length else (length - 1, length)
    return first, last


def _pixels(box: Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    # Outward rounding, so a box never loses a partly covered pixel: a redaction always covers
    # what was asked, and a crop is at least one pixel wide.
    width, height = size
    left, top, right, bottom = box
    left_px, right_px = _span(left, right, width)
    top_px, bottom_px = _span(top, bottom, height)
    return left_px, top_px, right_px, bottom_px


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class Layout:
    """What to send of each screen."""

    locate: Literal["full"] | Mapping[Orientation, Box]
    redact: Mapping[Orientation, tuple[Box, ...]]

    @classmethod
    def from_profile(cls, profile: Profile) -> Layout:
        # Profile.__post_init__ has already validated both and turned redact into a mapping with
        # both orientations, which its declared (input) type cannot say.
        redact = cast("Mapping[Orientation, tuple[Box, ...]]", profile.redact)
        return cls(locate=profile.locate, redact=redact)

    @classmethod
    def full(cls) -> Layout:
        """Send the whole screen as it is."""
        return cls(locate="full", redact={})

    @property
    def is_identity(self) -> bool:
        """Whether the screen is sent untouched, so an image's own bytes can be sent."""
        return self.locate == "full" and not any(self.redact.values())

    def apply(self, image: Image.Image) -> tuple[Image.Image, Crop]:
        """Redact, then crop, an upright screen; return the image to send and how it was cut.

        ``image`` is normalized first (see `normalize`). Raises `NoCropError` when the screen's
        orientation has no crop box.
        """
        screen = normalize(image)
        size = screen.size
        orientation = orientation_of(size)
        if self.locate == "full":
            bbox = _FULL
        else:
            box = self.locate.get(orientation)
            if box is None:
                raise NoCropError(orientation, size)
            bbox = box
        redact = self.redact.get(orientation, ())
        if redact:
            draw = ImageDraw.Draw(screen)
            black = (0, 0, 0, 255) if screen.mode == "RGBA" else (0, 0, 0)
            for area in redact:
                left, top, right, bottom = _pixels(area, screen.size)
                draw.rectangle((left, top, right - 1, bottom - 1), fill=black)
        if bbox != _FULL:
            screen = screen.crop(_pixels(bbox, screen.size))
        return screen, Crop(orientation=orientation, size=size, bbox=bbox, redact=list(redact))
