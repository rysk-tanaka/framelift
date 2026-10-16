from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageOps
from pydantic import BaseModel

from framelift import Profile
from framelift._internal.layout import Layout, NoCropError, orientation_of
from framelift._internal.sources import load_image
from framelift.profile import Box

RED, BLUE, WHITE, BLACK = (255, 0, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)
EXIF_ORIENTATION = 0x0112


class Out(BaseModel):
    value: int


def _layout(locate: object = "full", redact: object = ()) -> Layout:
    profile = Profile(
        name="p",
        output_type=Out,
        instructions="read",
        locate=locate,  # ty: ignore[invalid-argument-type]
        redact=redact,  # ty: ignore[invalid-argument-type]
    )
    return Layout.from_profile(profile)


def _close(a: Any, b: Any, tolerance: int = 40) -> bool:
    assert isinstance(a, tuple) and isinstance(b, tuple)
    return all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))


def _halves(size: tuple[int, int], left: tuple[int, int, int], right: tuple[int, int, int]):
    """An image whose left half is ``left`` and right half ``right``."""
    image = Image.new("RGB", size, left)
    image.paste(right, (size[0] // 2, 0, size[0], size[1]))
    return image


def _save(path: Path, image: Image.Image, **params: Any) -> Path:
    image.save(path, **params)
    return path


@pytest.mark.parametrize(
    ("size", "expected"),
    [((100, 200), "portrait"), ((200, 100), "landscape"), ((100, 100), "landscape")],
)
def test_orientation(size: tuple[int, int], expected: str) -> None:
    assert orientation_of(size) == expected


def test_redaction_paints_black_and_leaves_the_rest() -> None:
    view, crop = _layout(redact=[(0.0, 0.0, 0.5, 0.5)]).apply(Image.new("RGB", (100, 200), WHITE))

    assert view.size == (100, 200)
    assert view.getpixel((10, 10)) == BLACK
    assert view.getpixel((49, 99)) == BLACK
    assert view.getpixel((50, 100)) == WHITE
    assert view.getpixel((90, 190)) == WHITE
    assert crop.bbox == (0.0, 0.0, 1.0, 1.0)
    assert crop.redact == [(0.0, 0.0, 0.5, 0.5)]


def test_crop_rounds_outward() -> None:
    box: Box = (0.1, 0.2, 0.55, 0.9)
    view, crop = _layout(locate={"portrait": box}).apply(Image.new("RGB", (101, 200), WHITE))

    # floor(10.1) = 10, floor(40) = 40, ceil(55.55) = 56, ceil(180) = 180
    assert view.size == (46, 140)
    assert crop.orientation == "portrait"
    assert crop.size == (101, 200)  # the screen before cropping
    assert crop.bbox == box


def test_redaction_is_in_screen_coordinates_and_happens_before_cropping() -> None:
    layout = _layout(locate={"portrait": (0.0, 0.25, 1.0, 0.75)}, redact=[(0.0, 0.0, 1.0, 0.5)])
    view, _ = layout.apply(Image.new("RGB", (100, 200), WHITE))

    assert view.size == (100, 100)
    assert view.getpixel((50, 10)) == BLACK  # screen y = 60, inside the redaction
    assert view.getpixel((50, 90)) == WHITE  # screen y = 140


def test_redaction_can_be_given_per_orientation() -> None:
    layout = _layout(redact={"landscape": [(0.0, 0.0, 1.0, 1.0)]})

    portrait, _ = layout.apply(Image.new("RGB", (100, 200), WHITE))
    landscape, crop = layout.apply(Image.new("RGB", (200, 100), WHITE))

    assert portrait.getpixel((50, 50)) == WHITE
    assert landscape.getpixel((50, 50)) == BLACK
    assert crop.redact == [(0.0, 0.0, 1.0, 1.0)]


def test_an_orientation_without_a_crop_box_is_not_sent() -> None:
    layout = _layout(locate={"portrait": (0.0, 0.0, 1.0, 0.5)})

    with pytest.raises(NoCropError) as excinfo:
        layout.apply(Image.new("RGB", (200, 100), WHITE))

    assert excinfo.value.orientation == "landscape"
    assert excinfo.value.crop.bbox is None
    assert excinfo.value.crop.size == (200, 100)


def test_an_untouched_image_is_sent_as_its_own_bytes(tmp_path: Path) -> None:
    path = _save(tmp_path / "shot.jpg", Image.new("RGB", (100, 200), RED), format="JPEG")

    source = load_image(path, _layout())

    assert source.data == path.read_bytes()
    assert source.media_type == "image/jpeg"
    assert source.sha256 == source.source_sha256 == hashlib.sha256(source.data).hexdigest()


def test_a_cropped_image_is_sent_as_png(tmp_path: Path) -> None:
    path = _save(tmp_path / "shot.jpg", Image.new("RGB", (100, 200), RED), format="JPEG")

    source = load_image(path, _layout(locate={"portrait": (0.0, 0.0, 1.0, 0.5)}))

    assert source.media_type == "image/png"
    assert source.sha256 == hashlib.sha256(source.data).hexdigest() != source.source_sha256
    with Image.open(io.BytesIO(source.data)) as sent:
        assert sent.size == (100, 100)


@pytest.mark.parametrize("orientation", [2, 5, 6, 8])
def test_exif_orientation_is_applied_before_redacting(tmp_path: Path, orientation: int) -> None:
    stored = _halves((200, 100), RED, BLUE)  # stored landscape; 5, 6 and 8 display it portrait
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = orientation
    path = _save(tmp_path / "shot.jpg", stored, format="JPEG", exif=exif.tobytes(), quality=95)
    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image).convert("RGB")

    source = load_image(path, _layout(redact=[(0.0, 0.0, 0.2, 1.0)]))

    assert source.view.size == upright.size
    assert source.crop.orientation == orientation_of(upright.size)
    width, height = upright.size
    assert source.view.getpixel((1, height // 2)) == BLACK
    # Outside the redaction the image is the upright one, not the stored one.
    point = (width - 2, height // 2)
    assert _close(source.view.getpixel(point), upright.getpixel(point))


def test_exif_rotation_makes_a_stored_landscape_portrait(tmp_path: Path) -> None:
    stored = _halves((200, 100), RED, BLUE)
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 6  # shown rotated 90 degrees clockwise
    path = _save(tmp_path / "shot.jpg", stored, format="JPEG", exif=exif.tobytes(), quality=95)

    source = load_image(path, _layout(redact=[(0.0, 0.0, 1.0, 0.1)]))

    assert source.crop.orientation == "portrait"
    assert source.crop.size == (100, 200)  # upright, not as stored
    assert source.view.size == (100, 200)
    assert source.view.getpixel((50, 5)) == BLACK
    assert _close(source.view.getpixel((50, 50)), RED)  # the stored left half is now on top
    assert _close(source.view.getpixel((50, 150)), BLUE)


def _gray(tmp_path: Path, fmt: str) -> Path:
    return _save(tmp_path / f"gray.{fmt.lower()}", Image.new("L", (100, 200), 200), format=fmt)


def _cmyk(tmp_path: Path) -> Path:
    image = Image.new("CMYK", (100, 200), (0, 255, 255, 0))
    return _save(tmp_path / "cmyk.jpg", image, format="JPEG")


def _rgba(tmp_path: Path) -> Path:
    return _save(tmp_path / "rgba.png", Image.new("RGBA", (100, 200), (255, 0, 0, 0)))


def _palette(tmp_path: Path) -> Path:
    image = Image.new("P", (100, 200), 0)
    image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    return _save(tmp_path / "palette.png", image, transparency=0)


def _rgb_with_black_transparent(tmp_path: Path) -> Path:
    # Black is the transparent color, so a black mask must not turn transparent.
    image = Image.new("RGB", (100, 200), WHITE)
    return _save(tmp_path / "trns-rgb.png", image, transparency=BLACK)


def _gray_with_black_transparent(tmp_path: Path) -> Path:
    image = Image.new("L", (100, 200), 255)
    return _save(tmp_path / "trns-l.png", image, transparency=0)


@pytest.mark.parametrize(
    "make",
    [
        lambda tmp: _gray(tmp, "PNG"),
        lambda tmp: _gray(tmp, "JPEG"),
        _cmyk,
        _rgba,
        _palette,
        _rgb_with_black_transparent,
        _gray_with_black_transparent,
    ],
    ids=["gray-png", "gray-jpeg", "cmyk", "rgba", "palette", "trns-rgb", "trns-l"],
)
def test_any_color_mode_is_redacted_with_opaque_black(tmp_path: Path, make) -> None:
    path = make(tmp_path)

    source = load_image(path, _layout(redact=[(0.0, 0.0, 0.5, 1.0)]))

    assert source.media_type == "image/png"
    with Image.open(io.BytesIO(source.data)) as sent:
        assert sent.convert("RGBA").getpixel((10, 100)) == (0, 0, 0, 255)


def test_an_edge_on_a_pixel_boundary_takes_no_extra_row() -> None:
    # 0.07 * 100 is 7.000000000000001 in floating point.
    view, _ = _layout(locate={"landscape": (0.0, 0.0, 1.0, 0.07)}).apply(
        Image.new("RGB", (200, 100), WHITE)
    )

    assert view.size == (200, 7)


def test_a_16_bit_grayscale_png_is_scaled_not_clipped(tmp_path: Path) -> None:
    image = Image.fromarray(np.full((200, 100), 30000, dtype=np.uint16))
    assert image.mode == "I;16"
    path = _save(tmp_path / "gray16.png", image)

    source = load_image(path, _layout(redact=[(0.0, 0.0, 0.5, 1.0)]))

    assert source.view.getpixel((90, 100)) == (117, 117, 117)  # 30000 / 256, not white
    assert source.view.getpixel((10, 100)) == BLACK


@pytest.mark.parametrize(
    "box",
    [(0.1, 0.0, 0.1 + 1e-9, 1.0), (1.0 - 1e-9, 0.0, 1.0, 1.0)],
    ids=["inside", "at-the-right-edge"],
)
def test_a_box_narrower_than_the_snapping_tolerance_keeps_one_pixel(box: Box) -> None:
    layout = _layout(locate={"portrait": box}, redact=[box])

    view, _ = layout.apply(Image.new("RGB", (100, 200), WHITE))

    assert view.size == (1, 200)
    assert view.getpixel((0, 100)) == BLACK
