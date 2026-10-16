"""Resolve user inputs into image files and load them."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import imagehash
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from collections.abc import Iterable

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_GLOB_CHARS = frozenset("*?[")


@dataclass(frozen=True)
class ImageSource:
    path: Path
    data: bytes
    media_type: str
    sha256: str
    phash: str


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _glob(pattern: str) -> list[Path]:
    p = Path(pattern)
    if p.is_absolute():
        root = Path(p.anchor)
        relative = str(p.relative_to(root))
    else:
        root, relative = Path(), pattern
    return sorted(root.glob(relative))


def expand_inputs(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files, directories and glob patterns into a de-duplicated list of image paths.

    Directories contribute their image files (non-recursive, sorted). Raises `FileNotFoundError`
    when an input matches nothing, so typos are not silently ignored.
    """
    seen: set[Path] = set()
    result: list[Path] = []
    for item in inputs:
        text = str(item)
        if any(c in text for c in _GLOB_CHARS):
            candidates = [p for p in _glob(text) if _is_image(p)]
        else:
            path = Path(text)
            if path.is_dir():
                candidates = sorted(p for p in path.iterdir() if _is_image(p))
            elif path.is_file():
                candidates = [path]
            else:
                candidates = []
        if not candidates:
            raise FileNotFoundError(f"No image matched input: {text}")
        for candidate in candidates:
            key = candidate.resolve()
            if key not in seen:
                seen.add(key)
                result.append(candidate)
    return result


def load_image(path: Path) -> ImageSource:
    """Read an image, verify it decodes, and compute its SHA-256 and perceptual hash."""
    data = path.read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            media_type = _MEDIA_TYPES.get(image.format or "")
            phash = str(imagehash.phash(image))
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a readable image: {path}") from exc
    if media_type is None:
        raise ValueError(f"Unsupported image format for {path}")
    return ImageSource(
        path=path,
        data=data,
        media_type=media_type,
        sha256=hashlib.sha256(data).hexdigest(),
        phash=phash,
    )
