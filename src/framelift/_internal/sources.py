"""Resolve user inputs into image and video files, and load images."""

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
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm"})
_MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_GLOB_CHARS = frozenset("*?[")


_HASH_CHUNK = 1 << 20


def error_reason(exc: BaseException) -> str:
    """What went wrong, without the file name.

    OS and FFmpeg errors quote the path in their message, so their ``strerror`` is used. Other
    errors keep their message: Pillow reads from memory, so its messages name no path.
    """
    strerror = getattr(exc, "strerror", None)
    return strerror if isinstance(strerror, str) and strerror else str(exc)


class UnreadableInputError(ValueError):
    """An image or video that cannot be read or decoded.

    The message names the path; ``reason`` does not, so a record that must not contain paths
    can still say why (replacing the path in the message misses escaped or repeated forms).
    """

    def __init__(self, kind: str, path: Path, reason: str, *, sha256: str | None = None) -> None:
        super().__init__(f"Not a readable {kind}: {path} ({reason})")
        self.kind = kind
        self.path = path
        self.reason = reason
        self.sha256 = sha256
        """SHA-256 of the file, when its bytes were read before decoding failed."""

    @property
    def without_path(self) -> str:
        """The message with the path left out."""
        return f"Not a readable {self.kind} ({self.reason})"


@dataclass(frozen=True)
class ImageSource:
    """An image to send to the model, with where it came from."""

    path: Path
    """The input file: the image itself, or the video the frame was taken from."""
    data: bytes
    media_type: str
    sha256: str
    """SHA-256 of ``data``."""
    phash: str
    source_sha256: str
    """SHA-256 of the input file; the same as ``sha256`` for an image."""
    frame_index: int = 0
    timestamp_ms: int | None = None


def is_video(path: Path) -> bool:
    """Whether ``path`` is a video, judged by its extension."""
    return path.suffix.lower() in VIDEO_SUFFIXES


def _is_input(path: Path) -> bool:
    return path.is_file() and (path.suffix.lower() in IMAGE_SUFFIXES or is_video(path))


def _glob(pattern: str) -> list[Path]:
    p = Path(pattern)
    if p.is_absolute():
        root = Path(p.anchor)
        relative = str(p.relative_to(root))
    else:
        root, relative = Path(), pattern
    return sorted(root.glob(relative))


def expand_inputs(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files, directories and glob patterns into a de-duplicated list of image and video
    paths.

    A path that exists is used as is, even if it contains glob characters. Directories contribute
    their image and video files (non-recursive, sorted). Raises `FileNotFoundError` when an input
    matches nothing, so typos are not silently ignored.
    """
    seen: set[Path] = set()
    result: list[Path] = []
    for item in inputs:
        text = str(item)
        path = Path(text)
        # An existing path is taken literally, so names like "shot[1].png" are not globs.
        if path.is_dir():
            candidates = sorted(p for p in path.iterdir() if _is_input(p))
        elif path.is_file():
            candidates = [path]
        elif any(c in text for c in _GLOB_CHARS):
            candidates = [p for p in _glob(text) if _is_input(p)]
        else:
            candidates = []
        if not candidates:
            raise FileNotFoundError(f"No image or video matched input: {text}")
        for candidate in candidates:
            key = candidate.resolve()
            if key not in seen:
                seen.add(key)
                result.append(candidate)
    return result


def _decode(path: Path, data: bytes, sha256: str) -> tuple[Image.Image, str]:
    # Every decoding failure becomes an UnreadableInputError (a ValueError), so callers report it
    # per image (a truncated PNG raises OSError, an oversized one DecompressionBombError) instead
    # of stopping a run.
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise UnreadableInputError("image", path, error_reason(exc), sha256=sha256) from exc
    try:
        image.load()
    except (Image.DecompressionBombError, OSError) as exc:
        image.close()
        raise UnreadableInputError("image", path, error_reason(exc), sha256=sha256) from exc
    media_type = _MEDIA_TYPES.get(image.format or "")
    if media_type is None:
        image.close()
        raise UnreadableInputError("image", path, "unsupported image format", sha256=sha256)
    return image, media_type


def check_image(path: Path) -> str:
    """Verify that an image decodes and is supported; return its SHA-256 (no perceptual hash)."""
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    image, _ = _decode(path, data, sha256)
    image.close()
    return sha256


def hash_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks so that long videos are not loaded into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> ImageSource:
    """Read an image, verify it decodes, and compute its SHA-256 and perceptual hash."""
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    image, media_type = _decode(path, data, sha256)
    with image:
        phash = str(imagehash.phash(image))
    return ImageSource(
        path=path,
        data=data,
        media_type=media_type,
        sha256=sha256,
        phash=phash,
        source_sha256=sha256,
    )
