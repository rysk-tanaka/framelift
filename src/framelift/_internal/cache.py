"""On-disk cache of extractions and screen checks, so a run or an evaluation repeated on the same
images does not pay for them again.

Entries live in ``cache_dir/<extract|screen>/<key[:2]>/<key>.json``. A key hashes `CACHE_REVISION`,
the image sent, the model string, the server's URL, the model settings when there are any and
what else decides the answer (the profile's fingerprint, or the screen description, the profile's
version and, when the check is held to a schema, that it is). Only results that succeeded are
stored.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings

type Kind = Literal["extract", "screen"]

CACHE_REVISION = "1"
"""Part of every key. Bump it whenever framelift changes what it asks or how it reads the answer
for every result: the prompts, the screen check's instructions or the shape of its output, or the
entry format; entries made the old way are then not used. A change that only some models go
through can instead add a part to their keys, so that the other entries are still used, as
asking local servers' screen checks through NativeOutput does."""

# Warnings point at the first frame outside framelift: the user's call, however deep inside.
_PACKAGE_DIR = str(Path(__file__).parent.parent) + os.sep


def model_cache_id(model: Model | str) -> str | None:
    """What identifies ``model`` in a cache key, or None when it cannot be identified.

    A model string names the provider, the API and the model. A `Model` instance can carry
    settings, a profile, wrapped models or another HTTP client that change its answers and that
    cannot be read off it reliably, so it is not cached.
    """
    return model if isinstance(model, str) else None


def cache_key(*parts: str) -> str:
    """SHA-256 of `CACHE_REVISION` and ``parts``, joined unambiguously."""
    # ASCII, escaping everything else: a lone surrogate, which UTF-8 cannot encode, must not stop
    # a run that would go on without a cache.
    payload = json.dumps([CACHE_REVISION, *parts])
    return hashlib.sha256(payload.encode()).hexdigest()


def json_sha256(data: Any) -> str:
    """SHA-256 of ``data`` as canonical JSON."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))  # ASCII, as in cache_key
    return hashlib.sha256(text.encode()).hexdigest()


def same_state(a: Any, b: Any) -> bool:
    """Whether ``a`` and ``b`` are the same down to types, fields set, extras and private
    attributes.

    Stricter than ``==``: pydantic's ``==`` takes ``False`` for ``0`` and ignores which fields
    were set, either of which can change what is dumped or keyed from a model. NaN is never the
    same as itself, so an output holding one is not cached.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, BaseModel):
        return (
            a.model_fields_set == b.model_fields_set
            and all(same_state(getattr(a, name), getattr(b, name)) for name in type(a).model_fields)
            and same_state(a.model_extra, b.model_extra)
            and same_state(a.__pydantic_private__, b.__pydantic_private__)
        )
    if isinstance(a, dict):
        # In order: keys are compared by type too, which dict lookup does not do (0 == False).
        return len(a) == len(b) and all(
            same_state(key_a, key_b) and same_state(value_a, value_b)
            for (key_a, value_a), (key_b, value_b) in zip(a.items(), b.items(), strict=True)
        )
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, set | frozenset):
        return len(a) == len(b) and all(any(same_state(x, y) for y in b) for x in a)
    try:
        return bool(a == b)
    except Exception:  # e.g. an array whose == is elementwise
        return False


class ExtractEntry(BaseModel):
    """A cached extraction."""

    input: str
    """The output as JSON, by field name, to validate again into the output model."""
    data_sha256: str
    """SHA-256 of the output as observations record it; an entry whose output now dumps
    differently, e.g. after a library update, is not used."""
    input_tokens: int
    output_tokens: int
    requests: int
    retries: int
    latency_ms: int


class ScreenEntry(BaseModel):
    """A cached screen check."""

    is_target: bool
    confidence: float
    input_tokens: int
    output_tokens: int
    requests: int
    latency_ms: int


class Cache:
    """A cache directory. Reads that fail are misses; writes that fail warn once and are
    skipped, as the result is already paid for and the run goes on."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._warned_model = False
        self._warned_settings = False
        self._warned_write = False

    def model_id(
        self, model: Model | str, resolved: Model, settings: ModelSettings | None = None
    ) -> tuple[str, ...] | None:
        """`model_cache_id`, the URL of the server ``resolved`` (``model`` as a `Model`) asks and
        the hash of ``settings``, as parts of a key, warning once when ``model`` or ``settings``
        cannot be cached.

        The server is there because a model string does not name it: ``OPENAI_BASE_URL``, for
        one, points the same string at another server. No settings add no part, so that entries
        made before settings could be given are still used.
        """
        model_id = model_cache_id(model)
        if model_id is None:
            if not self._warned_model:
                self._warned_model = True
                warnings.warn(
                    "framelift caches only models given as strings such as 'anthropic:<model>'; "
                    "a Model instance is not cached, as its settings cannot be told apart",
                    skip_file_prefixes=(_PACKAGE_DIR,),
                )
            return None
        parts = (model_id, resolved.base_url or "")
        if not settings:
            return parts
        try:
            return (*parts, json_sha256(settings))
        except (TypeError, ValueError):  # e.g. an httpx.Timeout, which JSON cannot hold
            if not self._warned_settings:
                self._warned_settings = True
                warnings.warn(
                    "framelift caches results only for model settings that JSON can hold; with "
                    "these, no extraction or screen check is cached, as they cannot be told apart",
                    skip_file_prefixes=(_PACKAGE_DIR,),
                )
            return None

    def _path(self, kind: Kind, key: str) -> Path:
        return self.root / kind / key[:2] / f"{key}.json"

    def read[EntryT: BaseModel](
        self, kind: Kind, key: str, entry_type: type[EntryT]
    ) -> EntryT | None:
        """The entry stored under ``key``, or None when there is none or it cannot be read."""
        try:
            text = self._path(kind, key).read_text(encoding="utf-8")
            return entry_type.model_validate_json(text)
        except (OSError, ValueError, ValidationError):
            return None

    def write(self, kind: Kind, key: str, entry: BaseModel) -> None:
        """Store ``entry`` under ``key``, replacing any entry there."""
        path = self._path(kind, key)
        temp: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written aside and moved in, so a run stopped midway leaves no half-written entry.
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
            ) as fh:
                temp = Path(fh.name)
                fh.write(entry.model_dump_json())
            temp.replace(path)
            temp = None
        except OSError as exc:
            if not self._warned_write:
                self._warned_write = True
                warnings.warn(
                    f"framelift could not write to the cache in {self.root} ({exc}); results "
                    "that cannot be written are not cached",
                    skip_file_prefixes=(_PACKAGE_DIR,),
                )
        finally:
            # Also on Ctrl-C, which is not an OSError: no temporary file is left behind.
            if temp is not None:
                with contextlib.suppress(OSError):
                    temp.unlink(missing_ok=True)
