"""`Profile`: what to read from a screen and how to read it."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal, get_args

from pydantic import BaseModel

from framelift._internal.fields import item_model_of

type Validator[OutputT: BaseModel] = Callable[[OutputT], OutputT | None]
"""A check run on every extracted output.

Raise `pydantic_ai.ModelRetry` with a message to make the model try again; return a (possibly
modified) output, or `None` to keep it as is.
"""

type Box = tuple[float, float, float, float]
"""``(left, top, right, bottom)`` as fractions (0-1) of the upright screen: a video frame with its
display rotation applied, an image with its EXIF orientation applied."""

type Orientation = Literal["portrait", "landscape"]
"""``portrait`` when the upright screen is taller than wide, ``landscape`` otherwise."""

ORIENTATIONS: tuple[Orientation, ...] = get_args(Orientation.__value__)


def _box(value: Any, where: str) -> Box:
    is_four_numbers = (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and len(value) == 4
        and all(isinstance(v, Real) and not isinstance(v, bool) for v in value)
    )
    if not is_four_numbers:
        raise ValueError(f"{where} must be (left, top, right, bottom), got {value!r}")
    left, top, right, bottom = (float(v) for v in value)
    is_in_range = all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (left, top, right, bottom))
    if not is_in_range:
        raise ValueError(f"{where} must be fractions of the screen between 0 and 1, got {value!r}")
    if not (left < right and top < bottom):
        raise ValueError(f"{where} must have left < right and top < bottom, got {value!r}")
    return (left, top, right, bottom)


def _orientation_keys(value: Mapping[Any, Any], where: str) -> None:
    # Literal annotations are not checked at runtime; a typo such as "landscpae" would otherwise
    # leave every landscape screen unredacted or uncropped.
    unknown = [key for key in value if key not in ORIENTATIONS]
    if unknown:
        raise ValueError(f"{where} keys must be 'portrait' or 'landscape', got {unknown!r}")


def _locate(value: Any) -> Literal["full"] | dict[Orientation, Box]:
    if value == "full":
        return "full"
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "Profile.locate must be 'full' or a crop box per orientation, e.g. "
            "{'portrait': (0.0, 0.1, 1.0, 0.9)}"
        )
    _orientation_keys(value, "Profile.locate")
    return {key: _box(box, f"Profile.locate[{key!r}]") for key, box in value.items()}


def _redact(value: Any) -> dict[Orientation, tuple[Box, ...]]:
    if isinstance(value, Mapping):
        _orientation_keys(value, "Profile.redact")
        by_orientation = {key: value.get(key, ()) for key in ORIENTATIONS}
    else:
        by_orientation = dict.fromkeys(ORIENTATIONS, value)
    result: dict[Orientation, tuple[Box, ...]] = {}
    for key, boxes in by_orientation.items():
        if isinstance(boxes, str) or not isinstance(boxes, Sequence):
            raise ValueError(f"Profile.redact must be a list of boxes, got {boxes!r}")
        result[key] = tuple(_box(box, f"Profile.redact for {key}") for box in boxes)
    return result


def item_field(output_type: type[BaseModel]) -> tuple[str, type[BaseModel]]:
    """Find the one top-level ``list[SomeModel]`` field that `Profile.item_key` applies to.

    ``list[SomeModel] | None`` counts too; ``None`` is read as an empty list.

    Returns the field name and the item model. Raises `ValueError` unless exactly one such field
    exists, because the key alone cannot tell which list it is meant for.
    """
    candidates: list[tuple[str, type[BaseModel]]] = []
    for name, info in output_type.model_fields.items():
        item_model = item_model_of(info.annotation)
        if item_model is not None:
            candidates.append((name, item_model))
    if len(candidates) != 1:
        found = ", ".join(name for name, _ in candidates) or "none"
        raise ValueError(
            "Profile.item_key needs exactly one top-level list-of-models field in "
            f"{output_type.__name__} (found: {found})"
        )
    return candidates[0]


@dataclass(frozen=True, kw_only=True)
class Profile[OutputT: BaseModel]:
    """Defines what to extract from a screen.

    Args:
        name: Identifier written to every observation, e.g. ``"fantasy-shop"``.
        output_type: The Pydantic model to fill in. Field descriptions are passed to the model.
        instructions: How to read the screen.
        version: Bump this whenever the instructions or the output model change, so that
            observations, caches and evaluation results can be told apart.
        validators: Extra checks run on every output; see `Validator`.
        locate: Which part of the screen to send. ``"full"`` sends the whole screen; a mapping
            such as ``{"portrait": (0.0, 0.1, 1.0, 0.9), "landscape": ...}`` crops each
            orientation to a `Box`. A screen whose orientation has no box is not sent and is
            recorded as an error. Required, so that sending the whole screen is a choice.
        redact: Boxes painted opaque black before cropping, in whole-screen coordinates: a list
            for every orientation, or a mapping per orientation. Stored as a mapping with both
            orientations.
        item_key: Returns the identity of one list item, e.g. ``lambda item: item.name``. It is
            applied to the items of the output's only ``list[SomeModel]`` field, and is used to
            match extracted items with the expected ones during evaluation.
    """

    name: str
    output_type: type[OutputT]
    instructions: str
    version: str = "1"
    validators: Sequence[Validator[OutputT]] = field(default_factory=tuple)
    locate: Literal["full"] | Mapping[Orientation, Box]
    redact: Sequence[Box] | Mapping[Orientation, Sequence[Box]] = ()
    item_key: Callable[[Any], Hashable] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Profile.name must not be empty")
        if not self.version.strip():
            raise ValueError("Profile.version must not be empty")
        if not (isinstance(self.output_type, type) and issubclass(self.output_type, BaseModel)):
            raise TypeError("Profile.output_type must be a pydantic BaseModel subclass")
        object.__setattr__(self, "validators", tuple(self.validators))
        object.__setattr__(self, "locate", _locate(self.locate))
        object.__setattr__(self, "redact", _redact(self.redact))
        if self.item_key is not None:
            item_field(self.output_type)

    @property
    def fingerprint(self) -> str:
        """A stable hash of everything that affects extraction results."""
        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "instructions": self.instructions,
                "schema": self.output_type.model_json_schema(),
                # What is sent changes with these, and so may what is read.
                "locate": self.locate,
                "redact": self.redact,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
