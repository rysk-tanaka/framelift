"""`Profile`: what to read from a screen and how to read it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

type Validator[OutputT: BaseModel] = Callable[[OutputT], OutputT | None]
"""A check run on every extracted output.

Raise `pydantic_ai.ModelRetry` with a message to make the model try again; return a (possibly
modified) output, or `None` to keep it as is.
"""


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
    """

    name: str
    output_type: type[OutputT]
    instructions: str
    version: str = "1"
    validators: Sequence[Validator[OutputT]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Profile.name must not be empty")
        if not self.version.strip():
            raise ValueError("Profile.version must not be empty")
        if not (isinstance(self.output_type, type) and issubclass(self.output_type, BaseModel)):
            raise TypeError("Profile.output_type must be a pydantic BaseModel subclass")
        object.__setattr__(self, "validators", tuple(self.validators))

    @property
    def fingerprint(self) -> str:
        """A stable hash of everything that affects extraction results."""
        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "instructions": self.instructions,
                "schema": self.output_type.model_json_schema(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
