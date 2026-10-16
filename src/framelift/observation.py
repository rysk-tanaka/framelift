"""The `framelift/observation/v2` record written to ``observations.jsonl``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OBSERVATION_SCHEMA = "framelift/observation/v2"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class ProfileRef(_Record):
    name: str
    version: str


class SourceRef(_Record):
    """The input file: an image or a video."""

    path: str | None = Field(description="Input path; omitted when paths are anonymized")
    sha256: str | None = Field(description="SHA-256 of the file; null when it could not be read")


class FrameRef(_Record):
    """The image sent to the model: the image itself, or one frame picked from a video."""

    index: int = Field(description="Position among the frames picked from the input; 0 for images")
    timestamp_ms: int | None = Field(description="Time in the video; null for images")
    sha256: str = Field(description="SHA-256 of the image bytes sent to the model")
    phash: str = Field(description="Perceptual hash (hex) of the image")


class Usage(_Record):
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    latency_ms: int = 0


class Observation(_Record):
    """What was read from one image or video frame."""

    schema_id: Literal["framelift/observation/v2"] = Field(
        default=OBSERVATION_SCHEMA, alias="schema"
    )
    status: Literal["ok", "error"]
    profile: ProfileRef
    source: SourceRef
    frame: FrameRef | None = Field(
        description="The image sent; null when the input could not be read at all"
    )
    data: dict[str, Any] | None = Field(description="The validated output; null on error")
    error: str | None = None
    model: str
    usage: Usage
    retries: int = Field(description="How many times validation asked the model to try again")

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)
