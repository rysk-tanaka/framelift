"""The `framelift/observation/v3` record written to ``observations.jsonl``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OBSERVATION_SCHEMA = "framelift/observation/v3"


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


type _Box = tuple[float, float, float, float]


class Crop(_Record):
    """How the image sent was cut from the screen (see `Profile.locate` and `Profile.redact`)."""

    orientation: Literal["portrait", "landscape"] = Field(
        description="Orientation of the upright screen"
    )
    size: tuple[int, int] = Field(
        description="(width, height) of the upright screen in pixels, before cropping. Boxes are "
        "fractions of it, so a screen of another aspect ratio may need other boxes"
    )
    bbox: _Box | None = Field(
        description="(left, top, right, bottom) cropped, as fractions of the upright screen; "
        "[0, 0, 1, 1] for locate='full'; null when no crop box fits the orientation"
    )
    redact: list[_Box] = Field(description="Boxes painted black before cropping")


class Usage(_Record):
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    latency_ms: int = 0


class Screen(_Record):
    """The screen check asked before extraction (see `Profile.screen_description`)."""

    is_target: bool | None = Field(
        description="Whether the image shows the described screen; null when the check failed"
    )
    confidence: float | None = Field(description="How sure the check was (0-1); null on failure")
    model: str
    usage: Usage = Field(description="Tokens the check took, apart from the extraction's")
    error: str | None = None


class Observation(_Record):
    """What was read from one image or video frame."""

    schema_id: Literal["framelift/observation/v3"] = Field(
        default=OBSERVATION_SCHEMA, alias="schema"
    )
    status: Literal["ok", "skipped", "error"] = Field(
        description="skipped: the screen check found another screen, so nothing was extracted"
    )
    profile: ProfileRef
    source: SourceRef
    frame: FrameRef | None = Field(description="The image sent; null when nothing was sent")
    crop: Crop | None = Field(
        description="How the image sent was cut from the screen; null when the input could not "
        "be read at all"
    )
    screen: Screen | None = Field(
        description="The screen check; null when the profile has no screen_description or "
        "nothing was sent"
    )
    data: dict[str, Any] | None = Field(description="The validated output; null unless ok")
    error: str | None = None
    model: str
    usage: Usage = Field(description="Tokens the extraction took; zero when it was not asked")
    retries: int = Field(description="How many times validation asked the model to try again")

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)
