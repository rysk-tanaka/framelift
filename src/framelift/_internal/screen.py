"""Screen check: whether an image shows the screen `Profile.screen_description` describes, asked
before extraction so that other screens are not read."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRetries, BinaryContent, ToolOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.usage import RunUsage

from framelift._internal.extract import model_label
from framelift._internal.layout import encode_png
from framelift.observation import Screen, Usage

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from framelift._internal.sources import ImageSource

SCREEN_SIZE = 512
"""Longest side, in pixels, of the image checked: telling screens apart needs no fine detail."""

USER_PROMPT = "Does this image show the screen described in the instructions?"


class ScreenCheck(BaseModel):
    """The model's answer."""

    is_target: bool = Field(description="Whether the image shows the described screen")
    confidence: float = Field(ge=0, le=1, description="How sure the answer is, from 0 to 1")


def screen_image(view: Image.Image) -> bytes:
    """``view`` shrunk to `SCREEN_SIZE` on its longest side (never enlarged), as PNG."""
    width, height = view.size
    scale = SCREEN_SIZE / max(width, height)
    if scale >= 1:
        return encode_png(view)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # Resized straight to the small size, without a full-size copy of a large frame first.
    return encode_png(view.resize(size, Image.Resampling.BICUBIC, reducing_gap=2.0))


class ScreenClassifier:
    """Asks ``model`` whether an image shows the screen ``description`` describes."""

    def __init__(self, description: str, model: Model | str, *, max_retries: int) -> None:
        self._model = model
        self._agent: Agent[None, ScreenCheck] = Agent(
            # A name of its own: with the default, the check and the extraction would both
            # answer through "final_result", and a model given both could not tell them apart.
            output_type=ToolOutput(ScreenCheck, name="screen_check"),
            instructions=f"Decide whether the image shows this screen: {description}",
            retries=AgentRetries(output=max_retries),
            name="framelift:screen",
        )

    def classify(self, image: ImageSource) -> Screen:
        """Check one image. A failure is returned, not raised, with the usage it took."""
        # The upright, redacted and cropped image, not ``data``: an image sent as it is keeps
        # its EXIF rotation and color mode there, which a PNG could not hold as they are.
        prompt = [USER_PROMPT, BinaryContent(data=screen_image(image.view), media_type="image/png")]
        started = time.perf_counter()
        usage = RunUsage()
        check: ScreenCheck | None = None
        error: str | None = None
        try:
            check = self._agent.run_sync(prompt, model=self._model, usage=usage).output
        except UnexpectedModelBehavior as exc:
            error = str(exc)
        return Screen(
            is_target=None if check is None else check.is_target,
            confidence=None if check is None else check.confidence,
            model=model_label(self._model),
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                requests=usage.requests,
                latency_ms=round((time.perf_counter() - started) * 1000),
            ),
            error=error,
        )
