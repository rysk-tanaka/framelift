"""Screen check: whether an image shows the screen `Profile.screen_description` describes, asked
before extraction so that other screens are not read."""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRetries, BinaryContent, ToolOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.usage import RunUsage

from framelift._internal.cache import ScreenEntry, cache_key
from framelift._internal.extract import model_label
from framelift._internal.layout import encode_png
from framelift.observation import Screen, Usage

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from framelift._internal.cache import Cache
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

    def __init__(
        self,
        description: str,
        model: Model | str,
        *,
        max_retries: int,
        version: str,
        cache: Cache | None = None,
    ) -> None:
        self._description = description
        self._version = version
        self._model = model
        self._max_retries = max_retries
        self._cache = cache
        self._model_id = None if cache is None else cache.model_id(model)
        self._agent: Agent[None, ScreenCheck] = Agent(
            # A name of its own: with the default, the check and the extraction would both
            # answer through "final_result", and a model given both could not tell them apart.
            output_type=ToolOutput(ScreenCheck, name="screen_check"),
            instructions=f"Decide whether the image shows this screen: {description}",
            retries=AgentRetries(output=max_retries),
            name="framelift:screen",
        )

    def classify(self, image: ImageSource) -> Screen:
        """Check one image, or take the answer from the cache. A check that gets no valid answer
        after every retry is returned, not raised, with the usage it took."""
        # The upright, redacted and cropped image, not ``data``: an image sent as it is keeps
        # its EXIF rotation and color mode there, which a PNG could not hold as they are.
        data = screen_image(image.view)
        # The image checked, not the one extracted: they differ once shrunk. The description,
        # not the rest of the profile, decides the answer; the version is there so that bumping
        # it checks every image again, as it reads every image again.
        key = (
            None
            if self._model_id is None
            else cache_key(
                hashlib.sha256(data).hexdigest(), self._model_id, self._description, self._version
            )
        )
        cached = self._cached(key)
        if cached is not None:
            return cached
        screen = self._classify(data)
        self._store(key, screen)
        return screen

    def _classify(self, data: bytes) -> Screen:
        prompt = [USER_PROMPT, BinaryContent(data=data, media_type="image/png")]
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
            cached=False,
        )

    def _cached(self, key: str | None) -> Screen | None:
        if key is None or self._cache is None:
            return None
        entry = self._cache.read("screen", key, ScreenEntry)
        # Every request after the first was a retry; more than allowed now is a miss.
        if entry is None or entry.requests - 1 > self._max_retries:
            return None
        return Screen(
            is_target=entry.is_target,
            confidence=entry.confidence,
            model=model_label(self._model),
            usage=Usage(
                input_tokens=entry.input_tokens,
                output_tokens=entry.output_tokens,
                requests=entry.requests,
                latency_ms=entry.latency_ms,
            ),
            error=None,
            cached=True,
        )

    def _store(self, key: str | None, screen: Screen) -> None:
        if key is None or self._cache is None:
            return
        if screen.is_target is None or screen.confidence is None:  # failed: may succeed later
            return
        entry = ScreenEntry(
            is_target=screen.is_target,
            confidence=screen.confidence,
            input_tokens=screen.usage.input_tokens,
            output_tokens=screen.usage.output_tokens,
            requests=screen.usage.requests,
            latency_ms=screen.usage.latency_ms,
        )
        self._cache.write("screen", key, entry)
