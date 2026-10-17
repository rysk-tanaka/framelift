"""Structured extraction from one image with Pydantic AI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel
from pydantic_ai import Agent, AgentRetries, BinaryContent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.usage import RunUsage

from framelift._internal.cache import ExtractEntry, cache_key, json_sha256, same_state
from framelift.observation import FrameRef, Observation, ProfileRef, SourceRef, Usage

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model

    from framelift._internal.cache import Cache
    from framelift._internal.sources import ImageSource
    from framelift.observation import Crop, Screen
    from framelift.profile import Profile, Validator

USER_PROMPT = "Extract the data shown in this screenshot."


@dataclass(frozen=True)
class Extraction[OutputT: BaseModel]:
    output: OutputT | None
    """The validated output, or None when validation failed after every retry."""
    error: str | None
    input_tokens: int
    output_tokens: int
    requests: int
    retries: int
    latency_ms: int
    cached: bool = False
    """Read from the cache: the usage, retries and latency are those of the original run."""


def _adapt[OutputT: BaseModel](validator: Validator[OutputT]):
    def run(output: OutputT) -> OutputT:
        result = validator(output)
        return output if result is None else result

    return run


def model_label(model: Model | str) -> str:
    if isinstance(model, str):
        return model
    return f"{model.system}:{model.model_name}"


def _prompt(image: ImageSource) -> list[str | BinaryContent]:
    return [USER_PROMPT, BinaryContent(data=image.data, media_type=image.media_type)]


def _extraction[OutputT: BaseModel](
    output: OutputT | None,
    error: str | None,
    usage: RunUsage,
    messages: list[ModelMessage],
    started: float,
) -> Extraction[OutputT]:
    retries = sum(
        isinstance(part, RetryPromptPart) for message in messages for part in message.parts
    )
    return Extraction(
        output=output,
        error=error,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
        retries=retries,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


class Extractor[OutputT: BaseModel]:
    """Reads one image into ``profile.output_type``, retrying when validation fails."""

    def __init__(
        self,
        profile: Profile[OutputT],
        model: Model | str,
        *,
        max_retries: int,
        cache: Cache | None = None,
    ) -> None:
        self._profile = profile
        self._model = model
        self._max_retries = max_retries
        self._cache = cache
        self._model_id = None if cache is None else cache.model_id(model)
        # Computed once: it builds the output model's JSON schema.
        self._fingerprint = profile.fingerprint if self._model_id is not None else ""
        self._agent: Agent[None, OutputT] = Agent(
            output_type=profile.output_type,
            instructions=profile.instructions,
            retries=AgentRetries(output=max_retries),
            name=f"framelift:{profile.name}",
        )
        for validator in profile.validators:
            self._agent.output_validator(_adapt(validator))

    def extract(self, image: ImageSource) -> Extraction[OutputT]:
        """Read one image, or take it from the cache. A failure after every retry is returned,
        not raised, with the usage and retries it took, so failed images still count toward
        cost."""
        key = self._cache_key(image)
        cached = self._cached(key)
        if cached is not None:
            return cached
        extraction = self._extract(image)
        self._store(key, extraction)
        return extraction

    async def extract_async(self, image: ImageSource) -> Extraction[OutputT]:
        """`extract` for callers that already run an event loop, such as pydantic-evals."""
        key = self._cache_key(image)
        cached = self._cached(key)
        if cached is not None:
            return cached
        extraction = await self._extract_async(image)
        self._store(key, extraction)
        return extraction

    def _extract(self, image: ImageSource) -> Extraction[OutputT]:
        started = time.perf_counter()
        usage = RunUsage()
        with capture_run_messages() as messages:
            try:
                result = self._agent.run_sync(_prompt(image), model=self._model, usage=usage)
            except UnexpectedModelBehavior as exc:
                return _extraction(None, str(exc), usage, messages, started)
        return _extraction(result.output, None, usage, messages, started)

    async def _extract_async(self, image: ImageSource) -> Extraction[OutputT]:
        started = time.perf_counter()
        usage = RunUsage()
        with capture_run_messages() as messages:
            try:
                result = await self._agent.run(_prompt(image), model=self._model, usage=usage)
            except UnexpectedModelBehavior as exc:
                return _extraction(None, str(exc), usage, messages, started)
        return _extraction(result.output, None, usage, messages, started)

    def _cache_key(self, image: ImageSource) -> str | None:
        if self._model_id is None:
            return None
        return cache_key(image.sha256, self._model_id, self._fingerprint)

    def _cached(self, key: str | None) -> Extraction[OutputT] | None:
        if key is None or self._cache is None:
            return None
        entry = self._cache.read("extract", key, ExtractEntry)
        # One that took more retries than allowed now would not have succeeded under this limit.
        if entry is None or entry.retries > self._max_retries:
            return None
        # Anything the output model's own validators or serializers raise is a miss: the cache
        # must never stop a run that would succeed without it.
        try:
            output = self._profile.output_type.model_validate_json(entry.input, by_name=True)
            if json_sha256(output.model_dump(mode="json")) != entry.data_sha256:
                return None
        except Exception:
            return None
        return Extraction(
            output=output,
            error=None,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            requests=entry.requests,
            retries=entry.retries,
            latency_ms=entry.latency_ms,
            cached=True,
        )

    def _store(self, key: str | None, extraction: Extraction[OutputT]) -> None:
        # Failures are not stored: the next run may succeed.
        output = extraction.output
        if key is None or self._cache is None or output is None:
            return
        # By field name, so a model that dumps by alias still validates back, and only the
        # fields that were set, so they are the ones set again. Fields excluded from dumps and
        # private attributes are not stored, so what observations leave out stays out of the
        # cache too.
        try:
            text = output.model_dump_json(by_alias=False, round_trip=True, exclude_unset=True)
            restored = self._profile.output_type.model_validate_json(text, by_name=True)
        except Exception:  # e.g. a validator that reads an alias key; not stored, as above
            return
        # Only an output that comes back exactly as it was is stored, so a hit gives the same
        # data, items and keys as the run that paid for it. One that does not (an excluded field
        # or private attribute set, a validator that changes its input, NaN) is just not cached.
        try:
            is_same = same_state(restored, output)
        except Exception:  # e.g. RecursionError on a private attribute that refers to itself
            return
        if not is_same:
            return
        entry = ExtractEntry(
            input=text,
            data_sha256=json_sha256(output.model_dump(mode="json")),
            input_tokens=extraction.input_tokens,
            output_tokens=extraction.output_tokens,
            requests=extraction.requests,
            retries=extraction.retries,
            latency_ms=extraction.latency_ms,
        )
        self._cache.write("extract", key, entry)

    def observe(
        self, image: ImageSource, *, anonymize_paths: bool = False, screen: Screen | None = None
    ) -> tuple[Observation, OutputT | None]:
        """Extract one image into an `Observation`; a failed extraction becomes an error record.

        Also returns the validated output (``None`` on error), so callers that need the model,
        such as evaluation, do not have to validate the stored JSON again. ``screen`` is the
        screen check that let the image through, if one was asked.
        """
        extraction = self.extract(image)
        return self._observation(image, extraction, anonymize_paths, screen), extraction.output

    async def observe_async(
        self, image: ImageSource, *, anonymize_paths: bool = False
    ) -> tuple[Observation, OutputT | None]:
        """`observe` for callers that already run an event loop."""
        extraction = await self.extract_async(image)
        return self._observation(image, extraction, anonymize_paths), extraction.output

    def unreadable(
        self,
        path: Path,
        sha256: str | None,
        error: str,
        *,
        anonymize_paths: bool = False,
        crop: Crop | None = None,
    ) -> Observation:
        """An error record for an input that could not be read, or a screen that could not be
        cropped (``crop`` then says its orientation), so nothing was sent."""
        return Observation(
            status="error",
            profile=ProfileRef(name=self._profile.name, version=self._profile.version),
            source=SourceRef(path=None if anonymize_paths else path.as_posix(), sha256=sha256),
            frame=None,
            crop=crop,
            screen=None,
            cached=False,
            data=None,
            error=error,
            model=model_label(self._model),
            usage=Usage(),
            retries=0,
        )

    def screened_out(
        self, image: ImageSource, screen: Screen, *, anonymize_paths: bool = False
    ) -> Observation:
        """The record of an image not extracted because of its screen check: skipped when the
        check found another screen, an error when the check itself failed."""
        return self._observation(
            image,
            Extraction(
                output=None,
                error=None if screen.error is None else f"Screen check failed: {screen.error}",
                input_tokens=0,
                output_tokens=0,
                requests=0,
                retries=0,
                latency_ms=0,
            ),
            anonymize_paths,
            screen,
            status="error" if screen.error is not None else "skipped",
        )

    def _observation(
        self,
        image: ImageSource,
        extraction: Extraction[OutputT],
        anonymize_paths: bool,
        screen: Screen | None = None,
        *,
        status: Literal["skipped", "error"] | None = None,
    ) -> Observation:
        output = extraction.output
        return Observation(
            status=status or ("ok" if output is not None else "error"),
            profile=ProfileRef(name=self._profile.name, version=self._profile.version),
            source=SourceRef(
                path=None if anonymize_paths else image.path.as_posix(),
                sha256=image.source_sha256,
            ),
            frame=FrameRef(
                index=image.frame_index,
                timestamp_ms=image.timestamp_ms,
                sha256=image.sha256,
                phash=image.phash,
            ),
            crop=image.crop,
            screen=screen,
            cached=extraction.cached,
            data=None if output is None else output.model_dump(mode="json"),
            error=extraction.error,
            model=model_label(self._model),
            usage=Usage(
                input_tokens=extraction.input_tokens,
                output_tokens=extraction.output_tokens,
                requests=extraction.requests,
                latency_ms=extraction.latency_ms,
            ),
            retries=extraction.retries,
        )
