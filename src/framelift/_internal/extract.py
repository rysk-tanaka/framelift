"""Structured extraction from one image with Pydantic AI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic_ai import Agent, AgentRetries, BinaryContent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.usage import RunUsage

from framelift.observation import Observation, ProfileRef, SourceRef, Usage

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model

    from framelift._internal.sources import ImageSource
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

    def __init__(self, profile: Profile[OutputT], model: Model | str, *, max_retries: int) -> None:
        self._profile = profile
        self._model = model
        self._agent: Agent[None, OutputT] = Agent(
            output_type=profile.output_type,
            instructions=profile.instructions,
            retries=AgentRetries(output=max_retries),
            name=f"framelift:{profile.name}",
        )
        for validator in profile.validators:
            self._agent.output_validator(_adapt(validator))

    def extract(self, image: ImageSource) -> Extraction[OutputT]:
        """Read one image. A failure after every retry is returned, not raised, with the usage
        and retries it took, so failed images still count toward cost."""
        started = time.perf_counter()
        usage = RunUsage()
        with capture_run_messages() as messages:
            try:
                result = self._agent.run_sync(_prompt(image), model=self._model, usage=usage)
            except UnexpectedModelBehavior as exc:
                return _extraction(None, str(exc), usage, messages, started)
        return _extraction(result.output, None, usage, messages, started)

    async def extract_async(self, image: ImageSource) -> Extraction[OutputT]:
        """`extract` for callers that already run an event loop, such as pydantic-evals."""
        started = time.perf_counter()
        usage = RunUsage()
        with capture_run_messages() as messages:
            try:
                result = await self._agent.run(_prompt(image), model=self._model, usage=usage)
            except UnexpectedModelBehavior as exc:
                return _extraction(None, str(exc), usage, messages, started)
        return _extraction(result.output, None, usage, messages, started)

    def observe(
        self, image: ImageSource, *, anonymize_paths: bool = False
    ) -> tuple[Observation, OutputT | None]:
        """Extract one image into an `Observation`; a failed extraction becomes an error record.

        Also returns the validated output (``None`` on error), so callers that need the model,
        such as evaluation, do not have to validate the stored JSON again.
        """
        extraction = self.extract(image)
        return self._observation(image, extraction, anonymize_paths), extraction.output

    async def observe_async(
        self, image: ImageSource, *, anonymize_paths: bool = False
    ) -> tuple[Observation, OutputT | None]:
        """`observe` for callers that already run an event loop."""
        extraction = await self.extract_async(image)
        return self._observation(image, extraction, anonymize_paths), extraction.output

    def _observation(
        self, image: ImageSource, extraction: Extraction[OutputT], anonymize_paths: bool
    ) -> Observation:
        output = extraction.output
        return Observation(
            status="ok" if output is not None else "error",
            profile=ProfileRef(name=self._profile.name, version=self._profile.version),
            source=SourceRef(
                path=None if anonymize_paths else image.path.as_posix(),
                sha256=image.sha256,
                phash=image.phash,
            ),
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
