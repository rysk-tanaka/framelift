"""Structured extraction from one image with Pydantic AI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic_ai import Agent, AgentRetries, BinaryContent
from pydantic_ai.messages import RetryPromptPart

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from framelift._internal.sources import ImageSource
    from framelift.profile import Profile, Validator

USER_PROMPT = "Extract the data shown in this screenshot."


@dataclass(frozen=True)
class Extraction[OutputT: BaseModel]:
    output: OutputT
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


class Extractor[OutputT: BaseModel]:
    """Reads one image into ``profile.output_type``, retrying when validation fails."""

    def __init__(self, profile: Profile[OutputT], model: Model | str, *, max_retries: int) -> None:
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
        started = time.perf_counter()
        result = self._agent.run_sync(
            [USER_PROMPT, BinaryContent(data=image.data, media_type=image.media_type)],
            model=self._model,
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        retries = sum(
            isinstance(part, RetryPromptPart)
            for message in result.all_messages()
            for part in getattr(message, "parts", ())
        )
        usage = result.usage
        return Extraction(
            output=result.output,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            requests=usage.requests,
            retries=retries,
            latency_ms=latency_ms,
        )
