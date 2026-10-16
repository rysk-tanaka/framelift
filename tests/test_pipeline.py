from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from examples.fantasy_shop import profile as shop_profile
from framelift import Observation, Pipeline, Profile


def _image_sha(messages: list[ModelMessage]) -> str:
    for message in messages:
        for part in getattr(message, "parts", ()):
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                for item in part.content:
                    if isinstance(item, BinaryContent):
                        return hashlib.sha256(item.data).hexdigest()
    raise AssertionError("no image in the request")


def _truth_by_sha(root: Path, truths: list) -> dict[str, dict]:
    return {
        hashlib.sha256((root / t.file).read_bytes()).hexdigest(): {
            "shop_name": t.shop_name,
            "items": [asdict(i) for i in t.items],
        }
        for t in truths
    }


def test_run_reads_every_image_and_writes_jsonl(samples, tmp_path: Path) -> None:
    root, truths = samples
    answers = _truth_by_sha(root, truths)

    def read_screen(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, answers[_image_sha(messages)])]
        )

    out = tmp_path / "out"
    result = Pipeline(
        shop_profile, model=FunctionModel(read_screen, model_name="fake"), out_dir=out
    ).run(root)

    assert len(result.succeeded) == len(truths) and not result.failed
    assert result.output_path == out / "observations.jsonl"
    lines = result.output_path.read_text().splitlines()
    observations = [Observation.model_validate_json(line) for line in lines]
    assert [o.data for o in observations] == [answers[o.source.sha256] for o in observations]
    first = observations[0]
    assert first.profile.name == "fantasy-shop"
    assert first.model == "function:fake"
    assert first.retries == 0
    assert '"schema":"framelift/observation/v1"' in lines[0]


def test_validator_failure_triggers_a_retry(samples) -> None:
    root, truths = samples
    answers = _truth_by_sha(root, truths)
    calls: dict[str, int] = {}

    def duplicate_then_correct(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sha = _image_sha(messages)
        calls[sha] = calls.get(sha, 0) + 1
        answer = dict(answers[sha])
        if calls[sha] == 1:  # first attempt repeats a row, which the profile validator rejects
            answer["items"] = answer["items"] + answer["items"][:1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    model = FunctionModel(duplicate_then_correct, model_name="fake")
    result = Pipeline(shop_profile, model=model).run(root / truths[0].file)

    (observation,) = result.observations
    assert observation.status == "ok"
    assert observation.retries == 1
    assert observation.data == answers[observation.source.sha256]
    assert result.output_path is None


def test_exhausted_retries_are_recorded_as_errors(samples) -> None:
    root, truths = samples

    def always_empty(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"shop_name": "x", "items": []})]
        )

    result = Pipeline(shop_profile, model=FunctionModel(always_empty), max_retries=1).run(root)

    assert len(result.failed) == len(truths)
    assert all(o.data is None and o.error for o in result.failed)
    # The failed attempts still cost tokens, which evaluation compares.
    assert all(o.retries == 1 and o.usage.requests == 2 for o in result.failed)
    assert all(o.usage.input_tokens > 0 for o in result.failed)


def test_anonymized_paths(samples) -> None:
    root, truths = samples
    answers = _truth_by_sha(root, truths)

    def read_screen(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, answers[_image_sha(messages)])]
        )

    result = Pipeline(shop_profile, model=FunctionModel(read_screen), anonymize_paths=True).run(
        root
    )
    assert all(o.source.path is None for o in result.observations)


class Aliased(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    shop_name: str = Field(alias="shopName")


def test_observations_keep_the_models_own_serialization(samples) -> None:
    root, truths = samples
    profile = Profile(name="p", output_type=Aliased, instructions="read")

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"shopName": "A"})])

    result = Pipeline(profile, model=FunctionModel(respond)).run(root / truths[0].file)

    assert result.observations[0].data == {"shopName": "A"}
