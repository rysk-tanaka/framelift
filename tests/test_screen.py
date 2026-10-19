from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from framelift._internal.extract import resolve_model
from framelift._internal.layout import Layout
from framelift._internal.screen import ScreenClassifier, output_mode, screen_image
from framelift._internal.sources import load_image
from tests.answers import output_name, respond_with


def _image(path: Path, size: tuple[int, int]) -> Path:
    Image.new("RGB", size, (200, 30, 30)).save(path)
    return path


@pytest.mark.parametrize(
    ("size", "expected"),
    [((720, 1280), (288, 512)), ((1280, 720), (512, 288)), ((300, 200), (300, 200))],
    ids=["portrait", "landscape", "small"],
)
def test_the_checked_image_is_shrunk_to_512_on_its_longest_side(
    size: tuple[int, int], expected: tuple[int, int]
) -> None:
    with Image.open(io.BytesIO(screen_image(Image.new("RGB", size)))) as image:
        assert image.format == "PNG"
        assert image.size == expected  # never enlarged


def test_the_check_is_asked_with_the_description(tmp_path: Path) -> None:
    seen: list[tuple[str | None, str, tuple[int, int]]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        (part,) = [p for m in messages for p in m.parts if isinstance(p, UserPromptPart)]
        (content,) = [c for c in part.content if isinstance(c, BinaryContent)]
        with Image.open(io.BytesIO(content.data)) as image:
            seen.append((info.instructions, output_name(info), image.size))
        return respond_with(info, {"is_target": True, "confidence": 0.8})

    source = load_image(_image(tmp_path / "a.png", (720, 1280)), Layout.full())
    classifier = ScreenClassifier(
        "a shop's item list", FunctionModel(respond), max_retries=0, version="1"
    )

    screen = classifier.classify(source)

    ((instructions, tool, size),) = seen
    assert instructions is not None and "a shop's item list" in instructions
    assert tool == "screen_check"
    assert size == (288, 512)
    assert screen.is_target is True
    assert screen.confidence == 0.8
    assert screen.error is None
    assert screen.usage.requests == 1


def test_a_check_that_never_validates_is_a_failure_with_its_usage(tmp_path: Path) -> None:
    def out_of_range(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return respond_with(info, {"is_target": True, "confidence": 2.0})

    source = load_image(_image(tmp_path / "a.png", (100, 200)), Layout.full())
    classifier = ScreenClassifier("a shop", FunctionModel(out_of_range), max_retries=1, version="1")

    screen = classifier.classify(source)

    assert screen.is_target is None
    assert screen.confidence is None
    assert screen.error is not None
    assert screen.usage.requests == 2  # the first try and one retry


def test_the_check_is_held_to_a_schema_on_a_local_server(tmp_path: Path, answering_model) -> None:
    seen: list[tuple[str, int]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append((info.model_request_parameters.output_mode, len(info.output_tools)))
        return respond_with(info, {"is_target": False, "confidence": 0.7})

    source = load_image(_image(tmp_path / "a.png", (100, 200)), Layout.full())
    model = answering_model(respond, answered=None)
    classifier = ScreenClassifier("a shop", model, max_retries=0, version="1")

    screen = classifier.classify(source)

    # No tool, so no text before calling one: a local model wrote its reasons there.
    assert seen == [("native", 0)]
    assert screen.is_target is False


@pytest.mark.parametrize(
    "kwargs",
    [
        # OpenAI's profile claims a schema for every model, older ones too.
        {"base_url": "https://api.openai.com/v1"},
        {"system": "anthropic", "base_url": "https://api.anthropic.com"},
        {"profile": {"supports_json_schema_output": False}},
    ],
)
def test_the_check_is_asked_through_a_tool_elsewhere_or_without_a_schema(
    tmp_path: Path, answering_model, kwargs: dict[str, Any]
) -> None:
    seen: list[list[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append([t.name for t in info.output_tools])
        return respond_with(info, {"is_target": True, "confidence": 0.8})

    source = load_image(_image(tmp_path / "a.png", (100, 200)), Layout.full())
    model = answering_model(respond, answered=None, **kwargs)
    classifier = ScreenClassifier("a shop", model, max_retries=0, version="1")

    screen = classifier.classify(source)

    assert seen == [["screen_check"]]
    assert screen.is_target is True


def test_a_model_without_a_profile_of_its_own_is_asked_through_a_tool(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append([t.name for t in info.output_tools])
        return respond_with(info, {"is_target": True, "confidence": 0.8})

    source = load_image(_image(tmp_path / "a.png", (100, 200)), Layout.full())
    # Its models may differ in what they take, so it has no profile to read the mode from.
    model = FallbackModel(FunctionModel(respond), FunctionModel(respond))
    classifier = ScreenClassifier("a shop", model, max_retries=0, version="1")

    screen = classifier.classify(source)

    assert seen == [["screen_check"]]
    assert screen.is_target is True


@pytest.mark.parametrize(
    ("base_url", "mode"),
    [("http://localhost:1234/v1", "native"), ("https://api.openai.com/v1", "tool")],
)
def test_a_model_string_is_held_to_a_schema_only_on_a_local_server(
    monkeypatch: pytest.MonkeyPatch, base_url: str, mode: str
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # As a run resolves it: on a local server, wrapped to check the model's name.
    assert output_mode(resolve_model("openai:qwen3")) == mode
