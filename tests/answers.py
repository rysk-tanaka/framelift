"""Responses for tests' `FunctionModel`s in the shape each output is asked for: the screen check
answers in JSON text when the model can be held to a schema, as a `FunctionModel` can, and the
extraction through a tool."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

if TYPE_CHECKING:
    from pydantic_ai.models.function import AgentInfo


def output_name(info: AgentInfo) -> str:
    """The name of the output asked for: "screen_check" or "final_result"."""
    output = info.model_request_parameters.output_object
    return info.output_tools[0].name if output is None else output.name or ""


def respond_with(info: AgentInfo, data: dict[str, Any] | str) -> ModelResponse:
    """``data``, a dict or its JSON, as the answer to the output asked for."""
    if info.model_request_parameters.output_mode == "native":
        return ModelResponse(parts=[TextPart(data if isinstance(data, str) else json.dumps(data))])
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])
