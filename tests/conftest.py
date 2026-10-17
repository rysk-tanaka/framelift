from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.function import FunctionModel

from framelift._internal import synth

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

# Tests must never reach a real model provider.
models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def samples(tmp_path: Path) -> tuple[Path, list[synth.SampleTruth]]:
    """Three synthetic shop screenshots and their ground truth."""
    out = tmp_path / "samples"
    truths = synth.generate(out, n=3, seed=42)
    return out, truths


@pytest.fixture
def scrolling_video(tmp_path: Path) -> tuple[Path, synth.SampleVideo]:
    """A synthetic video of a shop list that scrolls between still screens."""
    pytest.importorskip("av")
    path = tmp_path / "video" / "shop-scroll.mp4"
    path.parent.mkdir()
    return path, synth.generate_video(path, seed=7)


LOCAL_SERVER = "http://localhost:1234/v1/"


class AnsweringModel(FunctionModel):
    """A FunctionModel asked through ``system`` at ``base_url``, whose responses name
    ``answered`` (none when it is None), as a server that answers with another model does."""

    def __init__(
        self,
        function: Any,
        *,
        model_name: str | None = None,
        answered: str | None = "gemma-3",
        system: str = "openai",
        base_url: str | None = LOCAL_SERVER,
    ) -> None:
        super().__init__(function, model_name=model_name)
        self._answering_system = system
        self._base_url = base_url
        self._answered = answered

    @property
    def system(self) -> str:
        return self._answering_system

    @property
    def base_url(self) -> str | None:
        return self._base_url

    async def request(self, *args: Any, **kwargs: Any) -> ModelResponse:
        response = await super().request(*args, **kwargs)
        # After FunctionModel, which names every response after itself.
        response.model_name = self._answered
        return response


@pytest.fixture
def answering_model() -> type[AnsweringModel]:
    """`AnsweringModel`, for tests to build models that name the server they ask and the model
    that answers."""
    return AnsweringModel
