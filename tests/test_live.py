"""Tests against a real model, named by FRAMELIFT_LIVE_MODEL (e.g. 'anthropic:<model-name>').

They are deselected by default (``-m "not live"``) and skipped without the variable; the
provider's API key is read from its usual environment variable. Each run reads a few synthetic
screens, so it is billed.
"""

from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from pydantic_ai import models

from examples.fantasy_shop import profile
from framelift import Observation, Pipeline
from framelift._internal import synth

pytestmark = pytest.mark.live

SHOP_SCREEN = "A shop screen listing items with their prices"
# What pydantic_ai raises when a request is made while ALLOW_MODEL_REQUESTS is False.
MODEL_REQUESTS_BLOCKED = "Model requests are not allowed"

# A screen of another kind, for the screen check to skip.
SETTINGS_LINES = ["Settings", "Sound: On", "Vibration: Off", "Language: English"]
SETTINGS_FONT_SIZE = 40
SETTINGS_LEFT, SETTINGS_TOP, SETTINGS_LINE_HEIGHT = 40, 60, 90
SETTINGS_BACKGROUND, SETTINGS_TEXT = (240, 240, 240), (20, 20, 20)


@pytest.fixture
def live_model(monkeypatch: pytest.MonkeyPatch) -> str:
    model = os.environ.get("FRAMELIFT_LIVE_MODEL")
    if not model:
        pytest.skip("set FRAMELIFT_LIVE_MODEL to run tests against a real model")
    # conftest.py forbids model requests for every other test.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    return model


def _expected(truth: synth.SampleTruth) -> dict[str, object]:
    return {"shop_name": truth.shop_name, "items": [asdict(item) for item in truth.items]}


def _file(observation: Observation) -> str:
    assert observation.source.path is not None
    return Path(observation.source.path).name


def _settings_screen(path: Path) -> None:
    # Portrait like the shop screen, so Profile.locate has a box for it and it reaches the check.
    image = Image.new("RGB", synth._SIZES["portrait"], SETTINGS_BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=SETTINGS_FONT_SIZE)
    for index, line in enumerate(SETTINGS_LINES):
        top = SETTINGS_TOP + index * SETTINGS_LINE_HEIGHT
        draw.text((SETTINGS_LEFT, top), line, font=font, fill=SETTINGS_TEXT)
    image.save(path)


def test_every_item_is_read_from_the_cropped_screens(live_model: str, tmp_path: Path) -> None:
    truths = synth.generate(tmp_path / "samples", n=2)  # one portrait, one landscape
    expected = {truth.file: _expected(truth) for truth in truths}

    result = Pipeline(profile, model=live_model).run(tmp_path / "samples")

    read = {_file(o): o.data if o.status == "ok" else o.error for o in result.observations}
    assert read == expected


def test_the_screen_check_skips_another_screen(live_model: str, tmp_path: Path) -> None:
    (truth,) = synth.generate(tmp_path / "samples", n=1)
    _settings_screen(tmp_path / "samples" / "settings.png")
    screened = replace(profile, screen_description=SHOP_SCREEN)

    result = Pipeline(screened, model=live_model).run(tmp_path / "samples")

    by_file = {_file(o): o for o in result.observations}
    shop, other = by_file[truth.file], by_file["settings.png"]
    # Only that it was extracted: what was read is the first test's concern.
    assert shop.status == "ok", shop.error or shop.screen
    assert other.status == "skipped", other.error or other.screen
    assert other.screen is not None and other.screen.is_target is False


def test_a_second_run_is_read_from_the_cache(live_model: str, tmp_path: Path) -> None:
    synth.generate(tmp_path / "samples", n=1)
    screened = replace(profile, screen_description=SHOP_SCREEN)  # the check is cached too
    pipeline = Pipeline(screened, model=live_model, cache_dir=tmp_path / "cache")

    (before,) = pipeline.run(tmp_path / "samples").observations
    # Failures are not cached, so the second run could only ask again.
    assert before.status == "ok", before.error or before.screen
    # Any request on the second run raises, so a hit is known not to have been billed.
    with models.override_allow_model_requests(False):
        try:
            second = pipeline.run(tmp_path / "samples")
        except RuntimeError as exc:  # raised by pydantic_ai, which framelift does not catch
            if MODEL_REQUESTS_BLOCKED not in str(exc):
                raise
            pytest.fail(f"the second run asked the model, so the cache missed: {exc}")

    (after,) = second.observations
    assert not before.cached
    assert before.screen is not None and not before.screen.cached
    assert after.cached
    assert after.screen is not None and after.screen.cached
    assert after.screen.usage == before.screen.usage
    assert after.data == before.data
    assert after.usage == before.usage  # the original run's
