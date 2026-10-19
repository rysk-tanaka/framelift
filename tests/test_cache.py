from __future__ import annotations

import datetime as dt
import enum
import hashlib
import json
import math
import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModelSettings

from examples.fantasy_shop import ShopItem, ShopScreen
from examples.fantasy_shop import profile as fantasy_shop_profile
from framelift import Pipeline, Profile
from framelift._internal import cache as cache_module
from framelift._internal.cache import Cache, ExtractEntry, ScreenEntry, cache_key, same_state
from framelift._internal.extract import Extractor, check_model_name, is_local_server
from framelift._internal.layout import Layout
from framelift._internal.screen import screen_image
from framelift._internal.sources import load_image
from tests.answers import output_name, respond_with

if TYPE_CHECKING:
    from pydantic_ai.profiles import ModelProfile

shop_profile = replace(fantasy_shop_profile, locate="full")
ANSWER = {"shop_name": "A", "items": [{"name": "a", "price": 1}]}


@pytest.fixture
def model_ids(monkeypatch: pytest.MonkeyPatch) -> dict[int, str]:
    """Give FunctionModels a cache id, as a model string has; tests may set it per model."""
    ids: dict[int, str] = {}
    monkeypatch.setattr(
        cache_module, "model_cache_id", lambda model: ids.get(id(model), "function-model")
    )
    return ids


class Calls:
    """A model that answers with ``answers`` in turn (the last one repeated) and counts calls."""

    def __init__(self, *answers: dict[str, Any]) -> None:
        self.answers = answers or (ANSWER,)
        self.count = 0
        self.model = FunctionModel(self._respond)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answer = self.answers[min(self.count, len(self.answers) - 1)]
        self.count += 1
        # As JSON text, as a provider sends it: strict models take dates and enums from JSON only.
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(answer))])


def _entries(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("*.json"))


def test_a_second_run_reads_from_the_cache(samples, tmp_path: Path, model_ids) -> None:
    root, _ = samples
    cache_dir = tmp_path / "cache"
    first, second = Calls(), Calls()

    before = Pipeline(shop_profile, model=first.model, cache_dir=cache_dir).run(root)
    after = Pipeline(shop_profile, model=second.model, cache_dir=cache_dir).run(root)

    assert first.count == 3
    assert second.count == 0
    assert [o.cached for o in before.observations] == [False] * 3
    assert [o.cached for o in after.observations] == [True] * 3
    assert after.cached == after.observations
    for old, new in zip(before.observations, after.observations, strict=True):
        assert new.data == old.data
        assert new.usage == old.usage  # the original run's, so costs still compare
        assert new.retries == old.retries
    assert len(after.items) == len(before.items)
    assert len(_entries(cache_dir)) == 3
    assert {p.parent.parent.name for p in _entries(cache_dir)} == {"extract"}


@pytest.mark.parametrize(
    "change",
    [
        lambda p: replace(p, version="2"),
        lambda p: replace(p, instructions="Read it all."),
        lambda p: replace(p, locate={"portrait": (0.0, 0.0, 1.0, 0.6), "landscape": (0, 0, 1, 1)}),
    ],
    ids=["version", "instructions", "crop"],
)
def test_a_changed_profile_is_not_read_from_the_cache(
    samples, tmp_path: Path, model_ids, change
) -> None:
    root, truths = samples
    path = root / truths[0].file
    Pipeline(shop_profile, model=Calls().model, cache_dir=tmp_path).run(path)
    again = Calls()

    (observation,) = (
        Pipeline(change(shop_profile), model=again.model, cache_dir=tmp_path).run(path).observations
    )

    assert again.count == 1
    assert observation.cached is False


def test_another_model_is_not_read_from_the_cache(samples, tmp_path: Path, model_ids) -> None:
    root, truths = samples
    path = root / truths[0].file
    first, second = Calls(), Calls()
    model_ids[id(second.model)] = "another-model"

    Pipeline(shop_profile, model=first.model, cache_dir=tmp_path).run(path)
    Pipeline(shop_profile, model=second.model, cache_dir=tmp_path).run(path)

    assert second.count == 1


def test_model_instances_are_not_cached_and_warn_once(samples, tmp_path: Path) -> None:
    root, _ = samples
    calls = Calls()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline = Pipeline(shop_profile, model=calls.model, cache_dir=tmp_path / "cache")
        pipeline.run(root)
        pipeline.run(root)

    assert calls.count == 6
    assert [str(w.message) for w in caught if "caches only models" in str(w.message)] == [
        "framelift caches only models given as strings such as 'anthropic:<model>'; a Model "
        "instance is not cached, as its settings cannot be told apart"
    ]
    assert _entries(tmp_path) == []


def test_errors_are_not_cached(samples, tmp_path: Path, model_ids) -> None:
    root, truths = samples
    path = root / truths[0].file
    invalid = {"shop_name": "A", "items": [{"name": "a", "price": 0}]}  # price must be > 0
    failing, fixed = Calls(invalid), Calls()

    (error,) = (
        Pipeline(shop_profile, model=failing.model, cache_dir=tmp_path).run(path).observations
    )
    assert error.status == "error"
    assert _entries(tmp_path) == []

    (ok,) = Pipeline(shop_profile, model=fixed.model, cache_dir=tmp_path).run(path).observations

    assert fixed.count == 1
    assert ok.status == "ok" and ok.cached is False


def test_an_answer_that_took_more_retries_than_allowed_is_not_used(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    path = root / truths[0].file
    invalid = {"shop_name": "A", "items": [{"name": "a", "price": 0}]}
    slow = Calls(invalid, invalid, ANSWER)  # passes on the second retry

    (paid,) = (
        Pipeline(shop_profile, model=slow.model, max_retries=2, cache_dir=tmp_path)
        .run(path)
        .observations
    )
    strict = Calls(invalid)
    (strict_run,) = (
        Pipeline(shop_profile, model=strict.model, max_retries=0, cache_dir=tmp_path)
        .run(path)
        .observations
    )
    lenient = Calls()
    (lenient_run,) = (
        Pipeline(shop_profile, model=lenient.model, max_retries=3, cache_dir=tmp_path)
        .run(path)
        .observations
    )

    assert paid.retries == 2
    assert strict.count == 1 and strict_run.status == "error"
    assert lenient.count == 0 and lenient_run.cached is True


def test_a_broken_entry_is_a_miss_and_is_written_again(samples, tmp_path: Path, model_ids) -> None:
    root, truths = samples
    path = root / truths[0].file
    Pipeline(shop_profile, model=Calls().model, cache_dir=tmp_path).run(path)
    (entry,) = _entries(tmp_path)
    entry.write_text("{not json")
    again = Calls()

    (observation,) = (
        Pipeline(shop_profile, model=again.model, cache_dir=tmp_path).run(path).observations
    )

    assert again.count == 1
    assert observation.status == "ok"
    assert json.loads(entry.read_text())["input"]


def test_an_unwritable_cache_warns_once_and_the_run_goes_on(
    samples, tmp_path: Path, model_ids
) -> None:
    root, _ = samples
    blocked = tmp_path / "cache"
    blocked.write_text("a file, not a directory")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Pipeline(shop_profile, model=Calls().model, cache_dir=blocked).run(root)

    assert [o.status for o in result.observations] == ["ok"] * 3
    assert len([w for w in caught if "could not write to the cache" in str(w.message)]) == 1


class Tagged(BaseModel):
    shop_name: str
    items: list[ShopItem]
    tag: int | bool = Field(default=0, exclude=True)


def _tagged_profile(output_type: type[BaseModel] = Tagged, **kwargs: Any) -> Profile:
    return Profile(
        name="tagged",
        output_type=output_type,
        instructions="read",
        locate="full",
        item_key=lambda item: item.name,
        **kwargs,
    )


@pytest.mark.parametrize("tag", [7, False], ids=["value", "false-equals-0"])
def test_an_output_with_an_excluded_field_set_is_not_cached(
    samples, tmp_path: Path, model_ids, tag: int | bool
) -> None:
    root, truths = samples
    path = root / truths[0].file
    calls = Calls({**ANSWER, "tag": tag})

    Pipeline(_tagged_profile(), model=calls.model, cache_dir=tmp_path).run(path)
    Pipeline(_tagged_profile(), model=calls.model, cache_dir=tmp_path).run(path)

    assert calls.count == 2
    assert _entries(tmp_path) == []


class Noted(BaseModel):
    shop_name: str
    items: list[ShopItem]
    note: str = ""


@pytest.mark.parametrize(
    "answer", [ANSWER, {**ANSWER, "note": ""}], ids=["left-out", "set-to-the-default"]
)
def test_fields_left_at_their_default_or_set_to_it_are_cached(
    samples, tmp_path: Path, model_ids, answer: dict[str, Any]
) -> None:
    root, truths = samples
    path = root / truths[0].file
    reread = Calls()

    (before,) = (
        Pipeline(_tagged_profile(Noted), model=Calls(answer).model, cache_dir=tmp_path)
        .run(path)
        .observations
    )
    (after,) = (
        Pipeline(_tagged_profile(Noted), model=reread.model, cache_dir=tmp_path)
        .run(path)
        .observations
    )

    assert reread.count == 0
    assert after.data == before.data


class Suffixed(BaseModel):
    shop_name: str
    items: list[ShopItem]

    @field_validator("shop_name")
    @classmethod
    def suffix(cls, value: str) -> str:
        return value + "!"  # not idempotent: validating its own output changes it


def test_an_output_whose_validators_change_it_again_is_not_cached(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    calls = Calls()

    for _ in range(2):
        (observation,) = (
            Pipeline(_tagged_profile(Suffixed), model=calls.model, cache_dir=tmp_path)
            .run(root / truths[0].file)
            .observations
        )
        assert observation.data == {**ANSWER, "shop_name": "A!"}

    assert calls.count == 2


class Special(ShopItem):
    pass


def _specialize(screen: ShopScreen) -> ShopScreen:
    return screen.model_copy(update={"items": [Special(**i.model_dump()) for i in screen.items]})


def test_an_output_whose_profile_validator_swaps_item_types_is_not_cached(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    profile = replace(shop_profile, validators=[_specialize])
    calls = Calls()

    Pipeline(profile, model=calls.model, cache_dir=tmp_path).run(root / truths[0].file)
    Pipeline(profile, model=calls.model, cache_dir=tmp_path).run(root / truths[0].file)

    assert calls.count == 2


class Rarity(enum.Enum):
    COMMON = "common"
    RARE = "rare"


class Listed(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    rarity: Rarity
    since: dt.date


class Catalog(BaseModel):
    model_config = ConfigDict(strict=True)

    title: str = Field(alias="Title")
    items: list[Listed]


def test_strict_models_with_dates_enums_and_aliases_come_back_the_same(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    answer = {
        "Title": "T",
        "items": [{"name": "a", "rarity": "rare", "since": "2026-01-02"}],
    }
    first, second = Calls(answer), Calls()

    before = Pipeline(_tagged_profile(Catalog), model=first.model, cache_dir=tmp_path).run(
        root / truths[0].file
    )
    after = Pipeline(_tagged_profile(Catalog), model=second.model, cache_dir=tmp_path).run(
        root / truths[0].file
    )

    assert second.count == 0
    assert after.observations[0].data == before.observations[0].data
    assert after.observations[0].data == {
        "title": "T",
        "items": [{"name": "a", "rarity": "rare", "since": "2026-01-02"}],
    }
    assert [i.key for i in after.items] == [i.key for i in before.items] == ["a"]


def test_screen_checks_are_cached(samples, tmp_path: Path, model_ids) -> None:
    root, _ = samples
    profile = replace(shop_profile, screen_description="a shop's item list")
    asked: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = output_name(info)
        asked.append(name)
        answer = {"is_target": True, "confidence": 0.9} if name == "screen_check" else ANSWER
        return respond_with(info, answer)

    Pipeline(profile, model=FunctionModel(respond), cache_dir=tmp_path).run(root)
    first = list(asked)
    asked.clear()
    result = Pipeline(profile, model=FunctionModel(respond), cache_dir=tmp_path).run(root)

    assert first.count("screen_check") == 3
    assert asked == []
    assert all(o.screen is not None and o.screen.cached for o in result.observations)
    assert all(o.screen is not None and o.screen.usage.requests == 1 for o in result.observations)
    assert {p.parent.parent.name for p in _entries(tmp_path)} == {"extract", "screen"}


def test_another_screen_description_is_checked_again(samples, tmp_path: Path, model_ids) -> None:
    root, truths = samples
    asked: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = output_name(info)
        asked.append(name)
        answer = {"is_target": True, "confidence": 0.9} if name == "screen_check" else ANSWER
        return respond_with(info, answer)

    for description in ("a shop", "a shop's item list"):
        profile = replace(shop_profile, screen_description=description)
        Pipeline(profile, model=FunctionModel(respond), cache_dir=tmp_path).run(
            root / truths[0].file
        )

    # The extraction is the same, as the description is not in the fingerprint.
    assert asked == ["screen_check", "final_result", "screen_check"]


def test_another_server_is_asked_again(samples, tmp_path: Path, model_ids, answering_model) -> None:
    root, truths = samples
    profile = replace(shop_profile, screen_description="a shop's item list")
    asked: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = output_name(info)
        asked.append(name)
        answer = {"is_target": True, "confidence": 0.9} if name == "screen_check" else ANSWER
        return respond_with(info, answer)

    # The same model id, as the same model string would give, on two servers.
    for url in (
        "http://localhost:1234/v1/",
        "http://192.0.2.1:1234/v1/",
        "http://localhost:1234/v1/",
    ):
        # Not an OpenAI-compatible server, so that only the URL differs.
        model = answering_model(respond, system="function", answered=None, base_url=url)
        Pipeline(profile, model=model, cache_dir=tmp_path).run(root / truths[0].file)

    # Both the check and the extraction are asked again on the second server, and not on the
    # first one again.
    assert asked == ["screen_check", "final_result"] * 2


def test_a_model_string_takes_its_server_from_the_environment(
    samples, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, truths = samples
    image = load_image(root / truths[0].file, Layout.full())
    monkeypatch.setenv("OPENAI_API_KEY", "not-sent")  # nothing is asked
    keys = []
    for url in ("http://localhost:1234/v1", "http://192.0.2.1:1234/v1"):
        monkeypatch.setenv("OPENAI_BASE_URL", url)
        extractor = Extractor(
            shop_profile, "openai-chat:qwen3", max_retries=0, cache=Cache(tmp_path)
        )
        keys.append(extractor._cache_key(image))
        # The server found is one the model name is checked on.
        assert is_local_server(extractor._model)
        with pytest.raises(UserError, match="gemma-3"):
            check_model_name(extractor._model, ModelResponse(parts=[], model_name="gemma-3"))

    assert None not in keys
    assert keys[0] != keys[1]


class Private(BaseModel):
    value: int
    _note: str = PrivateAttr(default="")


class Extra(BaseModel):
    model_config = ConfigDict(extra="allow")
    value: int


class Defaulted(BaseModel):
    value: int = 0


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        (1, 1, True),
        (1, True, False),
        (1, 1.0, False),
        (math.nan, math.nan, False),
        ({0: "x"}, {False: "x"}, False),
        ([1, 2], [1, 2], True),
        ([1, 2], (1, 2), False),
        ({1, 2}, {2, 1}, True),
        (Defaulted(), Defaulted(value=0), False),  # value is set in one only
        (Extra(value=1, more=1), Extra(value=1, more=True), False),
    ],
)
def test_same_state(a: Any, b: Any, same: bool) -> None:
    assert same_state(a, b) is same


def test_same_state_compares_private_attributes() -> None:
    a, b = Private(value=1), Private(value=1)
    assert same_state(a, b)
    b._note = "x"
    assert not same_state(a, b)


def test_an_entry_whose_output_now_dumps_differently_is_a_miss(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    path = root / truths[0].file
    Pipeline(shop_profile, model=Calls().model, cache_dir=tmp_path).run(path)
    (entry,) = _entries(tmp_path)
    stored = json.loads(entry.read_text())
    entry.write_text(json.dumps({**stored, "data_sha256": "0" * 64}))
    again = Calls()

    Pipeline(shop_profile, model=again.model, cache_dir=tmp_path).run(path)

    assert again.count == 1


def test_an_evaluation_repeated_with_a_cache_calls_no_model(
    samples, tmp_path: Path, model_ids
) -> None:
    evaluation = pytest.importorskip("framelift._internal.evaluation")
    root, _ = samples
    dataset = evaluation.build_dataset(root / "dataset.yaml", shop_profile)
    first, second = Calls(), Calls()

    evaluation.run_evaluation(
        dataset, shop_profile, first.model, progress=False, cache_dir=tmp_path
    )
    report = evaluation.run_evaluation(
        dataset, shop_profile, second.model, progress=False, cache_dir=tmp_path
    )

    assert first.count == 3
    assert second.count == 0
    assert all(case.output.observation.cached for case in report.cases)


def test_each_video_frame_is_cached_under_the_image_sent(
    scrolling_video, tmp_path: Path, model_ids
) -> None:
    path, sample = scrolling_video
    asked: list[str] = []

    def by_image(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # A different answer per frame, so frames that shared one entry would show.
        image = next(
            item
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content
            if isinstance(item, BinaryContent)
        )
        asked.append(image.identifier)
        answer = {**ANSWER, "shop_name": hashlib.sha256(image.data).hexdigest()[:8]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(answer))])

    before = Pipeline(shop_profile, model=FunctionModel(by_image), cache_dir=tmp_path).run(path)
    calls = len(asked)
    after = Pipeline(shop_profile, model=FunctionModel(by_image), cache_dir=tmp_path).run(path)

    assert calls == len(sample.holds) > 1
    assert len(asked) == calls  # none the second time
    assert [o.data for o in after.observations] == [o.data for o in before.observations]
    assert len({o.data["shop_name"] for o in after.observations if o.data}) == calls
    assert len(_entries(tmp_path)) == calls


def _screen_answers(*checks: dict[str, Any], model_profile: ModelProfile | None = None):
    """A model that answers the screen check with ``checks`` in turn, and the extraction with
    `ANSWER`; it counts the checks asked."""
    asked = {"screen_check": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = output_name(info)
        if name != "screen_check":
            return respond_with(info, json.dumps(ANSWER))
        answer = checks[min(asked["screen_check"], len(checks) - 1)]
        asked["screen_check"] += 1
        return respond_with(info, json.dumps(answer))

    return FunctionModel(respond, profile=model_profile), asked


def test_failed_screen_checks_are_not_cached(samples, tmp_path: Path, model_ids) -> None:
    root, truths = samples
    path = root / truths[0].file
    profile = replace(shop_profile, screen_description="a shop")
    failing, _ = _screen_answers({"is_target": True, "confidence": 2.0})
    passing, asked = _screen_answers({"is_target": True, "confidence": 0.9})

    (error,) = Pipeline(profile, model=failing, cache_dir=tmp_path).run(path).observations
    assert error.status == "error"
    assert [p for p in _entries(tmp_path) if p.parent.parent.name == "screen"] == []

    (ok,) = Pipeline(profile, model=passing, cache_dir=tmp_path).run(path).observations

    assert asked["screen_check"] == 1
    assert ok.screen is not None and ok.screen.cached is False


def test_a_screen_check_that_took_more_retries_than_allowed_is_not_used(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    path = root / truths[0].file
    profile = replace(shop_profile, screen_description="a shop")
    invalid, valid = {"is_target": True, "confidence": 2.0}, {"is_target": True, "confidence": 0.9}
    slow, _ = _screen_answers(invalid, valid)  # passes on the first retry

    Pipeline(profile, model=slow, max_retries=1, cache_dir=tmp_path).run(path)
    # The same limit first: the stricter run below checks again and replaces the entry.
    same, same_asked = _screen_answers(valid)
    (same_run,) = (
        Pipeline(profile, model=same, max_retries=1, cache_dir=tmp_path).run(path).observations
    )
    strict, strict_asked = _screen_answers(valid)
    (strict_run,) = (
        Pipeline(profile, model=strict, max_retries=0, cache_dir=tmp_path).run(path).observations
    )

    assert same_asked["screen_check"] == 0  # the same limit as the run that paid for it
    assert same_run.screen is not None and same_run.screen.cached is True
    assert strict_asked["screen_check"] == 1
    assert strict_run.screen is not None and strict_run.screen.cached is False


class AliasKeyed(BaseModel):
    shop_name: str = Field(alias="shopName")
    items: list[ShopItem]

    @model_validator(mode="before")
    @classmethod
    def strip(cls, data: Any) -> Any:
        # Reads the alias key only, so the by-name JSON the cache keeps raises KeyError.
        return {**data, "shopName": data["shopName"].strip()}


def test_an_output_model_that_raises_on_its_cached_json_does_not_stop_the_run(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    path = root / truths[0].file
    profile = _tagged_profile(AliasKeyed)
    answer = {"shopName": " A ", "items": [{"name": "a", "price": 1}]}
    calls = Calls(answer)

    # Storing: the round trip raises, so nothing is written.
    first = Pipeline(profile, model=calls.model, cache_dir=tmp_path).run(path)
    assert _entries(tmp_path) == []
    # Reading: an entry in the by-name form, which this model cannot validate, is a miss.
    key = cache_key(
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "function-model",
        "",  # a FunctionModel names no server
        profile.fingerprint,
    )
    entry = ExtractEntry(
        input=json.dumps({"shop_name": "A", "items": answer["items"]}),
        data_sha256="0" * 64,
        input_tokens=1,
        output_tokens=1,
        requests=1,
        retries=0,
        latency_ms=1,
    )
    Cache(tmp_path).write("extract", key, entry)
    second = Pipeline(profile, model=calls.model, cache_dir=tmp_path).run(path)

    assert [o.status for o in first.observations + second.observations] == ["ok", "ok"]
    assert calls.count == 2


class SelfReferring(BaseModel):
    shop_name: str
    items: list[ShopItem]
    _me: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def remember(self) -> SelfReferring:
        self._me = self  # comparing it with its copy would recurse without end
        return self


def test_an_output_that_cannot_be_compared_is_not_cached_and_the_run_goes_on(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    calls = Calls()

    result = Pipeline(_tagged_profile(SelfReferring), model=calls.model, cache_dir=tmp_path).run(
        root / truths[0].file
    )

    assert [o.status for o in result.observations] == ["ok"]
    assert _entries(tmp_path) == []


def test_a_new_cache_revision_reads_every_image_again(
    samples, tmp_path: Path, model_ids, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, truths = samples
    path = root / truths[0].file
    Pipeline(shop_profile, model=Calls().model, cache_dir=tmp_path).run(path)
    monkeypatch.setattr(cache_module, "CACHE_REVISION", "next")
    again = Calls()

    Pipeline(shop_profile, model=again.model, cache_dir=tmp_path).run(path)

    assert again.count == 1


def test_a_new_profile_version_checks_every_screen_again(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    path = root / truths[0].file
    profile = replace(shop_profile, screen_description="a shop")
    first, _ = _screen_answers({"is_target": False, "confidence": 0.6})
    Pipeline(profile, model=first, cache_dir=tmp_path).run(path)
    again, asked = _screen_answers({"is_target": True, "confidence": 0.9})

    same = Pipeline(profile, model=again, cache_dir=tmp_path).run(path)
    bumped = Pipeline(replace(profile, version="2"), model=again, cache_dir=tmp_path).run(path)

    assert [o.status for o in same.observations] == ["skipped"]  # the cached answer
    assert [o.status for o in bumped.observations] == ["ok"]
    assert asked["screen_check"] == 1


def test_a_check_asked_through_a_tool_is_not_answered_from_one_held_to_a_schema(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    path = root / truths[0].file
    profile = replace(shop_profile, screen_description="a shop")
    native, _ = _screen_answers({"is_target": False, "confidence": 0.6})
    Pipeline(profile, model=native, cache_dir=tmp_path).run(path)
    tool, asked = _screen_answers(
        {"is_target": True, "confidence": 0.9}, model_profile={"supports_json_schema_output": False}
    )

    result = Pipeline(profile, model=tool, cache_dir=tmp_path).run(path)

    assert [o.status for o in result.observations] == ["ok"]
    assert asked["screen_check"] == 1


def test_a_check_asked_through_a_tool_keeps_the_key_it_had_before_native_output(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    path = root / truths[0].file
    profile = replace(shop_profile, screen_description="a shop")
    tool, asked = _screen_answers(
        {"is_target": True, "confidence": 0.9}, model_profile={"supports_json_schema_output": False}
    )
    # An entry as 0.6 wrote it, keyed without the output mode.
    data = screen_image(load_image(path, Layout.full()).view)
    key = cache_key(
        hashlib.sha256(data).hexdigest(), "function-model", "", "a shop", profile.version
    )
    entry = ScreenEntry(
        is_target=False, confidence=0.6, input_tokens=1, output_tokens=1, requests=1, latency_ms=1
    )
    Cache(tmp_path).write("screen", key, entry)

    result = Pipeline(profile, model=tool, cache_dir=tmp_path).run(path)

    assert [o.status for o in result.observations] == ["skipped"]
    assert asked["screen_check"] == 0


def test_an_interrupted_write_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(self: Path, target: Any) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "replace", interrupted)
    entry = ExtractEntry(
        input="{}",
        data_sha256="0" * 64,
        input_tokens=0,
        output_tokens=0,
        requests=1,
        retries=0,
        latency_ms=0,
    )

    with pytest.raises(KeyboardInterrupt):
        Cache(tmp_path).write("extract", "ab" * 32, entry)

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_cache_warnings_point_at_the_callers_code(samples, tmp_path: Path) -> None:
    root, truths = samples
    blocked = tmp_path / "cache"
    blocked.write_text("a file, not a directory")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Pipeline(shop_profile, model=Calls().model, cache_dir=tmp_path / "other")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(cache_module, "model_cache_id", lambda model: "function-model")
            Pipeline(shop_profile, model=Calls().model, cache_dir=blocked).run(
                root / truths[0].file
            )

    assert len(caught) == 2  # a Model instance, then a cache it cannot write to
    assert [w.filename for w in caught] == [__file__, __file__]


def test_a_lone_surrogate_in_the_description_does_not_stop_the_run(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    profile = replace(shop_profile, screen_description="a shop \ud800")
    model, asked = _screen_answers({"is_target": True, "confidence": 0.9})

    for _ in range(2):
        result = Pipeline(profile, model=model, cache_dir=tmp_path).run(root / truths[0].file)
        assert [o.status for o in result.observations] == ["ok"]

    assert asked["screen_check"] == 1  # and the check was cached


def test_a_lone_surrogate_in_the_instructions_does_not_stop_a_cached_run(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    profile = replace(shop_profile, instructions="Read the list \ud800")
    calls = Calls()

    for _ in range(2):
        result = Pipeline(profile, model=calls.model, cache_dir=tmp_path).run(root / truths[0].file)
        assert [o.status for o in result.observations] == ["ok"]

    assert calls.count == 1


def test_model_settings_are_sent_with_the_screen_check_and_the_extraction(
    samples, tmp_path: Path
) -> None:
    root, truths = samples
    profile = replace(shop_profile, screen_description="a shop's item list")
    sent: dict[str, Any] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = output_name(info)
        sent[name] = info.model_settings
        answer = {"is_target": True, "confidence": 0.9} if name == "screen_check" else ANSWER
        return respond_with(info, answer)

    settings: OpenAIChatModelSettings = {"temperature": 0.0, "openai_reasoning_effort": "none"}
    Pipeline(profile, model=FunctionModel(respond), model_settings=settings).run(
        root / truths[0].file
    )

    assert sent == {"screen_check": settings, "final_result": settings}


def test_model_settings_changed_after_construction_are_not_sent(
    samples, tmp_path: Path, model_ids
) -> None:
    root, truths = samples
    profile = replace(shop_profile, screen_description="a shop's item list")
    sent: list[Any] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = output_name(info)
        sent.append(info.model_settings)
        answer = {"is_target": True, "confidence": 0.9} if name == "screen_check" else ANSWER
        return respond_with(info, answer)

    settings: OpenAIChatModelSettings = {
        "openai_reasoning_effort": "none",
        "extra_body": {"top_k": 1},
    }
    pipeline = Pipeline(
        profile, model=FunctionModel(respond), cache_dir=tmp_path, model_settings=settings
    )
    # The cache key was made of the settings as given, so what is changed afterwards, nested
    # dicts included, must not be what is sent and stored under it.
    settings["openai_reasoning_effort"] = "high"
    settings["extra_body"]["top_k"] = 2
    pipeline.run(root / truths[0].file)

    assert sent == [{"openai_reasoning_effort": "none", "extra_body": {"top_k": 1}}] * 2


def test_other_model_settings_are_asked_again(samples, tmp_path: Path, model_ids) -> None:
    root, truths = samples
    profile = replace(shop_profile, screen_description="a shop's item list")
    asked: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = output_name(info)
        asked.append(name)
        answer = {"is_target": True, "confidence": 0.9} if name == "screen_check" else ANSWER
        return respond_with(info, answer)

    runs: list[OpenAIChatModelSettings | None] = [
        None,
        # No settings, as before they could be given: the entries made without them are used.
        {},
        {"openai_reasoning_effort": "none"},
        {"openai_reasoning_effort": "high"},
        {"openai_reasoning_effort": "none"},
    ]
    counts = []
    for settings in runs:
        asked.clear()
        Pipeline(
            profile,
            model=FunctionModel(respond),
            cache_dir=tmp_path,
            model_settings=settings,
        ).run(root / truths[0].file)
        counts.append(len(asked))

    assert counts == [2, 0, 2, 2, 0]


def test_model_settings_json_cannot_hold_are_not_cached(samples, tmp_path: Path, model_ids) -> None:
    root, truths = samples
    calls = Calls()
    settings: Any = {"extra_headers": {"x": object()}}

    with pytest.warns(UserWarning, match="model settings that JSON can hold") as caught:
        for _ in range(2):
            Pipeline(
                shop_profile, model=calls.model, cache_dir=tmp_path, model_settings=settings
            ).run(root / truths[0].file)

    assert calls.count == 2
    assert _entries(tmp_path) == []
    # Once per cache, which each pipeline makes for itself.
    assert len(caught) == 2
    assert caught[0].filename == __file__
