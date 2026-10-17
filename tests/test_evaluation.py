from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pytest
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_core import to_json

from examples.fantasy_shop import ShopScreen
from examples.fantasy_shop import profile as fantasy_shop_profile
from framelift import Profile
from framelift._internal import cache as cache_module
from framelift._internal.dataset import load_dataset, write_dataset
from framelift._internal.evaluation import (
    ItemPairs,
    _matcher,
    build_dataset,
    expected_output,
    field_mismatches,
    normalize_expected,
    run_evaluation,
)

# Most tests look up the answer by the hash of the input image, so the screen is sent whole.
shop_profile = replace(fantasy_shop_profile, locate="full")

type Answer = Callable[[dict[str, Any]], dict[str, Any]]


def _image_sha(messages: list[ModelMessage]) -> str:
    for message in messages:
        for part in getattr(message, "parts", ()):
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                for item in part.content:
                    if isinstance(item, BinaryContent):
                        return hashlib.sha256(item.data).hexdigest()
    raise AssertionError("no image in the request")


def _model(
    dataset_path: Path,
    answer: Answer,
    on_request: Callable[[], None] | None = None,
    *,
    model_type: type[FunctionModel] = FunctionModel,
    model_name: str = "fake",
) -> FunctionModel:
    """A model that answers each image with ``answer(expected)`` for that image."""
    dataset, _ = load_dataset(dataset_path)
    expected = {case.sha256: case.expected for case in dataset.cases}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if on_request is not None:
            on_request()
        data = answer(expected[_image_sha(messages)])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])

    return model_type(respond, model_name=model_name)


def _evaluate(dataset_path: Path, answer: Answer, **kwargs: Any):
    model = _model(dataset_path, answer)
    dataset = build_dataset(dataset_path, shop_profile)
    return run_evaluation(dataset, shop_profile, model, progress=False, **kwargs)


def _mismatches(
    profile: Profile, expected: dict[str, Any], raw_output: dict[str, Any]
) -> tuple[int, int, list[str]]:
    """Score ``raw_output`` the way FieldScores does, after validating it like an extraction."""
    case = expected_output(profile, expected)
    output = profile.output_type.model_validate_json(to_json(raw_output), by_name=True)
    data = output.model_dump(mode="json", by_alias=False)
    matcher = _matcher(profile)
    items = None
    if matcher is not None and case.item_keys is not None:
        output_items = matcher.output_items(output)
        items = ItemPairs(matcher.field_name, case.item_keys, output_items)
    return field_mismatches(case.values, data, items)


def test_perfect_answers_score_one(samples) -> None:
    root, truths = samples
    report = _evaluate(root / "dataset.yaml", lambda expected: expected)

    assert len(report.cases) == len(truths) and not report.failures
    for case in report.cases:
        assert case.assertions["ValidationPassed"].value is True
        assert case.scores["FieldAccuracy"].value == 1.0
        assert case.scores["item_precision"].value == 1.0
        assert case.scores["item_recall"].value == 1.0
        assert case.scores["retries"].value == 0


def test_a_wrong_price_is_a_misread_field_not_a_missing_item(samples) -> None:
    root, _ = samples

    def wrong_first_price(expected: dict[str, Any]) -> dict[str, Any]:
        items = [dict(item) for item in expected["items"]]
        items[0]["price"] += 1
        return {**expected, "items": items}

    report = _evaluate(root / "dataset.yaml", wrong_first_price)

    for case in report.cases:
        first_name = case.expected_output.values["items"][0]["name"]
        accuracy = case.scores["FieldAccuracy"]
        assert 0.0 < accuracy.value < 1.0
        assert accuracy.reason == f"misread: items[{first_name!r}].price"
        assert case.scores["item_precision"].value == 1.0
        assert case.scores["item_recall"].value == 1.0


def test_a_dropped_item_lowers_recall(samples) -> None:
    root, _ = samples

    def drop_last(expected: dict[str, Any]) -> dict[str, Any]:
        return {**expected, "items": expected["items"][:-1]}

    report = _evaluate(root / "dataset.yaml", drop_last)

    for case in report.cases:
        n_items = len(case.expected_output.values["items"])
        last_name = case.expected_output.values["items"][-1]["name"]
        assert case.scores["item_precision"].value == 1.0
        assert case.scores["item_recall"].value == (n_items - 1) / n_items
        accuracy = case.scores["FieldAccuracy"]
        assert accuracy.reason == f"misread: items[{last_name!r}] (missing)"
        # shop_name plus name and price per item; both leaves of the dropped item are misread.
        n_leaves = 1 + 2 * n_items
        assert accuracy.value == (n_leaves - 2) / n_leaves


def test_failed_validation_scores_zero_without_stopping(samples) -> None:
    root, truths = samples

    def no_items(expected: dict[str, Any]) -> dict[str, Any]:
        return {**expected, "items": []}  # the profile validator rejects an empty list

    report = _evaluate(root / "dataset.yaml", no_items, max_retries=1)

    assert len(report.cases) == len(truths) and not report.failures
    for case in report.cases:
        assert case.assertions["ValidationPassed"].value is False
        assert case.scores["FieldAccuracy"].value == 0.0
        assert case.scores["item_recall"].value == 0.0


def test_required_fields_must_be_listed(samples) -> None:
    root, _ = samples
    path = root / "dataset.yaml"
    dataset, _ = load_dataset(path)
    for case in dataset.cases:
        case.expected = {"shop_name": case.expected["shop_name"]}
    write_dataset(path, dataset.cases)

    with pytest.raises(ValueError, match=re.escape("shop-000: items: Field required")):
        build_dataset(path, shop_profile)


def test_fields_with_defaults_may_be_left_out() -> None:
    # RichTable.title and RichRow.note have defaults, so they are not compared.
    assert normalize_expected(RichTable, {"rows": [RICH]}) == {"rows": [RICH]}


def test_image_names_are_not_treated_as_globs(samples) -> None:
    root, _ = samples
    path = root / "dataset.yaml"
    dataset, _ = load_dataset(path)
    first = dataset.cases[0]
    (root / first.image).rename(root / "shop [0].png")
    first.image = "shop [0].png"
    write_dataset(path, dataset.cases)

    report = _evaluate(path, lambda expected: expected)

    assert not report.failures
    assert all(case.scores["FieldAccuracy"].value == 1.0 for case in report.cases)


class AliasedRow(BaseModel):
    name: str = Field(alias="Name")
    price: int


class AliasedTable(BaseModel):
    rows: list[AliasedRow]


def test_items_of_aliased_models_are_matched() -> None:
    profile = Profile(
        locate="full",
        name="p",
        output_type=AliasedTable,
        instructions="read",
        item_key=lambda r: r.name,
    )
    raw_output = {"rows": [{"Name": "a", "price": 1}]}
    expected = {"rows": [{"name": "a", "price": 2}]}  # datasets use field names

    assert _mismatches(profile, expected, raw_output) == (2, 1, ["rows['a'].price"])


class Rarity(Enum):
    COMMON = "common"
    RARE = "rare"


class Ident(BaseModel):
    code: str


class RichRow(BaseModel):
    model_config = ConfigDict(strict=True)

    ident: Ident
    rarity: Rarity
    since: date
    note: str = ""


class RichTable(BaseModel):
    title: str = ""
    rows: list[RichRow]


RICH = {"ident": {"code": "a"}, "rarity": "rare", "since": "2026-01-02"}


def _rich_profile(key: Callable[[Any], Any]) -> Profile:
    return Profile(
        locate="full", name="p", output_type=RichTable, instructions="read", item_key=key
    )


@pytest.mark.parametrize(
    "key", [lambda r: r.ident.code, lambda r: (r.ident.code, r.rarity, r.since)]
)
def test_nested_enum_and_date_keys_match_on_both_sides(key: Callable[[Any], Any]) -> None:
    profile = _rich_profile(key)
    # Three leaves: ident.code, rarity and since.
    assert _mismatches(profile, {"rows": [RICH]}, {"rows": [RICH]}) == (3, 0, [])


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"titel": "x", "rows": []}, "titel: not in the output"),
        (
            {"rows": [{**RICH, "ident": {"code": "a", "cod": "b"}}]},
            "rows[0].ident.cod: not in the output",
        ),
        ({"title": False, "rows": []}, "title"),  # an unquoted YAML `no`
    ],
)
def test_expected_outputs_that_do_not_fit_the_model_are_rejected(
    expected: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        normalize_expected(RichTable, expected)


def test_lists_without_item_key_compare_listed_fields_by_position() -> None:
    expected = {"rows": [{"ident": {"code": "a"}}]}
    assert field_mismatches(expected, {"rows": [{**RICH, "note": ""}]}, None) == (1, 0, [])
    assert field_mismatches(expected, {"rows": []}, None) == (1, 1, ["rows"])


def test_duplicate_keys_are_paired_in_order() -> None:
    profile = Profile(
        locate="full",
        name="p",
        output_type=ShopScreen,
        instructions="read",
        item_key=lambda i: i.name,
    )
    raw_output = {"shop_name": "s", "items": [{"name": "a", "price": 1}, {"name": "a", "price": 2}]}

    assert _mismatches(profile, raw_output, raw_output) == (5, 0, [])


def test_extra_items_lower_precision(samples) -> None:
    root, _ = samples

    def add_item(expected: dict[str, Any]) -> dict[str, Any]:
        return {**expected, "items": [*expected["items"], {"name": "Made Up", "price": 5}]}

    report = _evaluate(root / "dataset.yaml", add_item)

    for case in report.cases:
        n_items = len(case.expected_output.values["items"])
        assert case.scores["item_precision"].value == n_items / (n_items + 1)
        assert case.scores["item_recall"].value == 1.0


def test_an_image_replaced_during_the_run_fails_its_case(samples) -> None:
    root, truths = samples
    path = root / "dataset.yaml"

    def replace_last_image() -> None:
        # Runs while the first case is extracted, after every image was checked.
        (root / truths[-1].file).write_bytes((root / truths[0].file).read_bytes())

    model = _model(path, lambda expected: expected, on_request=replace_last_image)
    report = run_evaluation(build_dataset(path, shop_profile), shop_profile, model, progress=False)

    assert [failure.name for failure in report.failures] == [truths[-1].file.removesuffix(".png")]
    assert "changed after the dataset was checked" in report.failures[0].error_message
    assert len(report.cases) == len(truths) - 1


class Part(BaseModel):
    model_config = ConfigDict(str_to_lower=True)

    name: str
    price: int

    @field_validator("price", mode="before")
    @classmethod
    def _thousands(cls, value: Any) -> Any:
        return int(value.replace(",", "")) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _non_negative(self) -> Part:
        if self.price < 0:
            raise ValueError("price must not be negative")
        return self


class Kit(BaseModel):
    title: str = ""
    main: Part | None = None
    parts: list[Part] | None = None


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"main": {"nam": "x"}}, "main.nam"),  # a typo inside an Optional nested model
        ({"parts": [{"name": "a", "price": -1}]}, "price must not be negative"),
    ],
)
def test_optional_models_and_model_validators_report_problems(
    expected: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        normalize_expected(Kit, expected)


class Order(BaseModel):
    title: str
    parts: list[Part]


def test_nested_models_and_items_must_be_complete() -> None:
    partial_item = {"rows": [{"ident": {"code": "a"}, "rarity": "rare"}]}
    with pytest.raises(ValueError, match=re.escape("rows.0.since: Field required")):
        normalize_expected(RichTable, partial_item)


def test_incomplete_expected_items_are_rejected_before_running(samples) -> None:
    root, _ = samples
    path = root / "dataset.yaml"
    dataset, _ = load_dataset(path)
    dataset.cases[0].expected["items"][0].pop("name")
    write_dataset(path, dataset.cases)

    with pytest.raises(ValueError, match=re.escape("shop-000: items.0.name: Field required")):
        build_dataset(path, shop_profile)


@pytest.mark.parametrize("output_type", [Kit, Order])
def test_nested_values_go_through_their_model_validators(output_type: type[BaseModel]) -> None:
    expected = {"title": "t", "parts": [{"name": "BOLT", "price": "1,200"}]}
    assert normalize_expected(output_type, expected) == {
        "title": "t",
        "parts": [{"name": "bolt", "price": 1200}],
    }


def test_items_are_keyed_after_item_validators() -> None:
    profile = Profile(
        locate="full", name="p", output_type=Order, instructions="read", item_key=lambda p: p.name
    )
    raw_output = {"title": "x", "parts": [{"name": "bolt", "price": 5}]}
    expected = {"title": "x", "parts": [{"name": "BOLT", "price": 5}]}

    assert _mismatches(profile, expected, raw_output) == (3, 0, [])


def test_model_validators_reject_expected_items() -> None:
    with pytest.raises(ValueError, match="price must not be negative"):
        normalize_expected(Order, {"title": "t", "parts": [{"name": "a", "price": -1}]})


class AliasDumped(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    shop_name: str = Field(alias="shopName")


def test_expected_values_keep_field_names_when_models_serialize_by_alias() -> None:
    assert normalize_expected(AliasDumped, {"shop_name": "A"}) == {"shop_name": "A"}


class Row(BaseModel):
    name: str


class Tagged(BaseModel):
    rows: list[Row]

    @model_validator(mode="after")
    def _tag(self) -> Tagged:
        # Not idempotent: validating the dumped output again would tag the names twice.
        for row in self.rows:
            row.name = f"tag:{row.name}"
        return self


def test_non_idempotent_validators_are_applied_once_on_both_sides() -> None:
    profile = Profile(
        locate="full", name="p", output_type=Tagged, instructions="read", item_key=lambda r: r.name
    )
    raw = {"rows": [{"name": "a"}]}

    assert _mismatches(profile, raw, raw) == (1, 0, [])


class Counted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[Row]

    @computed_field
    @property
    def count(self) -> int:
        return len(self.rows)


def test_outputs_that_cannot_be_validated_again_are_scored() -> None:
    profile = Profile(
        locate="full", name="p", output_type=Counted, instructions="read", item_key=lambda r: r.name
    )
    raw = {"rows": [{"name": "a"}]}

    assert _mismatches(profile, raw, raw) == (1, 0, [])


class MaybeNamed(BaseModel):
    name: str | None


class MaybeTable(BaseModel):
    rows: list[MaybeNamed]


def test_item_key_errors_on_expected_items_are_dataset_errors() -> None:
    profile = Profile(
        locate="full",
        name="p",
        output_type=MaybeTable,
        instructions="read",
        item_key=lambda r: r.name.lower(),
    )
    with pytest.raises(ValueError, match="item_key failed on an expected item"):
        expected_output(profile, {"rows": [{"name": None}]})


def test_item_key_errors_on_extracted_items_count_as_misreads() -> None:
    profile = Profile(
        locate="full",
        name="p",
        output_type=MaybeTable,
        instructions="read",
        item_key=lambda r: r.name.lower(),
    )
    raw_output = {"rows": [{"name": None}, {"name": "a"}]}

    assert _mismatches(profile, {"rows": [{"name": "a"}]}, raw_output) == (1, 0, [])


def test_any_item_key_error_on_expected_items_is_a_dataset_error() -> None:
    profile = Profile(
        locate="full",
        name="p",
        output_type=MaybeTable,
        instructions="read",
        item_key=lambda r: r.name[5],
    )
    with pytest.raises(ValueError, match="IndexError"):
        expected_output(profile, {"rows": [{"name": "a"}]})


class Lowered(BaseModel):
    model_config = ConfigDict(str_to_lower=True, strict=True)

    title: str
    count: int


def test_model_config_applies_to_expected_values() -> None:
    assert normalize_expected(Lowered, {"title": "ABC", "count": 3}) == {"title": "abc", "count": 3}
    with pytest.raises(ValueError, match="count"):
        normalize_expected(Lowered, {"title": "a", "count": "3"})  # strict: a string is no int


class WithExcluded(BaseModel):
    title: str
    secret: str = Field(exclude=True)
    note: str = Field(default="", exclude=True)


def test_excluded_fields_are_validated_but_not_compared() -> None:
    expected = {"title": "a", "secret": "s", "note": "n"}  # secret is required
    assert normalize_expected(WithExcluded, expected) == {"title": "a"}


def test_a_failed_case_scores_zero_even_when_no_items_are_expected(samples) -> None:
    root, _ = samples
    path = root / "dataset.yaml"
    dataset, _ = load_dataset(path)
    dataset.cases[0].expected["items"] = []
    write_dataset(path, dataset.cases)

    def no_items(expected: dict[str, Any]) -> dict[str, Any]:
        return {**expected, "items": []}  # the profile validator rejects an empty list

    report = _evaluate(path, no_items, max_retries=0)

    scores = {case.name: case.scores for case in report.cases}
    assert scores["shop-000"]["item_precision"].value == 0.0
    assert scores["shop-000"]["item_recall"].value == 0.0


class Cat(BaseModel):
    kind: Literal["cat"]
    name: str


class Dog(BaseModel):
    kind: Literal["dog"]
    name: str


class Pet(BaseModel):
    model_config = ConfigDict(extra="allow")

    entry: Cat | Dog


def test_unknown_fields_inside_unions_are_rejected() -> None:
    with pytest.raises(ValueError, match=re.escape("entry.naem: not in the output")):
        normalize_expected(Pet, {"entry": {"kind": "cat", "name": "a", "naem": "a"}})


def test_extra_fields_of_models_that_allow_them_are_compared() -> None:
    expected = {"entry": {"kind": "dog", "name": "b"}, "age": 3}
    assert normalize_expected(Pet, expected) == expected


def test_an_expected_empty_list_counts_extra_items_as_a_misread() -> None:
    profile = Profile(
        locate="full",
        name="p",
        output_type=ShopScreen,
        instructions="read",
        item_key=lambda i: i.name,
    )
    raw_output = {"shop_name": "A", "items": [{"name": "Made Up", "price": 5}]}

    assert _mismatches(profile, {"shop_name": "A", "items": []}, raw_output) == (2, 1, ["items"])


class Flag(BaseModel):
    value: bool | int


def test_booleans_and_numbers_are_different_values() -> None:
    assert field_mismatches({"value": False}, {"value": 0}, None) == (1, 1, ["value"])
    assert field_mismatches({"value": 1}, {"value": True}, None) == (1, 1, ["value"])
    assert normalize_expected(Flag, {"value": False}) == {"value": False}


class Sorted(BaseModel):
    rows: list[Order]

    @field_serializer("rows")
    def _by_title(self, rows: list[Order]) -> list[dict[str, Any]]:
        return [row.model_dump() for row in sorted(rows, key=lambda row: row.title)]


def test_a_reordering_list_serializer_keeps_keys_and_values_together() -> None:
    profile = Profile(
        locate="full", name="p", output_type=Sorted, instructions="read", item_key=lambda o: o.title
    )
    expected = {"rows": [{"title": "b", "parts": []}, {"title": "a", "parts": []}]}
    raw_output = {"rows": [{"title": "a", "parts": []}, {"title": "b", "parts": []}]}

    assert _mismatches(profile, expected, raw_output) == (4, 0, [])


def test_a_none_key_is_a_valid_key() -> None:
    profile = Profile(
        locate="full",
        name="p",
        output_type=MaybeTable,
        instructions="read",
        item_key=lambda r: r.name,
    )
    raw = {"rows": [{"name": None}, {"name": "a"}]}

    assert _mismatches(profile, raw, raw) == (2, 0, [])


def test_unhashable_keys_on_extracted_items_count_as_misreads() -> None:
    profile = Profile(
        locate="full",
        name="p",
        output_type=MaybeTable,
        instructions="read",
        item_key=lambda r: r.name if r.name is not None else [],  # ty: ignore[invalid-argument-type]
    )
    raw_output = {"rows": [{"name": None}, {"name": "a"}]}

    assert _mismatches(profile, {"rows": [{"name": "a"}]}, raw_output) == (1, 0, [])


class Tag(RootModel[str]):
    pass


class Tags(BaseModel):
    tags: list[Tag]


def test_items_that_are_not_objects_are_paired_by_key() -> None:
    profile = Profile(
        locate="full", name="p", output_type=Tags, instructions="read", item_key=lambda t: t.root
    )
    expected = {"tags": ["a", "b"]}

    assert _mismatches(profile, expected, {"tags": ["b", "a"]}) == (2, 0, [])
    assert _mismatches(profile, expected, {"tags": ["a"]}) == (2, 1, ["tags['b'] (missing)"])


def _image_sizes(sizes: list[tuple[int, int]]) -> FunctionModel:
    """A model that notes the size of every image it is sent and gives one valid answer."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in getattr(message, "parts", ()):
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    for item in part.content:
                        if isinstance(item, BinaryContent):
                            with Image.open(io.BytesIO(item.data)) as image:
                                sizes.append(image.size)
        data = {"shop_name": "A", "items": [{"name": "a", "price": 1}]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])

    return FunctionModel(respond)


def test_evaluation_sends_the_cropped_image_like_a_run(samples) -> None:
    root, _ = samples
    sizes: list[tuple[int, int]] = []
    dataset = build_dataset(root / "dataset.yaml", fantasy_shop_profile)

    report = run_evaluation(dataset, fantasy_shop_profile, _image_sizes(sizes), progress=False)

    # The input file's hash is checked, so a cropped image (a new PNG) is not taken as replaced.
    assert not report.failures
    assert sizes == [(720, 768), (1280, 648), (720, 768)]


def test_a_case_without_a_crop_box_scores_like_a_failed_extraction(samples) -> None:
    root, _ = samples  # shop-001 is landscape
    profile = replace(fantasy_shop_profile, locate={"portrait": (0.0, 0.0, 1.0, 0.6)})
    sizes: list[tuple[int, int]] = []
    dataset = build_dataset(root / "dataset.yaml", profile)

    report = run_evaluation(dataset, profile, _image_sizes(sizes), progress=False)

    assert not report.failures
    landscape = next(case for case in report.cases if case.name == "shop-001")
    assert landscape.assertions["ValidationPassed"].value is False
    assert landscape.scores["FieldAccuracy"].value == 0.0
    assert len(sizes) == 2


def test_another_model_answering_stops_the_evaluation_after_one_request(
    samples, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answering_model
) -> None:
    root, truths = samples
    assert len(truths) > 1  # so that a later case could have been sent
    dataset_path = root / "dataset.yaml"
    requests: list[None] = []
    model = _model(
        dataset_path,
        lambda expected: expected,
        lambda: requests.append(None),
        model_type=answering_model,
        model_name="qwen3",
    )
    # A cache id, as a model string has, so that the answer could be cached.
    monkeypatch.setattr(cache_module, "model_cache_id", lambda model: "openai-chat:qwen3")
    dataset = build_dataset(dataset_path, shop_profile)

    with pytest.raises(UserError, match="gemma-3") as raised:
        run_evaluation(dataset, shop_profile, model, progress=False, cache_dir=tmp_path / "cache")

    assert len(requests) == 1  # the later cases are not sent
    (note,) = raised.value.__notes__
    assert note.startswith("Found while reading ") and note.endswith(tuple(t.file for t in truths))
    assert not (tmp_path / "cache").exists()
