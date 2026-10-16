from __future__ import annotations

from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from framelift import Item, Profile
from framelift._internal.aggregate import Aggregator
from framelift._internal.items import item_keyer
from framelift.observation import FrameRef, Observation, ProfileRef, SourceRef, Usage


class Row(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    price: int | None = None


class Table(BaseModel):
    rows: list[Row]


def _observation(index: int, *, status: str = "ok") -> Observation:
    return Observation.model_validate(
        {
            "status": status,
            "profile": ProfileRef(name="p", version="1"),
            "source": SourceRef(path=None, sha256="video"),
            "frame": FrameRef(
                index=index, timestamp_ms=index * 1000, sha256=f"f{index}", phash="0"
            ),
            "data": {} if status == "ok" else None,
            "error": None if status == "ok" else "failed",
            "model": "m",
            "usage": Usage(),
            "retries": 0,
        }
    )


def _profile(output_type: type[BaseModel] = Table, key: Any = lambda r: r.name) -> Profile:
    return Profile(name="p", output_type=output_type, instructions="read", item_key=key)


def _aggregate(profile: Profile, frames: list[Any]) -> Aggregator:
    keyer = item_keyer(profile)
    assert keyer is not None
    aggregator = Aggregator(keyer)
    for index, raw in enumerate(frames):
        aggregator.add(_observation(index), profile.output_type.model_validate(raw))
    return aggregator


def _items(profile: Profile, frames: list[Any]) -> dict[Any, Item]:
    return {item.key: item for item in _aggregate(profile, frames).items()}


def _rows(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"rows": list(rows)}


def test_the_majority_value_wins_and_disagreements_are_kept() -> None:
    frames = [
        _rows({"name": "a", "price": 120}),
        _rows({"name": "a", "price": 12}),
        _rows({"name": "a", "price": 120}),
    ]
    (item,) = _items(_profile(), frames).values()

    assert item.data == {"name": "a", "price": 120}
    assert item.votes == 3
    (conflict,) = item.conflicts
    assert conflict.field == "price"
    assert [(c.value, c.votes) for c in conflict.values] == [(120, 2), (12, 1)]


def test_a_tie_is_left_null_instead_of_guessed() -> None:
    frames = [_rows({"name": "a", "price": 2}), _rows({"name": "a", "price": 1})]
    (item,) = _items(_profile(), frames).values()

    assert item.data == {"name": "a", "price": None}
    assert [(c.value, c.votes) for c in item.conflicts[0].values] == [(1, 1), (2, 1)]


def test_a_missing_field_abstains_but_an_explicit_null_votes() -> None:
    frames = [
        _rows({"name": "a", "note": "x"}),
        _rows({"name": "a"}),  # no note: abstains, so "x" is not outvoted
        _rows({"name": "b", "note": "y"}),
        _rows({"name": "b", "note": None}),
    ]
    items = _items(_profile(), frames)
    reversed_items = _items(_profile(), frames[::-1])

    assert items["a"].data["note"] == "x"
    assert not [c for c in items["a"].conflicts if c.field == "note"]
    assert items["b"].data["note"] is None  # one vote each: a tie
    for key, item in items.items():
        assert reversed_items[key].data == item.data
        assert reversed_items[key].conflicts == item.conflicts


def test_votes_count_frames_and_a_frame_votes_once_per_item() -> None:
    frames = [
        _rows({"name": "a", "price": 1}, {"name": "a", "price": 9}),  # listed twice: first counts
        _rows({"name": "a", "price": 1}, {"name": "b", "price": 2}),
    ]
    items = _items(_profile(), frames)

    assert items["a"].votes == 2 and items["a"].data == {"name": "a", "price": 1}
    assert not items["a"].conflicts
    assert items["b"].votes == 1


def test_items_keep_the_order_their_keys_first_appeared() -> None:
    frames = [_rows({"name": "b"}, {"name": "a"}), _rows({"name": "c"}, {"name": "a"})]
    keys = [item.key for item in _aggregate(_profile(), frames).items()]
    assert keys == ["b", "a", "c"]


def test_evidence_points_at_the_frames() -> None:
    (item,) = _items(_profile(), [_rows({"name": "a"}), _rows({"name": "a"})]).values()
    assert [(e.source_sha256, e.frame_index, e.timestamp_ms) for e in item.evidence] == [
        ("video", 0, 0),
        ("video", 1, 1000),
    ]


class Color(Enum):
    RED = "red"


def test_keys_with_the_same_json_are_one_item() -> None:
    profile = _profile(key=lambda r: Color.RED if r.name == "enum" else "red")
    items = _aggregate(profile, [_rows({"name": "enum"}), _rows({"name": "red"})]).items()

    assert [item.key for item in items] == ["red"]
    assert items[0].votes == 2


def test_failed_keys_are_counted_and_a_none_key_is_kept() -> None:
    def key(row: Row) -> Any:
        if row.name == "broken":
            raise KeyError(row.name)
        return None if row.name == "none" else row.name

    aggregator = _aggregate(_profile(key=key), [_rows({"name": "broken"}, {"name": "none"})])

    assert aggregator.unkeyed == 1
    assert [item.key for item in aggregator.items()] == [None]


class Tag(RootModel[str]):
    pass


class Tags(BaseModel):
    tags: list[Tag]


def test_items_that_are_not_objects_are_voted_on_whole() -> None:
    profile = _profile(Tags, key=lambda t: t.root.lower())
    (item,) = _items(profile, [{"tags": ["Red"]}, {"tags": ["red"]}, {"tags": ["Red"]}]).values()

    assert item.data == "Red"
    assert item.conflicts[0].field is None


class Cell(RootModel[dict[str, int] | str]):
    pass


class Cells(BaseModel):
    cells: list[Cell]


def test_mixed_object_and_plain_items_are_voted_on_whole() -> None:
    profile = _profile(Cells, key=lambda c: "k")
    frames = [{"cells": [{"price": 1}]}, {"cells": ["unreadable"]}, {"cells": [{"price": 1}]}]

    for ordered in (frames, frames[::-1]):
        (item,) = _items(profile, ordered).values()
        assert item.data == {"price": 1}
        assert item.conflicts[0].field is None
        assert [(c.value, c.votes) for c in item.conflicts[0].values] == [
            ({"price": 1}, 2),
            ("unreadable", 1),
        ]

    (tied,) = _items(profile, frames[:2]).values()
    assert tied.data is None


def test_failed_frames_are_ignored() -> None:
    keyer = item_keyer(_profile())
    assert keyer is not None
    aggregator = Aggregator(keyer)
    aggregator.add(_observation(0, status="error"), None)
    assert aggregator.items() == []


def test_item_records_serialize_with_their_schema() -> None:
    (item,) = _items(_profile(), [_rows({"name": "a"})]).values()
    assert item.to_json().startswith('{"schema":"framelift/item/v1","key":"a"')


class Point:
    """A hashable key pydantic cannot serialize."""

    def __init__(self, x: int) -> None:
        self.x = x

    def __hash__(self) -> int:
        return hash(self.x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and other.x == self.x

    def __str__(self) -> str:
        return f"point-{self.x}"


def test_keys_pydantic_cannot_serialize_fall_back_to_str() -> None:
    profile = _profile(key=lambda r: Point(len(r.name)))
    items = _aggregate(profile, [_rows({"name": "ab"}), _rows({"name": "cd"})]).items()

    assert [(item.key, item.votes) for item in items] == [("point-2", 2)]


def test_the_first_item_key_failure_is_kept() -> None:
    def key(row: Row) -> str:
        raise KeyError(row.name)

    aggregator = _aggregate(_profile(key=key), [_rows({"name": "a"}, {"name": "b"})])

    assert aggregator.unkeyed == 2
    assert aggregator.unkeyed_error == "KeyError('a')"


def test_items_need_one_evidence_entry_per_vote() -> None:
    (item,) = _items(_profile(), [_rows({"name": "a"})]).values()
    with pytest.raises(ValidationError, match="votes"):
        Item.model_validate({**item.model_dump(), "votes": 2})


class Label:
    """Hashable, not serializable, and printed alike for different values."""

    def __init__(self, x: int) -> None:
        self.x = x

    def __hash__(self) -> int:
        return hash(self.x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Label) and other.x == self.x

    def __str__(self) -> str:
        return "label"


def test_keys_that_only_print_alike_are_not_merged() -> None:
    profile = _profile(key=lambda r: Label(len(r.name)))
    items = _aggregate(profile, [_rows({"name": "a"}), _rows({"name": "bb"})]).items()

    assert [(item.key, item.votes) for item in items] == [("label", 1), ("label", 1)]


class AliasedRow(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    item_name: str = Field(alias="itemName")


class AliasedTable(BaseModel):
    rows: list[AliasedRow]


def test_items_are_written_as_the_model_serializes_them() -> None:
    profile = _profile(AliasedTable, key=lambda r: r.item_name)
    raw = {"rows": [{"itemName": "a"}]}
    (item,) = _items(profile, [raw]).values()

    # The same form as the item in observations.jsonl.
    assert item.data == AliasedTable.model_validate(raw).model_dump(mode="json")["rows"][0]
    assert item.data == {"itemName": "a"}
