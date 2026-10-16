from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from framelift import Profile
from framelift.profile import item_field


class Out(BaseModel):
    value: int


def test_validators_are_frozen_into_a_tuple() -> None:
    profile = Profile(
        locate="full", name="p", output_type=Out, instructions="read", validators=[lambda o: o]
    )
    assert isinstance(profile.validators, tuple)


@pytest.mark.parametrize(("name", "version", "field"), [("  ", "1", "name"), ("p", " ", "version")])
def test_empty_identifiers_are_rejected(name: str, version: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        Profile(locate="full", name=name, version=version, output_type=Out, instructions="read")


def test_output_type_must_be_a_model() -> None:
    with pytest.raises(TypeError):
        Profile(locate="full", name="p", output_type=dict, instructions="read")  # ty: ignore[invalid-argument-type]


def test_fingerprint_changes_with_version_and_instructions() -> None:
    base = Profile(locate="full", name="p", output_type=Out, instructions="read")
    assert (
        base.fingerprint
        == Profile(locate="full", name="p", output_type=Out, instructions="read").fingerprint
    )
    assert (
        base.fingerprint
        != Profile(
            locate="full", name="p", output_type=Out, instructions="read", version="2"
        ).fingerprint
    )
    assert (
        base.fingerprint
        != Profile(locate="full", name="p", output_type=Out, instructions="look").fingerprint
    )


class Row(BaseModel):
    name: str


class Table(BaseModel):
    title: str
    rows: list[Row]


class TwoTables(BaseModel):
    left: list[Row]
    right: list[Row]


def test_item_key_finds_the_list_field() -> None:
    Profile(
        locate="full",
        name="p",
        output_type=Table,
        instructions="read",
        item_key=lambda row: row.name,
    )
    assert item_field(Table) == ("rows", Row)


@pytest.mark.parametrize("output_type", [Out, TwoTables])
def test_item_key_needs_exactly_one_list_of_models(output_type: type[BaseModel]) -> None:
    with pytest.raises(ValueError, match="item_key"):
        Profile(
            locate="full",
            name="p",
            output_type=output_type,
            instructions="read",
            item_key=lambda r: r,
        )


def test_item_key_is_optional_for_any_output() -> None:
    Profile(locate="full", name="p", output_type=TwoTables, instructions="read")


def test_locate_is_required() -> None:
    with pytest.raises(TypeError, match="locate"):
        Profile(name="p", output_type=Out, instructions="read")  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    "locate",
    [
        "whole",
        {},
        {"portrait": (0.0, 0.0, 1.5, 1.0)},
        {"portrait": (0.5, 0.0, 0.4, 1.0)},
        {"portrait": (0.0, 0.0, float("nan"), 1.0)},
        {"portrait": (0.0, 0.0, 1.0)},
        {"portrait": (True, 0.0, 1.0, 1.0)},
        {"portrait": "0, 0, 1, 1"},
        {"portrat": (0.0, 0.0, 1.0, 1.0)},
    ],
)
def test_invalid_crop_boxes_are_rejected(locate: Any) -> None:
    with pytest.raises(ValueError, match=r"Profile\.locate"):
        Profile(locate=locate, name="p", output_type=Out, instructions="read")


@pytest.mark.parametrize(
    "redact",
    [
        {"landscpae": [(0.0, 0.0, 1.0, 1.0)]},
        [(0.0, 0.0, 2.0, 1.0)],
        (0.0, 0.0, 1.0, 1.0),  # one box, not a list of boxes
        "top",
    ],
)
def test_invalid_redactions_are_rejected(redact: Any) -> None:
    with pytest.raises(ValueError, match=r"Profile\.redact"):
        Profile(locate="full", redact=redact, name="p", output_type=Out, instructions="read")


def test_redaction_is_stored_per_orientation() -> None:
    every = Profile(
        locate="full", redact=[(0, 0, 1, 0.5)], name="p", output_type=Out, instructions="r"
    )
    landscape = Profile(
        locate="full",
        redact={"landscape": [(0, 0, 1, 0.5)]},
        name="p",
        output_type=Out,
        instructions="r",
    )

    box = (0.0, 0.0, 1.0, 0.5)
    assert every.redact == {"portrait": (box,), "landscape": (box,)}
    assert landscape.redact == {"portrait": (), "landscape": (box,)}


def test_fingerprint_changes_with_locate_and_redact() -> None:
    def fingerprint(**kwargs: Any) -> str:
        return Profile(name="p", output_type=Out, instructions="read", **kwargs).fingerprint

    full = fingerprint(locate="full")
    assert full == fingerprint(locate="full", redact=[])
    assert full != fingerprint(locate={"portrait": (0, 0, 1, 0.5)})
    assert full != fingerprint(locate="full", redact=[(0, 0, 1, 0.5)])


@pytest.mark.parametrize("description", ["", "  "])
def test_an_empty_screen_description_is_rejected(description: str) -> None:
    with pytest.raises(ValueError, match="screen_description"):
        Profile(
            name="p",
            output_type=Out,
            instructions="read",
            locate="full",
            screen_description=description,
        )


def test_the_screen_description_does_not_change_the_fingerprint() -> None:
    # It decides which images are read, not what is read from them.
    base = Profile(name="p", output_type=Out, instructions="read", locate="full")
    checked = Profile(
        name="p", output_type=Out, instructions="read", locate="full", screen_description="a shop"
    )

    assert checked.fingerprint == base.fingerprint
