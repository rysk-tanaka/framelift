from __future__ import annotations

import pytest
from pydantic import BaseModel

from framelift import Profile


class Out(BaseModel):
    value: int


def test_validators_are_frozen_into_a_tuple() -> None:
    profile = Profile(name="p", output_type=Out, instructions="read", validators=[lambda o: o])
    assert isinstance(profile.validators, tuple)


@pytest.mark.parametrize(("name", "version", "field"), [("  ", "1", "name"), ("p", " ", "version")])
def test_empty_identifiers_are_rejected(name: str, version: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        Profile(name=name, version=version, output_type=Out, instructions="read")


def test_output_type_must_be_a_model() -> None:
    with pytest.raises(TypeError):
        Profile(name="p", output_type=dict, instructions="read")  # ty: ignore[invalid-argument-type]


def test_fingerprint_changes_with_version_and_instructions() -> None:
    base = Profile(name="p", output_type=Out, instructions="read")
    assert base.fingerprint == Profile(name="p", output_type=Out, instructions="read").fingerprint
    assert (
        base.fingerprint
        != Profile(name="p", output_type=Out, instructions="read", version="2").fingerprint
    )
    assert base.fingerprint != Profile(name="p", output_type=Out, instructions="look").fingerprint
