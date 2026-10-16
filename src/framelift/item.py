"""The `framelift/item/v1` record written to ``items.jsonl``."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ITEM_SCHEMA = "framelift/item/v1"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Candidate(_Record):
    """One value the frames gave for a field, and how many frames gave it."""

    value: Any
    votes: int


class Conflict(_Record):
    """A field whose frames disagreed, with every value seen."""

    field: str | None = Field(description="Field name; null when the whole item was voted on")
    values: list[Candidate] = Field(description="Most votes first")


class Evidence(_Record):
    """One frame the item was read from."""

    source_sha256: str
    frame_index: int
    timestamp_ms: int | None


class Item(_Record):
    """One list item, merged from every frame that showed it (matched by `Profile.item_key`)."""

    schema_id: Literal["framelift/item/v1"] = Field(default=ITEM_SCHEMA, alias="schema")
    key: Any = Field(description="The item's Profile.item_key, as JSON")
    data: Any = Field(
        description="The item by majority vote; a field (or the whole item) is null on a tie"
    )
    votes: int = Field(description="How many frames showed the item")
    conflicts: list[Conflict] = Field(description="Fields whose frames disagreed")
    evidence: list[Evidence] = Field(description="One entry per frame that voted")

    @model_validator(mode="after")
    def _one_evidence_per_vote(self) -> Self:
        if self.votes != len(self.evidence):
            raise ValueError("votes must equal the number of evidence entries")
        return self

    def to_json(self) -> str:
        """The record as one line of ``items.jsonl``."""
        return self.model_dump_json(by_alias=True)
