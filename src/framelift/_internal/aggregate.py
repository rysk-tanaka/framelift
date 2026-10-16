"""Merge the same list item seen in several frames into one `Item`, by majority vote."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_core import PydanticSerializationError, to_jsonable_python

from framelift.item import Candidate, Conflict, Evidence, Item

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pydantic import BaseModel

    from framelift._internal.items import ItemKey, ItemKeyer
    from framelift.observation import Observation


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _key_json(key: ItemKey) -> tuple[Hashable, Any]:
    """How to group an item by its key, and the key as written to ``items.jsonl``.

    Keys are grouped by their JSON form, so keys that look the same in items.jsonl (an Enum and
    its value, a date and its ISO string) are one item rather than two rows with the same "key".
    A key with no JSON form is written as ``str(key)`` but grouped by the key itself, so two
    different keys that print alike are not merged.
    """
    try:
        value = to_jsonable_python(key)
    except PydanticSerializationError:
        return ("key", key), str(key)
    return ("json", _canonical(value)), value


def _vote(values: list[Any]) -> tuple[Any, list[Candidate]]:
    """The value with the most votes (None on a tie) and every candidate, most votes first.

    Values are compared as canonical JSON, so nested values count as one value, and candidates
    with equal votes are ordered by that JSON so the result does not depend on frame order.
    """
    counts = Counter(_canonical(value) for value in values)
    first_value = {}
    for value in values:
        first_value.setdefault(_canonical(value), value)
    ranked = sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    candidates = [Candidate(value=first_value[text], votes=votes) for text, votes in ranked]
    has_winner = len(candidates) == 1 or candidates[0].votes > candidates[1].votes
    return (candidates[0].value if has_winner else None), candidates


@dataclass
class _Group:
    key: Any
    dumps: list[Any] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    def merge(self) -> Item:
        conflicts: list[Conflict] = []
        if all(isinstance(dump, dict) for dump in self.dumps):
            # Field by field over every field seen; a frame without a field abstains, while an
            # explicit null is a vote like any other value.
            names = list(dict.fromkeys(name for dump in self.dumps for name in dump))
            data: Any = {}
            for name in names:
                value, candidates = _vote([dump[name] for dump in self.dumps if name in dump])
                data[name] = value
                if len(candidates) > 1:
                    conflicts.append(Conflict(field=name, values=candidates))
            conflicts.sort(key=lambda conflict: conflict.field or "")
        else:
            # Items that are not objects (e.g. RootModel[str]), or a mix, are voted on whole.
            data, candidates = _vote(self.dumps)
            if len(candidates) > 1:
                conflicts.append(Conflict(field=None, values=candidates))
        return Item(
            key=self.key,
            data=data,
            votes=len(self.dumps),
            conflicts=conflicts,
            evidence=self.evidence,
        )


class Aggregator:
    """Collects the items of every successful frame of a run, then merges them by key."""

    def __init__(self, keyer: ItemKeyer) -> None:
        self._keyer = keyer
        self._groups: dict[Hashable, _Group] = {}
        self.unkeyed = 0
        """Items skipped because `Profile.item_key` raised or returned an unhashable value."""
        self.unkeyed_error: str | None = None
        """The first of those failures, so the cause can be found."""

    def add(self, observation: Observation, output: BaseModel | None) -> None:
        source_sha256 = observation.source.sha256
        is_read = observation.status == "ok" and output is not None
        # A frame that was read always has both; the checks keep error records out.
        if not is_read or observation.frame is None or source_sha256 is None:
            return
        evidence = Evidence(
            source_sha256=source_sha256,
            frame_index=observation.frame.index,
            timestamp_ms=observation.frame.timestamp_ms,
        )
        seen: set[Hashable] = set()
        # Dumped as the model serializes itself, so items.jsonl matches observations.jsonl even
        # for models that serialize by alias.
        for item in self._keyer.items(output, by_alias=None):
            if not item.has_key:
                self.unkeyed += 1
                if self.unkeyed_error is None:
                    self.unkeyed_error = item.error
                continue
            group_id, key = _key_json(item.key)
            # A frame votes once per item, even if it lists the item twice.
            if group_id in seen:
                continue
            seen.add(group_id)
            group = self._groups.setdefault(group_id, _Group(key))
            group.dumps.append(item.dump)
            group.evidence.append(evidence)

    def items(self) -> list[Item]:
        """Merged items, in the order their keys first appeared."""
        return [group.merge() for group in self._groups.values()]
