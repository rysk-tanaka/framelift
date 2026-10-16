"""Key the items of an output's list field with `Profile.item_key`.

Shared by evaluation (pairing extracted items with expected ones) and aggregation (merging the
same item seen in several frames), so both key and dump items the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from framelift.profile import item_field

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from pydantic import BaseModel

    from framelift.profile import Profile

type ItemKey = Hashable


def dump_item(model: BaseModel, *, by_alias: bool | None = False) -> Any:
    """An item (or output) in JSON form: by field name by default, as datasets are written, or
    with ``by_alias=None`` as the model serializes itself, as observations are written.

    Not always a dict: an item can be e.g. a ``RootModel[str]``.
    """
    return model.model_dump(mode="json", by_alias=by_alias)


@dataclass(frozen=True)
class KeyedItem:
    """One list item with its `Profile.item_key` and its JSON form."""

    key: Any
    """The item's `Profile.item_key`; meaningless when ``has_key`` is false."""
    dump: Any
    has_key: bool
    """False when ``item_key`` raised or returned an unhashable value. A returned ``None`` is a
    valid key."""
    error: str | None = None
    """Why ``item_key`` failed, when it did."""


@dataclass(frozen=True)
class ItemKeyer:
    """Applies `Profile.item_key` to the items of the output's list field."""

    field_name: str
    key: Callable[[Any], ItemKey]

    def key_of(self, model: Any) -> ItemKey:
        """The item's key. Raises whatever ``item_key`` raises, or `TypeError` if unhashable."""
        key = self.key(model)
        hash(key)  # raises TypeError now rather than deep inside the scoring or merging
        return key

    def items(self, output: BaseModel, *, by_alias: bool | None = False) -> list[KeyedItem]:
        """Key and dump every item of an extracted output (``by_alias`` as in `dump_item`).

        Each dump comes from the item's own model, like its key, so a list serializer that
        reorders items cannot mix them up.
        """
        # An optional list may be None, which means no items.
        models = getattr(output, self.field_name) or []
        keyed: list[KeyedItem] = []
        for model in models:
            dump = dump_item(model, by_alias=by_alias)
            try:
                key = self.key_of(model)
            # item_key is user code; an item it cannot handle (e.g. a None it does not expect)
            # is a misread, not a crash.
            except Exception as exc:
                keyed.append(KeyedItem(None, dump, has_key=False, error=repr(exc)))
                continue
            keyed.append(KeyedItem(key, dump, has_key=True))
        return keyed


def item_keyer(profile: Profile) -> ItemKeyer | None:
    """The keyer for ``profile``, or None when it has no `Profile.item_key`."""
    if profile.item_key is None:
        return None
    return ItemKeyer(item_field(profile.output_type)[0], profile.item_key)
