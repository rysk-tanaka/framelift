"""Recognize model-typed fields in a Pydantic output model."""

from __future__ import annotations

from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel


def unwrap(annotation: Any) -> Any:
    """Strip ``Annotated`` and ``| None`` so ``Sub | None`` is treated like ``Sub``."""
    if get_origin(annotation) is Annotated:
        return unwrap(get_args(annotation)[0])
    if get_origin(annotation) in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return unwrap(args[0])
    return annotation


def model_of(annotation: Any) -> type[BaseModel] | None:
    """The model class of a ``SomeModel`` (or ``SomeModel | None``) annotation."""
    annotation = unwrap(annotation)
    is_model = isinstance(annotation, type) and issubclass(annotation, BaseModel)
    return annotation if is_model else None


def item_model_of(annotation: Any) -> type[BaseModel] | None:
    """The item model of a ``list[SomeModel]`` (or ``list[SomeModel] | None``) annotation."""
    annotation = unwrap(annotation)
    args = get_args(annotation)
    return model_of(args[0]) if get_origin(annotation) is list and len(args) == 1 else None
