"""The `framelift/dataset/v1` evaluation dataset: images with their expected output.

Image paths are relative to the dataset file's directory, or to ``--root`` when given, so
private screenshots can live outside the repository. Every image is checked against its
recorded SHA-256 before a run, so a wrong root fails early instead of scoring other images.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import to_jsonable_python

from framelift._internal.sources import check_image

DATASET_SCHEMA = "framelift/dataset/v1"


class DatasetCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    image: str = Field(description="Image path, relative to the dataset root")
    sha256: str
    expected: dict[str, Any] = Field(
        description="Expected output; every required field, fields with defaults may be left out"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image")
    @classmethod
    def _inside_root(cls, value: str) -> str:
        # A dataset should not make framelift read (and send to a model) files outside its root.
        normalized = Path(os.path.normpath(value))
        is_inside_root = not normalized.is_absolute() and normalized.parts[:1] != ("..",)
        if not is_inside_root:
            raise ValueError("must be a relative path inside the dataset root")
        return value

    @field_validator("sha256")
    @classmethod
    def _hex_digest(cls, value: str) -> str:
        # Some tools print uppercase hex; compare in hashlib's lowercase form.
        return value.strip().lower()

    @field_validator("expected")
    @classmethod
    def _as_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Extracted data is compared in JSON form, but YAML loads e.g. 2026-10-16 as a date.
        return to_jsonable_python(value)


class DatasetFile(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: Literal["framelift/dataset/v1"] = Field(default=DATASET_SCHEMA, alias="schema")
    cases: list[DatasetCase]


def load_dataset(path: Path, root: Path | None = None) -> tuple[DatasetFile, Path]:
    """Read a dataset and check that every image exists, decodes and matches its SHA-256.

    Returns the dataset and the resolved root. Raises `ValueError` listing every problem at once.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    try:
        dataset = DatasetFile.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path} is not a {DATASET_SCHEMA} dataset:\n{exc}") from exc

    counts = Counter(case.name for case in dataset.cases)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate case names in {path}: {', '.join(duplicates)}")

    # No environment variable: a root exported for one dataset would silently apply to others.
    resolved_root = root if root is not None else path.parent
    problems: list[str] = []
    for case in dataset.cases:
        image = resolved_root / case.image
        if not image.is_file():
            problems.append(f"{case.name}: image not found: {image}")
            continue
        # check_image also decodes the image, so a corrupt or unsupported file fails here,
        # before any case is sent to a paid model.
        try:
            sha256 = check_image(image)
        except ValueError as exc:
            problems.append(f"{case.name}: {exc}")
            continue
        if sha256 != case.sha256:
            problems.append(f"{case.name}: sha256 mismatch: {image}")
    if problems:
        details = "\n".join(problems)
        raise ValueError(f"Dataset {path} does not match root {resolved_root}:\n{details}")
    return dataset, resolved_root


def write_dataset(path: Path, cases: list[DatasetCase]) -> None:
    """Write ``cases`` as a dataset YAML file."""
    dataset = DatasetFile(cases=cases)
    text = yaml.safe_dump(
        dataset.model_dump(mode="json", by_alias=True), sort_keys=False, allow_unicode=True
    )
    path.write_text(text, encoding="utf-8")
