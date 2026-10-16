"""Evaluate a `Profile` and a model against a `framelift/dataset/v1` dataset with pydantic-evals.

Needs the ``eval`` extra (``pip install "framelift[eval]"``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError
from pydantic_core import to_json
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from framelift._internal.dataset import load_dataset
from framelift._internal.extract import Extractor, model_label
from framelift._internal.items import ItemKeyer, dump_item, item_keyer
from framelift._internal.sources import load_image

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai.models import Model
    from pydantic_evals.reporting import EvaluationReport

    from framelift._internal.items import ItemKey
    from framelift.observation import Observation
    from framelift.profile import Profile

type Expected = dict[str, Any]
# pydantic-evals gives expected_output the task's output type, but here the task returns a
# `CaseOutput` and the expected output is an `ExpectedOutput`, so the types stay open.
type Output = Any
type Metadata = dict[str, Any]
"""A case's metadata (orientation, device...), pydantic-evals' third type parameter."""
type Context = EvaluatorContext[CaseInputs, Output, Metadata]

_MISSING = object()


@dataclass(frozen=True)
class CaseInputs:
    image: Path
    sha256: str


@dataclass(frozen=True)
class CaseOutput:
    observation: Observation
    # The validated output, so item keys come from the same model the extraction produced
    # instead of validating the stored JSON again, which may not be idempotent.
    output: BaseModel | None


@dataclass(frozen=True)
class ExpectedOutput:
    values: Expected
    """The listed fields in the JSON form observations use."""
    item_keys: list[ItemKey] | None
    """`Profile.item_key` of each expected item, or None when there are none to score."""


def _restrict(dumped: Any, shape: Any, path: str = "") -> Any:
    """Keep only the parts of ``dumped`` that ``shape`` (the expected output) mentions.

    A key that the dump does not have is an error: a misspelled or unknown field (ignored by
    validation), or one excluded from the output. Either way it could never match an extraction.
    Checking the dump instead of the model's fields covers unions, dicts and ``extra="allow"``.
    """
    if isinstance(shape, dict) and isinstance(dumped, dict):
        restricted: dict[str, Any] = {}
        for key, value in shape.items():
            if key not in dumped:
                raise ValueError(f"{path}{key}: not in the output (unknown or excluded field)")
            restricted[key] = _restrict(dumped[key], value, f"{path}{key}.")
        return restricted
    if isinstance(shape, list) and isinstance(dumped, list) and len(shape) == len(dumped):
        return [
            _restrict(d, s, f"{path.removesuffix('.')}[{i}].")
            for i, (d, s) in enumerate(zip(dumped, shape, strict=True))
        ]
    return dumped


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else error["msg"]


def _validate_expected(
    output_type: type[BaseModel], expected: Expected
) -> tuple[Expected, BaseModel]:
    """Validate an expected output as a whole; return it in JSON form and the validated model.

    Every required field must be listed, so the output model's own validators, serializers and
    config apply exactly as they do to an extraction. Fields with defaults may be left out and
    are then not compared, and neither are fields excluded from the output.
    """
    # Validated from JSON by field name, like stored observations: expected values are JSON
    # data, and strict models accept ISO dates and enum values only from JSON.
    try:
        output = output_type.model_validate_json(to_json(expected), by_name=True)
    except ValidationError as exc:
        raise ValueError(_first_error(exc)) from exc
    # Excluded fields never reach observations, so they are validated (a required one must be
    # listed) but not compared.
    compared = {
        key: value
        for key, value in expected.items()
        if key not in output_type.model_fields or not output_type.model_fields[key].exclude
    }
    # Dumped by field name, like the comparison in FieldScores, even for models that serialize
    # by alias.
    return _restrict(output.model_dump(mode="json", by_alias=False), compared), output


def normalize_expected(output_type: type[BaseModel], expected: Expected) -> Expected:
    """Check an expected output against the profile's model and convert it to JSON form.

    Raises `ValueError` for missing required fields, unknown field names and values the model
    would reject, e.g. an unquoted YAML ``no`` that loads as ``False`` for a text field.
    """
    return _validate_expected(output_type, expected)[0]


@dataclass(frozen=True)
class _ItemMatcher:
    """Keys list items with `Profile.item_key` so expected and extracted items can be paired."""

    keyer: ItemKeyer

    @property
    def field_name(self) -> str:
        return self.keyer.field_name

    def expected(self, output_type: type[BaseModel], raw: Expected) -> ExpectedOutput:
        values, output = _validate_expected(output_type, raw)
        if not isinstance(values.get(self.field_name), list):
            return ExpectedOutput(values, None)
        models = getattr(output, self.field_name) or []
        try:
            keys = [self.keyer.key_of(model) for model in models]
        # item_key is user code and may raise anything; report it as a dataset error.
        except Exception as exc:
            raise ValueError(f"item_key failed on an expected item: {exc!r}") from exc
        # Each item's value comes from its own model, like its key, so a list serializer that
        # reorders items cannot pair one item's key with another's values.
        values[self.field_name] = [
            _restrict(dump_item(model), item, f"{self.field_name}[{index}].")
            for index, (model, item) in enumerate(zip(models, raw[self.field_name], strict=True))
        ]
        return ExpectedOutput(values, keys)

    def output_items(self, output: BaseModel) -> list[tuple[ItemKey, Any]]:
        # An extracted item item_key cannot handle gets a fresh object as its key: it matches no
        # expected item and still counts as output.
        return [
            (item.key if item.has_key else object(), item.dump) for item in self.keyer.items(output)
        ]


def _matcher(profile: Profile) -> _ItemMatcher | None:
    keyer = item_keyer(profile)
    return _ItemMatcher(keyer) if keyer is not None else None


def expected_output(profile: Profile, raw: Expected) -> ExpectedOutput:
    """Validate a dataset case's expected output for ``profile`` (see `_validate_expected`)."""
    matcher = _matcher(profile)
    if matcher is None:
        return ExpectedOutput(normalize_expected(profile.output_type, raw), None)
    return matcher.expected(profile.output_type, raw)


@dataclass(frozen=True)
class ItemPairs:
    """Expected item keys and extracted items, for the list field `Profile.item_key` keys."""

    field_name: str
    expected_keys: list[ItemKey]
    output: list[tuple[ItemKey, Any]]


def _compare(expected: Any, actual: Any, path: str, mismatches: list[str]) -> int:
    """Compare the individual values in ``expected`` with ``actual``; return how many there were."""
    if isinstance(expected, dict):
        actual_dict = actual if isinstance(actual, dict) else {}
        return sum(
            _compare(value, actual_dict.get(key, _MISSING), f"{path}.{key}", mismatches)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        # A list without item_key counts as one value: it matches when every item matches by
        # position, comparing only the fields the expected items list.
        inner: list[str] = []
        is_same_length = isinstance(actual, list) and len(actual) == len(expected)
        if is_same_length:
            for index, (e, a) in enumerate(zip(expected, actual, strict=True)):
                _compare(e, a, f"{path}[{index}]", inner)
        if not is_same_length or inner:
            mismatches.append(path)
        return 1
    # False == 0 and True == 1 in Python, but they are different JSON values.
    is_same_kind = isinstance(expected, bool) == isinstance(actual, bool)
    if not is_same_kind or expected != actual:
        mismatches.append(path)
    return 1


def field_mismatches(
    expected: Expected, data: dict[str, Any], items: ItemPairs | None
) -> tuple[int, int, list[str]]:
    """Count the expected values and the misread ones, and list the paths to report.

    With item pairs, items of the list field are paired by key, so a wrong price shows up as one
    wrong value instead of a missing item. Without them, each list counts as a single value.
    """
    total = misread = 0
    reported: list[str] = []
    for key, value in expected.items():
        is_item_field = items is not None and key == items.field_name
        # An expected empty list has no items to pair, so it is one value like any list without
        # item_key: extra extracted items make it a misread.
        if items is None or not is_item_field or not value:
            actual = data.get(key, _MISSING)
            if is_item_field and actual is None:
                actual = []  # an optional list left empty
            paths: list[str] = []
            total += _compare(value, actual, key, paths)
            misread += len(paths)
            reported += paths
            continue
        # Items sharing a key are paired in order, matching how item_scores counts them.
        actual_by_key: defaultdict[ItemKey, list[Any]] = defaultdict(list)
        for item_key, item in items.output:
            actual_by_key[item_key].append(item)
        for item_key, item in zip(items.expected_keys, value, strict=True):
            path = f"{key}[{item_key!r}]"
            candidates = actual_by_key[item_key]
            actual = candidates.pop(0) if candidates else _MISSING
            paths = []
            total += _compare(item, actual, path, paths)
            misread += len(paths)
            # Every value of a missing item counts as misread, but one path is enough to report.
            reported += [f"{path} (missing)"] if actual is _MISSING else paths
    return total, misread, reported


def item_scores(items: ItemPairs) -> tuple[float, float]:
    """Precision and recall of list items, matched by `Profile.item_key`."""
    expected_keys = Counter(items.expected_keys)
    output_keys = Counter(key for key, _ in items.output)
    matched = sum((expected_keys & output_keys).values())
    n_expected, n_output = expected_keys.total(), output_keys.total()
    precision = matched / n_output if n_output else float(n_expected == 0)
    recall = matched / n_expected if n_expected else 1.0
    return precision, recall


@dataclass
class ValidationPassed(Evaluator[CaseInputs, Output, Metadata]):
    """Whether the output passed validation within the retry limit."""

    def evaluate(self, ctx: Context) -> EvaluationReason:
        observation = ctx.output.observation
        return EvaluationReason(value=observation.status == "ok", reason=observation.error)


@dataclass
class FieldScores(Evaluator[CaseInputs, Output, Metadata]):
    """``FieldAccuracy`` (share of expected values read correctly) and, when the profile
    has an `item_key` and the expected output lists items, ``item_precision`` and
    ``item_recall``."""

    matcher: _ItemMatcher | None = field(repr=False)

    def build_serialization_arguments(self) -> dict[str, Any]:
        return {}

    def evaluate(self, ctx: Context) -> dict[str, EvaluationReason | float]:
        case_output: CaseOutput = ctx.output
        expected: ExpectedOutput | None = ctx.expected_output
        if expected is None:
            return {}
        output = case_output.output
        if output is None:
            scores: dict[str, EvaluationReason | float] = {
                "FieldAccuracy": EvaluationReason(value=0.0, reason="no output")
            }
            # A failed case scores 0 everywhere, even when no items are expected, so failures
            # never raise the averages.
            if expected.item_keys is not None:
                scores |= {"item_precision": 0.0, "item_recall": 0.0}
            return scores

        # By field name like the dataset, even when the model serializes by alias.
        data = output.model_dump(mode="json", by_alias=False)

        items = None
        if self.matcher is not None and expected.item_keys is not None:
            items = ItemPairs(
                self.matcher.field_name,
                expected.item_keys,
                self.matcher.output_items(output),
            )
        total, misread, reported = field_mismatches(expected.values, data, items)
        accuracy = (total - misread) / total if total else 1.0
        reason = f"misread: {', '.join(reported)}" if reported else None
        scores = {"FieldAccuracy": EvaluationReason(value=accuracy, reason=reason)}
        if items is not None:
            precision, recall = item_scores(items)
            scores |= {"item_precision": precision, "item_recall": recall}
        return scores


@dataclass
class UsageMetrics(Evaluator[CaseInputs, Output, Metadata]):
    """Tokens, latency and retries, to compare cost and speed across runs."""

    def evaluate(self, ctx: Context) -> dict[str, int]:
        observation = ctx.output.observation
        usage = observation.usage
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": usage.latency_ms,
            "retries": observation.retries,
        }


def build_dataset(
    dataset_path: Path, profile: Profile, root: Path | None = None
) -> Dataset[CaseInputs, Output, Metadata]:
    """Load a dataset file as a pydantic-evals `Dataset`.

    Every image's SHA-256 is checked, and every expected output is checked against the
    profile's output model (and its items against `Profile.item_key`) before anything runs.
    Raises `ValueError` listing every case that does not fit.
    """
    dataset_file, resolved_root = load_dataset(dataset_path, root)
    cases: list[Case[CaseInputs, Output, Metadata]] = []
    problems: list[str] = []
    for case in dataset_file.cases:
        try:
            expected = expected_output(profile, case.expected)
        except ValueError as exc:
            problems.append(f"{case.name}: {exc}")
            continue
        cases.append(
            Case(
                name=case.name,
                inputs=CaseInputs(image=resolved_root / case.image, sha256=case.sha256),
                expected_output=expected,
                metadata=case.metadata,
            )
        )
    if problems:
        details = "\n".join(problems)
        raise ValueError(f"Expected outputs in {dataset_path} do not fit the profile:\n{details}")
    return Dataset(
        name=dataset_path.stem,
        cases=cases,
        evaluators=[ValidationPassed(), FieldScores(_matcher(profile)), UsageMetrics()],
    )


def run_evaluation(
    dataset: Dataset[CaseInputs, Output, Metadata],
    profile: Profile,
    model: Model | str,
    *,
    max_retries: int = 2,
    progress: bool = True,
) -> EvaluationReport[CaseInputs, Output, Metadata]:
    """Extract every case of a dataset from `build_dataset` and score the results."""
    extractor = Extractor(profile, model, max_retries=max_retries)

    # Async, so the agent runs on pydantic-evals' event loop: a sync task would run in worker
    # threads, each with its own loop, which a provider's cached HTTP client cannot share.
    async def extract(inputs: CaseInputs) -> CaseOutput:
        image = load_image(inputs.image)
        # The dataset was checked before the run; check again with the bytes actually sent, so
        # an image replaced in between fails its case instead of being scored as another one.
        if image.sha256 != inputs.sha256:
            raise ValueError(f"{inputs.image} changed after the dataset was checked")
        observation, output = await extractor.observe_async(image)
        return CaseOutput(observation, output)

    # Cases run one at a time, to keep cost and provider rate limits predictable.
    return dataset.evaluate_sync(
        extract,
        name=f"{profile.name}@{profile.version} {model_label(model)}",
        max_concurrency=1,
        progress=progress,
    )
