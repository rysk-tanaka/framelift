"""`Pipeline`: run a `Profile` over images and videos, writing ``observations.jsonl`` and
``items.jsonl``."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from framelift._internal.aggregate import Aggregator
from framelift._internal.cache import Cache
from framelift._internal.extract import Extractor, ModelMismatchError, note_input
from framelift._internal.items import item_keyer
from framelift._internal.layout import Layout, NoCropError
from framelift._internal.screen import ScreenClassifier
from framelift._internal.sources import (
    UnreadableInputError,
    error_reason,
    expand_inputs,
    hash_file,
    is_video,
    load_image,
)
from framelift._internal.video import (
    DEFAULT_FRAME_RATE,
    DEFAULT_MOTION_THRESHOLD,
    check_sampling,
    frame_to_source,
    import_av,
    sample_video,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from typing import TextIO

    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings

    from framelift._internal.sources import ImageSource
    from framelift.item import Item
    from framelift.observation import Observation
    from framelift.profile import Profile

OBSERVATIONS_FILE = "observations.jsonl"
ITEMS_FILE = "items.jsonl"


@dataclass(frozen=True)
class _Unreadable:
    """An input (or the rest of a video) that could not be read or decoded."""

    error: UnreadableInputError

    @property
    def path(self) -> Path:
        return self.error.path

    @property
    def sha256(self) -> str | None:
        """None when the file itself could not be read."""
        return self.error.sha256


@dataclass(frozen=True)
class _Uncropped:
    """A screen whose orientation has no box in `Profile.locate`, so it was not sent."""

    path: Path
    sha256: str | None
    error: NoCropError


@dataclass
class RunResult:
    """What a `Pipeline.run` produced."""

    observations: list[Observation] = field(default_factory=list)
    output_path: Path | None = None
    """``observations.jsonl``, when the pipeline has an ``out_dir``."""
    items: list[Item] = field(default_factory=list)
    """List items merged across frames; empty when the profile has no ``item_key``."""
    items_path: Path | None = None
    """``items.jsonl``, when the pipeline has an ``out_dir`` and the profile an ``item_key``."""
    unkeyed_items: int = 0
    """Extracted items left out of ``items`` because ``item_key`` failed on them."""
    unkeyed_error: str | None = None
    """The first such ``item_key`` failure, e.g. ``"KeyError('name')"``."""

    @property
    def succeeded(self) -> list[Observation]:
        return [o for o in self.observations if o.status == "ok"]

    @property
    def cached(self) -> list[Observation]:
        """Successful observations whose extraction was read from the cache."""
        return [o for o in self.observations if o.status == "ok" and o.cached]

    @property
    def skipped(self) -> list[Observation]:
        """Images the screen check found to show another screen."""
        return [o for o in self.observations if o.status == "skipped"]

    @property
    def failed(self) -> list[Observation]:
        return [o for o in self.observations if o.status == "error"]


class Pipeline[OutputT: BaseModel]:
    """Extract ``profile.output_type`` from every input image and from the still screens of
    every input video.

    Args:
        profile: What to read.
        model: A Pydantic AI model or a ``"provider:model"`` string.
        screen_model: The model for the screen check, when the profile has a
            ``screen_description``; ``None`` uses ``model``.
        model_settings: Pydantic AI model settings for every request, the screen check's too,
            e.g. ``{"openai_reasoning_effort": "none"}`` to turn off a local model's thinking.
            Settings for another provider are ignored, so one model can be given its own.
        out_dir: Where to write ``observations.jsonl`` (and ``items.jsonl`` when the profile has
            an ``item_key``). ``None`` keeps results in memory only.
        max_retries: How many times a failed validation may ask the model to try again, for the
            extraction and for the screen check, each.
        anonymize_paths: Omit input paths from observations, keeping only hashes.
        frame_rate: How many video frames per second to examine.
        motion_threshold: How much of the part of the screen that is sent (the share of 16x16
            tiles of the ``locate`` crop that ``redact`` leaves mostly visible, 0-1) may change
            between a video sample and the previous sample, and between it and the first sample
            of its still screen, for it to still belong to that screen. Only the sharpest sample
            of each still screen is sent to the model.
        cache_dir: Where to cache extractions and screen checks, so that images read before
            are not paid for again. ``None`` caches nothing. Only models given as strings are
            cached; see `Profile.version` for when to bump it so that old entries are not used.

    Raises:
        UserError: From Pydantic AI, when a model string cannot be resolved, e.g. as the
            provider's API key is not set. `run` raises one when a local server answers with
            another model than the one asked for.
    """

    def __init__(
        self,
        profile: Profile[OutputT],
        *,
        model: Model | str,
        screen_model: Model | str | None = None,
        model_settings: ModelSettings | None = None,
        out_dir: str | Path | None = None,
        max_retries: int = 2,
        anonymize_paths: bool = False,
        frame_rate: float = DEFAULT_FRAME_RATE,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
        cache_dir: str | Path | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if screen_model is not None and profile.screen_description is None:
            raise ValueError("screen_model needs a Profile.screen_description to check against")
        check_sampling(frame_rate, motion_threshold)
        self.profile = profile
        self.model = model
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.anonymize_paths = anonymize_paths
        self.frame_rate = frame_rate
        self.motion_threshold = motion_threshold
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        cache = Cache(self.cache_dir) if self.cache_dir is not None else None
        self.screen_model: Model | str | None = None
        """The model the screen check uses; None when the profile has no screen_description."""
        self.model_settings = model_settings
        self._extractor = Extractor(
            profile, model, max_retries=max_retries, cache=cache, model_settings=model_settings
        )
        self._classifier = None
        if profile.screen_description is not None:
            self.screen_model = model if screen_model is None else screen_model
            self._classifier = ScreenClassifier(
                profile.screen_description,
                self.screen_model,
                max_retries=max_retries,
                cache=cache,
                version=profile.version,
                model_settings=model_settings,
            )
        self._layout = Layout.from_profile(profile)

    def run(self, inputs: str | Path | Iterable[str | Path]) -> RunResult:
        """Process the inputs in order.

        A failure on one input does not stop the run: an image or video that cannot be read (or
        a video that breaks off partway) and a frame whose output fails validation are recorded
        as ``status: "error"`` observations, and frames read before a video broke off are kept.
        With a ``screen_description``, an image the screen check finds to show another screen
        is recorded as ``status: "skipped"``, and one the check fails on as an error.
        Any other error, such as the model provider refusing a request, stops the run and is
        raised; ``items.jsonl`` is still written for the frames read until then, like
        ``observations.jsonl``.

        Raises `FileNotFoundError` when an input matches nothing, and `MissingVideoExtraError`
        (an `ImportError`) when a video is given without the ``video`` extra, before any model is
        called.
        """
        items = [inputs] if isinstance(inputs, str | Path) else list(inputs)
        paths = expand_inputs(items)
        if any(is_video(path) for path in paths):
            # Directories and globs pick up videos too, so say how to read only the images.
            import_av(hint="; to read only the images, pass them alone (e.g. 'shots/*.png')")

        keyer = item_keyer(self.profile)
        aggregator = Aggregator(keyer) if keyer is not None else None
        result = RunResult()
        writer = None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # observations.jsonl is rewritten below; an items.jsonl left by an earlier run must
            # not sit next to it when this run writes none, because the profile has no item_key
            # or the output could not be written.
            (self.out_dir / ITEMS_FILE).unlink(missing_ok=True)
            result.output_path = self.out_dir / OBSERVATIONS_FILE
            writer = result.output_path.open("w", encoding="utf-8")
        try:
            # Closed explicitly, so an error that stops the run does not leave a video open for
            # as long as the caller holds the exception.
            with closing(self._images(paths)) as images:
                for image in images:
                    self._record(image, writer, aggregator, result)
        except BaseException as exc:
            # The items read so far are written even when an error stops the run, to match the
            # observations already written; if that fails too, the original error is kept.
            try:
                self._finish(writer, aggregator, result)
            except OSError as write_error:
                exc.add_note(f"framelift could not write its output either: {write_error}")
            raise
        self._finish(writer, aggregator, result)
        return result

    def _record(
        self,
        image: ImageSource | _Unreadable | _Uncropped,
        writer: TextIO | None,
        aggregator: Aggregator | None,
        result: RunResult,
    ) -> None:
        if isinstance(image, _Unreadable):
            # The reason alone when paths are anonymized: read errors quote the path.
            error = image.error.without_path if self.anonymize_paths else str(image.error)
            observation = self._extractor.unreadable(
                image.path, image.sha256, error, anonymize_paths=self.anonymize_paths
            )
            output = None
        elif isinstance(image, _Uncropped):
            # The message names no path, so it is the same with anonymized paths.
            observation = self._extractor.unreadable(
                image.path,
                image.sha256,
                str(image.error),
                anonymize_paths=self.anonymize_paths,
                crop=image.error.crop,
            )
            output = None
        else:
            try:
                observation, output = self._read(image)
            except ModelMismatchError as exc:
                note_input(exc, image, anonymize_paths=self.anonymize_paths)
                raise
        result.observations.append(observation)
        if aggregator is not None:
            aggregator.add(observation, output)
        if writer is not None:
            writer.write(observation.to_json() + "\n")
            writer.flush()

    def _read(self, image: ImageSource) -> tuple[Observation, BaseModel | None]:
        screen = None if self._classifier is None else self._classifier.classify(image)
        if screen is not None and screen.is_target is not True:
            # Another screen, or one the check could not tell: nothing is guessed at.
            observation = self._extractor.screened_out(
                image, screen, anonymize_paths=self.anonymize_paths
            )
            return observation, None
        return self._extractor.observe(image, anonymize_paths=self.anonymize_paths, screen=screen)

    def _finish(
        self, writer: TextIO | None, aggregator: Aggregator | None, result: RunResult
    ) -> None:
        if writer is not None:
            writer.close()
        if aggregator is not None:
            self._write_items(aggregator, result)

    def _write_items(self, aggregator: Aggregator, result: RunResult) -> None:
        result.items = aggregator.items()
        result.unkeyed_items = aggregator.unkeyed
        result.unkeyed_error = aggregator.unkeyed_error
        if self.out_dir is not None:
            result.items_path = self.out_dir / ITEMS_FILE
            with result.items_path.open("w", encoding="utf-8") as fh:
                fh.writelines(item.to_json() + "\n" for item in result.items)

    def _images(
        self, paths: list[Path]
    ) -> Generator[ImageSource | _Unreadable | _Uncropped, None, None]:
        for path in paths:
            # Only reading and decoding is guarded here; errors while extracting happen in the
            # caller, outside this generator, and are not mistaken for an unreadable input.
            # OSError covers a file that cannot be opened, was removed after the inputs were
            # expanded, or fails to read; like a corrupt file, it is one unreadable input.
            kind = "video" if is_video(path) else "image"
            if kind == "image":
                try:
                    image = load_image(path, self._layout)
                except NoCropError as exc:
                    yield _Uncropped(path, exc.sha256, exc)
                    continue
                except UnreadableInputError as exc:
                    yield _Unreadable(exc)
                    continue
                except OSError as exc:
                    yield _Unreadable(UnreadableInputError(kind, path, error_reason(exc)))
                    continue
                yield image
                continue
            try:
                source_sha256 = hash_file(path)
            except OSError as exc:
                yield _Unreadable(UnreadableInputError(kind, path, error_reason(exc)))
                continue
            frames = sample_video(
                path,
                frame_rate=self.frame_rate,
                motion_threshold=self.motion_threshold,
                layout=self._layout,
            )
            try:
                with closing(frames):
                    for frame in frames:
                        # One frame without a crop box does not stop the video: a recording
                        # may turn the screen, and the next frames can still be cropped.
                        try:
                            source = frame_to_source(path, source_sha256, frame, self._layout)
                        except NoCropError as exc:
                            yield _Uncropped(path, source_sha256, exc)
                            continue
                        yield source
            except UnreadableInputError as exc:
                exc.sha256 = source_sha256
                yield _Unreadable(exc)
            # Other ValueErrors come from decoding too, e.g. PyAV for a pixel format it cannot
            # convert.
            except (ValueError, OSError) as exc:
                error = UnreadableInputError(kind, path, error_reason(exc), sha256=source_sha256)
                yield _Unreadable(error)
