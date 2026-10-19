"""Command-line interface: ``framelift run``, ``eval``, ``frames`` and ``synth``."""

from __future__ import annotations

import importlib
import itertools
import json
import os
import re
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import imagehash
import typer
from pydantic_ai.exceptions import UserError

from framelift._internal import synth as _synth
from framelift._internal import video as _video
from framelift._internal.extract import ModelMismatchError
from framelift._internal.layout import Layout, NoCropError
from framelift._internal.sources import (
    UnreadableInputError,
    error_reason,
    is_video,
    load_image,
)
from framelift.pipeline import Pipeline
from framelift.profile import Profile

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from PIL import Image
    from pydantic_ai.settings import ModelSettings

app = typer.Typer(
    help="Extract structured data from screen recordings and screenshots with vision LLMs.",
    no_args_is_help=True,
    add_completion=False,
)


def _user_frame(exc: BaseException) -> str:
    """Where in the user's code ``exc`` was raised, as `` (file:line)``, or an empty string.

    Frames inside framelift and the import machinery are ignored, so a misconfigured
    ``Profile(...)`` points at the line in the user's module, not at framelift's own check.
    """
    # The separator keeps a sibling such as ``framelift_profiles/`` from counting as framelift.
    package_prefix = str(Path(__file__).resolve().parent) + os.sep
    frames = [
        frame
        for frame in traceback.extract_tb(exc.__traceback__)
        if not frame.filename.startswith((package_prefix, "<"))
    ]
    if not frames:
        return ""
    filename = Path(frames[-1].filename)
    # Profiles are imported from the current directory, so a relative path is usually enough
    # and keeps the message short when the terminal wraps it.
    if filename.is_relative_to(Path.cwd()):
        filename = filename.relative_to(Path.cwd())
    return f" ({filename}:{frames[-1].lineno})"


def load_profile(spec: str) -> Profile:
    """Load a `Profile` from ``"package.module:attribute"``.

    The current directory is put on ``sys.path`` so that local profile modules can be used.
    """
    module_name, sep, attribute = spec.partition(":")
    if not sep or not module_name or not attribute:
        raise typer.BadParameter("expected 'package.module:attribute'", param_hint="--profile")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, TypeError, ValueError) as exc:
        # Profile(...) raises TypeError or ValueError when it is misconfigured. Other errors are
        # bugs in the user's module and keep their full traceback.
        where = _user_frame(exc)
        raise typer.BadParameter(f"cannot import {module_name!r}: {exc}{where}") from exc
    profile = getattr(module, attribute, None)
    if not isinstance(profile, Profile):
        raise typer.BadParameter(f"{spec!r} is not a framelift Profile", param_hint="--profile")
    return profile


FRAMES_INDEX = "frames.jsonl"
# Frame files written by `frames`; four digits at least, more for a video with 10,000 frames.
FRAME_FILE = re.compile(r"frame-\d{4,}\.png")


def _checked(check: Callable[[float], float]) -> Callable[[float], float]:
    """A Typer callback that turns ``check``'s `ValueError` into a usage error for the option."""

    # Typer reads the callback's annotations at runtime, so they are spelled out as float.
    def callback(value: float) -> float:
        try:
            return check(value)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    return callback


def _checked_jpeg_quality(value: int | None) -> int | None:
    """`_checked` for ``--jpeg-quality``, whose value is an int or None."""
    try:
        return _synth.check_jpeg_quality(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


FrameRate = Annotated[
    float,
    typer.Option(
        callback=_checked(_video.check_frame_rate),
        help="Video frames examined per second (a finite number > 0)",
    ),
]
MotionThreshold = Annotated[
    float,
    typer.Option(
        callback=_checked(_video.check_motion_threshold),
        help="How much of the screen (share of 16x16 tiles, 0-1) may change between video "
        "frames that still count as one still screen; with a profile, of the part that is sent "
        "(the locate crop, leaving out tiles mostly redacted)",
    ),
]

ModelSetting = Annotated[
    list[str] | None,
    typer.Option(
        "--model-setting",
        metavar="KEY=VALUE",
        help="Pydantic AI model setting for every request, the screen check's too; repeatable. "
        "VALUE is read as JSON, or else as a string, e.g. openai_reasoning_effort=none or "
        "temperature=0; quote a string that would read as JSON, e.g. openai_user='\"12345\"'",
    ),
]


def _not_json(word: str) -> Any:
    """Refuse NaN and Infinity, which json.loads takes although JSON has no such values, so that
    they stay strings as the help says."""
    raise ValueError(word)


def _model_settings(values: list[str] | None) -> ModelSettings | None:
    """``--model-setting`` as Pydantic AI model settings, or None when none were given."""
    if not values:
        return None
    settings: dict[str, Any] = {}
    for value in values:
        key, sep, text = value.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(
                f"expected KEY=VALUE, got {value!r}", param_hint="--model-setting"
            )
        text = text.strip()
        try:
            settings[key.strip()] = json.loads(text, parse_constant=_not_json)
        except ValueError:
            # Bare words such as none are what most settings take; quoting them for JSON in a
            # shell would be a chore.
            settings[key.strip()] = text
    # Keys are not checked: ModelSettings and each provider's settings take different ones.
    return cast("ModelSettings", settings)


@app.command()
def run(
    inputs: Annotated[
        list[str], typer.Argument(help="Image or video files, directories or glob patterns")
    ],
    profile: Annotated[str, typer.Option(help="Profile to use, as 'package.module:attribute'")],
    model: Annotated[str, typer.Option(help="Pydantic AI model, e.g. 'anthropic:<model-name>'")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("out"),
    screen_model: Annotated[
        str | None,
        typer.Option(
            help="Pydantic AI model for the screen check of a profile with a "
            "screen_description [default: --model]"
        ),
    ] = None,
    model_setting: ModelSetting = None,
    max_retries: Annotated[
        int,
        typer.Option(
            min=0,
            help="Retries when validation fails, for the extraction and the screen check each",
        ),
    ] = 2,
    anonymize_paths: Annotated[
        bool, typer.Option(help="Write only hashes, not input paths, to the output")
    ] = False,
    frame_rate: FrameRate = _video.DEFAULT_FRAME_RATE,
    motion_threshold: MotionThreshold = _video.DEFAULT_MOTION_THRESHOLD,
    cache: Annotated[
        Path | None,
        typer.Option(
            help="Directory to cache extractions and screen checks in, so that images read "
            "before are not paid for again [default: no cache]"
        ),
    ] = None,
) -> None:
    """Extract data from images and videos; write observations.jsonl and items.jsonl."""
    loaded = load_profile(profile)
    settings = _model_settings(model_setting)
    try:
        pipeline = Pipeline(
            loaded,
            model=model,
            screen_model=screen_model,
            model_settings=settings,
            out_dir=out,
            max_retries=max_retries,
            anonymize_paths=anonymize_paths,
            frame_rate=frame_rate,
            motion_threshold=motion_threshold,
            cache_dir=cache,
        )
    except ValueError as exc:
        # The other settings are checked by their options first; only a --screen-model for a
        # profile with no screen_description is left to Pipeline.
        if screen_model is None or loaded.screen_description is not None:
            raise
        raise typer.BadParameter(str(exc), param_hint="--screen-model") from exc
    except UserError as exc:  # e.g. the provider's API key is not set
        _report(exc)
        raise typer.Exit(code=2) from exc
    try:
        result = pipeline.run(inputs)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="INPUTS") from exc
    except (_video.MissingVideoExtraError, ModelMismatchError) as exc:
        _report(exc)
        raise typer.Exit(code=2) from exc
    from_cache = f" ({len(result.cached)} from cache)" if cache is not None else ""
    typer.echo(
        f"{len(result.succeeded)} ok{from_cache}, {len(result.skipped)} skipped, "
        f"{len(result.failed)} failed -> {result.output_path}",
        err=True,
    )
    if result.items_path is not None:
        unkeyed = ""
        if result.unkeyed_items:
            unkeyed = (
                f" ({result.unkeyed_items} left out because item_key failed, "
                f"e.g. {result.unkeyed_error})"
            )
        typer.echo(f"{len(result.items)} items{unkeyed} -> {result.items_path}", err=True)
    if result.failed:
        raise typer.Exit(code=1)


def _report(exc: Exception) -> None:
    """Print ``exc`` and its notes, which say e.g. which image it was found in or that the
    output could not be written either."""
    typer.echo("\n".join([str(exc), *getattr(exc, "__notes__", [])]), err=True)


@app.command("eval")
def evaluate(
    dataset: Annotated[Path, typer.Argument(help="Dataset file (framelift/dataset/v1 YAML)")],
    profile: Annotated[str, typer.Option(help="Profile to use, as 'package.module:attribute'")],
    model: Annotated[str, typer.Option(help="Pydantic AI model, e.g. 'anthropic:<model-name>'")],
    root: Annotated[
        Path | None,
        typer.Option(
            help="Directory the dataset's image paths are relative to "
            "[default: the dataset's directory]"
        ),
    ] = None,
    model_setting: ModelSetting = None,
    max_retries: Annotated[int, typer.Option(min=0, help="Retries when validation fails")] = 2,
    cache: Annotated[
        Path | None,
        typer.Option(
            help="Directory to cache extractions in, so that a repeated evaluation does not pay "
            "for the same images again [default: no cache]"
        ),
    ] = None,
) -> None:
    """Score a profile and a model against a dataset of images with expected outputs."""
    # Imported here because pydantic-evals is an optional extra.
    try:
        from framelift._internal import evaluation
    except ImportError as exc:
        typer.echo(
            f'framelift eval needs the eval extra: pip install "framelift[eval]" ({exc})', err=True
        )
        raise typer.Exit(code=2) from exc

    loaded = load_profile(profile)
    settings = _model_settings(model_setting)
    try:
        cases = evaluation.build_dataset(dataset, loaded, root)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="DATASET") from exc
    try:
        report = evaluation.run_evaluation(
            cases,
            loaded,
            model,
            max_retries=max_retries,
            cache_dir=cache,
            model_settings=settings,
        )
    # Any UserError, not only ModelMismatchError: the model is resolved in there too, which fails
    # e.g. when the provider's API key is not set.
    except UserError as exc:
        _report(exc)
        raise typer.Exit(code=2) from exc
    report.print(include_averages=True, include_reasons=True)
    # Low scores are results, but a case or evaluator that crashed means the run is incomplete.
    is_incomplete = bool(report.failures) or any(c.evaluator_failures for c in report.cases)
    if is_incomplete:
        raise typer.Exit(code=1)


def _picked(
    path: Path, layout: Layout, frame_rate: float, motion_threshold: float
) -> Iterator[tuple[dict[str, Any], Image.Image]]:
    """The images run would send for ``path``, each with its frames.jsonl record."""
    if not is_video(path):
        try:
            source = load_image(path, layout)
        except OSError as exc:  # the input, not --out, which the caller's OSError handler means
            raise UnreadableInputError("image", path, error_reason(exc)) from exc
        record = {
            "index": 0,
            "timestamp_ms": None,
            "phash": source.phash,
            "sharpness": None,
            "crop": source.crop.model_dump(mode="json"),
        }
        yield record, source.view
        return
    for frame in _video.sample_video(
        path, frame_rate=frame_rate, motion_threshold=motion_threshold, layout=layout
    ):
        view, crop = layout.apply(frame.image)
        record = {
            "index": frame.index,
            "timestamp_ms": frame.timestamp_ms,
            # Of the image sent, as run records it (see frame_to_source).
            "phash": frame.phash if layout.is_identity else str(imagehash.phash(view)),
            "sharpness": round(frame.sharpness, 2),
            "crop": crop.model_dump(mode="json"),
        }
        yield record, view


@app.command()
def frames(
    input_path: Annotated[Path, typer.Argument(metavar="INPUT", help="Video or image file")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("frames"),
    profile: Annotated[
        str | None,
        typer.Option(
            help="Profile whose locate and redact to apply, as 'package.module:attribute' "
            "[default: the whole screen]"
        ),
    ] = None,
    frame_rate: FrameRate = _video.DEFAULT_FRAME_RATE,
    motion_threshold: MotionThreshold = _video.DEFAULT_MOTION_THRESHOLD,
) -> None:
    """Write the images run would send to the model (one per still screen of a video, cropped
    and redacted with --profile), without calling a model."""
    if not input_path.is_file():
        raise typer.BadParameter(f"not a file: {input_path}", param_hint="INPUT")
    layout = Layout.from_profile(load_profile(profile)) if profile else Layout.full()
    try:
        picked = _picked(input_path, layout, frame_rate, motion_threshold)
        # The generator reads the input on the first image; take it before touching the output,
        # so a missing extra or an unreadable file leaves an earlier run's frames as they were.
        first = next(picked, None)
        out.mkdir(parents=True, exist_ok=True)
        # Written to a staging directory and moved in only once the whole input is read, so a
        # video that breaks off partway leaves an earlier run's frames untouched, and a run that
        # picks fewer frames leaves no old ones behind.
        with tempfile.TemporaryDirectory(dir=out, prefix=".frames-") as staging:
            stage = Path(staging)
            lines: list[str] = []
            for record, view in itertools.chain([first] if first is not None else [], picked):
                file = f"frame-{record['index']:04d}.png"
                view.save(stage / file, format="PNG")
                lines.append(json.dumps({**record, "file": file}) + "\n")
            (stage / FRAMES_INDEX).write_text("".join(lines), encoding="utf-8")
            for old in out.glob("frame-*.png"):
                if FRAME_FILE.fullmatch(old.name):
                    old.unlink()
            for new in stage.iterdir():
                new.replace(out / new.name)
        count = len(lines)
    except _video.MissingVideoExtraError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except NoCropError as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="INPUT") from exc
    except OSError as exc:  # e.g. --out is a file, not writable, or the disk is full
        typer.echo(f"cannot write frames to {out}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {count} frames and frames.jsonl to {out}", err=True)


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("samples"),
    n: Annotated[int, typer.Option(min=1, help="Number of screenshots")] = 10,
    videos: Annotated[
        int, typer.Option(min=0, help="Number of scrolling videos (needs the video extra)")
    ] = 0,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    scale: Annotated[
        float,
        typer.Option(
            callback=_checked(_synth.check_scale),
            help=f"Resize the screenshots by this ({_synth.MIN_SCALE:g}-{_synth.MAX_SCALE:g})",
        ),
    ] = _synth.DEFAULT_SCALE,
    noise: Annotated[
        float,
        typer.Option(
            callback=_checked(_synth.check_noise),
            help="Standard deviation of the Gaussian noise added to the screenshots, in 0-255 "
            f"pixel values (0-{_synth.MAX_NOISE:g})",
        ),
    ] = _synth.DEFAULT_NOISE,
    jpeg_quality: Annotated[
        int | None,
        typer.Option(
            callback=_checked_jpeg_quality,
            help="Save the screenshots as JPEG of this quality "
            f"({_synth.MIN_JPEG_QUALITY}-{_synth.MAX_JPEG_QUALITY})",
        ),
    ] = None,
) -> None:
    """Generate screenshots of a fictional shop, with ground truth in dataset.yaml, and
    optionally videos of its list scrolling."""
    # Files of an earlier run that could not be removed are reported like the other messages,
    # even when Python's warnings are filtered out; other warnings are shown as they would be,
    # also when the run fails.
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", _synth.StaleFilesWarning)
            truths = _synth.generate(
                out,
                n=n,
                seed=seed,
                scale=scale,
                noise=noise,
                jpeg_quality=jpeg_quality,
                videos=videos,
            )
    except _video.MissingVideoExtraError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except OSError as exc:  # e.g. --out is a file, not writable, or the disk is full
        typer.echo(f"cannot write samples to {out}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        for warning in caught:
            if issubclass(warning.category, _synth.StaleFilesWarning):
                typer.echo(str(warning.message), err=True)
            else:
                warnings.showwarning(
                    warning.message, warning.category, warning.filename, warning.lineno
                )
    typer.echo(f"wrote {len(truths)} images and {_synth.DATASET_FILE} to {out}", err=True)
    if videos:
        typer.echo(f"wrote {videos} videos to {out}", err=True)


if __name__ == "__main__":
    app()
