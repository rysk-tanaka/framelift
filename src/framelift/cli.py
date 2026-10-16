"""Command-line interface: ``framelift run``, ``framelift eval`` and ``framelift synth``."""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import Annotated

import typer

from framelift._internal import synth as _synth
from framelift.pipeline import Pipeline
from framelift.profile import Profile

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


@app.command()
def run(
    inputs: Annotated[list[str], typer.Argument(help="Image files, directories or glob patterns")],
    profile: Annotated[str, typer.Option(help="Profile to use, as 'package.module:attribute'")],
    model: Annotated[str, typer.Option(help="Pydantic AI model, e.g. 'anthropic:<model-name>'")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("out"),
    max_retries: Annotated[int, typer.Option(min=0, help="Retries when validation fails")] = 2,
    anonymize_paths: Annotated[
        bool, typer.Option(help="Write only hashes, not input paths, to the output")
    ] = False,
) -> None:
    """Extract data from images and write observations.jsonl."""
    pipeline = Pipeline(
        load_profile(profile),
        model=model,
        out_dir=out,
        max_retries=max_retries,
        anonymize_paths=anonymize_paths,
    )
    try:
        result = pipeline.run(inputs)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="INPUTS") from exc
    typer.echo(
        f"{len(result.succeeded)} ok, {len(result.failed)} failed -> {result.output_path}",
        err=True,
    )
    if result.failed:
        raise typer.Exit(code=1)


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
    max_retries: Annotated[int, typer.Option(min=0, help="Retries when validation fails")] = 2,
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
    try:
        cases = evaluation.build_dataset(dataset, loaded, root)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="DATASET") from exc
    report = evaluation.run_evaluation(cases, loaded, model, max_retries=max_retries)
    report.print(include_averages=True, include_reasons=True)
    # Low scores are results, but a case or evaluator that crashed means the run is incomplete.
    is_incomplete = bool(report.failures) or any(c.evaluator_failures for c in report.cases)
    if is_incomplete:
        raise typer.Exit(code=1)


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("samples"),
    n: Annotated[int, typer.Option(min=1, help="Number of screenshots")] = 10,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
) -> None:
    """Generate screenshots of a fictional shop, with ground truth in dataset.yaml."""
    truths = _synth.generate(out, n=n, seed=seed)
    typer.echo(f"wrote {len(truths)} images and {_synth.DATASET_FILE} to {out}", err=True)


if __name__ == "__main__":
    app()
