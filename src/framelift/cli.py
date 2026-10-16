"""Command-line interface: ``framelift run`` and ``framelift synth``."""

from __future__ import annotations

import importlib
import sys
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
    except ImportError as exc:
        raise typer.BadParameter(f"cannot import {module_name!r}: {exc}") from exc
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


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("samples"),
    n: Annotated[int, typer.Option(min=1, help="Number of screenshots")] = 10,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
) -> None:
    """Generate screenshots of a fictional shop, with ground truth in truth.jsonl."""
    truths = _synth.generate(out, n=n, seed=seed)
    typer.echo(f"wrote {len(truths)} images and truth.jsonl to {out}", err=True)


if __name__ == "__main__":
    app()
