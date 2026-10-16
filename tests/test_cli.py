from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from framelift import Observation
from framelift.cli import app

runner = CliRunner()


def test_synth_then_run_with_the_test_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)  # so that `examples.fantasy_shop` is importable
    samples, out = tmp_path / "samples", tmp_path / "out"

    synth = runner.invoke(app, ["synth", "--out", str(samples), "--n", "2"])
    assert synth.exit_code == 0, synth.output

    # Pydantic AI's built-in "test" model returns schema-valid data without calling a provider.
    result = runner.invoke(
        app,
        [
            "run",
            str(samples),
            "--profile",
            "examples.fantasy_shop:profile",
            "--model",
            "test",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = (out / "observations.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert all(Observation.model_validate_json(line).status == "ok" for line in lines)


@pytest.mark.parametrize(
    "spec", ["no_colon", "examples.fantasy_shop:missing", "framelift:Pipeline"]
)
def test_invalid_profile_spec(spec: str, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path), "--profile", spec, "--model", "test"])
    assert result.exit_code == 2


def test_missing_input_is_a_usage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    result = runner.invoke(
        app,
        [
            "run",
            str(tmp_path / "nope.png"),
            "--profile",
            "examples.fantasy_shop:profile",
            "--model",
            "test",
        ],
    )
    assert result.exit_code == 2
