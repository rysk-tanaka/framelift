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


def _synth(samples: Path) -> None:
    result = runner.invoke(app, ["synth", "--out", str(samples), "--n", "2"])
    assert result.exit_code == 0, result.output


def test_synth_then_eval_with_the_test_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    samples = tmp_path / "samples"
    _synth(samples)

    result = runner.invoke(
        app,
        [
            "eval",
            str(samples / "dataset.yaml"),
            "--profile",
            "examples.fantasy_shop:profile",
            "--model",
            "test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "FieldAccuracy" in result.output


def test_eval_rejects_a_dataset_that_does_not_match_its_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    samples = tmp_path / "samples"
    _synth(samples)
    (samples / "shop-000.png").write_bytes(b"something else")

    result = runner.invoke(
        app,
        [
            "eval",
            str(samples / "dataset.yaml"),
            "--profile",
            "examples.fantasy_shop:profile",
            "--model",
            "test",
        ],
    )
    assert result.exit_code == 2


def test_eval_rejects_a_directory_as_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    result = runner.invoke(
        app,
        ["eval", str(tmp_path), "--profile", "examples.fantasy_shop:profile", "--model", "test"],
    )
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_a_profile_that_fails_on_import_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "broken_profile.py").write_text(
        "from pydantic import BaseModel\n"
        "from framelift import Profile\n"
        "class Out(BaseModel):\n"
        "    value: int\n"
        "profile = Profile(name='p', output_type=Out, instructions='read', item_key=len)\n"
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["run", str(tmp_path), "--profile", "broken_profile:profile", "--model", "test"]
    )
    assert result.exit_code == 2
    assert "item_key" in result.output
    assert "broken_profile.py:5" in result.output  # the user's line, not framelift's check
