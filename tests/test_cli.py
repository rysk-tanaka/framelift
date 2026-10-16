from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from framelift import Observation
from framelift import cli as cli_module
from framelift.cli import app

runner = CliRunner()

# Rich colors the output on CI (it detects GitHub Actions), splitting e.g. "--frame-rate" with
# escape codes, so text checks look at the output without them.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    return _ANSI.sub("", output)


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


def test_frames_writes_the_picked_frames(scrolling_video, tmp_path: Path) -> None:
    path, sample = scrolling_video
    out = tmp_path / "frames"

    result = runner.invoke(app, ["frames", str(path), "--out", str(out)])

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert len(records) == len(sample.holds)
    assert all((out / record["file"]).is_file() for record in records)
    assert set(records[0]) == {"index", "timestamp_ms", "phash", "sharpness", "file"}


def test_frames_reports_an_output_it_cannot_write(scrolling_video, tmp_path: Path) -> None:
    path, _ = scrolling_video
    out = tmp_path / "taken"
    out.write_text("a file, not a directory")

    result = runner.invoke(app, ["frames", str(path), "--out", str(out)])

    assert result.exit_code == 1
    assert "cannot write frames" in result.output
    assert "Traceback" not in result.output


def test_frames_rejects_unreadable_videos_without_touching_earlier_output(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "frames.jsonl").write_text("from an earlier run\n")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    result = runner.invoke(app, ["frames", str(broken), "--out", str(out)])

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert (out / "frames.jsonl").read_text() == "from an earlier run\n"
    assert sorted(p.name for p in out.iterdir()) == ["frames.jsonl"]


def test_synth_videos_then_run_writes_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("av")
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    samples, out = tmp_path / "samples", tmp_path / "out"

    synth = runner.invoke(app, ["synth", "--out", str(samples), "--n", "1", "--videos", "1"])
    assert synth.exit_code == 0, synth.output
    assert (samples / "shop-scroll-000.mp4").is_file()

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
    assert (out / "items.jsonl").read_text()


@pytest.mark.parametrize("command", ["run", "frames"])
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--frame-rate", "0"),
        ("--frame-rate", "nan"),
        ("--frame-rate", "inf"),
        ("--motion-threshold", "1.5"),
        ("--motion-threshold", "nan"),
    ],
)
def test_invalid_sampling_settings_are_usage_errors(
    command: str, option: str, value: str, tmp_path: Path
) -> None:
    args = ["run", str(tmp_path), "--profile", "x:y", "--model", "test"]
    if command == "frames":
        args = ["frames", str(tmp_path / "video.mp4")]
    result = runner.invoke(app, [*args, option, value])
    assert result.exit_code == 2
    assert option in _plain(result.output)


def test_run_reports_unreadable_inputs_as_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    result = runner.invoke(
        app,
        [
            "run",
            str(broken),
            "--profile",
            "examples.fantasy_shop:profile",
            "--model",
            "test",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "0 ok, 1 failed" in result.output
    assert "Traceback" not in result.output


def test_frames_replaces_an_earlier_run_only_when_the_video_is_fully_read(
    scrolling_video, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sample = scrolling_video
    out = tmp_path / "frames"
    out.mkdir()
    for index in [*range(9), 10_000]:  # an earlier run that picked more frames
        (out / f"frame-{index:04d}.png").write_bytes(b"old")
    (out / "frames.jsonl").write_text("old\n")
    (out / "notes.txt").write_text("not ours")
    (out / "frame-cover.png").write_text("not ours either")

    real_sample_video = cli_module._video.sample_video

    def breaks_after_one_frame(*args, **kwargs):
        yield next(iter(real_sample_video(*args, **kwargs)))
        raise ValueError("Not a readable video: truncated")

    monkeypatch.setattr(cli_module._video, "sample_video", breaks_after_one_frame)
    broken = runner.invoke(app, ["frames", str(path), "--out", str(out)])
    assert broken.exit_code == 2
    assert (out / "frames.jsonl").read_text() == "old\n"
    assert (out / "frame-0000.png").read_bytes() == b"old"

    monkeypatch.setattr(cli_module._video, "sample_video", real_sample_video)
    result = runner.invoke(app, ["frames", str(path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    expected = {f"frame-{index:04d}.png" for index in range(len(sample.holds))}
    assert {p.name for p in out.glob("frame-*.png")} == {*expected, "frame-cover.png"}
    assert len((out / "frames.jsonl").read_text().splitlines()) == len(sample.holds)
    assert (out / "notes.txt").read_text() == "not ours"
    assert not [p for p in out.iterdir() if p.name.startswith(".frames-")]


@pytest.mark.parametrize("command", ["run", "frames", "synth"])
def test_video_commands_without_the_extra_explain_how_to_install_it(
    command: str, scrolling_video, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = scrolling_video
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setitem(sys.modules, "av", None)
    args = {
        "run": [
            "run",
            str(path),
            "--profile",
            "examples.fantasy_shop:profile",
            "--model",
            "test",
            "--out",
            str(tmp_path / "out"),
        ],
        "frames": ["frames", str(path), "--out", str(tmp_path / "frames")],
        "synth": ["synth", "--out", str(tmp_path / "samples"), "--n", "1", "--videos", "1"],
    }[command]

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert 'pip install "framelift[video]"' in result.output
    assert "Traceback" not in result.output


def test_run_says_why_items_were_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, samples
) -> None:
    root, _ = samples
    (tmp_path / "picky_profile.py").write_text(
        "from examples.fantasy_shop import ShopScreen\n"
        "from framelift import Profile\n"
        "def key(item):\n"
        "    raise KeyError(item.name)\n"
        "profile = Profile(name='p', output_type=ShopScreen, instructions='read', item_key=key)\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    result = runner.invoke(
        app,
        [
            "run",
            str(root),
            "--profile",
            "picky_profile:profile",
            "--model",
            "test",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "skipped because item_key failed, e.g. KeyError(" in result.output


def test_an_import_error_from_profile_code_is_not_a_usage_error(
    samples, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, truths = samples
    (tmp_path / "importing_profile.py").write_text(
        "from pydantic import BaseModel\n"
        "from framelift import Profile\n"
        "class Out(BaseModel):\n"
        "    value: int\n"
        "def needs_a_module(out):\n"
        "    raise ImportError('a module the validator needs')\n"
        "profile = Profile(name='p', output_type=Out, instructions='read',\n"
        "                  validators=[needs_a_module])\n"
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(root / truths[0].file),
            "--profile",
            "importing_profile:profile",
            "--model",
            "test",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    # Raised as is, with its traceback, instead of a terse exit 2 meant for the video extra.
    assert result.exit_code == 1
    assert isinstance(result.exception, ImportError)
    assert "a module the validator needs" in str(result.exception)
