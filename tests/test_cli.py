from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import imagehash
import pytest
from PIL import Image
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo
from typer.testing import CliRunner

from framelift import Observation
from framelift import cli as cli_module
from framelift._internal import extract as extract_module
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


def test_run_twice_with_a_cache_reads_the_second_time_from_it(
    samples, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    root, _ = samples
    out = tmp_path / "out"
    args = [
        "run",
        str(root),
        "--profile",
        "examples.fantasy_shop:profile",
        "--model",
        "test",
        "--out",
        str(out),
        "--cache",
        str(tmp_path / "cache"),
    ]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert "3 ok (0 from cache)" in first.output
    assert second.exit_code == 0, second.output
    assert "3 ok (3 from cache)" in second.output
    lines = (out / "observations.jsonl").read_text().splitlines()
    assert all(Observation.model_validate_json(line).cached for line in lines)


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


def test_degraded_synth_then_run_and_eval_with_the_test_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    samples, out = tmp_path / "samples", tmp_path / "out"
    degrade = ["--scale", "0.5", "--noise", "8", "--jpeg-quality", "70"]
    synth = runner.invoke(app, ["synth", "--out", str(samples), "--n", "2", *degrade])
    assert synth.exit_code == 0, synth.output
    assert sorted(p.name for p in samples.iterdir()) == [
        "dataset.yaml",
        "shop-000.jpg",
        "shop-001.jpg",
        "synth.sha256",
    ]

    profile = ["--profile", "examples.fantasy_shop:profile", "--model", "test"]
    result = runner.invoke(app, ["run", str(samples), *profile, "--out", str(out)])
    assert result.exit_code == 0, result.output
    lines = (out / "observations.jsonl").read_text().splitlines()
    assert [Observation.model_validate_json(line).status for line in lines] == ["ok", "ok"]

    result = runner.invoke(app, ["eval", str(samples / "dataset.yaml"), *profile])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "option",
    [
        ["--scale", "0"],
        ["--scale", "nan"],
        ["--noise", "nan"],
        ["--noise", "65"],
        ["--jpeg-quality", "0"],
    ],
)
def test_synth_rejects_out_of_range_degradation(tmp_path: Path, option: list[str]) -> None:
    result = runner.invoke(app, ["synth", "--out", str(tmp_path / "samples"), "--n", "1", *option])
    assert result.exit_code == 2
    assert not (tmp_path / "samples").exists()


def test_synth_prints_files_it_cannot_remove_even_when_warnings_are_filtered_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def generate(*args: object, **kwargs: object) -> list[object]:
        warnings.warn("could not remove shop-001.png", cli_module._synth.StaleFilesWarning, 1)
        warnings.warn("something else", DeprecationWarning, stacklevel=1)
        return []

    monkeypatch.setattr(cli_module._synth, "generate", generate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = runner.invoke(app, ["synth", "--out", str(tmp_path), "--n", "1"])
    assert result.exit_code == 0, result.output
    assert "could not remove shop-001.png" in result.output
    assert "something else" not in result.output  # other warnings keep their filters


def test_synth_shows_other_warnings_also_when_it_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def generate(*args: object, **kwargs: object) -> list[object]:
        warnings.warn("something else", RuntimeWarning, stacklevel=1)
        raise OSError("disk full")

    shown: list[str] = []
    monkeypatch.setattr(cli_module._synth, "generate", generate)
    # pytest records warnings itself; see what the CLI hands on instead.
    monkeypatch.setattr(warnings, "showwarning", lambda message, *args: shown.append(str(message)))
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        result = runner.invoke(app, ["synth", "--out", str(tmp_path), "--n", "1"])
    assert result.exit_code == 1
    assert "disk full" in result.output
    assert shown == ["something else"]


def test_synth_reports_an_output_it_cannot_write(tmp_path: Path) -> None:
    out = tmp_path / "samples"
    out.write_text("a file, not a directory")
    result = runner.invoke(app, ["synth", "--out", str(out), "--n", "1"])
    assert result.exit_code == 1
    assert "cannot write samples to" in result.output


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
        "profile = Profile(locate='full', name='p', output_type=Out, instructions='read',\n"
        "                  item_key=len)\n"
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
    assert set(records[0]) == {"index", "timestamp_ms", "phash", "sharpness", "file", "crop"}


def test_frames_reports_an_output_it_cannot_write(scrolling_video, tmp_path: Path) -> None:
    path, _ = scrolling_video
    out = tmp_path / "taken"
    out.write_text("a file, not a directory")

    result = runner.invoke(app, ["frames", str(path), "--out", str(out)])

    assert result.exit_code == 1
    assert "cannot write frames" in result.output
    assert "Traceback" not in result.output


def test_frames_blames_the_input_for_an_image_it_cannot_read(
    samples, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, truths = samples
    unreadable = root / truths[0].file
    read_bytes = Path.read_bytes

    def denied(self: Path) -> bytes:
        if self == unreadable:
            raise PermissionError(13, "Permission denied", str(self))
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", denied)

    result = runner.invoke(app, ["frames", str(unreadable), "--out", str(tmp_path / "out")])

    assert result.exit_code == 2
    assert "Invalid value for INPUT" in _plain(result.output)
    assert "cannot write frames" not in result.output


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
    assert "0 ok, 0 skipped, 1 failed" in result.output
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
    if command == "synth":
        assert not (tmp_path / "samples").exists()  # checked before anything is drawn


def test_run_says_why_items_were_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, samples
) -> None:
    root, _ = samples
    (tmp_path / "picky_profile.py").write_text(
        "from examples.fantasy_shop import ShopScreen\n"
        "from framelift import Profile\n"
        "def key(item):\n"
        "    raise KeyError(item.name)\n"
        "profile = Profile(locate='full', name='p', output_type=ShopScreen, instructions='read',\n"
        "                  item_key=key)\n"
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
    assert "left out because item_key failed, e.g. KeyError(" in result.output


def _screened_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "screened_profile.py").write_text(
        "from dataclasses import replace\n"
        "from examples.fantasy_shop import profile as shop\n"
        'profile = replace(shop, screen_description="a shop\'s item list")\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))


def test_run_checks_the_screen_with_the_screen_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, samples
) -> None:
    root, _ = samples
    _screened_profile(tmp_path, monkeypatch)
    out = tmp_path / "out"
    # "test" is the only model that runs offline, so the screen model is given another name and
    # swapped for it once the CLI has passed it on, to tell the two options apart.
    passed: dict[str, object] = {}
    pipeline = cli_module.Pipeline

    def spy(profile, **kwargs: Any):
        passed.update(kwargs)
        kwargs["screen_model"] = "test"
        return pipeline(profile, **kwargs)

    monkeypatch.setattr(cli_module, "Pipeline", spy)

    result = runner.invoke(
        app,
        [
            "run",
            str(root),
            "--profile",
            "screened_profile:profile",
            "--model",
            "test",
            "--screen-model",
            "screen:model",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert passed["model"] == "test"
    assert passed["screen_model"] == "screen:model"
    observations = [
        Observation.model_validate_json(line)
        for line in (out / "observations.jsonl").read_text().splitlines()
    ]
    assert len(observations) == 3
    assert all(o.screen is not None and o.screen.model == "test" for o in observations)
    ok = sum(o.status == "ok" for o in observations)
    assert f"{ok} ok, {3 - ok} skipped, 0 failed" in result.output


def test_other_pipeline_errors_are_not_blamed_on_the_screen_model(
    samples, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = samples

    def broken(*args: Any, **kwargs: Any):
        raise ValueError("something else")

    monkeypatch.setattr(cli_module, "Pipeline", broken)

    result = runner.invoke(
        app, ["run", str(root), "--profile", "examples.fantasy_shop:profile", "--model", "test"]
    )

    assert isinstance(result.exception, ValueError)
    assert "--screen-model" not in _plain(result.output)


def test_a_screen_model_without_a_screen_description_is_a_usage_error(samples) -> None:
    root, _ = samples

    result = runner.invoke(
        app,
        [
            "run",
            str(root),
            "--profile",
            "examples.fantasy_shop:profile",
            "--model",
            "test",
            "--screen-model",
            "test",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for --screen-model" in _plain(result.output)


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
        "profile = Profile(locate='full', name='p', output_type=Out, instructions='read',\n"
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


def test_frames_writes_a_still_image_as_one_frame(samples, tmp_path: Path) -> None:
    root, truths = samples
    out = tmp_path / "frames"

    result = runner.invoke(app, ["frames", str(root / truths[0].file), "--out", str(out)])

    assert result.exit_code == 0, result.output
    (record,) = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert record["file"] == "frame-0000.png" and record["timestamp_ms"] is None
    assert record["crop"] == {
        "orientation": "portrait",
        "size": [720, 1280],
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "redact": [],
    }
    with Image.open(out / "frame-0000.png") as image:
        assert image.size == (720, 1280)


def test_frames_with_a_profile_writes_what_would_be_sent(
    samples, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    root, truths = samples
    out = tmp_path / "frames"

    result = runner.invoke(
        app,
        [
            "frames",
            str(root / truths[1].file),  # landscape
            "--profile",
            "examples.fantasy_shop:profile",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    (record,) = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert record["crop"]["bbox"] == [0.0, 0.0, 1.0, 0.9]
    with Image.open(out / "frame-0000.png") as image:
        assert image.size == (1280, 648)


def test_frames_of_a_video_with_a_profile_record_the_hash_of_what_would_be_sent(
    scrolling_video, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    path, _ = scrolling_video
    out = tmp_path / "frames"

    result = runner.invoke(
        app, ["frames", str(path), "--profile", "examples.fantasy_shop:profile", "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert records
    for record in records:
        with Image.open(out / record["file"]) as image:
            assert image.size == (1280, 648)
            assert record["phash"] == str(imagehash.phash(image))


def test_frames_reports_a_screen_the_profile_cannot_crop(
    samples, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "portrait_only.py").write_text(
        "from dataclasses import replace\n"
        "from examples.fantasy_shop import profile as shop\n"
        "profile = replace(shop, locate={'portrait': (0.0, 0.0, 1.0, 0.6)})\n"
    )
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.syspath_prepend(str(tmp_path))
    root, truths = samples

    result = runner.invoke(
        app,
        ["frames", str(root / truths[1].file), "--profile", "portrait_only:profile"],
    )

    assert result.exit_code == 2
    assert "no crop box for landscape" in _plain(result.output)


def _answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    data = {"shop_name": "A", "items": [{"name": "a", "price": 1}]}
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])


@pytest.mark.parametrize("command", ["run", "eval"])
def test_another_model_answering_is_reported_without_a_traceback(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answering_model
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    samples = tmp_path / "samples"
    _synth(samples)
    monkeypatch.setattr(
        extract_module, "infer_model", lambda model: answering_model(_answer, model_name="qwen3")
    )
    target = [str(samples), "--out", str(tmp_path / "out")]
    if command == "eval":
        target = [str(samples / "dataset.yaml")]

    result = runner.invoke(
        app,
        [command, *target, "--profile", "examples.fantasy_shop:profile", "--model", "openai:qwen3"],
    )

    assert result.exit_code == 2, result.output
    assert "'gemma-3'" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_missing_api_key_is_reported_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = runner.invoke(
        app,
        ["run", str(tmp_path), "--profile", "examples.fantasy_shop:profile", "--model", "openai:x"],
    )

    assert result.exit_code == 2, result.output
    assert "OPENAI_API_KEY" in result.output
