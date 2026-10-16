from __future__ import annotations

import hashlib
import io
import sys
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from examples.fantasy_shop import profile as fantasy_shop_profile
from framelift import Item, Observation, Pipeline, Profile
from framelift import pipeline as pipeline_module
from framelift._internal import video as video_module
from framelift.observation import FrameRef

# Most tests look up the answer by the hash of the input image, so the screen is sent whole.
shop_profile = replace(fantasy_shop_profile, locate="full")


def _frame(observation: Observation) -> FrameRef:
    assert observation.frame is not None
    return observation.frame


def _image_sha(messages: list[ModelMessage]) -> str:
    for message in messages:
        for part in getattr(message, "parts", ()):
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                for item in part.content:
                    if isinstance(item, BinaryContent):
                        return hashlib.sha256(item.data).hexdigest()
    raise AssertionError("no image in the request")


def _truth_by_sha(root: Path, truths: list) -> dict[str, dict]:
    return {
        hashlib.sha256((root / t.file).read_bytes()).hexdigest(): {
            "shop_name": t.shop_name,
            "items": [asdict(i) for i in t.items],
        }
        for t in truths
    }


def test_run_reads_every_image_and_writes_jsonl(samples, tmp_path: Path) -> None:
    root, truths = samples
    answers = _truth_by_sha(root, truths)

    def read_screen(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, answers[_image_sha(messages)])]
        )

    out = tmp_path / "out"
    result = Pipeline(
        shop_profile, model=FunctionModel(read_screen, model_name="fake"), out_dir=out
    ).run(root)

    assert len(result.succeeded) == len(truths) and not result.failed
    assert result.output_path == out / "observations.jsonl"
    lines = result.output_path.read_text().splitlines()
    observations = [Observation.model_validate_json(line) for line in lines]
    assert [o.data for o in observations] == [answers[_frame(o).sha256] for o in observations]
    first = observations[0]
    assert first.profile.name == "fantasy-shop"
    assert first.model == "function:fake"
    assert first.retries == 0
    assert '"schema":"framelift/observation/v3"' in lines[0]
    # An image is its own single frame.
    assert _frame(first).index == 0 and _frame(first).timestamp_ms is None
    assert _frame(first).sha256 == first.source.sha256


def test_validator_failure_triggers_a_retry(samples) -> None:
    root, truths = samples
    answers = _truth_by_sha(root, truths)
    calls: dict[str, int] = {}

    def duplicate_then_correct(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sha = _image_sha(messages)
        calls[sha] = calls.get(sha, 0) + 1
        answer = dict(answers[sha])
        if calls[sha] == 1:  # first attempt repeats a row, which the profile validator rejects
            answer["items"] = answer["items"] + answer["items"][:1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    model = FunctionModel(duplicate_then_correct, model_name="fake")
    result = Pipeline(shop_profile, model=model).run(root / truths[0].file)

    (observation,) = result.observations
    assert observation.status == "ok"
    assert observation.retries == 1
    assert observation.data == answers[_frame(observation).sha256]
    assert result.output_path is None


def test_exhausted_retries_are_recorded_as_errors(samples) -> None:
    root, truths = samples

    def always_empty(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"shop_name": "x", "items": []})]
        )

    result = Pipeline(shop_profile, model=FunctionModel(always_empty), max_retries=1).run(root)

    assert len(result.failed) == len(truths)
    assert all(o.data is None and o.error for o in result.failed)
    # The failed attempts still cost tokens, which evaluation compares.
    assert all(o.retries == 1 and o.usage.requests == 2 for o in result.failed)
    assert all(o.usage.input_tokens > 0 for o in result.failed)


def test_anonymized_paths(samples) -> None:
    root, truths = samples
    answers = _truth_by_sha(root, truths)

    def read_screen(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, answers[_image_sha(messages)])]
        )

    result = Pipeline(shop_profile, model=FunctionModel(read_screen), anonymize_paths=True).run(
        root
    )
    assert all(o.source.path is None for o in result.observations)


class Aliased(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    shop_name: str = Field(alias="shopName")


def test_observations_keep_the_models_own_serialization(samples) -> None:
    root, truths = samples
    profile = Profile(locate="full", name="p", output_type=Aliased, instructions="read")

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"shopName": "A"})])

    result = Pipeline(profile, model=FunctionModel(respond)).run(root / truths[0].file)

    assert result.observations[0].data == {"shopName": "A"}


def test_a_video_is_read_once_per_still_screen_and_items_are_merged(
    scrolling_video, tmp_path: Path
) -> None:
    path, sample = scrolling_video
    calls: list[int] = []

    def read_hold(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Frames arrive in order, one per still screen: answer with what that screen shows.
        hold = sample.holds[len(calls)]
        calls.append(1)
        data = {"shop_name": sample.truth.shop_name, "items": [asdict(i) for i in hold.items]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])

    out = tmp_path / "out"
    result = Pipeline(shop_profile, model=FunctionModel(read_hold), out_dir=out).run(path)

    assert len(result.observations) == len(sample.holds) and not result.failed
    assert [_frame(o).index for o in result.observations] == list(range(len(sample.holds)))
    video_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for observation, hold in zip(result.observations, sample.holds, strict=True):
        assert observation.source.sha256 == video_sha
        assert _frame(observation).sha256 != video_sha  # the PNG sent, not the video
        assert _frame(observation).timestamp_ms is not None
        assert hold.start_ms <= _frame(observation).timestamp_ms < hold.end_ms

    assert result.items_path == out / "items.jsonl"
    stored = [Item.model_validate_json(line) for line in result.items_path.read_text().splitlines()]
    assert stored == result.items
    votes = {item.key: item.votes for item in result.items}
    expected_votes = {
        item.name: sum(item in hold.items for hold in sample.holds) for item in sample.truth.items
    }
    assert votes == expected_votes
    assert any(count > 1 for count in votes.values())
    assert all(not item.conflicts for item in result.items)


def test_images_and_videos_can_be_mixed(samples, scrolling_video) -> None:
    root, truths = samples
    video, sample = scrolling_video

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        data = {"shop_name": "A", "items": [{"name": "a", "price": 1}]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])

    result = Pipeline(shop_profile, model=FunctionModel(respond)).run(
        [root / truths[0].file, video]
    )

    image, *frames = result.observations
    assert _frame(image).timestamp_ms is None
    assert len(frames) == len(sample.holds)
    assert all(_frame(frame).timestamp_ms is not None for frame in frames)
    (item,) = result.items
    assert item.votes == 1 + len(sample.holds)


def test_videos_without_the_extra_stop_before_any_model_call(scrolling_video, monkeypatch) -> None:
    path, _ = scrolling_video

    def never(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("the model must not be called")

    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(ImportError, match=r"framelift\[video\].*pass them alone"):
        Pipeline(shop_profile, model=FunctionModel(never)).run(path)


def test_invalid_settings_are_rejected() -> None:
    for frame_rate in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="frame_rate"):
            Pipeline(shop_profile, model="test", frame_rate=frame_rate)
    for motion_threshold in (-1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="motion_threshold"):
            Pipeline(shop_profile, model="test", motion_threshold=motion_threshold)
    with pytest.raises(ValueError, match="max_retries"):
        Pipeline(shop_profile, model="test", max_retries=-1)


def test_profiles_without_item_key_write_no_items(samples, tmp_path: Path) -> None:
    root, _ = samples
    profile = Profile(locate="full", name="p", output_type=Aliased, instructions="read")

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"shopName": "A"})])

    (tmp_path / "items.jsonl").write_text("left by an earlier run\n")
    result = Pipeline(profile, model=FunctionModel(respond), out_dir=tmp_path).run(root)

    assert result.items == [] and result.items_path is None
    assert not (tmp_path / "items.jsonl").exists()


def _fixed_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    data = {"shop_name": "A", "items": [{"name": "a", "price": 1}]}
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])


def test_unreadable_inputs_are_recorded_and_the_run_goes_on(samples, tmp_path: Path) -> None:
    root, truths = samples
    broken_video = tmp_path / "broken.mp4"
    broken_video.write_bytes(b"not a video")
    broken_image = tmp_path / "broken.png"
    broken_image.write_bytes(b"not an image")
    good = root / truths[0].file

    out = tmp_path / "out"
    result = Pipeline(shop_profile, model=FunctionModel(_fixed_answer), out_dir=out).run(
        [broken_video, broken_image, good]
    )

    video_error, image_error, ok = result.observations
    assert video_error.status == image_error.status == "error"
    assert video_error.frame is None and "Not a readable video" in (video_error.error or "")
    assert image_error.frame is None and "Not a readable image" in (image_error.error or "")
    assert video_error.source.sha256 == hashlib.sha256(b"not a video").hexdigest()
    assert image_error.source.sha256 == hashlib.sha256(b"not an image").hexdigest()
    assert ok.status == "ok"
    assert [item.key for item in result.items] == ["a"]
    assert result.items_path is not None and result.items_path.is_file()


def test_inputs_that_cannot_be_read_from_disk_are_recorded(
    samples, scrolling_video, monkeypatch
) -> None:
    root, truths = samples
    video_path, _ = scrolling_video
    locked_image, good = root / truths[0].file, root / truths[1].file
    real_load_image, real_hash_file = pipeline_module.load_image, pipeline_module.hash_file

    def load_image(path: Path, layout: Any) -> Any:
        if path == locked_image:
            raise PermissionError(f"Permission denied: {path}")
        return real_load_image(path, layout)

    def hash_file(path: Path) -> str:
        if path == video_path:
            raise FileNotFoundError(f"No such file: {path}")  # removed after expand_inputs
        return real_hash_file(path)

    monkeypatch.setattr(pipeline_module, "load_image", load_image)
    monkeypatch.setattr(pipeline_module, "hash_file", hash_file)
    result = Pipeline(shop_profile, model=FunctionModel(_fixed_answer)).run(
        [locked_image, video_path, good]
    )

    image_error, video_error, ok = result.observations
    assert image_error.status == "error" and "Permission denied" in (image_error.error or "")
    assert video_error.status == "error" and video_error.source.sha256 is None
    assert ok.status == "ok"
    assert [item.key for item in result.items] == ["a"]


def test_a_video_that_breaks_off_keeps_the_frames_read_before(scrolling_video, monkeypatch) -> None:
    path, _ = scrolling_video
    real_sample_video = pipeline_module.sample_video

    def breaks_after_one_frame(*args: Any, **kwargs: Any) -> Iterator[Any]:
        yield next(iter(real_sample_video(*args, **kwargs)))
        raise ValueError(f"Not a readable video: {path} (truncated)")

    monkeypatch.setattr(pipeline_module, "sample_video", breaks_after_one_frame)
    result = Pipeline(shop_profile, model=FunctionModel(_fixed_answer)).run(path)

    read, broken = result.observations
    assert read.status == "ok" and _frame(read).index == 0
    assert broken.status == "error" and broken.frame is None
    assert "truncated" in (broken.error or "")
    assert [item.votes for item in result.items] == [1]


def test_items_read_before_a_model_error_are_still_written(samples, tmp_path: Path) -> None:
    root, truths = samples
    calls: list[int] = []

    def fails_on_the_second_image(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("provider refused the request")
        return _fixed_answer(messages, info)

    (tmp_path / "items.jsonl").write_text("left by an earlier run\n")
    model = FunctionModel(fails_on_the_second_image)
    pipeline = Pipeline(shop_profile, model=model, out_dir=tmp_path)
    with pytest.raises(RuntimeError, match="provider refused"):
        pipeline.run([root / truths[0].file, root / truths[1].file])

    lines = (tmp_path / "items.jsonl").read_text().splitlines()
    assert [Item.model_validate_json(line).key for line in lines] == ["a"]
    assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("name", ["a", "tab\tname.png", "shot.png"])
def test_anonymized_errors_do_not_quote_the_input_path(tmp_path: Path, name: str) -> None:
    broken = tmp_path / "private" / name
    broken.parent.mkdir()
    broken.write_bytes(b"not an image")
    missing = tmp_path / "private" / "gone.png"

    pipeline = Pipeline(shop_profile, model=FunctionModel(_fixed_answer), anonymize_paths=True)
    images = list(pipeline._images([broken, missing]))  # missing: removed after expansion
    result = pipeline.run(broken)

    (observation,) = result.observations
    assert observation.source.path is None
    # The reason is kept whole, even for a one-letter name that a text replacement would mangle.
    assert (observation.error or "").startswith("Not a readable image (cannot identify image file")
    assert "private" not in observation.to_json()
    gone = images[1]
    assert isinstance(gone, pipeline_module._Unreadable)
    assert gone.error.without_path == "Not a readable image (No such file or directory)"


def test_anonymized_video_errors_do_not_quote_the_input_path(
    scrolling_video, tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    broken = private / "secret.mp4"
    broken.write_bytes(b"not a video")
    breaks_off = private / "recording.mp4"
    breaks_off.write_bytes(scrolling_video[0].read_bytes())
    av = pytest.importorskip("av")
    real_to_sample = video_module._to_sample
    samples: list[int] = []

    def breaks_after_the_first_screen(frame: Any, time: float) -> Any:
        samples.append(1)
        if len(samples) > 10:  # past the first still screen (1.5 s at 4 samples a second)
            # FFmpeg's message quotes the file name, as for a real truncated recording.
            raise av.error.FFmpegError(1094995529, "Invalid data", str(breaks_off))
        return real_to_sample(frame, time)

    monkeypatch.setattr(video_module, "_to_sample", breaks_after_the_first_screen)
    pipeline = Pipeline(shop_profile, model=FunctionModel(_fixed_answer), anonymize_paths=True)
    result = pipeline.run([broken, breaks_off])

    errors = [o for o in result.observations if o.status == "error"]
    assert len(errors) == 2
    assert any(o.status == "ok" for o in result.observations)  # read before it broke off
    assert all("private" not in o.to_json() for o in result.observations)
    assert [o.error for o in errors] == [
        "Not a readable video (Invalid data found when processing input)",
        "Not a readable video (Invalid data)",
    ]


def test_a_failed_items_write_does_not_hide_the_error_that_stopped_the_run(
    samples, tmp_path: Path, monkeypatch
) -> None:
    root, truths = samples

    def refuses(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("provider refused the request")

    def disk_full(*args: Any) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(Pipeline, "_write_items", disk_full)
    pipeline = Pipeline(shop_profile, model=FunctionModel(refuses), out_dir=tmp_path)

    with pytest.raises(RuntimeError, match="provider refused") as excinfo:
        pipeline.run(root / truths[0].file)
    assert any("No space left" in note for note in excinfo.value.__notes__)


def test_a_video_is_closed_when_an_error_stops_the_run(scrolling_video, monkeypatch) -> None:
    path, _ = scrolling_video
    real_sample_video = pipeline_module.sample_video
    closed: list[bool] = []

    def tracked(*args: Any, **kwargs: Any) -> Iterator[Any]:
        try:
            yield from real_sample_video(*args, **kwargs)
        finally:
            closed.append(True)

    def refuses(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("provider refused the request")

    monkeypatch.setattr(pipeline_module, "sample_video", tracked)
    with pytest.raises(RuntimeError):
        Pipeline(shop_profile, model=FunctionModel(refuses)).run(path)

    assert closed == [True]


def _sent_images(model_calls: list[Image.Image]):
    """A model that keeps every image it is sent and gives a fixed valid answer."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in getattr(message, "parts", ()):
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    for item in part.content:
                        if isinstance(item, BinaryContent):
                            model_calls.append(Image.open(io.BytesIO(item.data)).convert("RGB"))
        return _fixed_answer(messages, info)

    return FunctionModel(respond)


def test_the_sent_image_is_redacted_and_cropped(samples) -> None:
    root, truths = samples
    profile = replace(fantasy_shop_profile, redact={"portrait": [(0.0, 0.0, 1.0, 0.05)]})
    sent: list[Image.Image] = []

    result = Pipeline(profile, model=_sent_images(sent)).run(root / truths[0].file)  # portrait

    (observation,) = result.observations
    assert observation.crop is not None
    assert observation.crop.orientation == "portrait"
    assert observation.crop.bbox == (0.0, 0.0, 1.0, 0.6)
    assert observation.crop.size == (720, 1280)
    assert observation.crop.redact == [(0.0, 0.0, 1.0, 0.05)]
    (image,) = sent
    assert image.size == (720, 768)  # 60% of 1280
    assert image.getpixel((360, 10)) == (0, 0, 0)  # the redacted top 5%
    assert _frame(observation).sha256 != observation.source.sha256


def test_the_example_crop_keeps_every_row_of_the_synthetic_screens(samples) -> None:
    root, truths = samples
    sent: list[Image.Image] = []

    Pipeline(fantasy_shop_profile, model=_sent_images(sent)).run(root)

    # The crop must not cut a row: the bottom of the sent image is the plain background.
    for image, truth in zip(sent, truths, strict=True):
        width, height = image.size
        last_row_bottom = 150 + (len(truth.items) - 1) * 84 + 72
        assert last_row_bottom < height
        assert image.getpixel((width // 2, height - 1)) == (34, 30, 46)


def test_a_screen_without_a_crop_box_is_an_error_and_the_run_goes_on(samples) -> None:
    root, _ = samples  # shop-000 and shop-002 are portrait, shop-001 landscape
    profile = replace(fantasy_shop_profile, locate={"portrait": (0.0, 0.0, 1.0, 0.6)})
    sent: list[Image.Image] = []

    result = Pipeline(profile, model=_sent_images(sent), anonymize_paths=True).run(root)

    assert [o.status for o in result.observations] == ["ok", "error", "ok"]
    skipped = result.observations[1]
    assert skipped.frame is None
    assert skipped.crop is not None
    assert skipped.crop.orientation == "landscape" and skipped.crop.bbox is None
    assert skipped.crop.size == (1280, 720)
    assert skipped.source.sha256 is not None
    assert skipped.error == "Profile.locate has no crop box for landscape screens"
    assert len(sent) == 2


def test_video_frames_are_cropped(scrolling_video) -> None:
    path, sample = scrolling_video
    sent: list[Image.Image] = []

    result = Pipeline(fantasy_shop_profile, model=_sent_images(sent)).run(path)

    assert len(result.succeeded) == len(sample.holds)
    assert all(image.size == (1280, 648) for image in sent)  # 90% of 720, above the footer
    assert all(_frame(o).index == i for i, o in enumerate(result.observations))


def test_video_frames_without_a_crop_box_are_errors_and_the_video_goes_on(scrolling_video) -> None:
    path, sample = scrolling_video  # landscape
    profile = replace(fantasy_shop_profile, locate={"portrait": (0.0, 0.0, 1.0, 0.6)})
    sent: list[Image.Image] = []

    result = Pipeline(profile, model=_sent_images(sent)).run(path)

    assert len(result.observations) == len(sample.holds)
    assert all(o.status == "error" and o.frame is None for o in result.observations)
    assert all(o.crop is not None and o.crop.bbox is None for o in result.observations)
    assert {o.crop.orientation for o in result.observations if o.crop} == {"landscape"}
    video_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(o.source.sha256 == video_sha for o in result.observations)
    assert sent == []
