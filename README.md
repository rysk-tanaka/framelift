# framelift

Extract structured data from screen recordings and screenshots with vision LLMs.

You describe **what** to read as a Pydantic model and **how** to read it as a `Profile`;
framelift turns screenshots and screen recordings into validated, provenance-tagged JSON Lines.

> **Status:** pre-alpha (0.x). The public API is limited to `Profile` and `Pipeline`
> and may change between minor versions.

## Features (0.3)

- Image and video input (files, directories, glob patterns), each tagged with its SHA-256
- Videos are read once per still screen: frames are sampled, scrolling frames are skipped, and the
  sharpest frame of each still screen is sent
- Structured extraction with [Pydantic AI](https://ai.pydantic.dev/): your model is the output type,
  your validators trigger automatic retries
- Model-agnostic: pass any `provider:model` string supported by Pydantic AI
- Only the part of the screen you choose is sent: crop each orientation to a box and paint over
  areas such as a chat window before anything leaves your machine
- An optional screen check: images of other screens are skipped before extraction
- An optional cache, so a run or an evaluation repeated on the same images is not paid for again
- `observations.jsonl` output with source, frame, crop, usage and retry information per image or
  frame
- `items.jsonl`: list items seen in several frames merged by majority vote, with disagreements kept
- An evaluation harness built on [pydantic-evals](https://ai.pydantic.dev/evals/): score a profile
  and a model against images with expected outputs
- A synthetic sample generator (a fictional shop screen, still or scrolling) for demos and tests

## Install

```bash
pip install "framelift[anthropic]"   # or [openai], [google]
pip install "framelift[video,anthropic]"   # to read screen recordings too
```

## Quick start

```python
from pydantic import BaseModel, Field
from framelift import Pipeline, Profile


class ShopItem(BaseModel):
    name: str = Field(description="Item name as displayed")
    price: int = Field(description="Price as an integer", gt=0)


class ShopScreen(BaseModel):
    shop_name: str
    items: list[ShopItem]


shop = Profile(
    name="fantasy-shop",
    version="1",
    output_type=ShopScreen,
    instructions="Read the shop's item list exactly as displayed.",
    # Required: what to send. "full" sends the whole screen; boxes are (left, top, right, bottom)
    # as fractions of the upright screen, per orientation.
    locate={"portrait": (0.0, 0.0, 1.0, 0.6), "landscape": (0.0, 0.0, 1.0, 0.9)},
    redact=[(0.0, 0.0, 0.3, 0.05)],  # painted black before cropping, e.g. a player name
    item_key=lambda item: item.name,  # merges the same item across frames
)

Pipeline(shop, model="anthropic:<model-name>", out_dir="out").run(["shots/*.png", "scroll.mp4"])
```

Or from the command line:

```bash
framelift synth --out samples --n 3 --videos 1   # fictional shop screenshots and a scrolling video
framelift run samples --profile examples.fantasy_shop:profile \
  --model anthropic:<model-name> --out out
```

Each line of `out/observations.jsonl` is one image or video frame:

```json
{"schema": "framelift/observation/v3",
 "status": "ok",
 "profile": {"name": "fantasy-shop", "version": "1"},
 "source": {"path": "samples/shop-scroll-000.mp4", "sha256": "…"},
 "frame": {"index": 1, "timestamp_ms": 3000, "sha256": "…", "phash": "…"},
 "crop": {"orientation": "landscape", "size": [1280, 720], "bbox": [0.0, 0.0, 1.0, 0.9], "redact": []},
 "screen": null,
 "cached": false,
 "data": {"shop_name": "…", "items": [{"name": "…", "price": 120}]},
 "error": null,
 "model": "anthropic:<model-name>",
 "usage": {"input_tokens": 1534, "output_tokens": 96, "requests": 1, "latency_ms": 2104},
 "retries": 0}
```

For a video, `frame.timestamp_ms` counts from the start of the stream. For an image,
`frame.index` is 0 and `frame.timestamp_ms` is null. `frame.sha256` is the hash of the image sent:
the image file's own hash only when it is sent untouched (`locate="full"` and no `redact`).
An input that cannot be read, or a video that breaks off partway, is recorded as an observation
with `"status": "error"` and `"frame": null` (and `"source.sha256": null` when the file itself
cannot be opened), and the run goes on with the next input. So is a screen whose orientation has no
box in `locate`; its `crop` gives the orientation, so you can add a box for it.

## Videos and merged items

A video is sampled 4 times per second (`--frame-rate`). Samples that barely differ from their
neighbour and from the first sample of their run form a still screen: at most 5% of the screen
may change (`--motion-threshold`, the share of 16x16 tiles of a grayscale copy that changed).
Counting changed tiles catches a page or a list that changes only its text, and ignores a small
animation such as a clock. Only the sharpest sample of each still screen is sent to the model, so
scrolling frames are not read (a video that never stands still is read once, from its sharpest
sample). A screen shown for less than about two samples can be missed, and so can the screens
passed during a slow scroll that never stops: pause on each screen you want read.
To see which frames would be sent, without calling a model:

```bash
framelift frames samples/shop-scroll-000.mp4 --out frames
```

With `Profile.item_key`, items of the output's list field are merged across all frames of a run
into `out/items.jsonl`:

```json
{"schema": "framelift/item/v1", "key": "Moon Herb",
 "data": {"name": "Moon Herb", "price": 120}, "votes": 3,
 "conflicts": [{"field": "price", "values": [{"value": 120, "votes": 2}, {"value": 12, "votes": 1}]}],
 "evidence": [{"source_sha256": "…", "frame_index": 0, "timestamp_ms": 800},
              {"source_sha256": "…", "frame_index": 1, "timestamp_ms": 3000},
              {"source_sha256": "…", "frame_index": 2, "timestamp_ms": 5300}]}
```

Each field takes the value most frames agree on. On a tie the field is null rather than guessed,
and every field the frames disagreed on is listed in `conflicts`. Items are written the way the
output model serializes them, as in `observations.jsonl`. If a model error stops the run, the
items read until then are still written before the error is raised. `Item.data` is plain JSON,
not validated against the output model, so a required field can be null after a tie.

## Screen check

A recording often passes through screens you do not want read, such as menus or loading screens.
Describe the target screen, and each image is first checked against it:

```python
shop = Profile(..., screen_description="a shop's item list with prices")
Pipeline(shop, model="anthropic:<model-name>", screen_model="anthropic:<smaller-model>")
```

The check is sent the same cropped and redacted image as the extraction, shrunk to 512 pixels on
its longest side. An image of another screen is recorded with `"status": "skipped"` and is not
extracted; one the check cannot answer is an error. The check's answer and its token usage are in
the observation's `screen`, apart from the extraction's `usage`. `screen_model` (`--screen-model`
on the command line) defaults to the extraction model. Evaluation does not check screens: its
cases are all target screens.

## Cache

Pass a directory to cache extractions and screen checks in, and images read before are not paid
for again:

```bash
framelift run samples --profile examples.fantasy_shop:profile \
  --model anthropic:<model-name> --out out --cache .framelift-cache
```

(`Pipeline(..., cache_dir=".framelift-cache")` in Python, `framelift eval --cache` for
evaluations.) An extraction is keyed by the image sent, the model string and the profile's
`fingerprint`, and a screen check by the shrunk image checked, the model string, the screen
description and `Profile.version`. So changing the crop, the model, the instructions, the output
model or `Profile.version` reads the image again, and changing the description or the version
checks it again. A framelift release that changes its own prompts does not reuse older entries. Validators are code and
not part of the key: bump `Profile.version` when you change them, and keep them deterministic.
Only successful results are cached, and only for models given as strings: a `Model` instance
cannot be told apart from another with different settings, so it is not cached, and a warning says so.
An answer that needed more retries than the current `max_retries` is not reused.

A cached observation has `"cached": true` and keeps the usage, retries and latency of the run
that paid for it, so evaluation reports still compare models fairly; a screen check taken from the
cache has `"cached": true` in `screen`. The run summary says how many of the extractions came from
the cache. An output that would not come back exactly as it was from its JSON (for
example one with a `Field(exclude=True)` value or a private attribute set) is not cached; fields
excluded from dumps are never written to the cache. Neither is an output with a `set` or
`frozenset` field of two or more items in practice: their order, and so the JSON they dump to,
changes from one process to the next, so the entry a run writes is not found by the next one. Use
a list where the cache matters. Delete the directory to clear it.

## Evaluate

Install the `eval` extra, then score a profile and a model against a dataset of images with
expected outputs. `framelift synth` writes such a dataset (`dataset.yaml`) next to its images:

```bash
pip install "framelift[eval,anthropic]"
framelift synth --out samples --n 20
framelift eval samples/dataset.yaml --profile examples.fantasy_shop:profile \
  --model anthropic:<model-name>
```

To score reading under conditions closer to real screenshots, `synth` can resize the screens
(`--scale`, 0.1-4), add Gaussian noise (`--noise`, 0-64, a standard deviation in 0-255 pixel values)
and save them as JPEG (`--jpeg-quality`, 1-95), in that order. The ground truth stays the same
for the same `--seed`:

```bash
framelift synth --out samples-jpeg --n 20 --scale 0.5 --noise 8 --jpeg-quality 70
```

A dataset lists each image (relative to the dataset root), its SHA-256 and the expected output,
which lists every required field (fields with defaults may be left out):

```yaml
schema: framelift/dataset/v1
cases:
  - name: shop-000
    image: shop-000.png
    sha256: 3dbb5bc8…
    expected:
      shop_name: Amber Market
      items:
        - {name: Dew Rope, price: 1590}
    metadata: {orientation: portrait}
```

Image paths are relative to the dataset file; to keep images outside the repository, pass
`--root DIR`.
Before anything is sent to the model, every image is checked against its SHA-256 and every
expected output against the profile's output model, so a typo in a field name or a value YAML
reads differently (an unquoted `no` becomes `false`) is reported instead of scored as a misread.

The report shows, per case and on average, whether validation passed, field accuracy,
list-item precision and recall (items are matched with `Profile.item_key`, e.g.
`item_key=lambda item: item.name`), tokens, latency and retries.

## What is sent

`Profile.locate` decides what part of each screen is sent, and is required so that sending the
whole screen is a choice: `"full"`, or a box per orientation (`"portrait"` when the upright screen
is taller than wide, `"landscape"` otherwise). Boxes are fractions of the upright screen: a video
frame with its display rotation applied, an image with its EXIF orientation applied.
`Profile.redact` boxes, in the same whole-screen coordinates, are painted opaque black before
cropping. A screen whose orientation has no box is not sent.

Boxes are fractions of the screen and only the orientation tells screens apart, so the same
fractions land elsewhere on a screen of another aspect ratio: a tall phone and a tablet, or a
recording with black bars or a notch. A redaction that misses is not an error; what it should have
hidden is sent. Check what would be sent once for every device you capture on (below), and use a
profile per device if their layouts differ. Each observation's `crop.size` records the screen's
size in pixels, so screens of an unexpected aspect ratio can be found afterwards.

To check the boxes without calling a model, write what would be sent for a screenshot or a video:

```bash
framelift frames samples/shop-000.png --profile examples.fantasy_shop:profile --out frames
```

Evaluation crops and redacts the same way, so scores reflect what a run sends.

## Privacy

framelift runs locally and only sends the cropped and redacted images, from the files you pass
and the frames it picks from your videos, to the model provider you choose.
It has no telemetry. It writes only to the output and cache directories you give it.

## Development

```bash
uv sync --all-extras
uv run ruff check && uv run ruff format --check
uv run ty check
uv run pytest
```

Tests never call a real model (`ALLOW_MODEL_REQUESTS = False`); tests marked `live` are skipped by
default.

## License

MIT
