# framelift

Extract structured data from screen recordings and screenshots with vision LLMs.

You describe **what** to read as a Pydantic model and **how** to read it as a `Profile`;
framelift turns screenshots and screen recordings into validated, provenance-tagged JSON Lines.

> **Status:** pre-alpha (0.x). The public API is limited to `Profile` and `Pipeline`
> and may change between minor versions.

## Features (0.2)

- Image and video input (files, directories, glob patterns), each tagged with its SHA-256
- Videos are read once per still screen: frames are sampled, scrolling frames are skipped, and the
  sharpest frame of each still screen is sent
- Structured extraction with [Pydantic AI](https://ai.pydantic.dev/): your model is the output type,
  your validators trigger automatic retries
- Model-agnostic: pass any `provider:model` string supported by Pydantic AI
- `observations.jsonl` output with source, frame, usage and retry information per image or frame
- `items.jsonl`: list items seen in several frames merged by majority vote, with disagreements kept
- An evaluation harness built on [pydantic-evals](https://ai.pydantic.dev/evals/): score a profile
  and a model against images with expected outputs
- A synthetic sample generator (a fictional shop screen, still or scrolling) for demos and tests

Planned: screen classification, cropping and redaction, caching, and side-by-side comparison of
models and prompts (0.3).

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
{"schema": "framelift/observation/v2",
 "status": "ok",
 "profile": {"name": "fantasy-shop", "version": "1"},
 "source": {"path": "samples/shop-scroll-000.mp4", "sha256": "…"},
 "frame": {"index": 1, "timestamp_ms": 3000, "sha256": "…", "phash": "…"},
 "data": {"shop_name": "…", "items": [{"name": "…", "price": 120}]},
 "error": null,
 "model": "anthropic:<model-name>",
 "usage": {"input_tokens": 1534, "output_tokens": 96, "requests": 1, "latency_ms": 2104},
 "retries": 0}
```

For a video, `frame.timestamp_ms` counts from the start of the stream. For an image,
`frame.index` is 0, `frame.timestamp_ms` is null and `frame.sha256` is the image's.
An input that cannot be read, or a video that breaks off partway, is recorded as an observation
with `"status": "error"` and `"frame": null` (and `"source.sha256": null` when the file itself
cannot be opened), and the run goes on with the next input.

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

## Evaluate

Install the `eval` extra, then score a profile and a model against a dataset of images with
expected outputs. `framelift synth` writes such a dataset (`dataset.yaml`) next to its images:

```bash
pip install "framelift[eval,anthropic]"
framelift synth --out samples --n 20
framelift eval samples/dataset.yaml --profile examples.fantasy_shop:profile \
  --model anthropic:<model-name>
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

## Privacy

framelift runs locally and only sends the images you pass, and the frames it picks from your
videos, to the model provider you choose.
It has no telemetry. Cropping and redaction before sending are planned for 0.3.

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
