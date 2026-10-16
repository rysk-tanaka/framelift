# framelift

Extract structured data from screen recordings and screenshots with vision LLMs.

You describe **what** to read as a Pydantic model and **how** to read it as a `Profile`;
framelift turns images into validated, provenance-tagged JSON Lines.

> **Status:** pre-alpha (0.x). The public API is limited to `Profile` and `Pipeline`
> and may change between minor versions.

## Features (0.1)

- Image input (files, directories, glob patterns), each tagged with its SHA-256 and perceptual hash
- Structured extraction with [Pydantic AI](https://ai.pydantic.dev/): your model is the output type,
  your validators trigger automatic retries
- Model-agnostic: pass any `provider:model` string supported by Pydantic AI
- `observations.jsonl` output with source, usage and retry information per image
- A synthetic sample generator (a fictional shop screen) for demos and tests

Planned: video input with frame sampling and de-duplication (0.2), cross-frame aggregation (0.2),
screen classification, cropping and redaction (0.3), and an evaluation harness built on
`pydantic-evals`.

## Install

```bash
pip install "framelift[anthropic]"   # or [openai], [google]
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
)

Pipeline(shop, model="anthropic:<model-name>", out_dir="out").run(["shots/*.png"])
```

Or from the command line:

```bash
framelift synth --out samples --n 3                       # generate fictional shop screenshots
framelift run "samples/*.png" --profile examples.fantasy_shop:profile \
  --model anthropic:<model-name> --out out
```

Each line of `out/observations.jsonl` looks like:

```json
{"schema": "framelift/observation/v1",
 "profile": {"name": "fantasy-shop", "version": "1"},
 "source": {"path": "samples/shop-000.png", "sha256": "…", "phash": "…"},
 "data": {"shop_name": "…", "items": [{"name": "…", "price": 120}]},
 "model": "anthropic:<model-name>",
 "usage": {"input_tokens": 1534, "output_tokens": 96, "requests": 1, "latency_ms": 2104},
 "retries": 0}
```

## Privacy

framelift runs locally and only sends the images you pass to the model provider you choose.
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
