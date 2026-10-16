"""Synthetic sample generator: screenshots of a fictional shop with known ground truth.

The shop, item names and prices are invented, so the images can be shared, used in tests and
used as a public benchmark without any real game or app content.
"""

from __future__ import annotations

import hashlib
import io
import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from PIL import Image, ImageDraw, ImageFont

from framelift._internal.dataset import DatasetCase, write_dataset

if TYPE_CHECKING:
    from pathlib import Path

Orientation = Literal["portrait", "landscape"]

_SHOP_PREFIXES = ["Moonlit", "Copper", "Wandering", "Silverleaf", "Lantern", "Hollow", "Amber"]
_SHOP_SUFFIXES = ["Emporium", "Bazaar", "Trading Post", "Market", "Workshop", "Pantry"]
_ITEM_ADJECTIVES = ["Moon", "Ember", "Frost", "Dew", "Iron", "Velvet", "Starlit", "Mossy", "Sun"]
_ITEM_NOUNS = ["Herb", "Ingot", "Thread", "Lantern", "Tonic", "Feather", "Pebble", "Bread", "Rope"]

_SIZES: dict[Orientation, tuple[int, int]] = {"portrait": (720, 1280), "landscape": (1280, 720)}
_HEADER_HEIGHT, _LIST_TOP, _ROW_HEIGHT = 110, 150, 84
_MIN_ITEMS, _MAX_ITEMS = 4, 7


def _max_rows(orientation: Orientation) -> int:
    height = _SIZES[orientation][1]
    return min(_MAX_ITEMS, (height - _LIST_TOP) // _ROW_HEIGHT)


@dataclass(frozen=True)
class SampleItem:
    name: str
    price: int


@dataclass(frozen=True)
class SampleTruth:
    file: str
    orientation: Orientation
    shop_name: str
    items: list[SampleItem]


def _random_truth(rng: random.Random, file: str, orientation: Orientation) -> SampleTruth:
    shop = f"{rng.choice(_SHOP_PREFIXES)} {rng.choice(_SHOP_SUFFIXES)}"
    names: set[str] = set()
    count = rng.randint(_MIN_ITEMS, _max_rows(orientation))
    while len(names) < count:
        names.add(f"{rng.choice(_ITEM_ADJECTIVES)} {rng.choice(_ITEM_NOUNS)}")
    items = [SampleItem(name=n, price=rng.randrange(10, 2000, 5)) for n in sorted(names)]
    return SampleTruth(file=file, orientation=orientation, shop_name=shop, items=items)


def render(truth: SampleTruth) -> Image.Image:
    """Draw a simple shop screen for ``truth``."""
    width, height = _SIZES[truth.orientation]
    image = Image.new("RGB", (width, height), (34, 30, 46))
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=40)
    body = ImageFont.load_default(size=30)

    draw.rectangle((0, 0, width, _HEADER_HEIGHT), fill=(70, 52, 96))
    draw.text((32, 34), truth.shop_name, font=title, fill=(250, 236, 200))

    for index, item in enumerate(truth.items):
        y = _LIST_TOP + index * _ROW_HEIGHT
        fill = (52, 46, 70) if index % 2 == 0 else (44, 39, 60)
        draw.rounded_rectangle((24, y, width - 24, y + _ROW_HEIGHT - 12), radius=12, fill=fill)
        draw.text((48, y + 20), item.name, font=body, fill=(235, 235, 245))
        price = f"{item.price} G"
        price_width = draw.textlength(price, font=body)
        draw.text((width - 48 - price_width, y + 20), price, font=body, fill=(255, 214, 102))
    return image


DATASET_FILE = "dataset.yaml"


def _to_case(truth: SampleTruth, data: bytes) -> DatasetCase:
    width, height = _SIZES[truth.orientation]
    return DatasetCase(
        name=truth.file.removesuffix(".png"),
        image=truth.file,
        sha256=hashlib.sha256(data).hexdigest(),
        expected={"shop_name": truth.shop_name, "items": [asdict(i) for i in truth.items]},
        metadata={"orientation": truth.orientation, "width": width, "height": height},
    )


def generate(out_dir: Path, *, n: int, seed: int = 0) -> list[SampleTruth]:
    """Write ``n`` PNG screenshots and a ``dataset.yaml`` with their ground truth."""
    if n < 1:
        raise ValueError("n must be >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    truths: list[SampleTruth] = []
    cases: list[DatasetCase] = []
    for index in range(n):
        orientation: Orientation = "portrait" if index % 2 == 0 else "landscape"
        truth = _random_truth(rng, f"shop-{index:03d}.png", orientation)
        buffer = io.BytesIO()
        render(truth).save(buffer, format="PNG")
        data = buffer.getvalue()
        (out_dir / truth.file).write_bytes(data)
        truths.append(truth)
        cases.append(_to_case(truth, data))
    write_dataset(out_dir / DATASET_FILE, cases)
    return truths
