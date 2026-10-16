"""Example profile for the fictional shop screens produced by ``framelift synth``.

Usage:
    framelift synth --out samples --n 3
    framelift run "samples/*.png" --profile examples.fantasy_shop:profile \
        --model anthropic:<model-name> --out out
    framelift eval samples/dataset.yaml --profile examples.fantasy_shop:profile \
        --model anthropic:<model-name>
"""

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from framelift import Profile


class ShopItem(BaseModel):
    name: str = Field(description="Item name exactly as displayed")
    price: int = Field(description="Price as an integer, without the 'G' suffix", gt=0)


class ShopScreen(BaseModel):
    shop_name: str = Field(description="Shop name shown in the header")
    items: list[ShopItem] = Field(description="Every item row, top to bottom")


def items_are_unique(screen: ShopScreen) -> None:
    names = [item.name for item in screen.items]
    if not names:
        raise ModelRetry("No items were read; read every item row in the list.")
    if len(names) != len(set(names)):
        raise ModelRetry("Some item names are duplicated; read each row exactly once.")


profile = Profile(
    name="fantasy-shop",
    version="1",
    output_type=ShopScreen,
    instructions=(
        "The screenshot shows a shop's item list. Read the shop name and every item row, "
        "keeping names exactly as displayed."
    ),
    validators=[items_are_unique],
    item_key=lambda item: item.name,
)
