# Screen check

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
