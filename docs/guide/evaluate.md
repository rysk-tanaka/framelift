# Evaluate

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

`synth` lists what it writes in `synth.sha256`. Run again in the same directory, it removes the
files listed there that it did not write again, unless they were changed since; with
`--videos 0` (the default) the earlier videos are kept. Files it never listed are left alone,
apart from those it overwrites by name (`shop-000.png` and so on).

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
