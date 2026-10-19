# What is sent

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
