# Output

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
