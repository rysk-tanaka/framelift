# Videos and merged items

A video is sampled 4 times per second (`--frame-rate`). Samples that barely differ from their
neighbour and from the first sample of their run form a still screen: at most 5% of the screen
may change (`--motion-threshold`, the share of 16x16 tiles of a grayscale copy that changed).
Counting changed tiles catches a page or a list that changes only its text, and ignores a small
animation such as a clock. Only the sharpest sample of each still screen is sent to the model, so
scrolling frames are not read (a video that never stands still is read once, from its sharpest
sample). A screen shown for less than about two samples can be missed, and so can the screens
passed during a slow scroll that never stops: pause on each screen you want read.
Movement is measured on the part of the screen that is sent: the `locate` crop, leaving out the
tiles that `redact` mostly covers, so redacting an animated area, such as a character beside the
list, keeps it from splitting a still screen, and `--motion-threshold` is a share of those tiles.
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
