# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/); during 0.x, minor versions may contain breaking changes.

## [Unreleased]

## [0.3.0] - 2026-10-17

### Added

- `Profile.locate` and `Profile.redact`: send only a box of each screen orientation, with areas
  painted black first; boxes are fractions of the upright screen (display rotation for videos,
  EXIF orientation for images)
- `crop` in observations, recording the orientation, the screen size in pixels, the box cropped
  and the areas redacted
- `framelift frames --profile` writes the images a run would send, and `frames` accepts images
- Evaluation crops and redacts like a run
- `Profile.screen_description`: each image is first checked against it, on the cropped and
  redacted image shrunk to 512 pixels, and an image of another screen is recorded as
  `status: "skipped"` without extraction; `Pipeline(screen_model=...)` and `run --screen-model`
  choose the model for the check
- `screen` in observations (the check's answer, model, usage and error) and `RunResult.skipped`;
  `framelift run` reports skipped images
- A cache of extractions and screen checks, off unless asked for: `Pipeline(cache_dir=...)`,
  `run_evaluation(cache_dir=...)`, `framelift run --cache DIR` and `framelift eval --cache DIR`;
  keyed by the image sent, the model string and `Profile.fingerprint` (for a screen check, the
  shrunk image, the model string, the description and `Profile.version`), for models given as
  strings
- `cached` in observations and screen checks, and `RunResult.cached`

### Changed

- Breaking: `Profile.locate` is required; pass `locate="full"` to keep sending the whole screen
- Breaking: observations are `framelift/observation/v3` (adds `crop`, `screen` and `cached`, and
  the `skipped` status)
- A screen whose orientation has no crop box is recorded as an error observation and not sent
- `Profile.fingerprint` gives other values than in 0.2: it now covers `locate` and `redact`, and
  non-ASCII text is hashed in escaped form

## [0.2.0] - 2026-10-17

### Added

- Video input (`framelift[video]`): videos are sampled (`--frame-rate`), split into still screens by
  the share of the screen that changed (`--motion-threshold`), and the sharpest frame of each still
  screen is read
- `items.jsonl` and `RunResult.items`: list items merged across frames by `Profile.item_key`, by
  majority vote, with ties left null and disagreements kept in `conflicts` (`framelift/item/v1`)
- `Item` public record
- `framelift frames` to see which video frames would be sent, without calling a model
- `framelift synth --videos N` for synthetic scrolling videos

### Changed

- Breaking: observations are `framelift/observation/v2`; the perceptual hash moved from `source` to
  the new `frame` (`index`, `timestamp_ms`, `sha256`, `phash`), which describes the image sent
- An unreadable image or video (including one that cannot be opened) no longer stops
  `Pipeline.run`: it is recorded as an error observation with `frame: null`, and frames read before
  a video broke off are kept
- numpy is now a direct dependency

## [0.1.0] - 2026-10-16

### Added

- Evaluation harness on `pydantic-evals` (`framelift[eval]`): `framelift eval` scores a profile and
  a model against a `framelift/dataset/v1` dataset (validation pass rate, field accuracy, list-item
  precision and recall, tokens, latency, retries)
- `Profile.item_key` to match list items with the expected ones
- `--root` option to keep dataset images outside the repository

### Changed

- Breaking: `framelift synth` writes `dataset.yaml` (`framelift/dataset/v1`) instead of
  `truth.jsonl`
- PyYAML is now a dependency

## [0.0.1] - 2026-10-16

### Added

- `Profile` and `Pipeline` public API
- Image input (files, directories, glob patterns) with SHA-256 and perceptual hash
- Structured extraction with Pydantic AI, profile validators and retries
- `observations.jsonl` output (`framelift/observation/v1`)
- `framelift run` and `framelift synth` CLI commands
- Synthetic fictional-shop sample generator

[Unreleased]: https://github.com/rysk-tanaka/framelift/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/rysk-tanaka/framelift/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rysk-tanaka/framelift/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rysk-tanaka/framelift/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/rysk-tanaka/framelift/releases/tag/v0.0.1
