# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/); during 0.x, minor versions may contain breaking changes.

## [Unreleased]

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

[Unreleased]: https://github.com/rysk-tanaka/framelift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rysk-tanaka/framelift/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/rysk-tanaka/framelift/releases/tag/v0.0.1
