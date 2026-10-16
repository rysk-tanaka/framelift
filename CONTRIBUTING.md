# Contributing

Thanks for your interest in framelift.

- **Issues** — bug reports and feature requests are welcome. Please include the framelift version,
  Python version, the model you used, and (if possible) a minimal profile that reproduces the problem.
  Please do not attach screenshots that contain other people's personal information.
- **Pull requests** — welcome, but please open an issue first for anything beyond a small fix so we can
  agree on the approach. The public API is intentionally small during 0.x.
- **Response time** — this is a personal project maintained in spare time, so responses are irregular.

## Development setup

```bash
uv sync --all-extras
uvx pre-commit install      # optional
uv run ruff check && uv run ruff format --check
uv run ty check
uv run pytest
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `chore:` …); the changelog is generated from them.
