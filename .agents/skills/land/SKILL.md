---
name: land
description: 作業中の変更を検査し、コミットして origin に push し、プルリクエストを作って CI の結果を確かめる。マージは頼まれたときだけ行う。Delta の Land Changes や /land で、変更を main に取り込みたいときに使う。
disable-model-invocation: true
metadata:
  delta-action: land
---

# land

作業中の変更を、検査を通したうえで `main` 向けのプルリクエストにする。`main` に入った変更で `pyproject.toml` の `version` が変わっていると、`release.yml` がタグを作って PyPI に公開する。PyPI に公開した版は取り消せないので、版は頼まれたときだけ上げ、マージは頼まれたときだけ確認を取ってから行う。

## 手順

1. 状況を確かめる。`git status` と現在のブランチを見る。`main` にいるときは、変更内容に合ったブランチ（`feat/...`、`fix/...`、`docs/...`、`chore/...` など）を切る。画像・動画（`samples/` と `docs/assets/` 以外）、`.delta/`、`.env` などがコミット対象に入っていないことも確かめる。
2. 依存を揃える。`uv sync --locked --all-extras`
3. 検査する。CI と同じものを、すべて通るまで実行する。

   ```bash
   uv run ruff check
   uv run ruff format --check
   uv run ty check
   uv run pytest
   uv build
   ```

   失敗したら原因を直して検査をやり直す。直し方が変更の意図や公開 API に関わるときは、直さずに止めて報告する。
4. 利用者から見える変更（公開 API、CLI、出力の形式、依存）があれば、`CHANGELOG.md` の Unreleased に追記する。出力やデータセットの形式を変えたときは、スキーマの版の扱いが AGENTS.md の設計上の原則に沿っているか確かめる。
5. コミットする。メッセージは英語の Conventional Commits 形式で、単一行にする（本文や HEREDOC は使わない）。
6. push する。`git push -u origin <ブランチ名>`
7. プルリクエストを作る。`gh pr create --base main` で、タイトルは専門用語を避けた簡潔な日本語にし、本文は `.github/pull_request_template.md` の見出しに沿って、ですます調で書く。
8. CI を待つ。`gh pr checks --watch` で結果を確かめる。失敗したら手順3に戻って直し、push し直す。
9. マージは、このやり取りでマージまで頼まれているときだけ行う。
   - `pyproject.toml` の `version` が変わっているときは、新しい版と、`CHANGELOG.md` のその版の節を示し、PyPI に公開されることを伝えて確認を取ってからマージする
   - `gh pr merge --squash --delete-branch`

## 報告

最後に次をまとめて報告する。

- 取り込めたかどうか（プルリクエストを作った段階か、マージまで済んだか）
- 対象のブランチと、最後のコミット
- プルリクエストの URL
- CI の結果
- リリースになる場合は、その版
