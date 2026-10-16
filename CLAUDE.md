# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

framelift は、画面収録やスクリーンショットから、利用者が定義したスキーマどおりの構造化データを画像対応の LLM で取り出す Python ライブラリと CLI。利用者は「何を読み取るか（Pydantic モデル）」と「どう読むか（指示と検証）」を `Profile` として渡す。特定のゲームやアプリには依存しない。

最初の利用者は作者の別プロジェクト（非公開）。0.x の間は、そこで実際に必要になった機能だけを入れる。

## コマンド

```bash
# 依存関係のインストール（extras も含める）
uv sync --all-extras

# リンター・フォーマッター
uv run ruff check
uv run ruff format

# 型チェック（ty）
uv run ty check

# テスト
uv run pytest                          # 全テスト（live マーカーは除外）
uv run pytest tests/test_pipeline.py   # 特定ファイル
uv run pytest -k retry                 # 特定テスト
uv run pytest -m live                  # 実モデルを呼ぶテスト（API キーが必要）

# CLI の動作確認（"test" は Pydantic AI 組み込みの、プロバイダを呼ばないモデル）
uv run framelift synth --out /tmp/samples --n 3
uv run framelift run "/tmp/samples/*.png" --profile examples.fantasy_shop:profile --model test --out /tmp/out

# ビルド
uv build
```

Pydantic AI の起動バナーは `PYDANTIC_AI_NO_BANNER=1` で消せる。

## アーキテクチャ

```text
Pipeline.run(inputs)
  → expand_inputs: ファイル・ディレクトリ・glob を画像パスに展開（何も一致しなければ FileNotFoundError）
  → load_image: 読み込み、sha256 と知覚ハッシュ（pHash）を計算
  → Extractor.extract: Pydantic AI の Agent に output_type と画像（BinaryContent）を渡す
      Profile.validators を output_validator に登録し、ModelRetry で再試行
  → Observation を作る（再試行を使い切ったら status: "error"。1 枚の失敗で全体は止めない）
  → out_dir があれば observations.jsonl に 1 行ずつ追記
```

- `src/framelift/profile.py` — `Profile`（`name`・`version`・`output_type`・`instructions`・`validators`）と `Validator` 型。`fingerprint` は結果に影響する項目のハッシュ
- `src/framelift/pipeline.py` — `Pipeline` と `RunResult`
- `src/framelift/observation.py` — 出力レコード（`framelift/observation/v1`）。`schema` は Python 側では `schema_id`（alias で `schema` として出力）
- `src/framelift/cli.py` — Typer の CLI（`run`・`synth`）。`--profile` は `package.module:attribute` 形式で、カレントディレクトリを `sys.path` に追加して読み込む
- `src/framelift/_internal/` — 内部実装（公開 API ではない）
  - `sources.py` — 入力の展開と画像の読み込み
  - `extract.py` — Pydantic AI による抽出
  - `synth.py` — 架空のショップ画面と正解（`truth.jsonl`）の生成
- `examples/fantasy_shop.py` — 合成サンプル用の Profile の例

## 設計上の原則

- 公開 API は `framelift/__init__.py` の `__all__`（`Profile`・`Pipeline`・`Observation`・`RunResult`・`Validator`）だけ。`_internal` 配下は予告なく変えてよい。利用者から要望が出るまで、パイプラインの各段を公開しない
- `Profile` の項目は、それを使う機能と同じ版で足す。先回りして拡張点を作らない
- 出力の JSON 形式を変えるときは、スキーマの版（`framelift/observation/v1` の `v1`）を上げる
- ゲームやアプリの実在する名称・画像をリポジトリに入れない。例・テスト・ベンチマークは `synth` の架空の画面で作る
- 画像と動画はコミットしない。`.gitignore` で一括除外し、`samples/` と `docs/assets/` だけを許可している
- 外部に送るのは利用者が渡した画像だけ。テレメトリは持たない

## 今後の予定

設計の詳細と判断の理由は [docs/README.md](docs/README.md) から辿れる。

- 0.1 — 評価ハーネス（`pydantic-evals` 上に、画像と正解のデータセット形式、項目単位の評価器）
- 0.2 — 動画入力（PyAV）、一定間隔の抽出と pHash による重複除去、`item_key` による複数フレームの集約と `items.jsonl`、合成動画
- 0.3 — 画面判定、切り出し（`full`・`fixed`・`model`）とマスク、抽出結果のキャッシュ、設定の総当たり比較
- 1.0 — 公開 API の確定

## コーディング規約

### Python

- Python 3.12+。ジェネリクスは PEP 695 の構文（`class Profile[OutputT: BaseModel]`、`type X = ...`）
- 各モジュールの先頭に `from __future__ import annotations`。型注釈だけで使う import は `TYPE_CHECKING` ブロックに置く（ruff の `TC`。tests は対象外）
- パス操作は `pathlib`（ruff の `PTH`）
- 行幅 100。ruff のルールは `pyproject.toml` の `[tool.ruff.lint]` を参照
- 公開クラスと公開関数には docstring を付ける（英語）

### テスト

- 実モデルは呼ばない。`tests/conftest.py` で `models.ALLOW_MODEL_REQUESTS = False` にしている
- モデルの応答は `FunctionModel` で組み立てる。出力はツール呼び出しなので、`ToolCallPart(info.output_tools[0].name, {...})` で返す
- 再試行の確認は、1 回目に検証を通らない値、2 回目に正しい値を返す `FunctionModel` で行う
- CLI のテストは `--model test`（Pydantic AI 組み込みのテスト用モデル）と `typer.testing.CliRunner` を使う
- 画像は `samples` フィクスチャ（`synth.generate`）で作る。テスト用の画像ファイルを置かない
- 実モデルを使うテストには `@pytest.mark.live` を付ける（既定では実行されない）

### ドキュメント

- README は英語（OSS として読まれる範囲を広げるため）。CLAUDE.md は日本語
- Markdown の箇条書きの前にコロンを使わない（「以下の項目:」→「以下の項目。」）。太字は使わない

## コミット規約

- 英語で Conventional Commits 形式（`feat:`, `fix:`, `docs:`, `chore:` 等）。CHANGELOG はこれをもとに書く
- コミットメッセージは単一行にする（マルチライン・HEREDOC 不可）

## CI/CD

- `ci.yml` — push（main）と PR で実行
  - `lint` — ruff check、ruff format --check、ty check
  - `test` — Python 3.12・3.13・3.14 のマトリクスで pytest
  - `build` — wheel をビルドし、クリーンな環境にインストールして CLI を実行
- `release.yml` — main の `pyproject.toml` が変わったときに実行。共通ワークフロー（`rysk-tanaka/workflows` の `release-on-version-change.yml`）で版が変わっていれば `v<version>` のタグと GitHub Release を作り、テスト・ビルドの後、Trusted Publishing で PyPI に公開する。`GITHUB_TOKEN` で作ったタグは別のワークフローを起動しないので、タグ作成と公開は同じファイルに置く。`id-token: write` は publish ジョブだけに付ける。GitHub の `pypi` environment を使う。公開に失敗したときは `workflow_dispatch` で再実行する

リリース手順。

1. `pyproject.toml` の `version` を上げる（`uv version --bump minor` など）
2. `CHANGELOG.md` の Unreleased を新しい版の見出しに移す
3. コミットして main に入れる（タグは `release.yml` が作る）

## 依存関係の自動更新

Renovate で uv（`uv.lock`）と GitHub Actions の依存を更新する（GitHub Actions はコミットハッシュに固定）。

- ty はベータで、更新で診断が変わりうるため、`pyproject.toml` でバージョンを固定し、個別の PR で確認してから上げる
- `pydantic-ai-slim` と `pydantic-evals` は 1 本の PR にまとめる
- pre-commit のフックは `uv run` で uv.lock のバージョンを使うので、フック側のバージョン管理は不要
