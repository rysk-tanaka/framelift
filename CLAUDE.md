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
uv run framelift synth --out /tmp/samples --n 3 --videos 1
uv run framelift run /tmp/samples --profile examples.fantasy_shop:profile --model test --out /tmp/out
uv run framelift frames /tmp/samples/shop-scroll-000.mp4 --out /tmp/frames
uv run framelift eval /tmp/samples/dataset.yaml --profile examples.fantasy_shop:profile --model test

# ビルド
uv build
```

Pydantic AI の起動バナーは `PYDANTIC_AI_NO_BANNER=1` で消せる。

## アーキテクチャ

```text
Pipeline.run(inputs)
  → expand_inputs: ファイル・ディレクトリ・glob を画像・動画のパスに展開（実在するパスはそのまま使う。何も一致しなければ FileNotFoundError）
      動画があれば、モデルを呼ぶ前に PyAV を import できるか確かめる（なければ ImportError）
  → 画像: load_image で読み込み、sha256 と知覚ハッシュ（pHash）を計算
    動画: sample_video で一定間隔の標本を取り、縮小画像のタイルのうち変化した割合で静止区間に分け、区間ごとに最も鮮明な 1 枚を PNG にする（frame_to_source）
  → Extractor.observe: Pydantic AI の Agent に output_type と画像（BinaryContent）を渡し、Observation（v2。source と frame）を作る
      Profile.validators を output_validator に登録し、ModelRetry で再試行
      再試行を使い切ったら status: "error"。1 枚の失敗で全体は止めない
      読めない画像・動画（途中で壊れた動画、フレームのない動画を含む）は status: "error"、frame: null の観測にして次へ進む
  → out_dir があれば observations.jsonl に 1 行ずつ追記
  → item_key があれば Aggregator で項目をキーごとに束ねて多数決し、items.jsonl に書く

framelift eval（framelift[eval] が必要）
  → load_dataset: dataset.yaml（framelift/dataset/v1）を読み、全画像の存在・読み込み・sha256 を照合
  → 正解を output_type 全体で検証し JSON 形式に揃え、item_key のキーも計算しておく（必須フィールドは省けない。合わなければ実行前にエラー）
  → pydantic-evals の Dataset を組み、各ケースで load_image と Extractor.observe_async を 1 件ずつ実行（評価のイベントループ上で非同期に動かす）
      出力は Observation と検証済みのモデル。項目のキーはこのモデルから取り、保存した JSON を検証し直さない
      読んだ画像の sha256 をもう一度照合し、照合後に差し替わっていたらそのケースを失敗にする
  → 評価器（検証の通過、フィールド一致率、item_key による適合率・再現率、usage）でレポートを出す
      ケースや評価器が異常終了したら終了コード 1。スコアが低いだけなら 0
```

- `src/framelift/profile.py` — `Profile`（`name`・`version`・`output_type`・`instructions`・`validators`・`item_key`）と `Validator` 型。`fingerprint` は結果に影響する項目のハッシュ。`item_field` は `item_key` を適用する一覧フィールド（トップレベルでただ 1 つの `list[モデル]`。`| None` も可）を返す
- `src/framelift/pipeline.py` — `Pipeline` と `RunResult`
- `src/framelift/observation.py` — 出力レコード（`framelift/observation/v2`）。`schema` は Python 側では `schema_id`（alias で `schema` として出力）
- `src/framelift/item.py` — 集約後のレコード `Item`（`framelift/item/v1`）
- `src/framelift/cli.py` — Typer の CLI（`run`・`eval`・`frames`・`synth`）。`eval` は pydantic-evals を遅延 import する。`--profile` は `package.module:attribute` 形式で、カレントディレクトリを `sys.path` に追加して読み込む
- `src/framelift/_internal/` — 内部実装（公開 API ではない）
  - `sources.py` — 入力の展開、画像の読み込み、ファイルの sha256
  - `video.py` — 動画の標本取り、静止区間の判定、代表フレームの選択（PyAV を遅延 import する）
  - `items.py` — `item_key` による項目のキー計算と dump。評価と集約で共有する
  - `aggregate.py` — 複数フレームの項目の多数決と `Item` の組み立て
  - `extract.py` — Pydantic AI による抽出と Observation の組み立て
  - `fields.py` — 出力モデルのフィールドから、モデル型・モデルの一覧型を見分ける（`| None` と `Annotated` を外す）
  - `dataset.py` — 評価用データセット（`framelift/dataset/v1`）の読み書きと sha256 照合。pydantic-evals に依存しない
  - `evaluation.py` — pydantic-evals による評価と評価器
  - `synth.py` — 架空のショップ画面と正解（`dataset.yaml`）、スクロールする合成動画の生成
- `examples/fantasy_shop.py` — 合成サンプル用の Profile の例

## 設計上の原則

- 公開 API は `framelift/__init__.py` の `__all__`（`Profile`・`Pipeline`・`Observation`・`Item`・`RunResult`・`Validator`）だけ。`_internal` 配下は予告なく変えてよい。利用者から要望が出るまで、パイプラインの各段を公開しない
- `Profile` の項目は、それを使う機能と同じ版で足す。先回りして拡張点を作らない
- 出力やデータセットの形式を変えるときは、スキーマの版（`framelift/observation/v2`・`framelift/item/v1`・`framelift/dataset/v1` の末尾）を上げる
- ゲームやアプリの実在する名称・画像をリポジトリに入れない。例・テスト・ベンチマークは `synth` の架空の画面で作る
- 画像と動画はコミットしない。`.gitignore` で一括除外し、`samples/` と `docs/assets/` だけを許可している
- 外部に送るのは利用者が渡した画像と、動画から選んだフレームだけ。テレメトリは持たない

## 今後の予定

設計の詳細と判断の理由は [docs/README.md](docs/README.md) から辿れる。

- 0.3 — 画面判定、切り出し（`full`・`fixed`・`model`）と塗りつぶし、抽出結果のキャッシュ、設定の総当たり比較
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
- 画像は `samples` フィクスチャ（`synth.generate`）、動画は `scrolling_video` フィクスチャ（`synth.generate_video`）で作る。テスト用の画像・動画ファイルを置かない
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
