# AGENTS.md

このリポジトリで作業するコーディングエージェント（Claude Code、Delta など）への指示。Claude Code は `CLAUDE.md` からこのファイルを読み込む。

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
FRAMELIFT_LIVE_MODEL=anthropic:<model-name> uv run pytest -m live   # 実モデルを呼ぶテスト（API キーが必要、課金される。変数がなければ skip）

# CLI の動作確認（"test" は Pydantic AI 組み込みの、プロバイダを呼ばないモデル）
uv run framelift synth --out /tmp/samples --n 3 --videos 1
uv run framelift synth --out /tmp/samples-jpeg --n 3 --scale 0.5 --noise 8 --jpeg-quality 70
uv run framelift run /tmp/samples --profile examples.fantasy_shop:profile --model test --out /tmp/out
uv run framelift frames /tmp/samples/shop-scroll-000.mp4 --out /tmp/frames
uv run framelift frames /tmp/samples/shop-000.png --profile examples.fantasy_shop:profile --out /tmp/frames
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
  → 画像: load_image で読み込み、sha256 を計算し、Layout で送る画像を作る
    動画: sample_video で一定間隔の標本を取り、縮小画像のタイルのうち変化した割合で静止区間に分け、区間ごとに最も鮮明な 1 枚に Layout を適用して PNG にする（frame_to_source）
      変化の割合と鮮明さは、表示どおりの向きにした輝度を、Layout.mask の範囲と塗りつぶしで送る範囲に切り出して測る（layout.cut）。各タイルは塗りつぶしていない割合で重み付けして数え、鮮明さは塗りつぶしに接しない画素だけで測る。縮小は各辺 256px 以下、少なくとも 2.5 分の 1 で、16px 未満の辺だけ 16 に引き伸ばす。画面の向きと、切り出せたかどうかが同じなら、大きさの違う標本（解像度の変更）も小さいほうにそろえて比べ、どちらかが違えば閾値によらず別の画面とする（_same_view）。区間の 1 枚は画素の多い標本を優先し、一度も止まらない動画の代わりの 1 枚は切り出せる標本を優先する。切り出しの範囲と塗りつぶしのマスクは、デコードした大きさか回転が変わったときだけ作り、直近の 1 つを持つ（_Views）。比べるための標本（_Measured）はデコードしたフレームを持たず、デコード中のフレームのほかに持つのは、区間の 1 枚と代わりの 1 枚の候補だけ
    Layout: EXIF の向きと色空間を正規化し、redact を黒く塗ってから locate の範囲で切り出す。向きに合う範囲がなければ NoCropError（error の観測にして次へ進む）
    full かつ redact なしの静止画だけは、元のバイト列をそのまま送る
  → Profile.screen_description があれば ScreenClassifier.classify: 送る画像（ImageSource.view）を長辺 512px に縮小した PNG で対象画面か判定する
      ローカルのサーバー（is_local_server）では NativeOutput、ほかでは ToolOutput で答えさせる（output_mode）。Pydantic AI のプロファイルの supports_json_schema_output は OpenAI のどのモデルでも True で、対応しているかの目印にならないため。ローカルのサーバーでもプロファイルが False なら ToolOutput。どちらも名前は screen_check（抽出の final_result と見分けるため）。モデルは screen_model（省略時は model）
      model_settings（CLI は --model-setting KEY=VALUE。VALUE は JSON、読めなければ文字列）は判定と抽出の Agent の両方に渡す
      対象外なら status: "skipped"、判定の失敗なら status: "error" の観測にして抽出しない。判定の結果と usage は観測の screen に残す
  → Extractor.observe: Pydantic AI の Agent に output_type と画像（BinaryContent）を渡し、Observation（v3。source・frame・crop・screen・cached）を作る
      Profile.validators を output_validator に登録し、ModelRetry で再試行
      再試行を使い切ったら status: "error"。1 枚の失敗で全体は止めない
      モデルは Extractor・ScreenClassifier の構築時に infer_model で 1 度だけ解決し、その Model で問い合わせる
      ローカルのサーバー（system が openai で、ホストが localhost・プライベートや Tailscale のアドレス・ドットのない名前・.local などのローカルの名前。is_local_server）では、resolve_model がモデルを _NameChecked で包み、応答ごとに model_name が指定で始まるか（大文字・小文字は区別しない）を確かめる。始まらなければ再試行の前に ModelMismatchError（UserError）で実行を止める（判定も同じ）。CLI は、構築時の UserError（API キーがないなど）と実行中の ModelMismatchError を終了コード 2 のメッセージにする（eval は実行中の UserError もすべて）
      読めない画像・動画（途中で壊れた動画、フレームのない動画を含む）は status: "error"、frame: null の観測にして次へ進む
      cache_dir があれば、抽出と判定の結果をキャッシュ（_internal/cache.py）から読み、成功した結果を書く。キーは CACHE_REVISION・送る画像の sha256・モデルの文字列・接続先の URL（解決した Model の base_url）・モデルの設定（あるときだけ）・fingerprint（判定は縮小画像の sha256・モデルの文字列・接続先の URL・モデルの設定・screen_description・version・NativeOutput のときだけ native）。JSON にできない設定はキャッシュしない（警告 1 回）。framelift のプロンプトや判定の出力を変えたら CACHE_REVISION を上げる。一部の結果だけが変わるなら、代わりにその結果のキーに項目を足し、ほかのエントリを使い続ける（判定の出力の形はこちら）
      Model のインスタンスはキャッシュしない（警告 1 回）。復元したモデルが元と厳密に同じ状態（same_state）のときだけ書く。ヒットした観測は cached: true で、usage は元の実行のもの
  → out_dir があれば observations.jsonl に 1 行ずつ追記
  → item_key があれば Aggregator で項目をキーごとに束ねて多数決し、items.jsonl に書く

framelift eval（framelift[eval] が必要）
  → load_dataset: dataset.yaml（framelift/dataset/v1）を読み、全画像の存在・読み込み・sha256 を照合
  → 正解を output_type 全体で検証し JSON 形式に揃え、item_key のキーも計算しておく（必須フィールドは省けない。合わなければ実行前にエラー）
  → pydantic-evals の Dataset を組み、各ケースで load_image（Profile の Layout を適用）と Extractor.observe_async を 1 件ずつ実行（評価のイベントループ上で非同期に動かす）
      出力は Observation と検証済みのモデル。項目のキーはこのモデルから取り、保存した JSON を検証し直さない
      読んだ入力ファイルの sha256 をもう一度照合し、照合後に差し替わっていたらそのケースを失敗にする（送る画像は切り出すと別の PNG になるので、入力ファイルのハッシュで比べる）
      ローカルのサーバーが別のモデルで答えたら（ModelMismatchError）、後のケースを送らず、評価の後で投げ直す
  → 評価器（検証の通過、フィールド一致率、item_key による適合率・再現率、usage）でレポートを出す
      ケースや評価器が異常終了したら終了コード 1。スコアが低いだけなら 0
```

- `src/framelift/profile.py` — `Profile`（`name`・`version`・`output_type`・`instructions`・`validators`・`locate`・`redact`・`screen_description`・`item_key`）と `Validator`・`Box`・`Orientation` 型。`locate` は必須で、`redact` は向きごとの dict に正規化して持つ。`fingerprint` は結果に影響する項目のハッシュで、読む画像を選ぶだけの `screen_description` は入れない。`item_field` は `item_key` を適用する一覧フィールド（トップレベルでただ 1 つの `list[モデル]`。`| None` も可）を返す
- `src/framelift/pipeline.py` — `Pipeline` と `RunResult`（`succeeded`・`skipped`・`failed`）
- `src/framelift/observation.py` — 出力レコード（`framelift/observation/v3`）。v3 は 0.3.0 で初めてリリースするので、0.3 の間は v3 に項目を足しても版を上げない。`schema` は Python 側では `schema_id`（alias で `schema` として出力）
- `src/framelift/item.py` — 集約後のレコード `Item`（`framelift/item/v1`）
- `src/framelift/cli.py` — Typer の CLI（`run`・`eval`・`frames`・`synth`）。`frames` は静止画も受け付け、`--profile` で送る画像を確かめられる。`eval` は pydantic-evals を遅延 import する。`--profile` は `package.module:attribute` 形式で、カレントディレクトリを `sys.path` に追加して読み込む
- `src/framelift/_internal/` — 内部実装（公開 API ではない）
  - `sources.py` — 入力の展開、画像の読み込み、ファイルの sha256
  - `video.py` — 動画の標本取り、静止区間の判定、代表フレームの選択（PyAV を遅延 import する）
  - `layout.py` — 送る画像を作る（EXIF の向きと色空間の正規化、塗りつぶし、切り出し）。`NoCropError`
  - `screen.py` — 画面判定（`ScreenClassifier`・`ScreenCheck`・`screen_image`）
  - `cache.py` — 抽出と画面判定のキャッシュ（`Cache`・`model_cache_id`・`same_state`）。テストでは `model_cache_id` を差し替えて `FunctionModel` にも識別子を与える
  - `items.py` — `item_key` による項目のキー計算と dump。評価と集約で共有する
  - `aggregate.py` — 複数フレームの項目の多数決と `Item` の組み立て
  - `extract.py` — Pydantic AI による抽出と Observation の組み立て
  - `fields.py` — 出力モデルのフィールドから、モデル型・モデルの一覧型を見分ける（`| None` と `Annotated` を外す）
  - `dataset.py` — 評価用データセット（`framelift/dataset/v1`）の読み書きと sha256 照合。pydantic-evals に依存しない
  - `evaluation.py` — pydantic-evals による評価と評価器
  - `synth.py` — 架空のショップ画面と正解（`dataset.yaml`）、スクロールする合成動画の生成。静止画は拡大・縮小、ノイズ、JPEG の順に劣化させられる。書いたファイルは `synth.sha256` に記録し、生成し直すと、記録に載っていて sha256 が一致し、今回書かなかったファイルだけを消す（一時ディレクトリに書いてから移す）
- `examples/fantasy_shop.py` — 合成サンプル用の Profile の例

## 設計上の原則

- 公開 API は `framelift/__init__.py` の `__all__`（`Profile`・`Pipeline`・`Observation`・`Item`・`RunResult`・`Validator`）だけ。`_internal` 配下は予告なく変えてよい。利用者から要望が出るまで、パイプラインの各段を公開しない
- `Profile` の項目は、それを使う機能と同じ版で足す。先回りして拡張点を作らない
- 出力やデータセットの形式を変えるときは、スキーマの版（`framelift/observation/v3`・`framelift/item/v1`・`framelift/dataset/v1` の末尾）を上げる。ただし、まだどのリリースも書いていない版は、同じ版のまま項目を足してよい
- ゲームやアプリの実在する名称・画像をリポジトリに入れない。例・テスト・ベンチマークは `synth` の架空の画面で作る
- 画像と動画はコミットしない。`.gitignore` で一括除外し、`samples/` と `docs/assets/` だけを許可している
- 外部に送るのは、利用者が渡した画像と動画から選んだフレームを、`Profile.locate` で切り出し `Profile.redact` で塗りつぶしたものだけ。テレメトリは持たない

## 今後の予定

設計の詳細と判断の理由は [docs/README.md](docs/README.md) から辿れる。

- 0.3 — 切り出し（`full`・`fixed`）と塗りつぶし、画面判定、抽出結果のキャッシュ（実装済み）。総当たり比較と切り出しの `model` 戦略は要望が出てから
- 1.0 — 公開 API の確定

## エージェント向けの設定

- 指示はこの `AGENTS.md` に書く。`CLAUDE.md` は Claude Code 向けに `@AGENTS.md` で読み込むだけにする
- skill は `.agents/skills/<name>/SKILL.md` に置く
- `land` は Delta の Land Changes（`metadata.delta-action: land`）と `/land` で使う、検査からプルリクエスト・マージまでの手順。`pyproject.toml` の `version` が変わる変更を main に入れると PyPI に公開されるので、版は頼まれたときだけ上げる
- Delta は作業用のチェックアウトを `.delta/worktrees/`・`.delta/clones/` に作る。Git では `.git/info/exclude` で除外し、markdownlint の対象からも外している

## コーディング規約

### Python

- Python 3.12+。ジェネリクスは PEP 695 の構文（`class Profile[OutputT: BaseModel]`、`type X = ...`）
- 各モジュールの先頭に `from __future__ import annotations`。型注釈だけで使う import は `TYPE_CHECKING` ブロックに置く（ruff の `TC`。tests は対象外）
- パス操作は `pathlib`（ruff の `PTH`）
- 行幅 100。ruff のルールは `pyproject.toml` の `[tool.ruff.lint]` を参照
- 公開クラスと公開関数には docstring を付ける（英語）

### テスト

- `live` マーカーのテスト（`tests/test_live.py`）以外は実モデルを呼ばない。`tests/conftest.py` で `models.ALLOW_MODEL_REQUESTS = False` にしていて、live テストだけがテストの中で `True` に戻す
- ローカルのサーバーの接続先や応答のモデル名が要るテストは、`tests/conftest.py` の `answering_model` フィクスチャ（`AnsweringModel`）でモデルを作る
- モデルの応答は `FunctionModel` で組み立てる。抽出の出力はツール呼び出しなので、`ToolCallPart(info.output_tools[0].name, {...})` で返す。画面判定は、ローカルのサーバーとして作ったモデル（`answering_model`）では JSON のテキスト、ほかではツール呼び出しで答えるので、判定にも答えるモデルは `tests/answers.py` の `output_name`・`respond_with` を使う
- 再試行の確認は、1 回目に検証を通らない値、2 回目に正しい値を返す `FunctionModel` で行う
- CLI のテストは `--model test`（Pydantic AI 組み込みのテスト用モデル）と `typer.testing.CliRunner` を使う。モデルの文字列の解決を確かめるテストだけは `openai:<name>` などを渡し、`extract.infer_model` を差し替えるか、`OPENAI_*` の環境変数を `monkeypatch` で決める
- 画像は `samples` フィクスチャ（`synth.generate`）、動画は `scrolling_video` フィクスチャ（`synth.generate_video`）で作る。テスト用の画像・動画ファイルを置かない
- 実モデルを使うテストには `@pytest.mark.live` を付ける（既定では実行されない）
- DeprecationWarning・PendingDeprecationWarning・FutureWarning は、framelift のモジュールのものとして出たときだけテストでエラーになる（`pyproject.toml` の `filterwarnings`）。どのモジュールのものになるかは、警告を出すライブラリが `stacklevel` で指す呼び出し元で決まる。ライブラリが自分の中を指すと、framelift の古い呼び方が原因でもエラーにならないので、テストの警告の一覧も見る

### ドキュメント

- README は英語（OSS として読まれる範囲を広げるため）。CLAUDE.md は日本語
- README は概要・インストール・Quick start に留め、機能ごとの詳しい説明は `docs/guide/`（英語）に書く。README から docs へのリンクと画像は、PyPI でも表示されるよう `https://github.com/...` や `https://raw.githubusercontent.com/...` の絶対 URL にする
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
- pre-commit のフック（`pre-commit/pre-commit-hooks`）の `rev` も Renovate で更新する（`:enablePreCommit`）
