# TEKIJIN（適時／適材適所）

**暗黙知を、使った分だけ形式知に変える**ナレッジ蓄積プロダクト。社内の「これ誰に聞けばいいんだろう」に対し、
**蓄積済みのデータに答えがあれば出典付きで自己回答し、無ければ答えられる人へ取り次ぐ**。得られた回答は
共有できるノウハウとして蓄積し、次からは自己回答で返せるようにする（自己回答・取次ぎ・蓄積の三系統）。
取次ぎは蓄積を回す入口。**AIが根拠なしに答えることはない**（出典を示せないときは答えず人へ回す）。
大塚商会サマーインターン Aチーム。

> このリポジトリは、インターン終了後の個人継続開発版です。インターン期間中（〜2026-09）の記録とIssueは
> [shiyow5/TEKIJIN](https://github.com/shiyow5/TEKIJIN)（アーカイブ済み）にあります。本文中の「#番号」のうち
> このリポジトリに無いものは、元リポジトリのIssue・PRを指します。今後の移行方針は
> [個人継続開発への移行計画](docs/plans/2026-10-21-personal-continuation-plan.md)を参照してください。

> フロー（C1〜C8）と全画面が動作するプロトタイプ実装済み。DB＋埋め込み＋vLLM を
> 揃えれば質問→取り次ぎまで実データで通ります。ローカルでの起動方法は
> [アプリの起動（開発）](#アプリの起動開発)を参照してください。

## 主な機能（実装済み）

- **質問→意図解析→経路判定→推薦→依頼文（C1〜C8, LangGraph）**: 質問を構造化し、
  **データで答えられるか（自己回答）／人に取り次ぐか**を判定。データ由来の経路では C7' が
  過去回答・社内文書の根拠だけから**出典付きで回答**する（`self_answer_enabled`・現状は既定OFF）。
- **ナレッジ蓄積（主軸）**: 人に聞いて得た回答を「誰が・何に・どう答えたか」の形式知として蓄積し、
  専門性グラフを更新（C8）。使い続けるほど自己解決できる範囲が広がる。既存ノウハウは日報などから取り込み。
- **画面**: ハブ `/`・質問 `/questions`・思考過程 `/session/[id]`・結果 `/session/[id]/result`・
  受信箱 `/inbox`・回答 `/answer/[session_id]`・文書ビューア `/documents/[id]`・ダッシュボード `/dashboard`。
- **取り次ぎの堅牢性**: 受諾/辞退→リルート、stale-outcome ガード（generation token）、
  切断後の再接続 replay。
- **安全性**: C1 で個人情報要求・プロンプト注入・担当外照会を `out_of_scope` として拒否。
- **ダッシュボード**: 推薦精度・自己解決率・平均解決時間（実行時解決を反映）などの集計のみ表示。

## リポジトリ構成

```
TEKIJIN/
├─ backend/          Python 3.12 バックエンド（FastAPI + LangGraph）
│  ├─ src/tekijin/   アプリ本体（API / エージェント / 検索 / データ）
│  ├─ tests/         pytest（単体・結合）
│  ├─ pyproject.toml ruff / pytest 設定
│  └─ requirements-dev.txt
├─ frontend/         TypeScript フロントエンド（Next.js 15 + React 19）
│  ├─ src/
│  ├─ tests/
│  ├─ package.json   biome / vitest
│  └─ biome.json
├─ docs/             仕様書・設計文書の置き場（下記参照）
│  ├─ specs/         仕様書（プロダクト/技術仕様など）
│  └─ adr/           意思決定記録（Architecture Decision Records）
├─ scripts/          補助スクリプト
├─ .github/
│  ├─ workflows/     CI（format / lint / test）
│  └─ actions/       セットアップ用 composite action
└─ Makefile          開発タスクの入口
```

詳細は [docs/REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md) を参照。

## 開発の始め方

```bash
make setup        # backend/frontend の開発ツールを導入
make check        # フォーマット確認 + lint + テストを一括実行
make help         # 使えるタスク一覧
```

個別に:

```bash
make fmt          # 自動整形（backend: ruff / frontend: biome）
make lint         # lint
make test         # テスト
```

## アプリの起動（開発）

バックエンド（`:8000`）とフロントエンド（`:3000`）をまとめて 1 コマンドで立ち上げます。

```bash
make setup        # 初回のみ: backend/frontend の依存を導入
make serve        # backend(uvicorn --reload) + frontend(next dev) を同時起動
                  # 別名: make dev（make serve のエイリアス）
```

- 起動後: フロント <http://localhost:3000> ／ API <http://localhost:8000>（`/docs` に OpenAPI）。
- `Ctrl-C`（このターミナル上）で backend・frontend を**両方**停止します。片方が起動に失敗した
  ら（例: ポート使用中）もう片方も停止し、失敗が伝播します。
- 既定で LLM（C1/C2/C7）と checkpointer はスタブのため、**vLLM など外部 LLM なしでサーバは起動**します。
- ただし**質問を実際に処理するには DB と埋め込みが必要**です（下記）。未整備でも UI と両サーバは
  立ち上がりますが、質問送信はエラーになります。
- フロントは `NEXT_PUBLIC_API_BASE_URL`（既定 `http://localhost:8000`）で API を参照します。

質問処理まで一通り動かす（DB＋埋め込みを用意する）:

```bash
make db-up        # PostgreSQL 16 + pgvector を起動（healthy まで待機）
make seed         # 合成フィクスチャを投入
make setup-ml     # 実埋め込みモデル用の ML 依存を導入
make embed        # 密ベクトルを計算・格納
make serve        # 起動（LLM はスタブのまま）
```

片方だけ起動したいとき:

```bash
make run-backend  # backend のみ（uvicorn --reload）
make run-frontend # frontend のみ（next dev）
```

本番相当（実 vLLM + PostgreSQL/PostgresSaver、backend のみ）で動かすとき:

```bash
TEKIJIN_LLM_BACKEND=vllm TEKIJIN_CHECKPOINTER_BACKEND=postgres make serve-prod
```

チームで GPU サーバー（共有 vLLM）を使って立ち上げる手順は
[docs/gpu-server-setup.md](docs/gpu-server-setup.md) を参照。

## 開発フロー（重要）

Issue → ブランチ → 実装 → `make check` → PR（develop向け）→ CI → AIレビュー →（メンバーは PL レビュー）→ マージ。
**`main` への直 push は禁止。`develop` が統合ブランチ。**

- 手順の唯一の正: **[.claude/skills/dev-flow/SKILL.md](.claude/skills/dev-flow/SKILL.md)**
- 全エージェント共通指示: [AGENTS.md](AGENTS.md) / 人間向け要約: [CONTRIBUTING.md](CONTRIBUTING.md)

## ツールチェーン

| 対象 | 言語/FW | フォーマッタ | Linter | テスト |
|---|---|---|---|---|
| backend | Python 3.12（FastAPI + LangGraph + SQLAlchemy/pgvector） | ruff format | ruff（型検査は mypy を併用） | pytest |
| frontend | TypeScript（Next.js 15 + React 19 + Tailwind） | Biome | Biome（型検査は tsc を併用） | vitest + Playwright(E2E) |

## CI

`main` への push と Pull Request で、変更のあった領域だけ以下が走ります。

- **Format Check** … 整形崩れの検出
- **Lint** … 静的解析（ruff / Biome）＋ 型検査（backend: `mypy src` / frontend: `tsc --noEmit`）
- **Test** … backend pytest（PostgreSQL+pgvector 結合を含む）／frontend vitest
- **E2E** … frontend の Playwright（主要フローのブラウザテスト）

GitHub-hosted の `ubuntu-latest` で動作します（追加のランナー設定は不要）。

## 注意

- 秘密情報（`.env`、サービスアカウント JSON 等）はコミットしないこと。`.gitignore` で除外済み。
- 設定値のひな型は `.env.example` を参照。

## ライセンス

[MIT License](LICENSE)。`fixtures/` のデータはすべて合成で、実在の個人・顧客は含みません。
商品名や社内システム名（「おとどけーな」「まもりーな」「CRS」など）も架空です。
