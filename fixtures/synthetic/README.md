# fixtures/synthetic/ — 合成データ（ER スキーマ準拠）

デモ・評価・テストで使う **合成データ（synthetic data）** をここに置く。
TEKIJIN は実データに接続しない方針なので、**この合成データがそのまま「真実源」**になる。

> 生成スクリプト(`build_fixtures.py`、`build_eval*.py`、`eval_baselines.py`、`eval_label_agreement.py`、`import_human_labels.py`)は #89 で develop から消した。このファイルに出てくるスクリプトは `company-v1` タグで使える。

すべて **架空**。実在の社員・顧客・案件とは一切関係がない。
生成スクリプトは [`scripts/build_fixtures.py`](../../scripts/build_fixtures.py)
（`random.seed(42)` で再現可能）。reona 作のダミーデータ（社員 / 案件 / チャット / 日報）を
入力に、ER スキーマ（`docs/specs/model-definition.md` §4 データモデル、`docs/specs/technical-spec.md` §4）へ
写像しつつ、不足エンティティ（社員プロフィール / 資格 / 過去QA）を補完している。

再生成:

```bash
python3 scripts/build_fixtures.py
```

---

## 生成

```bash
python3 scripts/build_fixtures.py     # fixtures/source/ を入力に、ここを再生成する
```

一次データは [`fixtures/source/`](../source/README.md)（reona 作、#17）。
**以前は生成元がリポジトリ外の一時ディレクトリにしか無く、それが消えると再生成不能だった。**

---

## ファイル形式の規約

- **1エンティティ = 1つの JSON ファイル（`.json`）。中身は JSON 配列**（1要素 = 1レコード）。
  ※ 旧規約は JSON Lines（`.jsonl`）だったが、**JSON 配列に統一**した。
- フィールド名は **snake_case**。日付は ISO 8601（`2026-04-01` / `2026-04-01T09:00:00`）。
- **`embedding` 列は fixtures に含めない。** 埋め込みは取込（シード投入）時に
  アプリ側（C3 埋め込みモデル）が本文から計算してDBへ格納する。
  対象本文: `employee_profiles.description` / `questions.body` / `answers.body` /
  `projects.subject`・`remarks` / `daily_reports.content` など。

---

## ディレクトリ構成と ER（`docs/specs`）との対応

| ディレクトリ / ファイル | 件数 | ER 対応（technical-spec §4 / model-definition §4） | 主なキー・FK |
|---|---|---|---|
| `people/employees.json` | 40 | `employees(id, name, dept, role, branch, …)` | `id`（PK） |
| `people/employee_profiles.json` | 40 | `employees.self_intro`（人の自己紹介＝埋め込み対象） | `employee_id`→employees.id |
| `certifications/certifications.json` | 98 | `certifications(id, employee_id, name, acquired_at)` | `employee_id`→employees.id |
| `projects/projects.json` | 120 | `projects(id, industry, products, period, …)` | `id`（PK） |
| `projects/project_members.json` | 237 | `project_members(project_id, employee_id, role)` | `project_id`→projects.id, `employee_id`→employees.id |
| `chat/employee_chat_history.json` | 2000 | 社内チャットログ（行動痕跡・近接性の証拠） | `sender_employee_id`→employees.id |
| `daily_reports/daily_reports.json` | 3070 | 日報（`content` が類似検索の主要テキスト） | `employee_id`→employees.id |
| `questions/questions.json` | 150 | `questions(id, asker_id, body, topics, status, created_at)` | `asker_id`→employees.id |
| `answers/answers.json` | 150 | `answers(id, question_id, responder_id, body, reuse_count, was_helpful)` | `question_id`→questions.id, `responder_id`→employees.id |
| `documents/documents.json` | 30 | `documents(id, title, body, source, updated_at)` | `id`（PK） |
| `self_declared/skills.json` | 58 | `skills(id, employee_id, topic, level, source)`（自己申告＝最も弱い証拠、base_score 0.3） | `employee_id`→employees.id |
| `eval/` | 50 / 45 / 20 | 評価セット v2（`eval_person` / `eval_retrieval` / `eval_robustness`）＋人手ラベル `topic_experts_human.json`。生成は `scripts/build_eval_v2.py` | 下記「評価セット」参照 |

### 各エンティティのフィールド

- **employees**: `id, name, email, department, section, position, branch, role, hire_date, department_history[]`
  - `branch`（拠点）= 本社 / 東京 / 大阪 / 名古屋 / 福岡 から重み付き割当（スコア式 `proximity` の源）。
  - `role`（職種）= `department` から導出（営業 / 技術 / スタッフ）。
  - `section` = `department` + 役職ベースの簡易課（部長・課長=第1課、他=第2課）。
- **employee_profiles**: `employee_id, description, updated_at`
  — `description` は当人の案件（product/issue）・日報（content）から抽出した得意トピックを織り込んだ自己紹介文。
- **certifications**: `id, employee_id, name, acquired_at` — 資格は職種（role）に整合するリストから割当。
- **projects**: `id, subject, client_company, industry, company_size, client_issue, product, negotiation_count, status, remarks, start_date, end_date`
- **project_members**: `project_id, employee_id, role`（`lead` = 案件主担当、`member` = 同部署から0〜2名）。
- **employee_chat_history**: `id, sender_employee_id, receiver_employee_id(null), channel, message, sent_at`
- **daily_reports**: `id, employee_id, report_date, content, issue, created_at`（`issue` は content から簡易導出、無ければ `null`）。
  - **営業部の日報だけ CRS 訪問日報フォーマット（#326）**。ヒアリングで、営業部の日報は CRS（全社が閲覧できる
    顧客情報システム）に決まった様式で入力されると分かったため、営業部の従業員の日報を訪問日報の構造
    （`【訪問】訪問日 時刻／業種・規模／要件: 初期訪問|課題ヒアリング|提案|デモ・PoC|クロージング|導入フォロー／
    先方: ご担当／当社: 営業担当／所要○分。詳細（1〜2文）`）に差し替えている。**他部署の日報は不変**。
    詳細の担当トピックは、その社員の**案件由来の得意領域**（lead1.0/member0.6）から選び、日報が案件と同じ
    トピック証拠を補強する（案件に無い偽の専門性を作らない）。実データ・実顧客名は使わず、OA機器・IT商材販社の一般的な
    商材レンジ（CRM・営業支援 / 業務効率化コンサル / 契約書管理 / 基幹システム / ネットワーク・VPN 等）で合成。
    乱数は日報IDで決まる専用インスタンス（グローバル `random` 非消費＝他データの件数を一切ずらさない）。
    ガードは `backend/tests/test_data_unit.py::test_sales_daily_reports_use_crs_visit_format` ほか。
- **questions**: `id, asker_id, body, topics[], status(="answered"), created_at`
- **answers**: `id, question_id, responder_id, body, created_at, reuse_count(0〜8), was_helpful(≈7割 true), topic`
  - **全40名が最低1回は `responder_id` に登場**する（少数者の埋没を防ぐため、未カバー社員は
    当人が実際に活動しているトピックのQAへ差し替え。合計150は維持）。
- **documents**: `id, title, body, source, updated_at` — 22トピック語彙に沿った社内手順書・FAQ 風。
  C5 の「格下げ経路（文書で場所を指す）」の素材。`body` は取込時に埋め込み対象。
- **skills**（`self_declared/skills.json`）: `id, employee_id, topic, level, source(="self")`
  — 1人1〜2件。一部は本人の実活動トピックと一致、一部はあえて実活動と無関係にしてあり、
  「自己申告は弱い証拠（base_score 0.3）」であることを表現している。

---

## 「トピック → 得意な社員」と eval の復元

過去QA（questions/answers 150ペア）は、**行動データに裏打ちされた専門性**を種にしている。

1. 各社員の得意トピックを、その社員の **projects（product/issue）** と **daily_reports（content）** の
   頻出語から推定する（トピック語彙はOA機器・IT商材寄り: UTM / ネットワーク / セキュリティ / クラウド /
   基幹システム / CRM / Webマーケ / SNS など＋各部署業務）。証拠の重み: 案件 lead=1.0 / member=0.6、日報=0.15。
2. `questions.topics[]` と `answers.topic` にトピックを残し、`answers.responder_id` は
   **そのトピックの上位専門家**から選ぶ。

これにより **answers を `topic` でグルーピングして `responder_id` を集めるだけで、
専門家を employee_id 単位で復元**できる。

> **⚠ この復元性を評価の正解に使ってはいけない（Issue #43）。**
> 専門性推定（doc15）も同じ answers を最重量の証拠（有用回答1.0 / 過去回答0.7）に使うため、
> 「answers を数えるだけ」の実装が満点を取ってしまい、**スコアラーの良し悪しを一切測れない**。
> 評価セット v2 は gold を **projects + daily_reports からのみ**導出し、answers を使わない。
> 下記「評価セット（v2 / Issue #43）」を参照。
> なお、この復元性そのものは**データの整合確認**には引き続き有用。

---

## `data/` との違い（重要）

| | `fixtures/synthetic/`（ここ） | `data/`（gitignore 済み） |
|---|---|---|
| 中身 | 手で吟味した**種データ**・評価セット | 生成物・大容量・実データ・モデル重み |
| Git | **コミットする**（全員/毎デモで同一） | **コミットしない** |
| 再現性 | バージョン管理で固定 | ローカル限り・再生成可 |

---

## 既知の偏り（要チーム判断）

- **職種比率が仕様§8目標と乖離している。**
  現状 `employees.role`（department から導出）は **技術30% / 営業30% / スタッフ40%**（技術12・営業12・スタッフ16名）で、
  `docs/specs/technical-spec.md` §8 の目標比率 **技術38.7% / 営業32.0% / スタッフ27.7%** から外れている。
- **原因**は、入力元（reona 作 `employees.json`）が **10部署 × 各4名の均等構造**であることに起因する。
  役職の導出（顧客接点・技術・バックオフィス）だけでは目標比率に寄せられない。
- **#51 で部分的に是正した。** 支援部署（総務・経理・情シス等）を案件のメンバーに入れることで、
  バックオフィスの社員も案件の証拠を持てるようになり、部署をまたぐ案件が **43%**（52/120）になった。
  ただし**職種比率そのものは未是正**（下記）。
- **是正には社員・部署・案件の再生成が必要**（部署ごとの人数配分そのものを変えるか、
  role の割当規則を実態に合わせて再設計する）。案件（projects）・日報・チャットも人員構成に連動するため、
  影響範囲が広い。**この再生成を行うかはチーム判断**とし、本タスクでは既存構造を尊重して据え置いた。

## トピックごとの「答えの在り処」（#52）

以前は全22トピックが「回答6〜7件・文書1〜2件」と横並びで、
**route（`person` / `prior_answer` / `document`）をコーパスの状態から決められなかった**。
`scripts/build_eval_v2.py` に `PROCEDURAL_TOPICS` / `RECALL_TOPICS` という定数を持たざるを得ず、
経路判定精度を測っているとは言い切れない状態だった。

`scripts/build_fixtures.py` でトピックを3つの性格に分けた。**総件数はどれも変えていない。**

| 性格 | コーパス側 | トピック数 | 想定 route |
|---|---|---|---|
| `DOCUMENTED_TOPICS` | 文書3〜4件 / 有用回答の平均 reuse 0.0〜1.33 | 8 | `document` |
| `RECALL_RICH_TOPICS` | 文書0件 / 過去QA 12件 / 平均 reuse 4.89〜6.82 | 6 | `prior_answer` |
| それ以外 | 文書0件 / 過去QA 6件 / 平均 reuse 1.5〜3.0 | 5 | `person` |
| `NO_ANSWER_TOPICS` | 過去QA **0件** | 3 | `person`（現場判断のみ） |

これにより `build_eval_v2.py` の**定数を廃止**し、route を文書数と reuse から導出できるようになった。
評価セットの route 分布は `document` 10 / `prior_answer` 7 / `person` 28 / `none` 5。

退化の検出は `backend/tests/test_fixture_diversity.py` が担当する。

---

## 評価セット（v2 / Issue #43）

`scripts/build_eval_v2.py` が生成する。**旧 `eval_queries.json`（`scripts/build_eval.py`）は非推奨。**

### なぜ置き換えたか

旧セットは FK 整合・トピック網羅・route 分布という**機械的整合性**は満たしていたが、
**測定として成立していなかった**（2026-08-21 実測）:

| 検査項目 | 旧 eval_queries.json |
|---|---|
| クエリのユニークな文型数 | **6**（40件が6文型の穴埋め） |
| クエリに正解トピックの語が漏れている | **24/40（60%）** |
| キーワード5語だけで route を当てられる率 | **40/40（100%）** |
| 「answers を topic で数えるだけ」で正解と完全一致 | **40/40（100%）** |
| 独立サンプル数（ユニーク正解集合） | **21**（40件あるが実質21件） |
| 専門家不在・想定外入力・個人情報要求 | **0件** |

とくに、`correct_experts` を answers の responder 集計から作っていたため、
専門性推定も同じ answers を最重量の証拠に使う以上、
**「answers を数えるだけ」の実装が満点を取れた**。スコアラーの良し悪しを測れない。

### v2 の3ファイル

| ファイル | 件数 | 中身 |
|---|---|---|
| `eval/eval_person.json` | 50 | **主指標**。質問 → 正しい専門家。難易度 L1(10) / L2(25) / L3(10) / L4(5) |
| `eval/eval_retrieval.json` | 45 | 層1。質問 → 正しい根拠チャンク。**埋め込みモデルの横並び比較用**。`eval_person` の L1〜L3 と id で1対1 |
| `eval/eval_robustness.json` | 20 | 異常系。スコープ外5 / 機微4 / 情報不足5 / 専門家不在3 / 敵対的3。**全件 abstain が正解** |
| `eval/topic_experts_human.json` | 67トピック | **人手ラベル**（PR #46 由来）。topic → 専門家(employee_id)。gold の外部検証と L2 の材料 |

難易度の定義:

- **L1（10件）**: トピック語を明示。易しい床。回帰検出用
- **L2（25件）**: **症状のみ**。トピック語をクエリから除去
  - 15件は自動導出 gold。うち5件は拠点制約付き
  - **10件は人手ラベル（PR #46）由来**（`label_source: "human:pr46"`）。
    個人単位で鋭いトピック6件＋自前トピック体系に無い日常業務4件
- **L3（10件）**: 複数トピック横断 / 商材名だけ / 言い換え
- **L4（5件）**: 社内に証拠を持つ人が存在しない。**「わかりません＋エスカレーション」が正解**

### 設計上の3原則

1. **クエリにトピック名・キーワードを書かない**（L2以上）。症状で書く
2. **正解の導出経路をシステムの証拠と分ける**。gold は `projects`(lead1.0/member0.6) + `daily_reports`(0.15)
   からのみ導出し、**`answers` を使わない**
3. 難問・異常系はスクリプト内に定数として著述する（`label_source: "authored"`）。
   L4 の文面は **L2/L3 と同じ体裁**で書く（「〜な方はいますか」等の語尾で書くと表層だけで abstain を当てられる）

### ベースライン（`python3 scripts/eval_baselines.py`）

| baseline | L1 | L2 | L3 | 全体 | 意味 |
|---|---|---|---|---|---|
| random | 0.100 | 0.093 | 0.150 | 0.107 | 下限 |
| **answers_count** | 0.967 | **0.253** | 0.167 | **0.393** | **リークの残量**（旧セットでは 1.000） |
| lexical_profile | 0.600 | 0.080 | 0.067 | 0.193 | 語彙一致のみ。埋め込みはこれを超えないと採用理由が無い |
| lexical_answers | 0.333 | 0.100 | 0.067 | 0.144 | 強めの語彙ベースライン。実質的な打倒目標 |

*(Recall@3。L4 は abstain 判定なので Recall の対象外)*

**本システムの数字は、この表との差分でしか意味を持たない。** モデル比較の前にまずここを測る。

### 人手ラベルによる外部検証（PR #46 の取り込み）

gold が全て自動導出だと「合成データの中の別ルール」でしかなく、それ自体では妥当性を主張できない。
PR #46（reona 作）は同じ案件・日報・チャットを**人が読んで** 67トピックの専門家を挙げたもので、
**導出手順が完全に独立している**。これを外部検証に使う。

```
$ python3 scripts/eval_label_agreement.py
平均 Jaccard: 0.74
自動 gold が人手 gold に含まれる率（被覆）: 0.83
完全一致したトピック: 10/22
```

**取り込んだもの / 取り込まなかったもの:**

| PR #46 の中身 | 扱い | 理由 |
|---|---|---|
| topic → 専門家の人手マッピング（67トピック） | **取り込む**（`eval/topic_experts_human.json`） | 人手ラベルは自動導出では作れない資産 |
| 個人単位で鋭いトピック（正解1〜3名、13件） | **6件を L2 に採用** | 自動導出は部署4名を一括で拾い鈍い。人手側のほうが鋭い |
| 自前22トピック体系に無い日常業務トピック（11件） | **4件を L2 に採用** | 営業事務・CS・経理庶務・マーケ。TEKIJIN の実用途に最も近い層 |
| クエリ本文 402件 | **採らない** | 38文型の穴埋めで、**トピック語が92%のクエリに漏れている**（独立サンプルは実質22） |
| 氏名での `correct_experts` | **employee_id に変換して採用** | #24/#26 で employee_id 基準に統一済み |

採用した10件は、**クエリだけリーク遮断の原則に従って症状ベースで書き直し**、gold は人手ラベルをそのまま使っている。

**一致度が低かったトピック**（自動側が部署4名一括、人手側が案件実績で個人を絞り込み）:
ECサイト構築 0.25 / モバイルアプリ開発 0.25 / サーバー・インフラ運用 0.25 / ネットワーク・VPN 0.33。
このうち鋭い方（人手）を L2 に取り込んである。
**ネットワーク・VPN は自動側に営業部の2名が混ざっており**、日報由来の弱い証拠（重み0.15）が
誤って効いている可能性がある。スコアラー実装時の確認事項。

### 妥当性を守るテスト

`backend/tests/test_eval_quality.py` が、評価セットが「自分に甘いテスト」に戻っていないことを検証する:

- L2/L3 のクエリにトピック語が混入していない
- 表層キーワードによる route 的中率が、多数クラスのベースライン+10pt を超えない
- 「answers を数えるだけ」ベースラインの Recall@3 < 0.6
- 独立サンプル数 >= 30
- 人手ラベル由来の項目が10件あり `source_topic` を持つ

### 既知の限界（正直に開示する）

- **合成データである以上、gold を証拠から完全に独立させることはできない。**
  旧経路（answers 上位3）と新経路の平均重なりは **0.58**（#51 前は 0.69）。この数字はスライドでそのまま開示する。
  ベースライン `answers_count` が L2 で 0.253 まで落ちること、および
  **人手ラベルとの被覆 0.83** が、実質的な担保になっている。
- **独立サンプル数は 35。** #51 で支援部署のメンバーを入れて 31 → 35 まで増やしたが、目標の 40 には届いていない。
  **一次データで案件を持つのは顧客接点のある4部署（16名）だけ**で、残り24名は案件のリードを一度も務めない。
  区別できる正解集合の数はここで頭打ちになる。**超えるには `fixtures/source/case_history_dummy.json` 側に
  バックオフィス主導の案件を足す**しかなく、一次データの改変になるため本タスクでは行っていない。
- **案件のリードは付け替えていない。** #51 では当初「部署内でリードを分担させて専門特化させる」ことを試したが、
  PR #46 の人手ラベルは**元のリード割当**を人が読んで付けたものなので、付け替えると外部検証の土台が崩れる。
  実測で 人手ラベルとの一致 Jaccard が 0.74 → **0.68 に悪化**したため取り下げ、
  支援部署メンバーの追加のみで差を作っている（独立サンプルは 38 vs 35 で付け替えのほうが有利だが、
  外部検証のほうが価値が高いと判断した）。

---

## 注意

- **合成データのみ。** 実在の社員・顧客・案件を混ぜない。実データは受領しない前提。
- 秘密情報（トークン・実URL・実顧客名）は絶対に含めない。
