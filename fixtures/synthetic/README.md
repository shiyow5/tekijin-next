# fixtures/synthetic/ — 会社の合成データ(テスト用)

`make seed`(`python -m tekijin.data.seed`)が DB に入れる合成データと、評価セットを置く。
すべて**架空**で、実在の社員・顧客・案件とは関係がない。

もとは会社向けの版(`company-v1` タグ)のデモと評価のために作ったデータである。
任意の組織向けに作り替えた今は、主にバックエンドのテスト(`seed_counts` フィクスチャ)の土台として使う。
新しい前提の合成データは架空のサークル(`fixtures/circle/`、R2)のほうにある。

## 消したもの

- 生成スクリプト(`build_fixtures.py`、`build_eval*.py`、`eval_baselines.py`、`eval_label_agreement.py`、`import_human_labels.py`)は #89 で消した
- 日報(`daily_reports/`)、資格(`certifications/`)、案件と担当(`projects/`)、自己申告スキル(`self_declared/`)の合成データと表、社員の `branch`(拠点)は #97 で消した

どれも `company-v1` タグに残る。生成スクリプトが無いので、ここの JSON を作り直す手段は今の develop には無い。
直すときは JSON を直接編集し、件数を変えたら下記のテストも直す。

---

## ファイル形式

- 1エンティティ = 1つの JSON ファイル。中身は JSON 配列(1要素 = 1レコード)
- フィールド名は snake_case。日付は ISO 8601(`2026-04-01` / `2026-04-01T09:00:00`)
- `embedding` 列は含めない。埋め込みは `make embed`(`scripts/embed_fixtures.py`)が本文から計算して DB に入れる。
  対象は `employee_profiles.description` / `questions.body` / `answers.body` / `documents.body`

---

## ファイルと件数

| ファイル | 件数 | 表 | 主なキー・FK |
|---|---|---|---|
| `people/employees.json` | 40 | `employees` | `id`(PK) |
| `people/employee_profiles.json` | 40 | `employee_profiles`(自己紹介文。埋め込み対象) | `employee_id`→employees.id |
| `chat/employee_chat_history.json` | 2000 | `employee_chat_history`(社員間チャット) | `sender_employee_id`→employees.id |
| `questions/questions.json` | 150 | `questions` | `asker_id`→employees.id |
| `answers/answers.json` | 150 | `answers` | `question_id`→questions.id, `responder_id`→employees.id |
| `documents/documents.json` | 36 | `documents`(社内文書。うち型番付き6件は #296) | `id`(PK) |
| `eval/` | — | 評価セット(DB には入れない) | [`eval/README.md`](eval/README.md) |

seed が入れるのは上の6ファイル、合計 2416 行。読み込みの対応は
`backend/src/tekijin/data/loaders.py`(`FIXTURE_FILES`)と `mappers.py`(`_PLAN`)にある。
件数は `backend/tests/test_data_unit.py`(`EXPECTED_COUNTS`)と
`backend/tests/test_data_integration.py`(`test_seed_counts`)が見張る。

直接相談のふりかえり(`offline_consults`)は実行時にだけ書かれる表で、合成データは無い。

### 各エンティティのフィールド

- **employees**: `id, name, email, department, section, position, role, hire_date, department_history[]`
  - `role`(職種)は `department` から導出(営業 / 技術 / スタッフ)
  - `section` は `department` と役職からの簡易な課(部長・課長=第1課、他=第2課)
- **employee_profiles**: `employee_id, description, updated_at`
- **employee_chat_history**: `id, sender_employee_id, receiver_employee_id(null), channel, message, sent_at`
- **questions**: `id, asker_id, body, topics[], status(="answered"), created_at`
- **answers**: `id, question_id, responder_id, body, created_at, reuse_count(0〜8), was_helpful(約7割 true), topic`
  - 全40名が最低1回は `responder_id` に出てくる
- **documents**: `id, title, body, source, updated_at`。22トピック語彙に沿った社内手順書・FAQ 風

`topics` と `topic` は22トピック語彙(`TOPIC_VOCABULARY`)の語を使う。語彙は R6 で置き換える。

---

## トピックごとの「答えの在り処」(#52)

route(`person` / `prior_answer` / `document`)をコーパスの状態から決められるよう、
トピックごとに文書と過去QAの量に差を付けてある。

| 性格 | コーパス側 | トピック数 | 想定 route |
|---|---|---|---|
| 文書のあるトピック | 文書3〜4件 / 有用回答の平均 reuse 0.0〜1.33 | 8 | `document` |
| 過去QAの多いトピック | 文書0件 / 過去QA 12件 / 平均 reuse 4.89〜6.82 | 6 | `prior_answer` |
| それ以外 | 文書0件 / 過去QA 6件 / 平均 reuse 1.5〜3.0 | 5 | `person` |
| 過去QAの無いトピック | 過去QA 0件 | 3 | `person`(現場判断のみ) |

この差が崩れていないことは `backend/tests/test_fixture_diversity.py` が見張る。

---

## `data/` との違い

| | `fixtures/synthetic/`(ここ) | `data/`(gitignore 済み) |
|---|---|---|
| 中身 | 種データ・評価セット | 生成物・大容量・実データ・モデル重み |
| Git | コミットする | コミットしない |

---

## 注意

- 合成データだけを置く。実在の社員・顧客・案件を混ぜない
- 秘密情報(トークン・実URL・実顧客名)は入れない
