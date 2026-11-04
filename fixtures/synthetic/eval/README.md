# eval/ — 会社の合成データの評価セット(v2)

推薦精度・検索精度・異常系対応を測るための、正解ラベル付きの質問集。
会社向けの版(`company-v1` タグ)で作ったもので、設計書 4.8節では消す対象に入っている。
新しい前提(任意の組織)の評価セットは R7 で実組織の正解集として作り直す。それまではテストの床として残す。

## 生成元が無いこと

生成スクリプト(`build_eval_v2.py`、`eval_baselines.py`、`eval_label_agreement.py`、`import_human_labels.py`、
`research_e2e.py`)は #89 で、主な正解の出所だった日報・案件の合成データは #97 で develop から消した。
どれも `company-v1` タグに残る。ここの JSON は作り直せないので、直すときは JSON を直接編集する。

---

## ファイル

| ファイル | 件数 | 中身 | 読む側 |
|---|---|---|---|
| `eval_person.json` | 87 | 主指標。質問 → 正しい専門家・トピック・経路 | `tekijin.eval`、`test_eval_runner.py`、`test_eval_quality.py` |
| `eval_retrieval.json` | 72 | 質問 → 正しい根拠チャンク。`eval_person` の L1〜L3 と `id` で1対1 | `test_eval_quality.py` |
| `eval_robustness.json` | 20 | 異常系。全件 abstain(答えない・聞き返す)が正解 | `test_eval_quality.py` |
| `topic_experts_human.json` | 67トピック | 人手ラベル(PR #46)。topic → 専門家(`employee_id`) | なし(記録として残す) |
| `eval_queries.json` | 40 | 非推奨の旧セット(#26)。正解を `answers` から作っていたため測定にならなかった | なし |
| `route_calibration.json` | 87 | C5 の閾値較正用に、実 DB・実埋め込みで記録した類似度 | `test_route_calibration.py` |

---

## `eval_person.json` のスキーマ

| フィールド | 型 | 説明 |
|---|---|---|
| `id` | int | 連番 |
| `query` | string | 自然文の質問。L2 以上はトピック語を含まない(症状で書く) |
| `difficulty` | string | `L1`(トピック語を明示)/ `L2`(症状のみ)/ `L3`(複数トピック横断・商材名のみ)/ `L4`(専門家不在) |
| `gold_topics` | string[] | 正解トピック(22トピック語彙)。体系外の項目は空 |
| `gold_experts` | int[] | 正解の専門家 `employee_id`。順不同 |
| `gold_experts_alt` | int[] | 第2の正解。`answers` だけから導出(主 gold と経路が独立)。体系外トピックでは空 |
| `gold_route` | string | `person` / `prior_answer` / `document` / `none` |
| `expect_abstain` | bool | `true` なら「わかりません＋人へエスカレーション」が正解 |
| `constraint` | object\|null | 拠点の制約(例 `{"branch": "大阪"}`)。拠点の機能は #93 で消したので、今は誰も読まない |
| `source_topic` | string\|null | 人手ラベル由来の項目での PR #46 側のトピック名 |
| `label_source` | string | `auto:project_daily` / `authored` / `human:pr46` |
| `alt_label_source` | string | 第2の正解の導出経路。`auto:answers` 固定 |
| `note` | string | その項目の意図 |

`gold_route` の分布は全87件で `person` 49 / `document` 16 / `none` 15 / `prior_answer` 7。
`label_source` は `authored` 41 / `auto:project_daily` 25 / `human:pr46` 21。
`eval_robustness.json` の `category` は `out_of_scope`(5) / `pii`(4) / `insufficient`(5) / `no_expert`(3) / `adversarial`(3)。

## 正解の作り方(記録)

- 主 gold(`gold_experts`)は `projects`(lead 1.0 / member 0.6)と `daily_reports`(0.15)から作った。
  スコアラーが最も重く使う `answers` を正解の出所から外し、「answers を数えるだけ」の実装が満点を取れないようにするためである。
  #95 で案件をスコアラーの根拠から外したので、主 gold は今のスコアラーには当たりにくい(`test_eval_runner.py` の床の注記を参照)
- 第2の正解(`gold_experts_alt`)は `answers` だけから作った
- `gold_route` はクエリの言い回しを見ず、コーパスの状態(文書数と有用回答の平均 reuse)から決めた

---

## 変更するときの注意

`backend/tests/test_eval_quality.py` が、このセットが「自分に甘いテスト」に戻っていないことを検証する。

- L2/L3 のクエリにトピック語が混入していない
- 表層キーワードによる route 的中率が、多数クラスのベースライン+10pt を超えない
- 「answers を数えるだけ」ベースラインの Recall@3 < 0.6
- 独立サンプル数 >= 30

トピック語は `backend/tests/eval_topic_keywords.py` の表を使う。
クエリを足すときは、トピック語をクエリに書かないこと。

## `route_calibration.json`(#105)

C5 の閾値(`PRIOR_ANSWER_SIM` / `DOCUMENT_SIM` / `PERSON_WEAK_SIM`)が実際のコサイン分布と噛み合っているかを、
GPU も DB も使わずに CI で見張るための記録。コサインの絶対値はモデル依存なので、
`settings.embedding_model` を変えると `test_route_calibration.py` が明示的に落ちる。
記録し直すスクリプト(`research_e2e.py --task route`)は `company-v1` タグにある。
