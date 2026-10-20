# 判断専用モデルの比較調査

調査日: 2026-10-21

移行計画([2026-10-21-personal-continuation-plan.md](2026-10-21-personal-continuation-plan.md))の8章で挙げた研究テーマのために、Jevに似た判断専用モデルと、判断に転用できるモデルを洗い出した。値は一次情報(公式ドキュメント、モデルカード、GitHub)で確かめたもので、二次情報しか無いものはそう書いた。

## 1. 使いたい場面

| 記号 | 判断の中身 | 今のやり方と問題 |
|---|---|---|
| A | 質問と上位の知識単位を渡し、その知識だけで正しく答えられるかを確率で返す | cosineの閾値0.20では人に回すべき質問の18%で誤って発火し、0.30では一度も発火しない |
| B | 18トピックから選び、上位3件を確率付きで返す | C1(構造化出力)はacc@1が約0.75で、実行ごとに±0.09振れる。約1.5秒かかる |
| C | 曖昧な質問を見つけて逆質問する(#517) | 逆質問がほとんど発火しない |
| D | 会話スレッドに再利用できる知見があるかを判定する | 判定せずに、すべてLLMの抽出に回している |

## 2. 名前だけ把握していたものの確認結果

| 名称 | 結果 | 出典 |
|---|---|---|
| Jev 1.14(TypeSafe AI) | 2026-10-14提供(`jev-1.14.0`)。安定版の既定 `jev-latest` はまだ1.13。入力$0.042/100万トークン、出力は無料。CJKは英語ほどの精度ではないと明記。fine-tuneはできない | docs.typesafe.ai/models |
| OpenAI Decisions API | `POST /v1/decisions`、モデルは `gpt-6-luna` だけの公開ベータ。型はpredicate・choice・score。入力$0.10/100万トークン、出力は無料。較正の主張は無く、閾値は自分のラベル付きデータで決めるよう書かれている | developers.openai.com/api/docs/guides/decisions |
| Microsoft-Decision-1 | 2026-10-09発表。Qwen3.5-9Bを事後学習したもので、Foundryではプレビュー。入力$0.042/100万トークン、出力は無料。重みは公開されていない。OpenRouterからも呼べる | learn.microsoft.com(Foundry Models の Microsoft Decision の項) |
| OpenJev | 同じ名前のリポジトリが2つある。razorback16/openjev はJevとOpenAIの両方のAPIに互換な判断サーバー(Apache-2.0、既定モデルはVRAM 24GB以上)。SiliconLabAI/OpenJev は実験用(MIT) | GitHub |
| jevos | 確認できなかった | |

Jev、Microsoft-Decision-1、Clef、OpenJevは、質問の型(noul・choice・score)と要求の形式が共通している。比較用のクライアントを1つ書けば、APIとローカルのモデルを同じ条件で測れる。

## 3. 候補一覧

「日本語」の列の「未記載」は、一次情報に記述が無いという意味である。

| 区分 | 候補 | 形態・ライセンス | 規模 | 日本語 | 出力 | 費用・メモリ | 用途 |
|---|---|---|---|---|---|---|---|
| 判断専用API | Jev 1.14 | API | 非公開 | 英語より弱いと明記 | 確率。RLCDで較正 | $0.042/100万トークン | A B C D |
| 判断専用API | OpenAI Decisions | API(ベータ) | 非公開 | 未記載 | 確率 | $0.10/100万トークン | A B C D |
| 判断専用API | Microsoft-Decision-1 | API | 9B相当 | 未記載 | 選択肢ごとの確率 | $0.042/100万トークン | A B C D |
| 判断専用API | Fastino GLiDE | API | 非公開 | 未記載 | 確率 | $0.15〜0.30(二次情報のみ) | B C |
| 判断専用API | JJ-1(樹林AI) | 一部OSSと発表、公開先は確認できず | 未記載 | 日本語業務向け。自社ベンチの値のみ | 未記載 | 未記載 | B C |
| 判断専用OSS | Clef-flash(Cloudflare) | Apache-2.0 | 9B | 未記載 | 選択肢ごとのロジット。Brier 0.11 | Workers AIで$0.038/100万トークン。GGUFあり | A B C D |
| 判断専用OSS | sokudan-ja-310m | Apache-2.0 | 315M | 日本語で学習・評価。choice 0.880、真偽のAUROC 0.844(300問、1回のみ) | 真偽のみ温度補正済み | CPUで動く | B C(Aも試せる) |
| 判断専用OSS | Strands Decider 2B(AWS) | Apache-2.0 | 2B | 未記載 | ECE 0.064 | 4〜8GB(推定) | B |
| 判断専用OSS | Mapika decider-2b | Apache-2.0 | 2B | 英語のみ | ECE 0.164 | 約4GB | B |
| 判断専用OSS | Liquid d1-3B | LFM Open License | 3.1B | 16言語のタグのみ | Brier 0.14 | 約8GB(推定) | B C |
| 根拠判定 | HHEM-2.1-open(Vectara) | Apache-2.0 | 0.1B | 英語のみ。日本語は有償のHHEM-2.3 | 0〜1のスコア | 600MB未満 | A' |
| 根拠判定 | MiniCheck | MIT | 0.4〜0.8B | 英語 | 確率 | 2GB以下 | A' |
| 根拠判定 | LettuceDetect v2 mmBERT | Apache-2.0 | 0.3B | 7言語、日本語は未記載 | トークン単位 | 2GB | A' |
| 根拠判定 | Granite Guardian 4.1 | Apache-2.0 | 8B | 英語のみと明記 | yes/no | 16GB | A' |
| リランカー転用 | Qwen3-Reranker 0.6B/4B/8B | Apache-2.0 | 0.6〜8B | 100以上の言語 | yes/noのトークン確率。指示文を変えられる | 2〜16GB | A D |
| リランカー転用 | bge-reranker-v2-m3 | Apache-2.0 | 0.6B | 多言語 | ロジット | 2GB | A |
| リランカー転用 | ruri-v3-reranker-310m | Apache-2.0 | 310M | 日本語専用 | スコア | 1GB | A |
| リランカー転用 | japanese-reranker-large-v1 | MIT | 0.3B | 日本語専用 | スコア | 1GB | A |
| ゼロショット分類 | mDeBERTa-xnli-2mil7 | MIT | 0.3B | 学習データに日本語を含む | 含意の確率 | 1GB | B |
| ゼロショット分類 | xlm-roberta-large-jnli-jsick | ライセンス未確認 | 0.56B | JNLI 90.9% | 3値の確率 | 2GB | A B |
| 判定用LLM | Selene-1-Mini | Apache-2.0 | 8B | 英語が主 | 点数と講評 | 16GB(GGUFあり) | D |
| 判定用LLM | Prometheus 2 7B | Apache-2.0 | 7B | 英語 | 1〜5またはA/B | 14GB | D |

A' は、生成した回答文が文書に支えられているかを判定するモデルである。TEKIJINの知識回答は回答文を作るので、「回答文が根拠の知識単位に含意されるか」と言い換えればAに使える(推定)。ただし日本語に対応するのは有償のHHEM-2.3だけで、今回は候補から外す。

Bespoke-MiniCheck-7B、Lynx 8B、jina-reranker-v3、Gliderは非商用ライセンスなので除いた。Laya(Convai)はゼロショットの精度が低く、学習が前提なので除いた。

## 4. 用途ごとに比べる候補

どの用途でも、次の3つを比較の基準に入れる。

1. 今のやり方(Aならcosineの閾値、BならC1の構造化出力)
2. 汎用LLMに確率をJSONで答えさせる方式
3. 汎用LLMのyes/noのトークン確率を読む方式

| 用途 | 候補 | 理由 |
|---|---|---|
| A | Jev(noul)、OpenAI Decisions(predicate)、Microsoft-Decision-1、Qwen3-Reranker-4B/8B、ruri-v3-reranker | 判断専用モデルは、真偽の定義を指示文に書ける。Qwen3-Rerankerは同じ仕組みを手元で動かせる。日本語専用のリランカーは、「関連している」と「答えられる」の違いを測る対照になる |
| B | Jev(choice)、Microsoft-Decision-1、Clef-flash、sokudan-ja-310m、mDeBERTa-xnli | choiceは選択肢ごとの確率を返すので、上位3件をそのまま取れる。C1と、精度・速度・実行ごとのぶれで比べる |
| C | Jev・OpenAI(曖昧かどうか、何が足りないか)、sokudan-ja-310mの真偽、Bのトピック確率のエントロピー | エントロピーはBの副産物で、追加の費用がかからない |
| D | Jev・OpenAI(score)、Microsoft-Decision-1、Qwen3-Reranker、Selene-1-Mini | 判定の基準が主観的なので、段階で答えるscore型が合う |

## 5. 最初に試す5候補

| 順 | 候補 | 理由 |
|---|---|---|
| 1 | Microsoft-Decision-1 | Jevと同じ単価で、元のQwen3.5は多言語で学習されている。OpenRouterから呼べ、Jevと同じ形式で測れる |
| 2 | Jev 1.14 | 本命。CJKが弱いと公式が書いているので、日本語のまま渡す場合と英訳して渡す場合を比べる |
| 3 | OpenAI Decisions(gpt-6-luna) | 単価は2倍強。判断専用APIの上限の目安にする |
| 4 | Qwen3-Reranker-4B(8B) | 手元で動き、外部APIに出さずに済む。AとDの本命 |
| 5 | sokudan-ja-310m | 日本語で学習・評価された唯一の判断モデル。CPUで動く小型モデルがどこまで届くかの対照 |

補欠はClef-flash(Apache-2.0、9B、GGUFあり、Jevと同じ形式)とする。

## 6. 実験の進め方

評価データは87行だけなので、行を学習には使わず、閾値と温度を決めるためだけに使う。層化した5分割の交差検証で、4分割で閾値と温度を決め、残りの1分割で報告する。こうすると全行に「閾値決めに使っていない予測」が付く。

指標は、A・C・DではROC-AUCに95%信頼区間(bootstrap 1,000回)を付ける。あわせて、誤発火率を5%以下に抑えたときの再現率も出す。今の「誤発火18%か発火ゼロか」の二択を、同じ曲線の上の点として比べられるからだ。Bではacc@1、acc@3、正解トピックの順位の逆数の平均(MRR)を出す。較正はBrierとECEで見るが、87行ではECEのぶれが大きいことを書き添える。APIは同じ入力で3回流し、実行ごとの標準偏差と、判定が入れ替わった行数を報告する。

日本語の扱いは、同じ行を3通りで流して比べる。日本語のまま、機械翻訳した英語、指示文だけ英語で本文は日本語、の3つである。

ラベルは、Aなら評価セットのdata系の行を正例、abstain系とperson系の行を負例にできる見込みだ。CとDに使える行は今の評価セットに無い可能性が高く、評価用のラベルを新しく作る必要がある(推定)。

費用は、Aを1行あたり約3,000トークンとすると、87行×3回×3通りで約235万トークンになる。JevとMicrosoft-Decision-1で約$0.10、OpenAIで約$0.24に収まり、BからDを足しても1社あたり$1未満の見込みだ(推定)。ローカルのモデルはevo-x2で1つずつ動かす。

## 7. 二次情報しか無かったもの

JevBenchやDecision Indexの順位表は二次情報で、日本語のスコアは1件も載っていない。Microsoft-Decision-1の較正スコア92.2も二次情報だけである。日本のAzureリージョンでMicrosoft-Decision-1が使えるかは確認できなかったが、OpenRouter経由なら関係ない。
