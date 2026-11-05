# 架空のサークルの合成データ

架空の大学の天文サークル「星見ヶ丘天文同好会」1つ分のデータである。設計書 `docs/plans/2026-10-27-general-org-redesign.md` の4.7節にもとづき、回帰テストと閾値の調整に使う。人物、団体、店の名前はすべて架空である。実際のサーバーの発言は、公開リポジトリに置かない。

| ファイル | 内容 |
|---|---|
| `members.json` | メンバー20人 |
| `conversations.json` | Discordの会話。`chat-` で始まるものは雑談 |
| `documents.json`、`documents/` | 自己紹介スライド、企画書、マニュアル、引継ぎ書2本 |
| `questions.json` | 正解付きの質問 |

質問の `kind` は3種類ある。`answer` は資料で答えられる質問で、`sources` に出典、`contains` に答えの語句を持つ。`abstain` は答えるべきでない質問で、理由を `why` に書く。`person` は人に繋ぐ質問で、正しい人を `members` に書く。

会話と文書で内容が食い違う箇所は、新しい情報を選べるかを試すために残している。たとえば宿代は、引継ぎ書の7500円が去年の値で、会話の8000円が今年の値である。

投入には次のコマンドを使う。

```bash
cd backend && PYTHONPATH=src python -m tekijin.data.migrate
PYTHONPATH=src python -m tekijin.data.circle
```

専用の組織に入るので、ほかの組織のデータには触らない。何度流しても結果は同じになる。

質問をすべて実際の処理の流れに通して数えるには、`scripts/circle_eval.py` を使う。DBは読むだけで、書き込みはしない。LLMの設定を環境変数で渡さないとスタブで動くので、出力の1行目で使ったモデルを確かめる。

```bash
PYTHONPATH=backend/src TEKIJIN_LLM_BACKEND=vllm TEKIJIN_LLM_BASE_URL=... TEKIJIN_LLM_MODEL=... \
  python scripts/circle_eval.py --rows
```

種類ごとに、断った数、聞き返し、取次ぎ、根拠付きの答え、取次ぎと一緒に見せた答え、出典が正しい数、答えの語句を含む数、正しい人が3位以内に入った数、経路の内訳を出す。
