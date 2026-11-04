# `fixtures/source/` — 合成データの生成元（一次データ）

`scripts/build_fixtures.py` の**入力**だった。ここから `fixtures/synthetic/` を生成していた。

> 生成スクリプトは #89 で、日報・案件から作った合成データは #97 で develop から消した(`company-v1` タグに残る)。
> 今の develop にはここを読むコードが無い。会社向けの版の生成元の記録として残してある。

| ファイル | 中身 | 作成 |
|---|---|---|
| `employees.json` | 社員40名（氏名・部署・役職・入社日など） | reona（#17） |
| `case_history_dummy.json` | 案件履歴120件（顧客・商材・課題・担当者名） | reona（#17） |
| `daily_report_dummy.json` | 日報3,070件 | reona（#17） |
| `chat_history_dummy.json` | チャット2,000件 | reona（#17） |

```bash
python3 scripts/build_fixtures.py     # company-v1 タグで実行する。ここを入力に fixtures/synthetic/ を再生成
```

## なぜここに置くか

**以前は生成元がリポジトリ外の一時ディレクトリ（`/tmp/claude-1000/.../scratchpad`）にしか無かった。**
`scripts/build_fixtures.py` の `DEFAULT_INPUT` がそこを指しており、
**その一時ディレクトリが消えた時点で `fixtures/synthetic/` は誰にも再生成できなくなる**状態だった。

`fixtures/synthetic/` は生成物なので、生成元が失われると
「中身は分かるが、なぜそうなっているかを追えず、直すこともできない」データになる。
合成データなので秘密情報は含まれず、合計 1.2MB 程度なのでコミットして困らない。

## 注意

- **ここを直接編集しない。** 変更したい場合は `scripts/build_fixtures.py` の変換規則側で吸収する。
  一次データを書き換えると、reona 版との差分が追えなくなる
- 生成物（`fixtures/synthetic/`）を手で編集するのも同じ理由で禁止。必ずスクリプト経由で再生成する
