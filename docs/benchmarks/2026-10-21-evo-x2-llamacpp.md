# evo-x2 の llama.cpp で測り直した結果(2026-10-21)

インターン中のDGX Sparkが使えなくなったので、自宅のevo-x2でLLMを動かし、全グラフ評価をDGXでの基準と比べた。移行計画([2026-10-21-personal-continuation-plan.md](../plans/2026-10-21-personal-continuation-plan.md))の段階2にあたる。

## 1. 環境

| 項目 | DGX(基準) | evo-x2 |
|---|---|---|
| 機材 | DGX Spark(GB10) | Ryzen AI Max+ 395、メモリ64GB(GPUに回せる分は48GiB) |
| LLM の配信 | vLLM | llama.cpp b11756(Vulkan、ビルド済みバイナリ) |
| モデル | Qwen3.6-35B-A3B-NVFP4 | Qwen3.6-35B-A3B UD-Q4_K_XL(unsloth、20.8GiB) |
| 埋め込み | Nemotron-3-Embed-1B(CPU) | 同じモデル(CPU、torch 2.14.1+cpu) |
| DB | PostgreSQL 16 + pgvector | `pgvector/pgvector:0.8.6-pg16` |
| 評価 | `scripts/research_fullgraph_eval.py`、87行 | 同じ |

llama-serverは次の設定で起動した。待ち受けはevo-x2の中だけにしている。

```bash
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --alias qwen3.6-35b-a3b --jinja \
  -ngl 99 -c 32768 --parallel 4 -fa on --host 127.0.0.1 --port 8080
```

アプリ側の設定は次のとおり。

```bash
TEKIJIN_LLM_BACKEND=vllm
TEKIJIN_LLM_BASE_URL=http://127.0.0.1:8080/v1
TEKIJIN_LLM_MODEL=qwen3.6-35b-a3b
TEKIJIN_LLM_THINKING_CONTROL=chat_template
TEKIJIN_LLM_STRUCTURED_OUTPUT_METHOD=json_schema
```

## 2. 構造化出力は json_schema 方式にする

LangChainのfunction_calling方式は、`tool_choice`に関数を指定するオブジェクトを送って、ツールの呼び出しを強制する。llama-serverはこの形式を受け付けず、「Wrong type supplied for parameter 'tool_choice'」と警告して既定のautoに戻す。モデルがツールを呼ばなかった行ではC1の結果が空になり、評価が失敗する。function_callingで流した7回(自己回答ON 4回、本番の既定3回)のうち4回で、87行のうち同じ1行(id=50)が失敗した。

json_schema方式では、llama-serverが出力をスキーマで縛るので、この失敗が起きない。5回の実行で失敗はゼロだった。同じ構成の結果は、実行のたびに完全に一致した。

## 3. 品質

「自己回答ON」は`--self-answer`だけを立てた構成である。「本番の既定」は、これに`--question-fit --additive --daily-knowledge`を加えた構成で、今の既定値と同じ機能がONになる。

| 指標 | DGX | evo-x2 function_calling | evo-x2 json_schema |
|---|---|---|---|
| Hit@3(自己回答ON) | 0.727(ADR-0008の基準) | 0.697〜0.712(4回) | 0.727(3回とも) |
| Hit@3(本番の既定) | 0.788(#405) | 0.742〜0.758(3回) | 0.788(2回とも) |
| 人に回すべき質問の取りこぼしゼロ | 49/49 | 48〜49/49 | 49/49 |
| データで答えるべき質問を自己回答側へ | 11/23 | 11/23 | 11/23 |
| 経路の正解率(自己回答ON) | 下の注を参照 | 0.736〜0.750 | 0.778 |
| C1 トピック acc@1 | 0.72〜0.81 | 0.721〜0.735 | 0.721 |
| 自己回答の出典 recall / precision(自己回答ON) | 0.239 / 0.739 | 0.239 / 0.722 | 0.163 / 0.875 |
| 評価が失敗した行 | 0 | 0〜1 | 0 |

経路の正解率は、DGXの0.833とそのまま比べられない。DGXの0.833は、自己回答を配線しない構成の値である。指標は予測と正解の完全一致で数えるので、自己回答が出典付きで答えて「自己回答」で終わった行は、正解が文書経路でも不正解になる。evo-x2のjson_schemaで、文書経路が正解の11行は、すべて文書か自己回答に振り分けられた。自己回答で終わった4行を正解に数えると60/72 = 0.833になり、DGXと一致する。LLMを使わない`make eval`でも0.833だった。

json_schema方式では、自己回答の出典のprecisionが上がり、recallが下がった。引用を絞る方向に変わったということだ。誤った出典を出しにくくなった一方で、答えられる質問の一部に答えなくなっている。

## 4. 応答時間

`scripts/research_latency.py --repeat 3`で、4問を3回ずつ測った(中央値、ミリ秒)。

| ノード | DGX | evo-x2 json_schema | evo-x2 function_calling |
|---|---|---|---|
| C1 意図解析 | 約1,500 | 1,905 | 2,479 |
| C2 充足判定 | | 848 | 1,065 |
| C7 下書き | 約1,100 | 1,586 | 1,361 |
| compose(自己回答・追記型) | 約500 | 664 | 1,282 |
| 人へ取次ぎの合計(追記型あり) | | 5,003 | 6,187 |

C1は目安の4秒以内に収まった。全グラフ評価の1回(87行)は、json_schemaで約7〜9分かかった。

## 5. 結論

evo-x2のllama.cppでも、json_schema方式にすればDGXと同じ品質が出る。Hit@3、人に回すべき質問の取りこぼしゼロ、自己回答への振り分けは、DGXの値と一致した。違いは、自己回答の出典がやや慎重になったことと、応答が1.2〜1.4倍遅いことの2点である。

## 6. ファイル

| ファイル | 中身 |
|---|---|
| `ablation/evox2_fg_base_json_schema_run1.json` | 自己回答ON・json_schema。行ごとの記録つき |
| `ablation/evox2_fg_prod_json_schema_run1.json` | 本番の既定・json_schema。行ごとの記録つき |
| `ablation/evox2_fg_base_function_calling_run4.json` | 自己回答ON・function_calling。行ごとの記録つき(id=50 が失敗) |
| `ablation/evox2_fg_prod_run1.json` | 本番の既定・function_calling(行ごとの記録を入れる前の実行) |
| `ablation/evox2_latency_{json_schema,function_calling}.json` | 応答時間 |
