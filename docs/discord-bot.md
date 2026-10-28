# Discordボットの立ち上げ

Discordで `/ask 質問文` と打つと、TEKIJINが答えを返す(Issue #60)。`/tekijin join` などでTEKIJINへの参加や設定ができる(Issue #66)。返事は質問した本人にだけ見える。質問には顧客名や社内のトラブルが入りうるうえ、打ったチャンネルを誰が見ているか分からないからだ。

| TEKIJINの判断 | Discordへの返事 |
|---|---|
| 知識・文書・過去の回答で答えた | 回答と出典 |
| 人に相談するのがよい | 推薦された人(部署と理由)と、WebのURL。依頼を送るには、Webで同じ質問をし直す |
| 質問が足りない | 聞き返しの文と、`/ask` のし直しかWebで続ける案内 |
| 答えられない | その旨 |

取次ぎの送信、アカウント連携、ボタンはD2以降で足す。

## 仕組み

ボットはバックエンドとは別のプロセスで動く(`python -m tekijin.discord_bot`)。スラッシュコマンドはGateway経由で届くので、ボットのために公開URLを用意する必要は無い。メッセージ本文を読まないので、特権intent(MESSAGE_CONTENT)も要らない。

ボットは受けた質問を、バックエンドの内部API `POST /internal/discord/ask` に送る。認証は共有シークレット(ヘッダ `X-Tekijin-Bot-Secret`)で行う。バックエンドはDiscordのユーザーをTEKIJINのメンバーに対応づけ、Webの質問と同じグラフを最後まで流し、結果を1つにまとめて返す。今は、取次ぎの送信や聞き返しの続きをDiscordから行えない。そこで、返事を作ったあとにその実行を打ち切る。推薦された人の受信箱には、まだ送っていない依頼は載らない。

TEKIJINは、`TEKIJIN_DISCORD_GUILD_ID` で指定した1つのサーバーを1つの組織として受け持つ。ほかのサーバーからの呼び出しには「このサーバーでは使えません」と返す。サーバーのメンバーは、初めて `/ask` などを打った時点で自動でTEKIJINのメンバーになる。対応表を書く必要は無い。

| コマンド | 内容 |
|---|---|
| `/tekijin join` | 参加を登録し、Webにログインするための一回限りのURL(10分)を本人だけに返す |
| `/tekijin optout` | 詳しい人として推薦される対象から外れる |
| `/tekijin optin` | 推薦される対象に戻る |
| `/tekijin add` | 文章をTEKIJINの知識に追加する。参加した人だけが使える。題名と、元の文書の置き場所のURLは任意で、同じURLを送り直すと更新になる(Issue #70) |

## 手順

1. [Discord Developer Portal](https://discord.com/developers/applications) でアプリケーションを作り、Botタブでトークンを発行する。
2. OAuth2 の URL Generator で、スコープに `bot` と `applications.commands` を選ぶ。権限は無しでよい。生成されたURLを開き、ボットをサーバーに招待する。
3. 共有シークレットを作る。例: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
4. バックエンドの `.env` に次を書き、バックエンドを再起動する。

   ```dotenv
   TEKIJIN_DISCORD_BOT_SECRET=<3で作った値>
   TEKIJIN_DISCORD_WEB_URL=http://localhost:3000
   TEKIJIN_DISCORD_GUILD_ID=<サーバーID>
   ```

   サーバーIDは、Discordの設定で開発者モードを有効にし、サーバー名を右クリックして「サーバーIDをコピー」で取れる。既存のDBには、先に `cd backend && PYTHONPATH=src python -m tekijin.data.migrate` で新しい列とテーブルを足す(データは消えない)。

5. ボットのプロセスに、4と同じ値に加えて次を渡して起動する。

   ```dotenv
   TEKIJIN_DISCORD_BOT_TOKEN=<1で発行したトークン>
   TEKIJIN_DISCORD_BACKEND_URL=http://127.0.0.1:8000
   ```

   ```bash
   cd backend && PYTHONPATH=src python -m tekijin.discord_bot
   ```

   `TEKIJIN_DISCORD_GUILD_ID` を入れると、コマンドはそのサーバーにすぐ出る。

`.env` は `chmod 600` にしておく。トークンとシークレットはコミットしない。

## 止め方

`TEKIJIN_DISCORD_BOT_SECRET` を空にしてバックエンドを再起動すると、内部APIは404を返し、Discordからの質問はすべて止まる。既定は空なので、設定しない限りバックエンドの動作は変わらない。
