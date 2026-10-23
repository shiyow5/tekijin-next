# Discordボットの立ち上げ(D1: /ask)

Discordで `/ask 質問文` と打つと、TEKIJINが答えを返す。計画書 6章のD1にあたる(Issue #60)。返事は質問した本人にだけ見える。質問には顧客名や社内のトラブルが入りうるうえ、打ったチャンネルを誰が見ているか分からないからだ。

| TEKIJINの判断 | Discordへの返事 |
|---|---|
| 知識・文書・過去の回答で答えた | 回答と出典 |
| 人に相談するのがよい | 推薦された人(部署と理由)と、WebのURL。依頼を送るには、Webで同じ質問をし直す |
| 質問が足りない | 聞き返しの文と、`/ask` のし直しかWebで続ける案内 |
| 答えられない | その旨 |

取次ぎの送信、アカウント連携、ボタンはD2以降で足す。

## 仕組み

ボットはバックエンドとは別のプロセスで動く(`python -m tekijin.discord_bot`)。スラッシュコマンドはGateway経由で届くので、ボットのために公開URLを用意する必要は無い。メッセージ本文を読まないので、特権intent(MESSAGE_CONTENT)も要らない。

ボットは受けた質問を、バックエンドの内部API `POST /internal/discord/ask` に送る。認証は共有シークレット(ヘッダ `X-Tekijin-Bot-Secret`)で行う。バックエンドはDiscordのユーザーIDを社員IDに置き換え、Webの質問と同じグラフを最後まで流し、結果を1つにまとめて返す。D1では、取次ぎの送信や聞き返しの続きをDiscordから行えない。そこで、返事を作ったあとにその実行を打ち切る。推薦された人の受信箱には、まだ送っていない依頼は載らない。

DiscordのユーザーIDと社員IDの対応は、D1では設定 `TEKIJIN_DISCORD_USER_MAP` に書く。対応表に無い人には「連携されていません」と返す。正式なアカウント連携はD2で作る。

## 手順

1. [Discord Developer Portal](https://discord.com/developers/applications) でアプリケーションを作り、Botタブでトークンを発行する。
2. OAuth2 の URL Generator で、スコープに `bot` と `applications.commands` を選ぶ。権限は無しでよい。生成されたURLを開き、ボットをサーバーに招待する。
3. 共有シークレットを作る。例: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
4. バックエンドの `.env` に次を書き、バックエンドを再起動する。

   ```dotenv
   TEKIJIN_DISCORD_BOT_SECRET=<3で作った値>
   TEKIJIN_DISCORD_USER_MAP={"123456789012345678": 10}
   TEKIJIN_DISCORD_WEB_URL=http://localhost:3000
   ```

   DiscordのユーザーIDは、Discordの設定で開発者モードを有効にし、ユーザーを右クリックして「ユーザーIDをコピー」で取れる。

5. ボットのプロセスに、4と同じ値に加えて次を渡して起動する。

   ```dotenv
   TEKIJIN_DISCORD_BOT_TOKEN=<1で発行したトークン>
   TEKIJIN_DISCORD_BACKEND_URL=http://127.0.0.1:8000
   TEKIJIN_DISCORD_GUILD_ID=<サーバーID>
   ```

   ```bash
   cd backend && PYTHONPATH=src python -m tekijin.discord_bot
   ```

   `TEKIJIN_DISCORD_GUILD_ID` を入れると、`/ask` はそのサーバーにすぐ出る。入れないと全サーバー向けの登録になり、表示まで最大1時間かかる。

`.env` は `chmod 600` にしておく。トークンとシークレットはコミットしない。

## 止め方

`TEKIJIN_DISCORD_BOT_SECRET` を空にしてバックエンドを再起動すると、内部APIは404を返し、Discordからの質問はすべて止まる。既定は空なので、設定しない限りバックエンドの動作は変わらない。
