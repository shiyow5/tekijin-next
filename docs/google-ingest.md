# Googleドキュメントとスライドの取り込み

GoogleドキュメントとGoogleスライドを、TEKIJINの資料として定期的に取り込む(Issue #99)。取り込んだ文書は資料の一覧に出て、質問への回答の出典になる。

TEKIJINは、Googleのサービスアカウントとして文書を読む。サービスアカウントは、プログラムが使うGoogleのアカウントで、自分のメールアドレスを持つ。メンバーが文書やフォルダをこのアドレスに共有すると、TEKIJINはそれを読めるようになる。組織のGoogle管理コンソールの権限は要らない。

## 仕組み

取り込みは、バックエンドとは別のプロセスで動く(`python -m tekijin.ingest.poll`)。このプロセスは既定では15分ごとに、サービスアカウントが読めるドキュメントとスライドをDrive APIのfiles.listで一覧する。共有ドライブの中のものも含み、ゴミ箱の中のものは含まない。

前回から更新日時も題名も変わっていない文書は読まない。新しい文書と変わった文書は、files.exportでテキストに書き出して保存する。題名はファイル名、出典のURLはGoogleでその文書を開くURLになる。一覧に無くなった文書(共有を外した、ゴミ箱に入れた)は、TEKIJINからも消す。

一覧の取得に失敗した回は、何も消さない。1件の書き出しに失敗しても、残りの文書の取り込みは続ける。失敗した文書は次の回にもう一度読む。Googleが混み合っているという応答(429や5xx)には、間を空けて数回まで再試行する。10MBを超える文書はGoogleが書き出しを断るので、ログに記録して飛ばす。

取り込んだ文書は、組織のメンバー全員が見られる。サービスアカウントに共有するのは、メンバー全員に見せてよい文書だけにする。

## 準備(初回だけ)

### 1. Google Cloud のプロジェクトを作る

1. [Google Cloud コンソール](https://console.cloud.google.com/) に、組織の管理に使うGoogleアカウントでログインする
2. 上部のプロジェクト選択から「新しいプロジェクト」を選び、名前(例: `tekijin`)を付けて作る

### 2. Drive API を有効にする

1. 作ったプロジェクトを選んだ状態で、「APIとサービス」→「ライブラリ」を開く
2. 「Google Drive API」を検索して開き、「有効にする」を押す

Docs API と Slides API は要らない。書き出しは Drive API だけで行う。

### 3. サービスアカウントと鍵を作る

1. 「IAMと管理」→「サービスアカウント」を開き、「サービスアカウントを作成」を押す
2. 名前(例: `tekijin-reader`)を付けて作る。プロジェクトへのロールは付けなくてよい
3. 作ったサービスアカウントを開き、メールアドレス(`tekijin-reader@<プロジェクトID>.iam.gserviceaccount.com` の形)を控える。共有のときに使う
4. 「鍵」タブで「鍵を追加」→「新しい鍵を作成」→「JSON」を選ぶ。JSONファイルが1つダウンロードされる

このJSONファイルは、サービスアカウントとしてログインするための秘密の鍵である。メールやチャットで送らない。リポジトリに入れない。

組織のポリシーで鍵の作成が禁止されていると、4でエラーになる。そのときは、Google Workspace の管理者に `iam.disableServiceAccountKeyCreation` の制約を外してもらう必要がある。

### 4. 鍵をサーバーに置く

鍵は、リポジトリの外の、TEKIJINを動かすユーザーだけが読める場所に置く。

```bash
mkdir -p ~/secrets && chmod 700 ~/secrets
mv ~/Downloads/<ダウンロードした鍵>.json ~/secrets/google-sa.json
chmod 600 ~/secrets/google-sa.json
```

ほかのユーザーも読める権限のままだと、起動時に警告を出す。

### 5. 環境変数を設定する

リポジトリ直下の `.env` に、鍵のパスを書く。取り込み先の組織は `TEKIJIN_DISCORD_GUILD_ID` で決まるので、Discordボットと同じ値が入っていることを確かめる。

```dotenv
TEKIJIN_GOOGLE_SERVICE_ACCOUNT_FILE=/home/<ユーザー名>/secrets/google-sa.json
TEKIJIN_DISCORD_GUILD_ID=<サーバーID>
# 取り込みの間隔(秒)。省略すると900秒(15分)
TEKIJIN_POLL_INTERVAL_S=900
```

`TEKIJIN_GOOGLE_SERVICE_ACCOUNT_FILE` が空なら、Googleからは取り込まない。ほかに取り込み先が無ければ、取り込みのプロセスは理由を表示して終了する。

依存のパッケージ(`google-auth`)を入れていなければ、先に入れる。

```bash
cd backend && pip install -r requirements.txt
```

## 文書を共有する

取り込みたい文書かフォルダを開き、「共有」から、3で控えたサービスアカウントのメールアドレスを追加する。権限は「閲覧者」でよい。Googleが通知メールを送るか聞いてきたら、送らなくてよい。

フォルダを共有すると、その中のドキュメントとスライドがすべて対象になる。あとからフォルダに入れた文書も、次の回に取り込まれる。共有ドライブなら、サービスアカウントを共有ドライブのメンバー(閲覧者)に加える。

取り込むのは、Googleドキュメントとスライドだけである。PDF、Word、スプレッドシートなどは取り込まない。

TEKIJINから消したい文書は、サービスアカウントとの共有を外す。資料の画面で消しても、共有が残っていれば次の回にまた取り込まれる。

## 起動と停止

まず1回だけ取り込んで、動くことを確かめる。

```bash
cd backend && PYTHONPATH=src python -m tekijin.ingest.poll --once
```

ログの最後に、文書ごとの件数(`listed`、`added`、`updated`、`deleted` など)が出る。全部うまくいけば終了コードは0、一覧の取得や文書の取り込みに1件でも失敗すると1になる。設定が足りないときは2で終了し、足りない設定の名前を表示する。

定期的に取り込むときは、`--once` を付けずに起動する。

```bash
cd backend && PYTHONPATH=src python -m tekijin.ingest.poll
```

止めるときは、Ctrl+Cを押すか、プロセスにSIGTERMを送る。取り込みの途中なら、その回を終えてから止まる。常駐させるなら、Discordボットと同じように、systemdやtmuxなどで起動しておく。

最初の回は、共有された文書をすべて書き出して埋め込みを作るので、文書が多いと時間がかかる。2回目からは、変わった文書だけを読む。

## うまくいかないとき

| 症状 | 確かめること |
|---|---|
| `does not point to a file` で終了する | `TEKIJIN_GOOGLE_SERVICE_ACCOUNT_FILE` のパスが合っているか |
| `could not read the service account key file` で終了する | 鍵のファイルが、ダウンロードしたJSONのままか |
| `listing failed (HTTP 403)` が出る | プロジェクトで Drive API を有効にしたか |
| `listed=0` のまま増えない | 文書をサービスアカウントのメールアドレスに共有したか |
| `skipped ... export limit` が出る | その文書は10MBを超えている。分けるか、要る部分だけ別の文書にする |

ログには、鍵の中身もアクセストークンも出さない。
