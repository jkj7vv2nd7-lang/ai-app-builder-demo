# ローカルAIアプリ開発エージェント

アイデアを入力すると、ローカルのOllamaモデルが「要件定義 → デザイン設計 → 実装 → QA」の順に単一HTMLアプリを作ります。WindowsではGUIまたはデスクトップのバッチランチャーから起動できます。

> **GitHub Pagesについて:** このリポジトリのPagesサイトは、エージェントの紹介とセットアップ案内を表示する静的ページです。Python、Tkinter、Ollamaはブラウザー上では動きません。アプリを使うには、このリポジトリをダウンロードまたはcloneし、PythonとOllamaを準備してPC上で起動してください。

## 必要なもの

- Python 3.9以降。WindowsではTkinterを含むPythonをインストールしてください。
- [Ollama](https://ollama.com/download)をインストールし、生成中は起動しておきます。
- 既定の単一モデル: `qwen2.5-coder:7b`
- おすすめ自動ルーティング用: 要件・設計 `qwen3:8b`、実装・QA `qwen2.5-coder:7b`

モデルの初回ダウンロードにはインターネット通信とディスク容量が必要です。使用する正確なモデルタグのライセンスと利用条件を、配布元で確認してください。

## ダウンロードと起動

Gitがある場合はPowerShellでcloneします。

```powershell
git clone https://github.com/jkj7vv2nd7-lang/ai-app-builder-demo.git
cd ai-app-builder-demo
```

Gitを使わない場合は、GitHubの **Code > Download ZIP** から取得し、展開したフォルダーを開いてください。

単一モデルを取得してGUIを起動します。

```powershell
ollama pull qwen2.5-coder:7b
python .\app_gui.py
```

Windowsでは`launch_gui.bat`をダブルクリックしても起動できます。デスクトップに置く場合は、バッチファイルを右クリックして「ショートカットの作成」を選び、作成したショートカットをデスクトップへ移動します。

コマンドラインから直接実行することもできます。

```powershell
python .\app_builder.py "中学生向け英単語クイズアプリ"
```

### おすすめ自動ルーティング

GUIで「おすすめ自動ルーティング」を選ぶか、CLIで`--routing-mode auto`を指定します。必要モデルがない場合、アプリは自動ダウンロードや黙ったフォールバックをせず、必要なモデル名を案内して停止します。

```powershell
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
python .\app_builder.py "家計簿アプリ" --routing-mode auto
```

自動ルーティングは役割別に定めた固定表であり、ベンチマークでモデルを探索する機能ではありません。モデルはAPI呼び出し後にアンロードされるため、切り替え時にロード待ちがあります。16GB RAM・CPUの目安はコンテキスト長4096です。メモリ不足や遅さがある場合はGUIで2048に下げ、小さなモデルを検討してください。

## 入力例と成果物

最初の入力は、普段の言葉で書いた一文のアイデアで十分です。対象者や画面構成、技術用語を最初から決める必要はありません。分野別の短い例は[入力プロンプト例](./prompt_examples.md)を参照してください。

生成されたアプリは`runs\run_<日時>_<識別子>\`に保存されます。要件、設計、AIの実装・QA応答、静的チェック結果、ログ、状態ファイルを同じ実行フォルダーで確認できます。既存の実行結果やパッケージ直下のファイルは上書きしません。

## このパッケージのGitHub Pagesサイト

このリポジトリには、エージェントの紹介・セットアップ用の`index.html`が含まれます。自分の公開URLで同じ案内ページを表示するには、GitHubでこのリポジトリを開き、**Settings > Pages > Build and deployment**へ進みます。

1. Sourceで **Deploy from a branch** を選択
2. Branchで **main**、フォルダーで **/(root)** を選択
3. **Save**を押し、GitHubのデプロイ完了を待つ

公開後のURLは通常`https://<owner>.github.io/<repository>/`です。このページは説明用の静的サイトで、AIアプリ開発エージェントをWeb上で実行するものではありません。

## 生成したアプリを公開する場合

GUIには生成HTMLをGitHubへアップロードする機能があります。公開リポジトリはソースも誰でも閲覧できるため、内容を確認してから操作してください。アプリはリポジトリを作成せず、ユーザー作成済みの公開リポジトリURLを要求します。既定ブランチに`index.html`が既にある場合は上書きせず停止します。そのため、生成HTMLの公開先にはこの配布用リポジトリではなく、別の空の公開リポジトリを使ってください。

GitHub CLIとGitを導入し、PowerShellで認証します。

```powershell
gh auth login
gh auth setup-git
```

Git認証がすでに設定済みなら`gh auth setup-git`は不要です。GUIの確認画面で対象リポジトリと`index.html`だけが公開されることを確認してアップロードします。既存内容の上書き、強制push、Pagesの有効化は行いません。アップロード後は対象リポジトリの **Settings > Pages** で **Deploy from a branch**、既定ブランチ、`/(root)`を選んで保存してください。アプリはトークンを保存しません。

## 制約と安全上の注意

- 生成物は単一HTMLを前提とします。静的チェックとAI QAだけでは、実ブラウザーでの動作を保証しません。Playwrightによるローカル検証は任意です。
- 外部通信検出はヒューリスティックで、すべてのネットワークアクセスを検知できる保証はありません。Ollama生成リクエストはローカルの`localhost`へ送信します。
- GitHub Pagesや公開リポジトリを使うと、サイトとソースがインターネットから閲覧可能になります。個人情報、認証情報、秘密情報を含む内容を公開しないでください。

## テスト

追加パッケージなしで、パッケージのルートから実行できます。

```powershell
python -m unittest discover -s .\tests -v
```

公開機能のテストはGit/GitHub CLIをモックします。実際のGitHubへのアップロードやGitHub Pagesのデプロイを検証するものではありません。

## ライセンス

このソフトウェアは[MIT License](./LICENSE)で提供します。OllamaやAIモデルは別の利用条件に従います。
