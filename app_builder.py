#!/usr/bin/env python3
"""Generate a single-file web app with a staged Ollama agent workflow."""

import argparse
import json
import re
import secrets
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-coder:7b"
AUTO_MODEL_ROUTING = {
    "requirements": "qwen3:8b",
    "design": "qwen3:8b",
    "implementation": "qwen2.5-coder:7b",
    "qa": "qwen2.5-coder:7b",
}
EXTERNAL_URL_PATTERN = re.compile(r"(?:https?:)?//[^\s\"'<>`()]+", re.IGNORECASE)
NETWORK_API_PATTERN = re.compile(
    r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|WebTransport)\s*\(|"
    r"\bimport\s*\(|\bsendBeacon\s*\(|serviceWorker\s*\.\s*register\s*\(",
    re.IGNORECASE,
)

REQUIREMENTS_SYSTEM = """あなたはプロダクトマネージャーです。
入力されたアプリ案から、対象ユーザー、目的、機能要件、画面と操作、データ、
今回の対象外と前提を具体的に整理してください。必ず「機能要件」「非機能要件」
「受け入れ条件」の各見出しを作り、受け入れ条件はテスト可能な箇条書きにしてください。
不明な点は安全で一般的な前提を明示し、日本語で回答してください。"""

DESIGN_SYSTEM = """あなたはUI/UXデザイナー兼フロントエンド設計者です。
要件に基づき、画面構成、レスポンシブ対応、配色、操作と状態遷移、
アクセシビリティ上の配慮を設計してください。外部ライブラリやCDNに頼らない
ローカル実装の方針を含め、実装可能な具体性を保ち、日本語で回答してください。"""

IMPLEMENTATION_SYSTEM = """あなたは熟練したフロントエンドエンジニアです。
要件とデザインに基づく、ブラウザーで動く完全な単一HTMLアプリを作成してください。
HTML、CSS、JavaScriptを1ファイルに含め、必要な操作を実装してください。
外部CDN、外部フォント、外部画像、外部API、ネットワーク通信に依存せず、
オフラインのローカルファイルとして動作させてください。必要なデータや装飾は
HTML内に含め、ブラウザーのlocalStorageなどを使用してください。
説明やMarkdownのコードフェンスを避け、HTML全体を出力してください。
コードを省略せず、サーバー側テンプレート構文を使わないでください。"""

QA_SYSTEM = """あなたはQAエンジニアです。HTMLを確認し、必要なら問題を修正してください。
出力の最初に次のいずれかを1行で記してください:
QA_STATUS: APPROVED
QA_STATUS: NEEDS_REVISION
続けて確認事項を簡潔に書き、最後に修正後を含む完全なHTML全体を出力してください。
コード全体を返せない場合は、その理由と確認事項を記し、HTMLを捏造しないでください。
外部CDN/APIなどのネットワーク依存があれば指摘してください。
静的確認だけでブラウザー実行済みと主張しないでください。"""


class StageError(RuntimeError):
    """A failure tied to a specific workflow stage."""

    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def call_ollama(prompt, system_prompt, args, stage):
    model = selected_model(args, stage)
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except OSError:
            detail = ""
        raise StageError(stage, f"Ollama returned HTTP {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            message = f"connection timed out after {args.timeout:g} seconds"
        else:
            message = f"could not connect to Ollama: {reason}"
        raise StageError(stage, message) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise StageError(stage, f"request timed out after {args.timeout:g} seconds") from exc
    except (UnicodeDecodeError, OSError) as exc:
        raise StageError(stage, f"could not read Ollama response: {exc}") from exc

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise StageError(stage, f"Ollama returned invalid JSON: {exc}") from exc

    answer = result.get("response") if isinstance(result, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        detail = result.get("error") if isinstance(result, dict) else None
        detail_text = f": {detail}" if detail else ""
        raise StageError(stage, f"Ollama returned no response text{detail_text}")
    return answer.strip()


def extract_html(response):
    """Extract the first complete-looking HTML document from model output."""
    text = response.strip().lstrip("\ufeff")
    fences = re.findall(r"```(?:html|HTML)?\s*([\s\S]*?)```", text)
    candidates = fences + [text]
    for candidate in candidates:
        match = re.search(r"<!doctype\s+html\b|<html\b", candidate, re.IGNORECASE)
        if match:
            return candidate[match.start():].strip()
    return None


class _DocumentParser(HTMLParser):
    STRUCTURAL_TAGS = {"html", "head", "body"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = set()
        self.structural_stack = []
        self.errors = []
        self.inline_scripts = []
        self.body_has_content = False
        self.external_dependencies = set()
        self._script_parts = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self.tags.add(tag)
        attr_map = dict(attrs)
        for attr_name, value in attrs:
            if value and EXTERNAL_URL_PATTERN.search(value):
                self.external_dependencies.add(f"<{tag}> {attr_name}={value}")
        if "body" in self.structural_stack and (tag != "style") and (
            tag != "script" or "src" in attr_map
        ):
            self.body_has_content = True
        if tag in self.STRUCTURAL_TAGS:
            if tag in self.structural_stack:
                self.errors.append(f"duplicate or nested <{tag}> element")
            self.structural_stack.append(tag)
        if tag == "script":
            self._script_parts = [] if "src" not in attr_map else None

    def handle_startendtag(self, tag, attrs):
        tag = tag.lower()
        self.tags.add(tag)
        if "body" in self.structural_stack and tag not in {"script", "style"}:
            self.body_has_content = True

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "script" and self._script_parts is not None:
            self.inline_scripts.append("".join(self._script_parts))
            self._script_parts = None
        if tag in self.STRUCTURAL_TAGS:
            if not self.structural_stack or self.structural_stack[-1] != tag:
                self.errors.append(f"unexpected </{tag}> element")
            else:
                self.structural_stack.pop()

    def handle_data(self, data):
        if "body" in self.structural_stack and data.strip():
            self.body_has_content = True
        if self._script_parts is not None:
            self._script_parts.append(data)


def validate_html(html_content):
    """Run dependency-free HTML checks and an optional Node.js syntax check."""
    parser = _DocumentParser()
    errors = []
    try:
        parser.feed(html_content)
        parser.close()
    except Exception as exc:
        errors.append(f"HTML parsing failed: {exc}")

    errors.extend(parser.errors)
    external_dependencies = set(parser.external_dependencies)
    external_dependencies.update(
        url.rstrip(".,;:") for url in EXTERNAL_URL_PATTERN.findall(html_content)
    )
    network_apis = sorted(set(NETWORK_API_PATTERN.findall(html_content)))
    for tag in ("html", "head", "body"):
        if tag not in parser.tags:
            errors.append(f"missing <{tag}> element")
    if "body" in parser.tags and not parser.body_has_content:
        errors.append("<body> has no content")
    if parser.structural_stack:
        errors.append("unclosed structural elements: " + ", ".join(parser.structural_stack))

    js_status = "skipped"
    js_detail = "Node.js was not found; inline JavaScript syntax was not checked."
    node = shutil.which("node")
    if node:
        js_status = "checked"
        js_detail = "No inline JavaScript syntax errors found."
        for index, script in enumerate(parser.inline_scripts, start=1):
            if not script.strip():
                continue
            try:
                result = subprocess.run(
                    [node, "--check"],
                    input=script,
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"JavaScript check failed for inline script {index}: {exc}")
                js_detail = "Node.js check could not complete."
                break
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()
                errors.append(f"JavaScript syntax error in inline script {index}: {detail}")
                js_detail = "One or more inline scripts contain syntax errors."

    return {
        "passed": not errors,
        "errors": errors,
        "external_dependencies": sorted(external_dependencies),
        "network_api_usage": network_apis,
        "javascript": {"status": js_status, "detail": js_detail},
    }


def run_browser_check(html_content):
    """Optionally render the app with local Playwright while blocking network access."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {
            "status": "skipped",
            "reason": "Playwrightが未導入です。基本フローは静的検証で続行します。",
            "page_errors": [],
            "blocked_requests": [],
        }

    page_errors = []
    blocked_requests = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.on("pageerror", lambda error: page_errors.append(str(error)))

                def block_network(route):
                    scheme = urlparse(route.request.url).scheme.lower()
                    if scheme in {"http", "https"}:
                        blocked_requests.append(route.request.url)
                        route.abort()
                    else:
                        route.continue_()

                page.route("**/*", block_network)
                page.set_content(html_content, wait_until="load", timeout=15000)
                body_text = page.locator("body").inner_text(timeout=5000)
                if not body_text.strip() and not page.locator("body").locator("*").count():
                    page_errors.append("Rendered body is empty.")
            finally:
                browser.close()
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"ローカルブラウザー検証を実行できません: {exc}",
            "page_errors": page_errors,
            "blocked_requests": blocked_requests,
        }

    return {
        "status": "passed" if not page_errors else "failed",
        "reason": "実ブラウザーの描画を確認しました。ネットワーク要求は遮断しています。",
        "page_errors": page_errors,
        "blocked_requests": sorted(set(blocked_requests)),
    }


def selected_model(args, stage):
    if getattr(args, "routing_mode", "single") == "auto":
        return AUTO_MODEL_ROUTING[stage]
    option_name = {
        "requirements": "planning_model",
        "design": "planning_model",
        "implementation": "implementation_model",
        "qa": "qa_model",
    }.get(stage)
    return getattr(args, option_name, None) or args.model


def models_for_run(args):
    return {
        "planning": selected_model(args, "requirements"),
        "implementation": selected_model(args, "implementation"),
        "qa": selected_model(args, "qa"),
    }


def fetch_ollama_model_names(endpoint, timeout):
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise StageError("connection", "This workflow only sends model requests to local Ollama on localhost.")
    tags_url = f"http://{parsed.netloc}/api/tags"
    request = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise StageError("connection", f"Ollama model-list API returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise StageError("connection", f"Could not connect to local Ollama: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise StageError("connection", f"Ollama model check timed out after {timeout:g} seconds.") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("connection", f"Could not read local Ollama model list: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise StageError("connection", "Ollama returned an invalid model list.")
    return [
        item["name"]
        for item in data["models"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def extract_qa_status(response):
    match = re.search(
        r"^\s*QA_STATUS:\s*(APPROVED|NEEDS_REVISION)\s*$",
        response,
        re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).upper() if match else None


def resolve_output_root(out_arg):
    """Resolve --out and reject paths outside this portable package folder."""
    tool_dir = Path(__file__).resolve().parent
    requested = Path(out_arg)
    output_root = (requested if requested.is_absolute() else tool_dir / requested).resolve()
    try:
        output_root.relative_to(tool_dir)
    except ValueError as exc:
        raise ValueError(f"--out must stay inside {tool_dir}") from exc
    return output_root


def create_run_directory(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}_{secrets.token_hex(3)}"
    run_dir.mkdir()
    return run_dir


def write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Ollamaによる段階型AIアプリ開発ツール（成果物はこのパッケージ内に保存）"
    )
    parser.add_argument("request", nargs="*", help="作りたいアプリの説明。省略時は標準入力を読みます")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="ローカルOllamaの /api/generate URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollamaモデル名")
    parser.add_argument(
        "--routing-mode",
        choices=("single", "auto"),
        default="single",
        help="single: 全工程で同じモデル（既定）。auto: 役割ごとの固定おすすめモデルを使用",
    )
    parser.add_argument("--planning-model", help="要件・設計用モデル（省略時は--model）")
    parser.add_argument("--implementation-model", help="実装用モデル（省略時は--model）")
    parser.add_argument("--qa-model", help="QA用モデル（省略時は--model）")
    parser.add_argument("--timeout", type=float, default=600, help="APIタイムアウト秒数")
    parser.add_argument("--temperature", type=float, default=0.3, help="生成温度")
    parser.add_argument("--num-predict", type=int, default=4096, help="生成トークン上限")
    parser.add_argument("--num-ctx", type=int, default=4096, help="コンテキスト長（16GB RAM CPU向け既定値）")
    parser.add_argument("--max-qa-revisions", type=int, default=2, help="QAによる修正再試行の最大回数")
    parser.add_argument("--out", default="runs", help="パッケージ内の出力ルート")
    parser.add_argument(
        "--browser-check",
        action="store_true",
        help="Playwrightが導入済みの場合にChromiumでHTMLを任意検証する",
    )
    return parser


def validate_args(parser, args):
    parsed = urlparse(args.endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        parser.error("--endpointにはローカルOllamaのhttp://localhost形式のURLを指定してください")
    if args.timeout <= 0 or args.num_predict <= 0 or args.num_ctx <= 0:
        parser.error("--timeout、--num-predict、--num-ctxは正の値が必要です")
    if args.max_qa_revisions < 0:
        parser.error("--max-qa-revisionsは0以上を指定してください")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperatureは0から2の範囲で指定してください")
    if args.routing_mode == "auto" and (
        args.planning_model or args.implementation_model or args.qa_model
    ):
        parser.error("--routing-mode autoでは工程別モデル指定を使わないでください")


def run_pipeline(request_text, args, run_dir, progress_callback=None):
    models = models_for_run(args)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "request": request_text,
        "model": args.model,
        "models": models,
        "routing_mode": getattr(args, "routing_mode", "single"),
        "num_ctx": args.num_ctx,
        "endpoint": args.endpoint,
        "status": "running",
        "qa_approved": None,
        "stages": {},
    }
    log_path = run_dir / "run.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
        if progress_callback:
            progress_callback(message)

    def save_stage(stage, filename, content):
        (run_dir / filename).write_text(content.rstrip() + "\n", encoding="utf-8")
        manifest["stages"][stage] = "completed"
        write_json(run_dir / "run.json", manifest)

    write_json(run_dir / "run.json", manifest)
    log(f"[INFO] 出力先: {run_dir}")
    log(
        f"[INFO] モード: {'おすすめ自動ルーティング' if getattr(args, 'routing_mode', 'single') == 'auto' else '単一/手動モデル'}"
        f" / 工程別モデル: {models} / API: {args.endpoint}"
    )
    try:
        log("[1/4] 要件定義を作成中...")
        requirements = call_ollama(
            f"次のアプリ案について要件を定義してください。\n\nアプリ案:\n{request_text}",
            REQUIREMENTS_SYSTEM,
            args,
            "requirements",
        )
        save_stage("requirements", "01_requirements.md", requirements)

        log("[2/4] デザイン設計を作成中...")
        design = call_ollama(
            f"次の要件に基づいて画面と操作を設計してください。\n\n{requirements}",
            DESIGN_SYSTEM,
            args,
            "design",
        )
        save_stage("design", "02_design.md", design)

        log("[3/4] HTMLアプリを実装中...")
        raw_implementation = call_ollama(
            f"アプリ案:\n{request_text}\n\n要件:\n{requirements}\n\nデザイン:\n{design}",
            IMPLEMENTATION_SYSTEM,
            args,
            "implementation",
        )
        save_stage("implementation", "03_implementation_response.md", raw_implementation)
        current_html = extract_html(raw_implementation)
        if not current_html:
            raise StageError("implementation", "AI応答からHTML文書を抽出できませんでした")
        checks = validate_html(current_html)
        write_json(run_dir / "03_static_checks.json", checks)
        if not checks["passed"]:
            raise StageError("implementation", "初版HTMLの静的検証に失敗しました: " + "; ".join(checks["errors"]))
        (run_dir / "index.html").write_text(current_html.rstrip() + "\n", encoding="utf-8")

        log("[4/4] QAレビューを実行中...")
        qa_rounds = []
        qa_approved = False
        qa_note = ""
        for round_number in range(args.max_qa_revisions + 1):
            round_label = round_number + 1
            log(f"[QA] レビュー {round_label}/{args.max_qa_revisions + 1}")
            response = call_ollama(
                "コードを検証し、指定形式で判定してください。問題があれば修正し、"
                "必ず完全なHTMLを返してください。\n\n"
                f"現在のHTML:\n{current_html}",
                QA_SYSTEM,
                args,
                "qa",
            )
            (run_dir / f"04_qa_round_{round_label}.md").write_text(
                response.rstrip() + "\n",
                encoding="utf-8",
            )
            qa_status = extract_qa_status(response)
            candidate_html = extract_html(response)
            round_info = {"round": round_label, "ai_status": qa_status}
            if not candidate_html:
                round_info["result"] = "no_html_returned"
                qa_note = "QAがHTMLを返さなかったため、直近の静的検証済みHTMLを保持しました。"
                qa_rounds.append(round_info)
                break

            candidate_checks = validate_html(candidate_html)
            write_json(run_dir / f"04_qa_static_checks_{round_label}.json", candidate_checks)
            if not candidate_checks["passed"]:
                round_info["result"] = "static_check_failed"
                round_info["static_errors"] = candidate_checks["errors"]
                qa_note = "QA出力の静的検証に失敗したため、直近の検証済みHTMLを保持しました。"
                qa_rounds.append(round_info)
                break

            current_html = candidate_html
            (run_dir / "index.html").write_text(current_html.rstrip() + "\n", encoding="utf-8")
            round_info["result"] = "html_valid"
            qa_rounds.append(round_info)
            if qa_status == "APPROVED":
                qa_approved = True
                qa_note = "AI QAは承認を返しました。静的検証結果は別途記録されています。"
                break
            if qa_status != "NEEDS_REVISION":
                qa_note = "QAの判定形式が確認できないため、未承認として扱いました。"
                break
            if round_number == args.max_qa_revisions:
                qa_note = "QA修正回数の上限に達したため、未承認です。"

        report = [
            "# QA結果",
            "",
            f"- AI判定: {'APPROVED' if qa_approved else '未承認'}",
            f"- 静的検証: 初版HTMLは合格。詳細は `03_static_checks.json` を参照",
            f"- 外部依存: {len(checks['external_dependencies'])}件、ネットワークAPI利用: {len(checks['network_api_usage'])}件",
            f"- 状態: {qa_note or 'QAの承認を得られませんでした。'}",
            "- 注意: AIの判定と静的検証は、実ブラウザーでの動作テストを意味しません。",
            "",
            "## QAラウンド",
            "",
            json.dumps(qa_rounds, ensure_ascii=False, indent=2),
        ]
        (run_dir / "05_qa_result.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        manifest["stages"]["qa"] = "completed" if qa_approved else "unapproved"
        manifest["qa_approved"] = qa_approved
        manifest["qa_note"] = qa_note
        manifest["static_checks"] = checks
        browser_result = run_browser_check(current_html) if args.browser_check else {
            "status": "skipped",
            "reason": "オプション --browser-check が指定されていません。",
            "page_errors": [],
            "blocked_requests": [],
        }
        write_json(run_dir / "06_browser_check.json", browser_result)
        manifest["browser_check"] = browser_result
        manifest["status"] = "completed" if qa_approved else "completed_unapproved"
        write_json(run_dir / "run.json", manifest)
        with (run_dir / "05_qa_result.md").open("a", encoding="utf-8") as report_file:
            report_file.write(
                f"\n- 外部URL検出: {json.dumps(checks['external_dependencies'], ensure_ascii=False)}\n"
                f"- ネットワークAPI検出: {json.dumps(checks['network_api_usage'], ensure_ascii=False)}\n"
                f"- ブラウザー検証: {browser_result['status']} — "
                f"{browser_result.get('reason', '')}\n"
            )
        log(f"[DONE] 状態: {manifest['status']}")
        log(f"[DONE] HTML: {run_dir / 'index.html'}")
        if checks["external_dependencies"] or checks["network_api_usage"]:
            log("[WARN] 外部依存またはネットワークAPI利用を検出しました。03_static_checks.jsonを確認してください。")
        if browser_result["status"] not in {"passed", "skipped"}:
            log(f"[WARN] ブラウザー検証: {browser_result['status']} - {browser_result.get('reason', '')}")
        if not qa_approved:
            log(f"[WARN] {qa_note}")
        return 0 if qa_approved else 2
    except StageError as exc:
        manifest["status"] = "failed"
        manifest["failed_stage"] = exc.stage
        manifest["error"] = str(exc)
        write_json(run_dir / "run.json", manifest)
        log(f"[ERROR] 工程 '{exc.stage}' で失敗: {exc}")
        log("[ERROR] 後続工程は実行していません。")
        return 1
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_stage"] = "workflow"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "run.json", manifest)
        log(f"[ERROR] 実行を完了できませんでした: {type(exc).__name__}: {exc}")
        return 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    request_text = " ".join(args.request).strip()
    if not request_text:
        if sys.stdin.isatty():
            try:
                request_text = input("作りたいアプリの説明を入力してください: ").strip()
            except EOFError:
                request_text = ""
        else:
            request_text = sys.stdin.read().strip()
    if not request_text:
        print("[ERROR] アプリの説明が空です。処理を開始しません。", file=sys.stderr)
        return 1

    if args.routing_mode == "auto":
        try:
            installed = set(fetch_ollama_model_names(args.endpoint, args.timeout))
        except StageError as exc:
            print(f"[ERROR] 工程 '{exc.stage}' で失敗: {exc}", file=sys.stderr)
            return 1
        required = sorted(set(models_for_run(args).values()))
        absent = [model for model in required if model not in installed]
        if absent:
            print("[ERROR] 自動ルーティングに必要なモデルがありません:", file=sys.stderr)
            for model in absent:
                print(f"  - {model} (導入: ollama pull {model})", file=sys.stderr)
            print(
                "[INFO] 自動ダウンロードや単一モデルへの切替は行っていません。"
                "必要モデルを導入するか、--routing-mode single を明示して再実行してください。",
                file=sys.stderr,
            )
            return 1

    try:
        output_root = resolve_output_root(args.out)
        run_dir = create_run_directory(output_root)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] 出力先を準備できません: {exc}", file=sys.stderr)
        return 1
    return run_pipeline(request_text, args, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
