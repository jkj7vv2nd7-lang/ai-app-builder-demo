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
STAGES = ("requirements", "design", "implementation", "qa")
# Models we would ideally like to see installed; used only for advice in
# smart mode (smart mode never blocks on a missing "ideal" model).
IDEAL_MODELS = {
    "requirements": "qwen3:8b",
    "design": "qwen3:8b",
    "implementation": "qwen2.5-coder:7b",
    "qa": "qwen2.5-coder:7b",
}
_CODER_HINTS = ("coder", "code", "starcoder", "codellama", "deepseek-coder", "codegemma")
_REASONING_HINTS = ("qwen3", "deepseek-r1", "phi4", "phi-4", "llama3", "llama-3")
_JAPANESE_HINTS = ("qwen", "aya", "gemma", "llama3", "llama-3", "phi")


def _parse_param_billions(details, name):
    """Best-effort parameter size in billions from Ollama tag metadata."""
    raw = ""
    if isinstance(details, dict):
        raw = str(details.get("parameter_size") or "")
    if not raw:
        # Fall back to a size hint in the tag itself, e.g. qwen3:8b, llama3.1:70b
        import re as _re
        m = _re.search(r":(\d+(?:\.\d+)?)b\b", name.lower())
        raw = (m.group(1) + "B") if m else ""
    try:
        return float(raw.rstrip("Bb"))
    except (TypeError, ValueError):
        return 0.0


def _model_score(name, details, stage, file_size_gb):
    lname = name.lower()
    is_coder = any(h in lname for h in _CODER_HINTS)
    params = _parse_param_billions(details, name)
    score = 0.0
    reasons = []
    # Capability grows with size, but on a 16GB / CPU machine the practical
    # sweet spot is 7-9B. Very large models are slow and risk memory pressure.
    if params:
        if 6.5 <= params <= 9.5:
            score += 40; reasons.append("7〜9B級で16GB機の最適域")
        elif 2.5 <= params < 6.5:
            score += 26; reasons.append("軽量で高速")
        elif 9.5 < params <= 14:
            score += 18; reasons.append("高性能だがCPUでは低速")
        elif params > 14:
            score += 4; reasons.append("大型で16GB機には重い")
        else:
            score += 10
    elif file_size_gb:
        if 3.5 <= file_size_gb <= 6.5:
            score += 36; reasons.append("サイズが16GB機の最適域")
        elif file_size_gb < 3.5:
            score += 22; reasons.append("軽量で高速")
        elif file_size_gb <= 10:
            score += 14
        else:
            score += 2; reasons.append("ファイルが大きく16GB機には重い")
    if stage in ("implementation", "qa"):
        if is_coder:
            score += 100; reasons.append("コード特化モデル")
        if any(h in lname for h in _REASONING_HINTS):
            score += 15
    else:  # requirements / design: general reasoning + Japanese ability first
        if not is_coder:
            score += 70; reasons.append("汎用・推論モデル")
        else:
            score += 25
        if any(h in lname for h in _REASONING_HINTS):
            score += 25; reasons.append("推論が強い系統")
        if any(h in lname for h in _JAPANESE_HINTS):
            score += 15; reasons.append("日本語が得意な系統")
    if stage == "qa" and is_coder:
        reasons.append("コードの欠陥検出向き")
    return score, reasons


def smart_route_models(model_details_list):
    """Assign the best *installed* model to each stage (department).

    model_details_list: items like {"name": str, "size": int(bytes),
    "details": {...}} as returned by fetch_ollama_model_details.
    Returns (assignments, reasons, suggestions):
      assignments: {stage: model_name}
      reasons:     {stage: "why this model, in Japanese"}
      suggestions: [ollama pull commands for ideal-but-missing models]
    Never fails on an empty preference: with one model installed, every
    stage uses it. Raises StageError only when nothing is installed.
    """
    items = []
    for item in model_details_list or []:
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]:
            size_gb = 0.0
            try:
                size_gb = float(item.get("size") or 0) / (1024 ** 3)
            except (TypeError, ValueError):
                size_gb = 0.0
            items.append((item["name"], item.get("details") or {}, size_gb))
    if not items:
        raise StageError("connection", "Ollamaにモデルが1つも導入されていません。まず ollama pull qwen2.5-coder:7b を実行してください。")
    assignments, reasons = {}, {}
    for stage in STAGES:
        best = None
        for name, details, size_gb in items:
            score, why = _model_score(name, details, stage, size_gb)
            if best is None or score > best[0]:
                best = (score, name, why)
        assignments[stage] = best[1]
        reasons[stage] = "・".join(dict.fromkeys(best[2])) or "導入済みモデルの中で最適"
    installed_names = {name for name, _, _ in items}
    suggestions = []
    for stage in STAGES:
        ideal = IDEAL_MODELS[stage]
        if ideal not in installed_names and assignments[stage] != ideal:
            cmd = f"ollama pull {ideal}"
            if cmd not in suggestions:
                suggestions.append(cmd)
    return assignments, reasons, suggestions
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

QA_SYSTEM = """あなたはQAエンジニアです。与えられた単一HTMLアプリを審査してください。
重要: HTML全体は絶対に返さないでください。審査結果だけを返してください。
出力の最初に次のいずれかを1行で記してください:
QA_STATUS: APPROVED
QA_STATUS: NEEDS_REVISION
その後に「DEFECTS:」と書き、直すべき問題を具体的に箇条書きしてください
（どの要素・どの処理が、なぜ問題か。問題がなければ「なし」と書いてください）。
受け入れ条件を満たしているか、ボタンや入力が実際に動きそうか、
外部CDN/APIなどのネットワーク依存がないかを重点的に見てください。
静的確認だけでブラウザー実行済みと主張しないでください。"""

REPAIR_SYSTEM = """あなたは熟練したフロントエンドエンジニアです。
指摘された問題だけを直し、完全な単一HTMLアプリ全体を出力してください。
説明やMarkdownのコードフェンスを避け、HTML全体だけを出力してください。
コードを省略せず、外部CDN・外部API・ネットワーク通信には依存しないでください。
問題のない部分の機能や見た目は変えないでください。"""


class StageError(RuntimeError):
    """A failure tied to a specific workflow stage."""

    def __init__(self, stage, message, cancelled=False):
        super().__init__(message)
        self.stage = stage
        self.cancelled = cancelled


def generation_options(args, stage, kind=None):
    """Per-stage generation parameters.

    Planning stages stay short and a little creative; implementation and
    repair get a larger context and output budget so a full HTML file is
    far less likely to be truncated mid-file (the main cause of the old
    version stopping halfway); QA review stays short and strict.
    """
    base_ctx = int(getattr(args, "num_ctx", 4096) or 4096)
    base_predict = int(getattr(args, "num_predict", 4096) or 4096)
    base_temp = float(getattr(args, "temperature", 0.3))
    key = kind or stage
    if key in ("requirements", "design"):
        return {"temperature": 0.4, "num_predict": min(max(base_predict, 2048), 3072), "num_ctx": base_ctx}
    if key == "implementation":
        return {"temperature": 0.2, "num_predict": max(base_predict, 8192), "num_ctx": min(max(base_ctx, 8192), 16384)}
    if key == "repair":
        return {"temperature": 0.15, "num_predict": max(base_predict, 8192), "num_ctx": min(max(base_ctx, 8192), 16384)}
    if key == "qa":
        return {"temperature": 0.1, "num_predict": 1536, "num_ctx": min(max(base_ctx, 8192), 16384)}
    return {"temperature": base_temp, "num_predict": base_predict, "num_ctx": base_ctx}


def call_ollama(prompt, system_prompt, args, stage, kind=None, token_callback=None, cancel_event=None):
    """Call local Ollama. Streams tokens when streaming is enabled so a
    long generation shows progress and can be cancelled, instead of
    looking frozen until the whole answer arrives at once."""
    model = selected_model(args, stage)
    options = generation_options(args, stage, kind)
    use_stream = bool(getattr(args, "stream", False)) or token_callback is not None
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": use_stream,
        "keep_alive": 0,
        "options": options,
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise StageError(stage, "ユーザーにより中止されました。", cancelled=True)

    check_cancel()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            if not use_stream:
                response_text = response.read().decode("utf-8")
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
            # Streaming: one JSON object per line.
            parts = []
            total_chars = 0
            last_report = 0
            saw_line = False
            for raw_line in response:
                check_cancel()
                line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, (bytes, bytearray)) else str(raw_line).strip()
                if not line:
                    continue
                saw_line = True
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    raise StageError(stage, f"Ollama returned an error: {obj['error']}")
                chunk = obj.get("response") if isinstance(obj, dict) else None
                if isinstance(chunk, str) and chunk:
                    parts.append(chunk)
                    total_chars += len(chunk)
                    if token_callback is not None and total_chars - last_report >= 400:
                        last_report = total_chars
                        token_callback(stage, total_chars)
                if isinstance(obj, dict) and obj.get("done"):
                    break
            if not saw_line:
                # Some servers/mocks answer with a single non-streamed body.
                raise StageError(stage, "Ollama returned an empty stream")
            answer = "".join(parts).strip()
            if token_callback is not None:
                token_callback(stage, total_chars)
            if not answer:
                raise StageError(stage, "Ollama returned no response text")
            return answer
    except StageError:
        raise
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
    resolved = getattr(args, "resolved_models", None)
    if isinstance(resolved, dict) and resolved.get(stage):
        return resolved[stage]
    if getattr(args, "routing_mode", "single") == "auto":
        return AUTO_MODEL_ROUTING[stage]
    option_name = {
        "requirements": "planning_model",
        "design": "design_model",
        "implementation": "implementation_model",
        "qa": "qa_model",
    }.get(stage)
    return getattr(args, option_name, None) or args.model


def models_for_run(args):
    return {stage: selected_model(args, stage) for stage in STAGES}


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


def fetch_ollama_model_details(endpoint, timeout):
    """Like fetch_ollama_model_names but keeps size/details for routing."""
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
    details = []
    for item in data["models"]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            details.append({
                "name": item["name"],
                "size": item.get("size") or 0,
                "details": item.get("details") or {},
            })
    return details


def extract_qa_status(response):
    match = re.search(
        r"^\s*QA_STATUS:\s*(APPROVED|NEEDS_REVISION)\s*$",
        response,
        re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).upper() if match else None


def extract_qa_defects(response):
    """Return the free-text review that follows the QA status line."""
    lines = response.splitlines()
    kept = []
    for line in lines:
        if extract_qa_status(line) and line.strip().upper().startswith("QA_STATUS"):
            continue
        kept.append(line)
    text = "\n".join(kept).strip()
    return text


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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollamaモデル名（単一モード用）")
    parser.add_argument(
        "--routing-mode",
        choices=("smart", "single", "manual", "auto"),
        default="smart",
        help="smart: 部署ごとに導入済みモデルから最適なものを自動選択（既定）。"
             "single: 全工程で同じモデル。manual: 部署ごとに指定。auto: 従来の固定おすすめ表",
    )
    parser.add_argument("--planning-model", help="要件定義用モデル（manual時、省略時は--model）")
    parser.add_argument("--design-model", help="デザイン設計用モデル（manual時、省略時は--model）")
    parser.add_argument("--implementation-model", help="実装用モデル（manual時、省略時は--model）")
    parser.add_argument("--qa-model", help="QA用モデル（manual時、省略時は--model）")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                        help="生成をストリーミング表示する（既定: 有効）")
    parser.add_argument("--timeout", type=float, default=600, help="APIタイムアウト秒数")
    parser.add_argument("--temperature", type=float, default=0.3, help="生成温度")
    parser.add_argument("--num-predict", type=int, default=4096, help="生成トークン上限")
    parser.add_argument("--num-ctx", type=int, default=4096, help="コンテキスト長（16GB RAM CPU向け既定値）")
    parser.add_argument("--max-qa-revisions", type=int, default=2, help="QAによる修正再試行の最大回数")
    parser.add_argument("--max-impl-repairs", type=int, default=2, help="実装直後の静的検証に失敗したときの自動修復の最大回数")
    parser.add_argument("--resume", metavar="RUN_DIR", help="前回の実行フォルダーを指定し、完了した工程を飛ばして再開する")
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
    if args.max_impl_repairs < 0:
        parser.error("--max-impl-repairsは0以上を指定してください")
    if args.routing_mode in ("auto", "smart") and (
        args.planning_model or args.design_model or args.implementation_model or args.qa_model
    ):
        parser.error("--routing-mode auto/smartでは工程別モデル指定を使わないでください（manualで指定します）")


def run_pipeline(request_text, args, run_dir, progress_callback=None, token_callback=None, cancel_event=None, resume=False):
    """Run the four departments in order, with checkpointing.

    v2 behaviour changes versus the original:
    * streaming generation with progress + cancellation,
    * smart per-department model routing (resolved here when needed),
    * implementation output that fails static checks is *repaired* by the
      implementation model instead of aborting the whole run,
    * QA returns a review only; defects are fixed by the implementation
      model and re-checked, instead of QA re-emitting the whole file,
    * resume=True reuses stage outputs already saved in run_dir.
    """
    log_path = run_dir / "run.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
        if progress_callback:
            progress_callback(message)

    def check_cancel(stage="workflow"):
        if cancel_event is not None and cancel_event.is_set():
            raise StageError(stage, "ユーザーにより中止されました。", cancelled=True)

    # --- resolve models (smart routing needs the installed list) ---
    routing_reasons = dict(getattr(args, "routing_reasons", None) or {})
    routing_suggestions = []
    if getattr(args, "routing_mode", "single") == "smart" and not getattr(args, "resolved_models", None):
        details = fetch_ollama_model_details(args.endpoint, min(float(getattr(args, "timeout", 600) or 600), 30.0))
        assignments, routing_reasons, routing_suggestions = smart_route_models(details)
        args.resolved_models = assignments
        args.routing_reasons = routing_reasons
    models = models_for_run(args)

    manifest_path = run_dir / "run.json"
    manifest = {}
    if resume and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    manifest.update({
        "created_at": manifest.get("created_at") or datetime.now().astimezone().isoformat(),
        "updated_at": datetime.now().astimezone().isoformat(),
        "request": request_text,
        "model": args.model,
        "models": models,
        "routing_mode": getattr(args, "routing_mode", "single"),
        "routing_reasons": routing_reasons or manifest.get("routing_reasons") or {},
        "num_ctx": args.num_ctx,
        "endpoint": args.endpoint,
        "status": "running",
        "qa_approved": manifest.get("qa_approved") if resume else None,
        "stages": manifest.get("stages") if resume and isinstance(manifest.get("stages"), dict) else {},
    })

    def save_stage(stage, filename, content):
        (run_dir / filename).write_text(content.rstrip() + "\n", encoding="utf-8")
        manifest["stages"][stage] = "completed"
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(manifest_path, manifest)

    def load_stage(filename):
        path = run_dir / filename
        if resume and path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            return text or None
        return None

    def call(prompt, system, stage, kind=None):
        check_cancel(stage)
        return call_ollama(prompt, system, args, stage, kind=kind,
                           token_callback=token_callback, cancel_event=cancel_event)

    write_json(manifest_path, manifest)
    log(f"[INFO] 出力先: {run_dir}")
    mode_label = {"smart": "部署ごとに自動で最適モデルを選択", "auto": "固定おすすめルーティング",
                  "manual": "部署ごとに手動指定", "single": "単一モデル"}.get(getattr(args, "routing_mode", "single"), "単一モデル")
    log(f"[INFO] モード: {mode_label} / API: {args.endpoint}")
    for stage, label in (("requirements", "PM・要件定義"), ("design", "デザイン設計"), ("implementation", "実装"), ("qa", "QA")):
        reason = (routing_reasons or {}).get(stage)
        log(f"[INFO] {label}: {models[stage]}" + (f"（{reason}）" if reason else ""))
    for cmd in routing_suggestions:
        log(f"[INFO] 任意の改善案: {cmd} を導入すると、その部署はさらに得意なモデルに切り替わります（今のモデルでも続行できます）")
    if resume:
        log("[RESUME] 完了済みの工程は保存済みの成果物を使って飛ばします。")

    try:
        # --- 1/4 requirements ---
        requirements = load_stage("01_requirements.md")
        if requirements:
            manifest["stages"]["requirements"] = "completed"
            log("[1/4] 要件定義は前回の成果物を使用します（再開）")
        else:
            log(f"[1/4] 要件定義を作成中...（{models['requirements']}）")
            requirements = call(
                f"次のアプリ案について要件を定義してください。\n\nアプリ案:\n{request_text}",
                REQUIREMENTS_SYSTEM, "requirements")
            save_stage("requirements", "01_requirements.md", requirements)

        # --- 2/4 design ---
        design = load_stage("02_design.md")
        if design:
            manifest["stages"]["design"] = "completed"
            log("[2/4] デザイン設計は前回の成果物を使用します（再開）")
        else:
            log(f"[2/4] デザイン設計を作成中...（{models['design']}）")
            design = call(
                f"次の要件に基づいて画面と操作を設計してください。\n\n{requirements}",
                DESIGN_SYSTEM, "design")
            save_stage("design", "02_design.md", design)

        # --- 3/4 implementation, with automatic repair ---
        current_html = None
        checks = None
        saved_index = run_dir / "index.html"
        resumed_html = saved_index.read_text(encoding="utf-8").strip() if (resume and saved_index.is_file()) else None
        if resumed_html:
            resumed_checks = validate_html(resumed_html)
            if resumed_checks["passed"]:
                current_html, checks = resumed_html, resumed_checks
                manifest["stages"]["implementation"] = "completed"
                log("[3/4] 実装は前回保存された検証済みHTMLを使用します（再開）")
        if current_html is None:
            log(f"[3/4] HTMLアプリを実装中...（{models['implementation']}）")
            previous_text = ""
            problems = []
            for attempt in range(int(getattr(args, "max_impl_repairs", 2)) + 1):
                if attempt == 0:
                    raw = call(
                        f"アプリ案:\n{request_text}\n\n要件:\n{requirements}\n\nデザイン:\n{design}",
                        IMPLEMENTATION_SYSTEM, "implementation")
                else:
                    log(f"[REPAIR] 実装の自動修復 {attempt}/{int(getattr(args, 'max_impl_repairs', 2))}（問題: {'; '.join(problems[:3])}）")
                    raw = call(
                        "次のHTMLには問題があります。問題を直した完全なHTML全体を返してください。\n\n"
                        f"問題:\n" + "\n".join(f"- {item}" for item in problems) +
                        f"\n\n現在の応答・HTML（末尾が切れている場合は続きを含めて完成させてください）:\n{previous_text[-14000:]}",
                        REPAIR_SYSTEM, "implementation", kind="repair")
                (run_dir / f"03_implementation_response_{attempt + 1}.md").write_text(raw.rstrip() + "\n", encoding="utf-8")
                previous_text = raw
                candidate = extract_html(raw)
                if not candidate:
                    problems = ["AI応答からHTML文書を抽出できませんでした（出力が途中で切れた可能性があります）"]
                    continue
                candidate_checks = validate_html(candidate)
                write_json(run_dir / f"03_static_checks_{attempt + 1}.json", candidate_checks)
                if candidate_checks["passed"]:
                    current_html, checks = candidate, candidate_checks
                    break
                problems = candidate_checks["errors"]
                previous_text = candidate
            if current_html is None:
                partial = extract_html(previous_text)
                if partial:
                    (run_dir / "index_draft_unverified.html").write_text(partial.rstrip() + "\n", encoding="utf-8")
                    log("[WARN] 未検証の下書きHTMLを index_draft_unverified.html に保存しました。")
                raise StageError("implementation", "自動修復を試しましたが、静的検証に合格するHTMLが得られませんでした: " + "; ".join(problems))
            save_stage("implementation", "03_implementation_response.md", previous_text)
            write_json(run_dir / "03_static_checks.json", checks)
            (run_dir / "index.html").write_text(current_html.rstrip() + "\n", encoding="utf-8")
            log("[3/4] 実装の静的検証に合格しました。")

        # --- 4/4 QA review -> implementation repair loop ---
        log(f"[4/4] QAレビューを実行中...（{models['qa']}）")
        qa_rounds = []
        qa_approved = False
        qa_note = ""
        max_rounds = int(getattr(args, "max_qa_revisions", 2))
        for round_number in range(max_rounds + 1):
            round_label = round_number + 1
            check_cancel("qa")
            log(f"[QA] レビュー {round_label}/{max_rounds + 1}")
            response = call(
                "次の単一HTMLアプリを審査してください。HTML全体は返さず、判定とDEFECTSだけを返してください。\n\n"
                f"受け入れ条件を含む要件:\n{requirements}\n\n現在のHTML:\n{current_html}",
                QA_SYSTEM, "qa")
            (run_dir / f"04_qa_round_{round_label}.md").write_text(response.rstrip() + "\n", encoding="utf-8")
            qa_status = extract_qa_status(response)
            defects = extract_qa_defects(response)
            # Backward compatibility: a QA model that returns full HTML anyway.
            returned_html = extract_html(response)
            if returned_html and returned_html.strip() != current_html.strip():
                returned_checks = validate_html(returned_html)
                if returned_checks["passed"]:
                    current_html = returned_html
                    checks = returned_checks
                    (run_dir / "index.html").write_text(current_html.rstrip() + "\n", encoding="utf-8")
            round_info = {"round": round_label, "ai_status": qa_status}
            if qa_status == "APPROVED":
                qa_approved = True
                qa_note = "AI QAは承認を返しました。"
                round_info["result"] = "approved"
                qa_rounds.append(round_info)
                break
            round_info["result"] = "needs_revision" if qa_status == "NEEDS_REVISION" else "no_clear_verdict"
            round_info["defects"] = defects[:2000]
            qa_rounds.append(round_info)
            if round_number == max_rounds:
                qa_note = "QA修正回数の上限に達したため、未承認です。HTMLは保存済みです。"
                break
            log(f"[QA] 指摘を実装部署で修正します...（{models['implementation']}）")
            repaired = call(
                "QAから次の指摘を受けました。指摘だけを直した完全なHTML全体を返してください。\n\n"
                f"QAの指摘:\n{defects}\n\n現在のHTML:\n{current_html}",
                REPAIR_SYSTEM, "implementation", kind="repair")
            candidate = extract_html(repaired)
            if not candidate:
                qa_note = "修正応答からHTMLを抽出できなかったため、直近の検証済みHTMLを保持しました。"
                break
            candidate_checks = validate_html(candidate)
            write_json(run_dir / f"04_repair_static_checks_{round_label}.json", candidate_checks)
            if not candidate_checks["passed"]:
                qa_note = "修正後HTMLの静的検証に失敗したため、直近の検証済みHTMLを保持しました: " + "; ".join(candidate_checks["errors"])
                break
            current_html = candidate
            checks = candidate_checks
            (run_dir / "index.html").write_text(current_html.rstrip() + "\n", encoding="utf-8")

        report = [
            "# QA結果",
            "",
            f"- AI判定: {'APPROVED' if qa_approved else '未承認'}",
            f"- 静的検証: 最終HTMLは合格。詳細は `03_static_checks.json` などを参照",
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
        browser_result = run_browser_check(current_html) if getattr(args, "browser_check", False) else {
            "status": "skipped",
            "reason": "オプション --browser-check が指定されていません。",
            "page_errors": [],
            "blocked_requests": [],
        }
        write_json(run_dir / "06_browser_check.json", browser_result)
        manifest["browser_check"] = browser_result
        manifest["status"] = "completed" if qa_approved else "completed_unapproved"
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(manifest_path, manifest)
        with (run_dir / "05_qa_result.md").open("a", encoding="utf-8") as report_file:
            report_file.write(
                f"\n- 外部URL検出: {json.dumps(checks['external_dependencies'], ensure_ascii=False)}\n"
                f"- ネットワークAPI検出: {json.dumps(checks['network_api_usage'], ensure_ascii=False)}\n"
                f"- ブラウザー検証: {browser_result['status']} — {browser_result.get('reason', '')}\n"
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
        manifest["status"] = "cancelled" if exc.cancelled else "failed"
        manifest["failed_stage"] = exc.stage
        manifest["error"] = str(exc)
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(manifest_path, manifest)
        if exc.cancelled:
            log("[CANCELLED] 中止しました。完了した工程の成果物は保存済みで、「再開」で続きから実行できます。")
            return 3
        log(f"[ERROR] 工程 '{exc.stage}' で失敗: {exc}")
        log("[ERROR] 完了した工程の成果物は保存済みです。「再開」で続きから実行できます。")
        return 1
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_stage"] = "workflow"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(manifest_path, manifest)
        log(f"[ERROR] 実行を完了できませんでした: {type(exc).__name__}: {exc}")
        return 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    resume_dir = None
    if getattr(args, "resume", None):
        resume_dir = Path(args.resume).expanduser()
        if not resume_dir.is_absolute():
            resume_dir = (Path(__file__).resolve().parent / resume_dir)
        resume_dir = resume_dir.resolve()
        if not (resume_dir / "run.json").is_file():
            print(f"[ERROR] 再開先に run.json がありません: {resume_dir}", file=sys.stderr)
            return 1
        try:
            saved = json.loads((resume_dir / "run.json").read_text(encoding="utf-8"))
            request_text = str(saved.get("request") or "").strip()
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[ERROR] 再開先の run.json を読めません: {exc}", file=sys.stderr)
            return 1
        if not request_text:
            print("[ERROR] 再開先の run.json に元のアプリ案がありません。", file=sys.stderr)
            return 1
    else:
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

    # Pre-flight model check. Smart mode resolves against whatever is
    # installed and never blocks on an "ideal" model; other modes check
    # their explicit models, as before.
    try:
        if args.routing_mode == "smart":
            details = fetch_ollama_model_details(args.endpoint, min(args.timeout, 30.0))
            assignments, _reasons, _suggestions = smart_route_models(details)
            args.resolved_models = assignments
            args.routing_reasons = _reasons
        else:
            installed = set(fetch_ollama_model_names(args.endpoint, min(args.timeout, 30.0)))
            required = sorted(set(models_for_run(args).values()))
            absent = [model for model in required if model not in installed]
            if absent:
                print("[ERROR] 必要なモデルが導入されていません:", file=sys.stderr)
                for model in absent:
                    print(f"  - {model} (導入: ollama pull {model})", file=sys.stderr)
                print("[INFO] モデルの自動ダウンロードや勝手な切替は行っていません。導入するか、--routing-mode smart で導入済みモデルから自動選択してください。", file=sys.stderr)
                return 1
    except StageError as exc:
        print(f"[ERROR] 工程 '{exc.stage}' で失敗: {exc}", file=sys.stderr)
        return 1

    if resume_dir is not None:
        return run_pipeline(request_text, args, resume_dir, resume=True)
    try:
        output_root = resolve_output_root(args.out)
        run_dir = create_run_directory(output_root)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] 出力先を準備できません: {exc}", file=sys.stderr)
        return 1
    return run_pipeline(request_text, args, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
