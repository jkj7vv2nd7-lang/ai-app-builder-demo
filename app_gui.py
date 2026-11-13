#!/usr/bin/env python3
"""Tkinter desktop interface for the local Ollama app builder."""

import io
import json
import math
import os
import queue
import re
import time
import struct
import subprocess
import sys
import threading
import tkinter as tk
import wave
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from urllib.parse import urlparse

import app_builder


def _synth_wav_bytes(notes, volume=0.42, rate=22050):
    """Build a small chime in code (no bundled audio files needed).

    notes: list of (frequency_hz, seconds). Each note rings with an
    exponential decay, like a microwave / oven timer bell.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = bytearray()
        for freq, seconds in notes:
            count = int(rate * seconds)
            for i in range(count):
                t = i / rate
                envelope = math.exp(-3.2 * i / max(count, 1))
                value = (math.sin(2 * math.pi * freq * t)
                         + 0.35 * math.sin(2 * math.pi * freq * 2 * t)
                         + 0.12 * math.sin(2 * math.pi * freq * 3 * t))
                frames += struct.pack("<h", int(32767 * volume * envelope * value / 1.47))
        wav.writeframes(bytes(frames))
    return buffer.getvalue()


_CHIME_CACHE = {}


def play_chime(kind="success"):
    """Play the completion chime on Windows via winsound (stdlib, reliable).

    success: bright two-note 「チーン♪」. failed: softer descending tones.
    On non-Windows systems this quietly does nothing.
    """
    if sys.platform != "win32":
        return False

    def _play():
        try:
            import winsound
            if kind not in _CHIME_CACHE:
                if kind == "success":
                    _CHIME_CACHE[kind] = _synth_wav_bytes(
                        [(1567.98, 0.5), (2093.00, 0.85)])
                else:
                    _CHIME_CACHE[kind] = _synth_wav_bytes(
                        [(523.25, 0.26), (392.00, 0.42)], volume=0.28)
            winsound.PlaySound(_CHIME_CACHE[kind], winsound.SND_MEMORY)
        except Exception:
            try:
                import winsound
                if kind == "success":
                    winsound.Beep(1568, 220)
                    winsound.Beep(2093, 380)
                else:
                    winsound.Beep(523, 180)
                    winsound.Beep(392, 280)
            except Exception:
                pass

    threading.Thread(target=_play, daemon=True).start()
    return True


def _mix_color(color_from, color_to, ratio):
    """Interpolate between two #rrggbb colors."""
    def parts(color):
        return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    a, b = parts(color_from), parts(color_to)
    mixed = tuple(int(x + (y - x) * max(0.0, min(1.0, ratio))) for x, y in zip(a, b))
    return "#%02x%02x%02x" % mixed

import github_publish


EXAMPLE_REQUEST = "中学生向け英単語クイズ。出題・回答判定・正解数の表示・進捗保存をしたい"
PHASES = ("要件定義", "デザイン設計", "アプリ実装", "QA・保存")


class OllamaCheckError(RuntimeError):
    pass


def validate_local_endpoint(endpoint):
    parsed = urlparse(endpoint.strip())
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama接続先には http://localhost:11434/api/generate などのローカルURLを指定してください。")
    return parsed


def fetch_ollama_models(endpoint, timeout=5):
    try:
        return app_builder.fetch_ollama_model_names(endpoint, timeout)
    except app_builder.StageError as exc:
        raise OllamaCheckError(str(exc)) from exc


def fetch_ollama_model_details(endpoint, timeout=8):
    try:
        return app_builder.fetch_ollama_model_details(endpoint, timeout)
    except app_builder.StageError as exc:
        raise OllamaCheckError(str(exc)) from exc


STAGE_KEYS = ("requirements", "design", "implementation", "qa")


def selected_models(common_model, separate_models=None):
    common = common_model.strip() or app_builder.DEFAULT_MODEL
    separate = separate_models or {}
    return {
        key: (separate.get(key, "") or "").strip() or common
        for key in STAGE_KEYS
    }


def build_run_args(
    endpoint,
    common_model,
    stage_models,
    context_size,
    routing_mode="smart",
    timeout_seconds=600,
):
    validate_local_endpoint(endpoint)
    if routing_mode not in {"smart", "single", "manual", "auto"}:
        raise ValueError("モデルの使い方は「自動（部署ごとに切替）」「単一モデル」「部署ごとに手動」から選んでください。")
    if routing_mode in {"smart", "auto"} and any(
        (stage_models or {}).get(key, "").strip() for key in STAGE_KEYS
    ):
        raise ValueError("自動モードと部署別の手動指定は同時に使えません。")
    if str(context_size) not in {"2048", "4096", "8192"}:
        raise ValueError("コンテキスト長は2048、4096、8192のいずれかを指定してください。")
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("タイムアウトは秒数で入力してください（例: 600）。") from exc
    if timeout <= 0:
        raise ValueError("タイムアウトは1秒以上を指定してください。")
    parser = app_builder.build_parser()
    builder_mode = routing_mode
    arguments = [
        "--endpoint", endpoint,
        "--model", common_model.strip() or app_builder.DEFAULT_MODEL,
        "--num-ctx", str(context_size),
        "--timeout", str(timeout_seconds),
        "--routing-mode", builder_mode,
    ]
    if routing_mode == "manual":
        for option, key in (
            ("--planning-model", "requirements"),
            ("--design-model", "design"),
            ("--implementation-model", "implementation"),
            ("--qa-model", "qa"),
        ):
            model = (stage_models or {}).get(key, "").strip()
            if model:
                arguments.extend((option, model))
    args = parser.parse_args(arguments)
    return args


def models_for_mode(routing_mode, common_model, stage_models=None, installed_details=None):
    if routing_mode == "smart":
        if installed_details:
            assignments, _reasons, _suggestions = app_builder.smart_route_models(installed_details)
            return assignments
        return {key: "（接続確認後に自動決定）" for key in STAGE_KEYS}
    if routing_mode == "auto":
        return dict(app_builder.AUTO_MODEL_ROUTING)
    if routing_mode == "manual":
        return selected_models(common_model, stage_models)
    return selected_models(common_model, None)


def missing_models(required_models, installed_models):
    installed = set(installed_models)
    return sorted(set(required_models.values()) - installed)


def validate_request_text(text):
    request = text.strip()
    if not request:
        raise ValueError("作りたいアプリの説明を入力してください。")
    return request


def progress_phase(line):
    for marker, index in (("[1/4]", 0), ("[2/4]", 1), ("[3/4]", 2), ("[4/4]", 3), ("[QA]", 3)):
        if marker in line:
            return index
    return None


def error_guidance(stage, error, model_names=()):
    text = str(error)
    lowered = text.lower()
    steps = []
    if "could not connect" in lowered or "connection refused" in lowered or "接続できません" in text:
        steps = [
            "Ollamaアプリを起動し、準備ができるまで待ってください。",
            "この画面の「接続・モデルを確認」を押して再確認してください。",
            "接続先はローカルの http://localhost:11434/api/generate です。",
        ]
    elif "model not found" in lowered or "モデルが見つかりません" in text or "未導入" in text:
        names = model_names or (app_builder.DEFAULT_MODEL,)
        pull_commands = "\n".join(f"例: ollama pull {name}" for name in names)
        steps = [
            "指定したモデル名が正しいか、Ollamaのモデル一覧と照合してください。",
            "PowerShellで ollama pull <モデル名> を実行してモデルを取得してください。",
            pull_commands,
        ]
    elif "timeout" in lowered or "timed out" in lowered or "タイムアウト" in text:
        steps = [
            "CPUでの生成には時間がかかることがあります。まず数分待って再試行してください。",
            "設定のタイムアウトを長くするか、より小さいモデルを選んでください。",
            "応答が遅い場合はコンテキストを2048に下げ、他の重いアプリを閉じてください。",
        ]
    elif any(term in lowered for term in ("memory", "out of memory", "runner process", "メモリ")):
        steps = [
            "小さい量子化モデルまたは3Bクラスのモデルに切り替えてください。",
            "コンテキストを4096から2048へ下げてください。",
            "空きメモリを確保してから再試行してください。",
        ]
    elif "permission" in lowered or "access is denied" in lowered or "アクセスが拒否" in text:
        steps = [
            "保存先フォルダーへの書き込み権限を確認してください。",
            "他のアプリで同じ実行フォルダーのファイルを開いている場合は閉じてください。",
        ]
    elif stage:
        steps = [
            f"失敗した工程: {stage}。この工程の入力・モデル応答を確認してください。",
            "必要なら同じ条件でもう一度実行してください。生成済みのログと途中成果物は保存先に残ります。",
        ]
    else:
        steps = ["表示された内容を確認して修正し、もう一度お試しください。"]
    prefix = f"発生した工程: {stage}\n" if stage else ""
    return prefix + text + "\n\n次にお試しください:\n" + "\n".join(f"・{step}" for step in steps)


def publication_guidance(error):
    stage = getattr(error, "stage", "GitHub公開")
    text = str(error)
    if stage == "GitHub認証確認":
        advice = "PowerShellで gh auth login を実行し、ブラウザー認証を完了してください。"
    elif stage == "Git認証・ブランチ確認":
        advice = "PowerShellで gh auth setup-git を実行してGit認証を設定し、再確認してください。"
    elif stage == "書き込み権限確認":
        advice = "対象の公開リポジトリで、このGitHubアカウントに書き込み権限があるか確認してください。"
    elif stage in {"既存ファイル確認", "既存内容確認"}:
        advice = "既存内容を上書きしません。GitHub上のファイルを確認し、必要なら別の空リポジトリを用意してください。"
    elif "GitHubへのアップロード" in stage:
        advice = "競合時は上書きせず停止します。GitHub上のブランチを確認してから再試行してください。"
    else:
        advice = "URL、リポジトリの公開状態、GitHubログインと書き込み権限を確認してください。"
    details = f"発生した工程: {stage}\n{text}\n\n次にお試しください:\n・{advice}"
    repo_url = getattr(error, "repo_url", None)
    if repo_url:
        details += f"\n確認先リポジトリ: {repo_url}"
    return details


import tkinter.font as tkfont


# ---------------------------------------------------------------------------
# v3 GUI: a step-by-step wizard.
#
# Design rules (from the user's own feedback on v2):
#   * one job per screen, screens change as the work advances;
#   * large type everywhere (presbyopia-friendly), with a size switcher;
#   * the opening screen should promise something: the four AI departments
#     are shown as a "company" with their assigned models on cards;
#   * settings and GitHub publishing live on their own screens, never on
#     the main flow.
# ---------------------------------------------------------------------------

NAVY = "#1F2A5C"
NAVY2 = "#2B3A8F"
PRIMARY = "#3156C9"
PRIMARY_DARK = "#233F9B"
AMBER = "#F5A623"
BG = "#EEF1F8"
CARD = "#FFFFFF"
INK = "#172033"
MUTED = "#5A6579"
GREEN = "#1E8E4D"
RED = "#C0392B"
LINE = "#D8DEEC"
SOFT = "#EDF1FF"

DEPARTMENTS = (
    ("requirements", "🧭", "PM・要件定義", "アイデアを、作れる要件に整理します"),
    ("design", "🎨", "デザイン設計", "画面と操作の設計図を描きます"),
    ("implementation", "💻", "実装", "実際に動くアプリを書きます"),
    ("qa", "🔍", "QA・検査", "動くかどうかを検査し、直します"),
)
EXAMPLE_IDEAS = (
    ("英単語クイズ", "中学生向け英単語クイズ。出題・回答判定・正解数の表示・進捗保存をしたい"),
    ("家計簿メモ", "月末になると何にお金を使ったか分からなくなる。ざっくり記録して合計が見たい"),
    ("水やり記録", "植物にいつ水をあげたか忘れやすい。植物ごとに記録して、前回日が見たい"),
)


class AppBuilderWindow:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.busy = False
        self.checking = False
        self.last_output = None
        self.output_ready = False
        self.installed_models = None
        self.installed_details = None
        self.cancel_event = None
        self.active_models = {}
        self.current_page = "welcome"
        self.run_mode = "build"
        self.current_phase = 0
        self.run_started_at = None
        self.last_activity_at = None
        self.revision_target = None
        self.revision_return = "welcome"
        self.external_folder = None
        self.external_info = None
        self.sound_var = tk.BooleanVar(value=True)
        self.oven_frame = 0
        self._oven_job = None
        self.activity_name = ""
        self.qa_round = 0
        self._fonts = []
        self._scale_idx = 0
        self._scales = (1.0, 1.18, 1.36)
        self._scale_names = ("標準", "大", "特大")

        root.title("AIアプリ開発会社 ― ローカルAIアプリ開発エージェント v3.5")
        root.geometry("1060x830")
        root.minsize(920, 700)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---- shared variables ----
        self.endpoint_var = tk.StringVar(value=app_builder.DEFAULT_ENDPOINT)
        self.model_var = tk.StringVar(value=app_builder.DEFAULT_MODEL)
        self.context_var = tk.StringVar(value="4096")
        self.timeout_var = tk.StringVar(value="600")
        self.routing_var = tk.StringVar(value="smart")
        self.stage_model_vars = {key: tk.StringVar() for key in STAGE_KEYS}
        self.repository_url_var = tk.StringVar()
        self.publication_reviewed_var = tk.BooleanVar(value=False)
        self.show_log_var = tk.BooleanVar(value=True)

        self._build_header(root)
        self.container = tk.Frame(root, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.pages = {}
        self._scroll_canvases = {}
        for name, builder in (
            ("welcome", self._page_welcome),
            ("idea", self._page_idea),
            ("models", self._page_models),
            ("run", self._page_run),
            ("done", self._page_done),
            ("revision", self._page_revision),
            ("external", self._page_external),
            ("publish", self._page_publish),
            ("settings", self._page_settings),
        ):
            if name in ("welcome", "models", "run"):
                # v3.4: 内容がウィンドウの高さを超える画面はスクロール可能にする。
                # 固定 1060x830 のままでは、作品が増えた最初の画面や
                # 文字を大きくした2番目の画面で一番下が見えなくなるため。
                # v3.5: 作成中画面にもオーブンの演出が加わり高さが増えたため、
                # 同じ理由でスクロール可能にした（下のトグルやログが切れないように）。
                frame = self._scrollable_page(name, builder)
            else:
                frame = tk.Frame(self.container, bg=BG)
                builder(frame)
            frame.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = frame
        root.bind_all("<MouseWheel>", self._on_mousewheel)
        root.bind_all("<Button-4>", self._on_mousewheel)
        root.bind_all("<Button-5>", self._on_mousewheel)
        self.show_page("welcome")
        self.root.after(150, self.process_events)
        self.root.after(1000, self._tick_clock)
        self.root.after(400, self.check_connection)

    def _scrollable_page(self, name, builder):
        """縦スクロール付きの画面を作り、中のフレームに builder を実行する。"""
        outer = tk.Frame(self.container, bg=BG)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda _e, c=canvas: c.configure(scrollregion=c.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e, c=canvas, w=window: c.itemconfig(w, width=e.width))
        builder(inner)
        self._scroll_canvases[name] = canvas
        return outer

    def _on_mousewheel(self, event):
        canvas = self._scroll_canvases.get(self.current_page)
        if canvas is None:
            return
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        elif getattr(event, "delta", 0):
            canvas.yview_scroll(int(-event.delta / 120), "units")

    # ---- fonts (resizable) ----
    def F(self, size, bold=False):
        font = tkfont.Font(family="Yu Gothic UI", size=size,
                           weight="bold" if bold else "normal")
        self._fonts.append((font, size))
        return font

    def cycle_text_size(self):
        self._scale_idx = (self._scale_idx + 1) % len(self._scales)
        scale = self._scales[self._scale_idx]
        for font, base in self._fonts:
            font.configure(size=max(8, int(base * scale)))
        self.size_button.configure(text=f"文字の大きさ: {self._scale_names[self._scale_idx]}")

    # ---- shared widget helpers ----
    def _card(self, parent, **kw):
        return tk.Frame(parent, bg=CARD, highlightbackground=LINE,
                        highlightthickness=1, padx=18, pady=14, **kw)

    def _big_button(self, parent, text, command, primary=False, danger=False, width=None):
        bg = PRIMARY if primary else (RED if danger else CARD)
        fg = "#FFFFFF" if (primary or danger) else PRIMARY_DARK
        active = PRIMARY_DARK if primary else ("#96281B" if danger else SOFT)
        btn = tk.Button(parent, text=text, command=command, font=self.F(15, True),
                        bg=bg, fg=fg, activebackground=active,
                        activeforeground=fg if not (primary or danger) else "#FFFFFF",
                        relief="flat", padx=22, pady=12, cursor="hand2",
                        highlightbackground=LINE, highlightthickness=0 if (primary or danger) else 1)
        if width:
            btn.configure(width=width)
        return btn

    def _nav_row(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(side="bottom", fill="x", padx=26, pady=12)
        return row

    def _page_title(self, parent, text, sub=None):
        tk.Label(parent, text=text, font=self.F(22, True), fg=NAVY, bg=BG,
                 anchor="w").pack(fill="x", padx=26, pady=(16, 2))
        if sub:
            tk.Label(parent, text=sub, font=self.F(13), fg=MUTED, bg=BG,
                     anchor="w", wraplength=940).pack(fill="x", padx=26, pady=(0, 8))

    # ---- header ----
    def _build_header(self, root):
        header = tk.Frame(root, bg=NAVY)
        header.pack(fill="x")
        left = tk.Frame(header, bg=NAVY)
        left.pack(side="left", padx=20, pady=10)
        tk.Label(left, text="🏢", font=self.F(30), bg=NAVY, fg="#FFFFFF").pack(side="left")
        titles = tk.Frame(left, bg=NAVY)
        titles.pack(side="left", padx=(10, 0))
        tk.Label(titles, text="AIアプリ開発会社", font=self.F(22, True),
                 bg=NAVY, fg="#FFFFFF", anchor="w").pack(fill="x")
        tk.Label(titles, text="ローカルAIアプリ開発エージェント v3.5 ― あなたのPCの中で、4人のAI社員がアプリを作ります",
                 font=self.F(11), bg=NAVY, fg="#C9D2F2", anchor="w").pack(fill="x")
        right = tk.Frame(header, bg=NAVY)
        right.pack(side="right", padx=20, pady=10)
        self.step_label = tk.Label(right, text="", font=self.F(13, True),
                                   bg=NAVY, fg=AMBER, anchor="e")
        self.step_label.pack(fill="x")
        self.size_button = tk.Button(right, text="文字の大きさ: 標準", command=self.cycle_text_size,
                                     font=self.F(11), bg=NAVY2, fg="#FFFFFF", relief="flat",
                                     padx=12, pady=5, cursor="hand2", activebackground=PRIMARY,
                                     activeforeground="#FFFFFF")
        self.size_button.pack(anchor="e", pady=(6, 0))

    _STEP_TEXT = {
        "welcome": "準備の画面",
        "idea": "ステップ 1 / 4",
        "models": "ステップ 2 / 4",
        "run": "ステップ 3 / 4",
        "done": "ステップ 4 / 4",
        "revision": "ダメ出し（修正依頼）",
        "external": "既存アプリの改修",
        "publish": "GitHub公開",
        "settings": "設定",
    }

    def show_page(self, name):
        self.current_page = name
        if name == "run" and self._oven_job is None:
            self._oven_tick()
        self.pages[name].tkraise()
        self.step_label.configure(text=self._STEP_TEXT.get(name, ""))
        if name == "welcome" and hasattr(self, "runs_list_frame"):
            self.refresh_runs_list()

    def refresh_runs_list(self):
        for child in self.runs_list_frame.winfo_children():
            child.destroy()
        runs_root = Path(__file__).resolve().parent / "runs"
        run_dirs = []
        if runs_root.is_dir():
            run_dirs = sorted(
                (d for d in runs_root.iterdir() if d.is_dir() and (d / "run.json").is_file()),
                key=lambda d: d.stat().st_mtime, reverse=True)[:6]
        if not run_dirs:
            tk.Label(self.runs_list_frame, text="まだ作品はありません。「アプリを作りはじめる」からどうぞ。",
                     font=self.F(12), fg=MUTED, bg=BG, anchor="w").pack(fill="x")
            return
        status_text = {"completed": "完成", "completed_unapproved": "完成（QA未承認）",
                       "running": "作業中", "cancelled": "途中で中止", "failed": "途中で停止"}
        for run_dir in run_dirs:
            try:
                info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}
            status = str(info.get("status") or "")
            request = str(info.get("request") or run_dir.name)
            if info.get("mode") == "revision":
                request = "（ダメ出しの修正版）" + request
            elif info.get("mode") == "external_revision":
                request = "（既存アプリの改修）" + request
            row = self._card(self.runs_list_frame)
            row.pack(fill="x", pady=(0, 8))
            left = tk.Frame(row, bg=CARD)
            left.pack(side="left", fill="x", expand=True)
            tk.Label(left, text=request[:64], font=self.F(13, True), fg=INK, bg=CARD,
                     anchor="w").pack(fill="x")
            tk.Label(left, text=f"{status_text.get(status, status)} ／ {run_dir.name}",
                     font=self.F(11), fg=MUTED, bg=CARD, anchor="w").pack(fill="x")
            has_html = (run_dir / "index.html").is_file()
            if has_html:
                tk.Button(row, text="✎ ダメ出し",
                          command=lambda d=run_dir: self.goto_revision(d, "welcome"),
                          font=self.F(13, True), bg=PRIMARY, fg="#FFFFFF", relief="flat",
                          padx=18, pady=9, cursor="hand2", activebackground=PRIMARY_DARK,
                          activeforeground="#FFFFFF").pack(side="right")
            if status not in ("completed", "completed_unapproved") or not has_html:
                tk.Button(row, text="▶ つづき",
                          command=lambda d=run_dir: self.resume_generation(d),
                          font=self.F(13), bg=CARD, fg=PRIMARY_DARK, relief="flat",
                          padx=18, pady=9, cursor="hand2",
                          highlightbackground=LINE, highlightthickness=1).pack(side="right", padx=(0, 8))

    # ---- page: welcome ----
    def _page_welcome(self, page):
        hero = self._card(page)
        hero.pack(fill="x", padx=26, pady=(16, 10))
        tk.Label(hero, text="一言のアイデアから、動くアプリを作ります。",
                 font=self.F(23, True), fg=NAVY, bg=CARD, anchor="w").pack(fill="x")
        tk.Label(hero,
                 text="4人のAI社員が「要件 → 設計 → 実装 → 検査」の順に仕事を進めます。"
                      "部署ごとに、その仕事が一番得意なモデルへ自動で切り替わります。"
                      "生成はすべてあなたのPCの中だけで行われます。",
                 font=self.F(13), fg=MUTED, bg=CARD, anchor="w",
                 wraplength=920, justify="left").pack(fill="x", pady=(6, 0))

        cards = tk.Frame(page, bg=BG)
        cards.pack(fill="x", padx=26, pady=(2, 10))
        self.dept_card_models = {}
        for idx, (key, emoji, dept_name, role) in enumerate(DEPARTMENTS):
            card = self._card(cards)
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 6, 0))
            cards.grid_columnconfigure(idx, weight=1)
            tk.Label(card, text=emoji, font=self.F(24), bg=CARD).pack()
            tk.Label(card, text=dept_name, font=self.F(15, True), fg=INK, bg=CARD).pack()
            tk.Label(card, text=role, font=self.F(11), fg=MUTED, bg=CARD,
                     wraplength=190, justify="center").pack(pady=(0, 2))
            model_label = tk.Label(card, text="モデル確認中…", font=self.F(12, True),
                                   fg=PRIMARY_DARK, bg=SOFT, wraplength=190, justify="center")
            model_label.pack(fill="x", ipady=2)
            self.dept_card_models[key] = model_label

        conn = self._card(page)
        conn.pack(fill="x", padx=26, pady=(0, 12))
        self.conn_label = tk.Label(conn, text="● Ollamaを確認しています…",
                                   font=self.F(14, True), fg=MUTED, bg=CARD, anchor="w")
        self.conn_label.pack(side="left")
        self.conn_detail = tk.Label(conn, text="", font=self.F(11), fg=MUTED, bg=CARD, anchor="w")
        self.conn_detail.pack(side="left", padx=(12, 0))
        tk.Button(conn, text="再確認", command=self.check_connection, font=self.F(12),
                  bg=CARD, fg=PRIMARY_DARK, relief="flat", padx=14, pady=6,
                  cursor="hand2", highlightbackground=LINE, highlightthickness=1).pack(side="right")

        actions = tk.Frame(page, bg=BG)
        actions.pack(fill="x", padx=26)
        self.start_flow_button = self._big_button(
            actions, "▶ アプリを作りはじめる", lambda: self.show_page("idea"), primary=True)
        self.start_flow_button.pack(side="left")
        self._big_button(actions, "つづきから再開", self.resume_generation).pack(side="left", padx=(12, 0))
        self._big_button(actions, "✎ ダメ出し", self.pick_revision_target).pack(side="left", padx=(12, 0))
        self._big_button(actions, "⚙ 設定", lambda: self.show_page("settings")).pack(side="right")
        tk.Label(page,
                 text="「つづきから再開」は途中で止まった作業の続き、「✎ ダメ出し」は完成した作品に直してほしいことを伝えて直してもらうときに使います。",
                 font=self.F(11), fg=MUTED, bg=BG, anchor="w").pack(fill="x", padx=26, pady=(6, 0))
        ext_row = tk.Frame(page, bg=BG)
        ext_row.pack(fill="x", padx=26, pady=(12, 0))
        self._big_button(ext_row, "🔧 既存のアプリを直す・育てる", lambda: self.show_page("external")).pack(side="left")
        tk.Label(ext_row,
                 text="会社以外で作ったアプリや作りかけのアプリも、フォルダーを選んで依頼できます（元のフォルダーは変更しません）。",
                 font=self.F(11), fg=MUTED, bg=BG, anchor="w").pack(side="left", padx=(12, 0))
        tk.Label(page, text="これまでの作品", font=self.F(17, True), fg=NAVY, bg=BG,
                 anchor="w").pack(fill="x", padx=26, pady=(10, 2))
        tk.Label(page, text="作品ごとに、そのまま「ダメ出し」や「つづき」ができます。",
                 font=self.F(11), fg=MUTED, bg=BG, anchor="w").pack(fill="x", padx=26)
        self.runs_list_frame = tk.Frame(page, bg=BG)
        self.runs_list_frame.pack(fill="x", padx=26, pady=(6, 10))

    # ---- page: idea ----
    def _page_idea(self, page):
        nav = self._nav_row(page)
        self._big_button(nav, "← もどる", lambda: self.show_page("welcome")).pack(side="left")
        self._big_button(nav, "つぎへ：担当モデルを確認する →", self.goto_models, primary=True).pack(side="right")
        self._page_title(page, "① 作りたいアプリを教えてください",
                         "むずかしく書かなくて大丈夫です。「誰が・何に困っていて・どうなると嬉しいか」が伝われば、PMが要件に整理します。")
        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True, padx=26)
        self.request_text = scrolledtext.ScrolledText(
            body, height=8, wrap="word", font=self.F(15), padx=14, pady=12,
            highlightbackground=LINE, highlightthickness=1, relief="flat")
        self.request_text.pack(fill="both", expand=True)
        chips = tk.Frame(body, bg=BG)
        chips.pack(fill="x", pady=(12, 0))
        tk.Label(chips, text="例文を入れる:", font=self.F(12), fg=MUTED, bg=BG).pack(side="left")
        for label, example in EXAMPLE_IDEAS:
            tk.Button(chips, text=label,
                      command=lambda ex=example: self._set_idea(ex),
                      font=self.F(12), bg=CARD, fg=PRIMARY_DARK, relief="flat",
                      padx=14, pady=7, cursor="hand2",
                      highlightbackground=LINE, highlightthickness=1).pack(side="left", padx=(8, 0))

    def _set_idea(self, text):
        self.request_text.delete("1.0", "end")
        self.request_text.insert("1.0", text)
        self.request_text.focus_set()

    # ---- page: models ----
    def _page_models(self, page):
        nav = self._nav_row(page)
        self._big_button(nav, "← もどる", lambda: self.show_page("idea")).pack(side="left")
        self.create_button = self._big_button(nav, "▶ この内容で作成をはじめる", self.start_generation, primary=True)
        self.create_button.pack(side="right")
        self._page_title(page, "② どのモデルで進めますか",
                         "おすすめは「自動」です。部署ごとに一番得意なモデルを選びます。")
        modes = tk.Frame(page, bg=BG)
        modes.pack(fill="x", padx=26)
        mode_defs = (
            ("smart", "🤖 自動で部署ごとに切替（おすすめ）", "PM・設計は汎用モデル、実装・QAはコード特化モデルを自動で割り当てます"),
            ("single", "🧑‍🤝‍🧑 全員同じモデル（単一モデル）", "4部署とも同じモデルで進めます"),
            ("manual", "🎛 部署ごとに自分で選ぶ（手動）", "下の表で、部署ごとに使うモデルを指定します"),
        )
        self.mode_buttons = {}
        for value, title, desc in mode_defs:
            btn = tk.Radiobutton(
                modes, text=f"{title}\n{desc}", value=value, variable=self.routing_var,
                command=self.on_mode_changed, font=self.F(13), bg=CARD, fg=INK,
                selectcolor=SOFT, indicatoron=False, justify="left", anchor="w",
                wraplength=930, padx=16, pady=7, cursor="hand2",
                highlightbackground=LINE, highlightthickness=1,
                activebackground=SOFT)
            btn.pack(fill="x", pady=(0, 6))
            self.mode_buttons[value] = btn

        table = self._card(page)
        table.pack(fill="x", padx=26, pady=(4, 0))
        tk.Label(table, text="部署ごとの担当", font=self.F(14, True), fg=NAVY, bg=CARD,
                 anchor="w").pack(fill="x", pady=(0, 0))
        self.assign_rows = {}
        for key, emoji, dept_name, _role in DEPARTMENTS:
            row = tk.Frame(table, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{emoji} {dept_name}", font=self.F(14, True), fg=INK,
                     bg=CARD, width=16, anchor="w").pack(side="left")
            model_label = tk.Label(row, text="―", font=self.F(14, True), fg=PRIMARY_DARK,
                                   bg=CARD, anchor="w")
            model_label.pack(side="left")
            combo = ttk.Combobox(row, textvariable=self.stage_model_vars[key],
                                 font=self.F(13), width=30)
            reason_label = tk.Label(table, text="", font=self.F(10), fg=MUTED, bg=CARD, anchor="w")
            reason_label.pack(fill="x", padx=(190, 0))
            self.assign_rows[key] = (model_label, combo, reason_label)
        self.models_note = tk.Label(page, text="", font=self.F(11), fg=MUTED, bg=BG,
                                    anchor="w", wraplength=940)
        self.models_note.pack(fill="x", padx=26, pady=(4, 0))


    def on_mode_changed(self):
        manual = self.routing_var.get() == "manual"
        for key, (model_label, combo, _reason) in self.assign_rows.items():
            if manual:
                model_label.pack_forget()
                if not combo.winfo_ismapped():
                    combo.pack(side="left")
            else:
                combo.pack_forget()
                if not model_label.winfo_ismapped():
                    model_label.pack(side="left")
        self.refresh_models_display()

    def goto_models(self):
        try:
            validate_request_text(self.request_text.get("1.0", "end"))
        except ValueError:
            messagebox.showwarning(
                "アプリ案を入力してください",
                "作りたいアプリの説明が空です。\n入力欄にアプリ案を書いてから「つぎへ」を押してください。",
                parent=self.root)
            self.request_text.focus_set()
            return
        self.refresh_models_display()
        self.show_page("models")

    def refresh_models_display(self):
        # Welcome cards + models-page table share one computation.
        try:
            models = models_for_mode(
                self.routing_var.get(), self.model_var.get(),
                {k: v.get() for k, v in self.stage_model_vars.items()}
                if self.routing_var.get() == "manual" else {},
                installed_details=self.installed_details)
        except Exception:
            models = {key: "（未決定）" for key in STAGE_KEYS}
        reasons = {}
        if self.routing_var.get() == "smart" and self.installed_details:
            try:
                _a, reasons, _s = app_builder.smart_route_models(self.installed_details)
            except Exception:
                reasons = {}
        for key, _e, _n, _r in DEPARTMENTS:
            name = models.get(key, "―")
            if key in self.dept_card_models:
                self.dept_card_models[key].configure(text=name)
            if key in self.assign_rows:
                model_label, _combo, reason_label = self.assign_rows[key]
                model_label.configure(text=name)
                reason_label.configure(text=("選んだ理由: " + reasons[key]) if reasons.get(key) else "")
        if self.installed_models is not None:
            if self.routing_var.get() == "smart":
                self.models_note.configure(
                    text="自動モードでは、理想のモデルが入っていなくても止まらず、今あるモデルの中で一番良い割り当てで進みます。")
            else:
                absent = missing_models(models, self.installed_models)
                self.models_note.configure(
                    text=("注意: 未導入のモデルがあります ― " + ", ".join(absent) +
                          "（開始前に案内が出ます）") if absent else "選択中のモデルはすべて利用できます。")

    # ---- page: run ----
    def _page_run(self, page):
        nav = self._nav_row(page)
        self.stop_button = self._big_button(nav, "■ 中止する", self.stop_generation, danger=True)
        self.stop_button.pack(side="left")
        tk.Label(nav, text="中止しても、完了した部署の成果は残ります（「つづきから再開」で続けられます）",
                 font=self.F(11), fg=MUTED, bg=BG).pack(side="left", padx=(14, 0))
        self._page_title(page, "③ AI社員が作成中です",
                         "このままお待ちください。途中で止めても、できたところまでは保存されます。")
        self.run_status = tk.Label(page, text="準備中…", font=self.F(16, True),
                                   fg=PRIMARY_DARK, bg=BG, anchor="w")
        self.run_status.pack(fill="x", padx=26, pady=(2, 0))
        self.elapsed_label = tk.Label(page, text="", font=self.F(12), fg=MUTED, bg=BG, anchor="w")
        self.elapsed_label.pack(fill="x", padx=26, pady=(0, 10))
        rows = tk.Frame(page, bg=BG)
        rows.pack(fill="x", padx=26)
        self.phase_widgets = {}
        for idx, (key, emoji, dept_name, _role) in enumerate(DEPARTMENTS):
            card = self._card(rows)
            card.pack(fill="x", pady=(0, 8))
            icon = tk.Label(card, text="○", font=self.F(22, True), fg=MUTED, bg=CARD, width=2)
            icon.pack(side="left")
            middle = tk.Frame(card, bg=CARD)
            middle.pack(side="left", fill="x", expand=True)
            name_label = tk.Label(middle, text=f"{emoji} {dept_name}", font=self.F(16, True),
                                  fg=INK, bg=CARD, anchor="w")
            name_label.pack(side="left")
            model_line = tk.Label(middle, text="", font=self.F(12), fg=MUTED, bg=CARD, anchor="w")
            model_line.pack(side="left", padx=(14, 0))
            state = tk.Label(card, text="待機中", font=self.F(13, True), fg=MUTED, bg=CARD)
            state.pack(side="right")
            self.phase_widgets[key] = (icon, model_line, state, name_label)
        style = ttk.Style(self.root)
        try:
            style.configure("Big.Horizontal.TProgressbar", thickness=22)
        except tk.TclError:
            pass
        self.progress_bar = ttk.Progressbar(page, mode="determinate", maximum=100, value=0,
                                            style="Big.Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", padx=26, pady=(2, 10))
        self.oven_canvas = tk.Canvas(page, height=108, bg=BG, highlightthickness=0)
        self.oven_canvas.pack(fill="x", padx=26, pady=(0, 2))
        toggle_row = tk.Frame(page, bg=BG)
        toggle_row.pack(fill="x", padx=26)
        toggle = tk.Checkbutton(toggle_row, text="作業ログを見る", variable=self.show_log_var,
                                command=self._toggle_log, font=self.F(12), bg=BG, fg=MUTED,
                                activebackground=BG, anchor="w")
        toggle.pack(side="left")
        tk.Checkbutton(toggle_row, text="🔔 焼き上がったら音で知らせる（チーン♪）",
                       variable=self.sound_var, font=self.F(12), bg=BG, fg=MUTED,
                       activebackground=BG, anchor="w").pack(side="left", padx=(18, 0))
        tk.Label(toggle_row, text="オーブンの中で焼いています",
                 font=self.F(11), fg=MUTED, bg=BG, anchor="w").pack(side="left", padx=(14, 0))
        self.log_text = scrolledtext.ScrolledText(page, height=6, wrap="word",
                                                  font=self.F(11), state="disabled",
                                                  padx=10, pady=8)
        self.log_text.pack(fill="x", padx=26, pady=(4, 0))
        # 作成中画面はスクロール対応になったため、「中止する」を上部へ移す
        # （スクロールしなくてもすぐ押せるように。通常画面のナビは下部固定）
        nav.pack_forget()
        nav.pack(side="top", fill="x", padx=26, pady=(0, 2), before=self.run_status)

    # ---- oven animation (演出) ----
    def _oven_tick(self):
        if self.current_page != "run":
            self._oven_job = None
            return
        self.oven_frame += 1
        try:
            ratio = float(self.progress_bar.cget("value")) / 100.0
        except (tk.TclError, ValueError):
            ratio = 0.0
        self._draw_oven(ratio)
        self._oven_job = self.root.after(150, self._oven_tick)

    def _draw_oven(self, ratio):
        canvas = getattr(self, "oven_canvas", None)
        if canvas is None:
            return
        width = max(canvas.winfo_width(), 600)
        cx = width // 2
        canvas.delete("all")
        # oven body
        canvas.create_rectangle(cx - 185, 4, cx + 185, 104, fill="#39404d",
                                outline="#23272f", width=2)
        canvas.create_rectangle(cx - 165, 11, cx + 165, 29, fill="#2b3038", outline="")
        canvas.create_text(cx, 20, text="AI OVEN", fill="#cfd6df",
                           font=self.F(10, True))
        # door window; glass warms up as baking progresses
        glass = _mix_color("#5d6a7a", "#7a4a1d", ratio)
        canvas.create_rectangle(cx - 152, 34, cx + 152, 98, fill=glass,
                                outline="#20242b", width=2)
        # heater glow (flickers with the frame counter)
        glow = "#ff9a3c" if self.oven_frame % 2 == 0 else "#e87f24"
        if ratio > 0.03:
            canvas.create_line(cx - 142, 93, cx + 142, 93, fill=glow, width=3)
            canvas.create_line(cx - 142, 40, cx + 142, 40, fill=glow, width=2)
        # the app being baked: rises and browns with progress
        rise = ratio * 8
        item_fill = _mix_color("#f7ecc4", "#cf8433", ratio)
        canvas.create_rectangle(cx - 142, 88, cx + 142, 91, fill="#8f9aa8", outline="")
        canvas.create_oval(cx - 52, 62 - rise, cx + 52, 89 - rise,
                           fill=item_fill, outline="#a9713a", width=2)
        canvas.create_text(cx, 75 - rise, text="アプリ", fill="#5b3a12",
                           font=self.F(11, True))
        # steam once it is fairly baked
        if ratio > 0.35:
            for i in range(3):
                sx = cx - 44 + i * 44
                sy = 54 - ((self.oven_frame * 2 + i * 9) % 14)
                canvas.create_oval(sx - 5, sy - 7, sx + 5, sy + 7,
                                   fill="#e8edf2", outline="")
        # honest bake gauge inside the door window (top-right)
        canvas.create_text(cx + 144, 48, text=f"焼き加減 {int(ratio * 100)}%",
                           fill="#ffe9b0", font=self.F(10, True), anchor="e")
        if ratio >= 0.999:
            canvas.create_text(cx, 20, text="AI OVEN ― 焼き上がり！",
                               fill="#ffd98a", font=self.F(10, True))

    def _toggle_log(self):
        if self.show_log_var.get():
            self.log_text.pack(fill="x", padx=26, pady=(4, 0))
        else:
            self.log_text.pack_forget()

    def _set_phase(self, key, mode, state_text=None):
        icon, _model_line, state, _name = self.phase_widgets[key]
        if mode == "done":
            icon.configure(text="✓", fg=GREEN)
            state.configure(text=state_text or "完了", fg=GREEN)
        elif mode == "active":
            icon.configure(text="▶", fg=PRIMARY)
            state.configure(text=state_text or "作業中…", fg=PRIMARY_DARK)
        elif mode == "error":
            icon.configure(text="✖", fg=RED)
            state.configure(text=state_text or "失敗", fg=RED)
        else:
            icon.configure(text="○", fg=MUTED)
            state.configure(text=state_text or "待機中", fg=MUTED)

    def append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ---- page: done ----
    def _page_done(self, page):
        wrap = tk.Frame(page, bg=BG)
        wrap.pack(fill="both", expand=True, padx=26, pady=(12, 0))
        self.done_icon = tk.Label(wrap, text="✓", font=self.F(36, True), fg=GREEN, bg=BG)
        self.done_icon.pack()
        self.done_title = tk.Label(wrap, text="", font=self.F(24, True), fg=NAVY, bg=BG,
                                   wraplength=940, justify="center")
        self.done_title.pack(pady=(2, 2))
        self.done_message = tk.Label(wrap, text="", font=self.F(13), fg=MUTED, bg=BG,
                                     wraplength=900, justify="center")
        self.done_message.pack()
        self.done_path = tk.Label(wrap, text="", font=self.F(11), fg=MUTED, bg=BG,
                                  wraplength=900, justify="center")
        self.done_path.pack(pady=(4, 2))
        self.done_file_label = tk.Label(wrap, text="", font=self.F(15, True), fg=NAVY, bg=BG,
                                        wraplength=900, justify="center")
        self.done_file_label.pack(pady=(0, 8))
        self.done_feedback_card = self._card(wrap)
        self.done_feedback_card.pack(fill="x", pady=(2, 10))
        tk.Label(self.done_feedback_card, text="できあがりはどうですか？",
                 font=self.F(17, True), fg=NAVY, bg=CARD, anchor="w").pack(fill="x")
        tk.Label(self.done_feedback_card,
                 text="直してほしいことがあれば、下に書いて「この内容で直す」を押してください。その場で修正が始まります。",
                 font=self.F(12), fg=MUTED, bg=CARD, anchor="w", wraplength=880).pack(fill="x")
        self.done_feedback_text = scrolledtext.ScrolledText(
            self.done_feedback_card, height=2, wrap="word", font=self.F(14), padx=12, pady=8,
            highlightbackground=LINE, highlightthickness=1, relief="flat")
        self.done_feedback_text.pack(fill="x", pady=(8, 8))
        self.done_feedback_button = self._big_button(
            self.done_feedback_card, "▶ この内容で直す", self.start_revision_from_done, primary=True)
        self.done_feedback_button.pack(anchor="w")
        self.done_feedback_note = tk.Label(self.done_feedback_card, text="",
                                           font=self.F(11), fg=MUTED, bg=CARD, anchor="w")
        self.done_feedback_note.pack(fill="x", pady=(6, 0))
        buttons = tk.Frame(wrap, bg=BG)
        buttons.pack()
        self.preview_button = self._big_button(buttons, "🌐 ブラウザーで開く（完成したアプリ）", self.preview_output, primary=True)
        self.preview_button.pack(pady=(0, 8))
        row = tk.Frame(buttons, bg=BG)
        row.pack()
        self.open_button = self._big_button(row, "フォルダーを開く", self.open_output)
        self.open_button.pack(side="left")
        self._big_button(row, "GitHubへ公開する", self.goto_publish).pack(side="left", padx=(12, 0))
        self._big_button(row, "新しいアプリを作る", self.new_app).pack(side="left", padx=(12, 0))
        self.resume_done_button = self._big_button(buttons, "つづきから再開する", self.resume_generation, primary=True)
        self.revise_done_button = self._big_button(
            buttons, "✎ ダメ出し：直してほしいことを伝える",
            lambda: self.goto_revision(self.last_output, "done"))
        tk.Label(wrap,
                 text="「つづき」は同じ指示で残りの工程を行うもの、「ダメ出し」は直してほしい点を伝えて作り直すものです。",
                 font=self.F(11), fg=MUTED, bg=BG, wraplength=900, justify="center").pack(pady=(12, 0))

    def new_app(self):
        self.request_text.delete("1.0", "end")
        self.show_page("idea")

    def goto_publish(self):
        if not self.output_ready:
            messagebox.showinfo("GitHub公開", "先にアプリの生成が完了してから公開できます。", parent=self.root)
            return
        self.show_page("publish")

    def _show_done(self, code, output):
        output = Path(output)
        self.progress_bar.configure(value=100 if code in (0, 2) else self.progress_bar["value"])
        self.done_feedback_text.delete("1.0", "end")
        self.revise_done_button.pack_forget()
        if code == 0:
            self.done_icon.configure(text="✓", fg=GREEN)
            self.done_title.configure(text="アプリが完成しました！")
            self.done_message.configure(
                text="QA（検査）の承認も取れています。まず「生成したアプリを開く」で、実際に動くか確かめてみてください。")
            self.resume_done_button.pack_forget()
            self.preview_button.pack(pady=(0, 10))
        elif code == 2:
            self.done_icon.configure(text="⚠", fg=AMBER)
            self.done_title.configure(text="アプリは保存されました（QAは未承認）")
            self.done_message.configure(
                text="動くアプリはできていますが、検査の承認までは至りませんでした。開いて確かめ、気になるところがあれば教えてください。")
            self.resume_done_button.pack_forget()
            self.preview_button.pack(pady=(0, 10))
        elif code == 3:
            self.done_icon.configure(text="■", fg=MUTED)
            self.done_title.configure(text="中止しました")
            self.done_message.configure(
                text="完了した部署の成果は保存されています。「つづきから再開する」で、残りの部署だけを実行できます。")
            self.preview_button.pack_forget()
            self.resume_done_button.pack(pady=(0, 10))
        else:
            self.done_icon.configure(text="✖", fg=RED)
            self.done_title.configure(text="途中で止まりました")
            self.done_message.configure(
                text="完了した部署の成果は保存されています。もう一度「つづきから再開する」と、続きだけを実行できます。原因は作業ログと保存先の run.json に残っています。")
            self.preview_button.pack_forget()
            self.resume_done_button.pack(pady=(0, 10))
        if code == 2:
            try:
                manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
                defects = manifest.get("qa_defects_final") or []
            except (OSError, json.JSONDecodeError):
                defects = []
            if defects:
                listing = "\n".join(f"・{item}" for item in defects[:3])
                self.done_message.configure(
                    text=self.done_message.cget("text") +
                    f"\n\nQAに残っている指摘（上位{min(3, len(defects))}件）:\n{listing}\n"
                    "気になる点は「ダメ出し」から伝えて直せます。")
        self.revise_done_button.pack_forget()
        can_revise = output and ((output / "index.html").is_file()
                                 or (output / "index_draft_unverified.html").is_file())
        self.done_feedback_button.configure(state="normal" if can_revise else "disabled")
        self.done_feedback_note.configure(
            text="修正版は新しいフォルダーに保存され、今の作品はそのまま残ります。"
            if can_revise else
            "まだアプリ本体（index.html）がないため、ダメ出しはできません。「つづき」から完成させてください。")
        try:
            done_manifest_all = json.loads((output / "run.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            done_manifest_all = {}
        if done_manifest_all.get("mode") == "external_revision":
            self.done_message.configure(
                text=self.done_message.cget("text") +
                f"\n\n元のフォルダー： {done_manifest_all.get('source_folder')}（変更していません）"
                f"\n何を変えたか： {done_manifest_all.get('change_summary') or '（要約なし）'}"
                "\n修正版は上の保存先フォルダーの index.html です。問題なければ、その内容を元のアプリの場所へお使いください。")
        if self.run_mode == "revision":
            try:
                done_manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                done_manifest = {}
            if done_manifest.get("changed") is False:
                self.done_message.configure(
                    text=self.done_message.cget("text") +
                    "\n\n※今回は前回と変わりませんでした。指摘の書き方を変えて（例: どの部分が・どう動かないか）、もう一度ダメ出ししてみてください。")
        self.done_path.configure(text=f"保存先: {output}")
        if (output / "index.html").is_file():
            self.done_file_label.configure(
                text=f"完成したアプリのファイルは「{output.name}」フォルダーの中の index.html です")
        elif (output / "index_draft_unverified.html").is_file():
            self.done_file_label.configure(text="今回は下書き（index_draft_unverified.html）のみ保存されています")
        else:
            self.done_file_label.configure(text="")
        self.show_page("done")

    # ---- page: revision (ダメ出し) ----
    def _page_revision(self, page):
        nav = self._nav_row(page)
        self._big_button(nav, "← もどる", self._back_from_revision).pack(side="left")
        self.revision_start_button = self._big_button(
            nav, "▶ この内容で修正をはじめる", self.start_revision, primary=True)
        self.revision_start_button.pack(side="right")
        self._page_title(page, "✎ ダメ出し ― 直してほしいことを伝えます",
                         "完成した作品への修正依頼です。実装の社員が既存のファイルを直し、QAが再検査します。"
                         "修正版は新しい実行フォルダーに保存され、元の作品はそのまま残ります。")
        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True, padx=26)
        self.revision_target_label = tk.Label(body, text="", font=self.F(14, True),
                                              fg=NAVY, bg=BG, anchor="w")
        self.revision_target_label.pack(fill="x")
        self.revision_request_label = tk.Label(body, text="", font=self.F(12), fg=MUTED,
                                               bg=BG, anchor="w", wraplength=940)
        self.revision_request_label.pack(fill="x", pady=(2, 10))
        tk.Label(body, text="直してほしいこと（具体的に書くほど、その通りに直ります）",
                 font=self.F(14, True), fg=INK, bg=BG, anchor="w").pack(fill="x")
        self.revision_notes = scrolledtext.ScrolledText(
            body, height=7, wrap="word", font=self.F(15), padx=14, pady=12,
            highlightbackground=LINE, highlightthickness=1, relief="flat")
        self.revision_notes.pack(fill="both", expand=True, pady=(4, 0))
        chips = tk.Frame(body, bg=BG)
        chips.pack(fill="x", pady=(12, 0))
        tk.Label(chips, text="例文を入れる:", font=self.F(12), fg=MUTED, bg=BG).pack(side="left")
        for label, example in (
            ("停止ボタンを付けて", "読み上げを途中で止めるボタンがありません。停止ボタンを付けてください。"),
            ("文字を大きく", "文字とボタンが小さいです。全体に大きく、読みやすくしてください。"),
            ("動きがおかしい", "○○ボタンを押しても動きません。直してください。"),
        ):
            tk.Button(chips, text=label,
                      command=lambda ex=example: self._set_revision_notes(ex),
                      font=self.F(12), bg=CARD, fg=PRIMARY_DARK, relief="flat",
                      padx=14, pady=7, cursor="hand2",
                      highlightbackground=LINE, highlightthickness=1).pack(side="left", padx=(8, 0))

    def _set_revision_notes(self, text):
        self.revision_notes.delete("1.0", "end")
        self.revision_notes.insert("1.0", text)
        self.revision_notes.focus_set()

    def _back_from_revision(self):
        self.show_page(self.revision_return if self.revision_return in self.pages else "welcome")

    def goto_revision(self, target, return_to):
        if not target:
            return
        target = Path(target)
        if not ((target / "index.html").is_file() or (target / "index_draft_unverified.html").is_file()):
            messagebox.showwarning("ダメ出し", "対象のフォルダーに完成したアプリ（index.html）がありません。", parent=self.root)
            return
        self.revision_target = target
        self.revision_return = return_to
        request = ""
        try:
            manifest = json.loads((target / "run.json").read_text(encoding="utf-8"))
            request = str(manifest.get("request") or "")
        except (OSError, json.JSONDecodeError):
            request = ""
        self.revision_target_label.configure(text=f"対象の作品: {target.name}")
        self.revision_request_label.configure(
            text=f"元のアプリ案: {request[:140]}" if request else "元のアプリ案は記録されていません")
        self.show_page("revision")

    def pick_revision_target(self):
        if self.busy:
            return
        runs_root = Path(__file__).resolve().parent / "runs"
        chosen = filedialog.askdirectory(
            title="ダメ出しする作品の実行フォルダーを選んでください（index.html のあるフォルダー）",
            initialdir=str(runs_root) if runs_root.exists() else str(Path(__file__).resolve().parent),
            parent=self.root)
        if chosen:
            self.goto_revision(Path(chosen), "welcome")

    @staticmethod
    def _model_size_b(name, details):
        match = re.search(r":(\d+(?:\.\d+)?)b\b", str(name).lower())
        if match:
            return float(match.group(1))
        for item in details or []:
            if item.get("name") == name:
                size_text = str((item.get("details") or {}).get("parameter_size") or "")
                m2 = re.search(r"(\d+(?:\.\d+)?)\s*B", size_text)
                if m2:
                    return float(m2.group(1))
        return 0.0

    def _confirm_big_models(self, required):
        big = sorted({m for m in required.values()
                      if self._model_size_b(m, self.installed_details) >= 20})
        if not big:
            return True
        return messagebox.askyesno(
            "大きなモデルが含まれています",
            f"{'、'.join(big)} は大きなモデルです。\n"
            "16GBのPCでは1回の生成に非常に長い時間がかかり、ゲージが止まって見えることがあります。"
            "（前回の実機でも、実装に大きなモデルが選ばれていて止まって見えました）\n\n"
            "このまま続けますか？", parent=self.root)

    def _start_revision_with_notes(self, notes):
        try:
            validate_local_endpoint(self.endpoint_var.get())
            args, required = self._build_args_and_required()
        except (ValueError, SystemExit) as exc:
            if isinstance(exc, SystemExit):
                return
            messagebox.showerror("設定を確認してください", str(exc), parent=self.root)
            return
        if not self._confirm_big_models(required):
            return
        self._begin_revision(notes, required, args)

    def start_revision(self):
        if self.busy or not self.revision_target:
            return
        notes = self.revision_notes.get("1.0", "end").strip()
        if not notes:
            messagebox.showwarning("修正依頼を書いてください",
                                   "直してほしいことを入力欄に書いてから「修正をはじめる」を押してください。",
                                   parent=self.root)
            self.revision_notes.focus_set()
            return
        self._start_revision_with_notes(notes)

    def start_revision_from_done(self):
        if self.busy or not self.last_output:
            return
        notes = self.done_feedback_text.get("1.0", "end").strip()
        if not notes:
            messagebox.showinfo(
                "ダメ出し",
                "直してほしいことを上の欄に書いてから押してください。\n問題がなければ、そのままお使いください。",
                parent=self.root)
            self.done_feedback_text.focus_set()
            return
        self.revision_target = Path(self.last_output)
        self._start_revision_with_notes(notes)

    def _apply_phase_names(self, mode):
        if mode == "external":
            names = {"requirements": "📄 既存アプリの確認", "design": "📝 改修の方針",
                     "implementation": "💻 修正の実装", "qa": "🔍 QA・再検査"}
            for key, text in names.items():
                self.phase_widgets[key][3].configure(text=text)
            return
        if mode == "revision":
            names = {"requirements": "📄 元のアプリ確認", "design": "📝 修正方針を立てる",
                     "implementation": "💻 修正の実装", "qa": "🔍 QA・再検査"}
        else:
            names = {key: f"{emoji} {dept_name}" for key, emoji, dept_name, _r in DEPARTMENTS}
        for key, text in names.items():
            self.phase_widgets[key][3].configure(text=text)

    def _begin_revision(self, notes, required, args):
        self.busy = True
        self.run_mode = "revision"
        self.cancel_event = threading.Event()
        self.output_ready = False
        self.active_models = required
        self._apply_phase_names("revision")
        self.current_phase = 0
        self.activity_name = ""
        self.qa_round = 0
        self.run_started_at = time.monotonic()
        self.last_activity_at = self.run_started_at
        for key, _e, _n, _r in DEPARTMENTS:
            _icon, model_line, _state, _name = self.phase_widgets[key]
            model_line.configure(text=f"担当モデル: {required.get(key, '')}")
            self._set_phase(key, "wait")
        self.progress_bar.configure(value=0)
        self.run_status.configure(text="元のアプリと修正依頼を確認しています…")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.append_log(f"ダメ出しを受け付けました。対象: {self.revision_target}")
        self.show_page("run")
        threading.Thread(target=self._revision_worker,
                         args=(notes, required, args), daemon=True).start()

    def _revision_worker(self, notes, required, args):
        try:
            output_root = app_builder.resolve_output_root("runs")
            code, new_run = app_builder.run_revision(
                self.revision_target, notes, args, output_root,
                progress_callback=lambda line: self.events.put(("progress", line)),
                token_callback=lambda stage, chars: self.events.put(("tokens", (stage, chars))),
                cancel_event=self.cancel_event)
            self.events.put(("output", str(new_run)))
            self.events.put(("finished", (code, str(new_run))))
        except Exception as exc:
            self.events.put(("worker_error", str(exc)))

    # ---- page: external revision (既存のアプリを直す・育てる) ----
    def _page_external(self, page):
        nav = self._nav_row(page)
        self._big_button(nav, "← もどる", lambda: self.show_page("welcome")).pack(side="left")
        self.external_start_button = self._big_button(
            nav, "▶ この内容で直しはじめる", self.start_external_revision, primary=True)
        self.external_start_button.pack(side="right")
        self._page_title(page, "🔧 既存のアプリを直す・育てる",
                         "会社以外で作ったアプリや、作りかけのアプリのフォルダーを選んで、直してほしいこと・追加したい機能を書いてください。"
                         "元のフォルダーは変更しません。修正版は会社の runs フォルダーに新しく作ります（元のファイルの控えも一緒に保存されます）。"
                         "対象は1ファイルで完結する単体HTMLアプリです。")
        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True, padx=26)
        pick_row = tk.Frame(body, bg=BG)
        pick_row.pack(fill="x")
        self._big_button(pick_row, "📁 フォルダーを選ぶ", self.pick_external_folder).pack(side="left")
        self.external_folder_label = tk.Label(
            pick_row, text="まだ選ばれていません", font=self.F(13, True), fg=NAVY, bg=BG, anchor="w")
        self.external_folder_label.pack(side="left", padx=(14, 0))
        self.external_status_label = tk.Label(body, text="", font=self.F(12), fg=MUTED, bg=BG,
                                              anchor="w", wraplength=940, justify="left")
        self.external_status_label.pack(fill="x", pady=(6, 10))
        tk.Label(body, text="直してほしいこと・追加したい機能（具体的に書くほど、その通りに直ります）",
                 font=self.F(14, True), fg=INK, bg=BG, anchor="w").pack(fill="x")
        self.external_notes = scrolledtext.ScrolledText(
            body, height=7, wrap="word", font=self.F(15), padx=14, pady=12,
            highlightbackground=LINE, highlightthickness=1, relief="flat")
        self.external_notes.pack(fill="both", expand=True, pady=(4, 0))
        chips = tk.Frame(body, bg=BG)
        chips.pack(fill="x", pady=(10, 0))
        tk.Label(chips, text="例文を入れる:", font=self.F(12), fg=MUTED, bg=BG).pack(side="left")
        for label, example in (
            ("ボタンを大きく", "ボタンと文字が小さいので、全体的に大きくして、押しやすくしてください。"),
            ("機能を追加", "（追加したい機能を具体的に書いてください。例: 結果を保存できるボタンを付けてください）"),
            ("色合いを変更", "色合いを、目にやさしい落ち着いた色に変えてください。"),
        ):
            tk.Button(chips, text=label,
                      command=lambda ex=example: self._set_external_notes(ex),
                      font=self.F(12), bg=CARD, fg=PRIMARY_DARK, relief="flat",
                      padx=14, pady=7, cursor="hand2",
                      highlightbackground=LINE, highlightthickness=1).pack(side="left", padx=(8, 0))

    def _set_external_notes(self, text):
        self.external_notes.delete("1.0", "end")
        self.external_notes.insert("1.0", text)
        self.external_notes.focus_set()

    def pick_external_folder(self):
        chosen = filedialog.askdirectory(
            title="直したいアプリのフォルダーを選んでください（単体HTMLアプリ）",
            parent=self.root)
        if not chosen:
            return
        self.external_folder = Path(chosen)
        self.external_info = app_builder.inspect_external_target(self.external_folder)
        self.external_folder_label.configure(text=f"対象のフォルダー: {self.external_folder}")
        if self.external_info["ok"]:
            self.external_status_label.configure(
                text=f"✓ 対象ファイル: {self.external_info['name']}（単体HTML）― このまま依頼できます。",
                fg=GREEN)
        else:
            self.external_status_label.configure(
                text=f"✗ {self.external_info['reason']}", fg=RED)

    def start_external_revision(self):
        if self.busy:
            return
        if not self.external_folder or not self.external_info or not self.external_info.get("ok"):
            messagebox.showwarning(
                "フォルダーを選んでください",
                "先に「フォルダーを選ぶ」で、単体HTMLアプリのフォルダーを選んでください。"
                "対象外の場合は、その理由が画面に表示されます。",
                parent=self.root)
            return
        notes = self.external_notes.get("1.0", "end").strip()
        if not notes:
            messagebox.showwarning("依頼内容を書いてください",
                                   "直してほしいこと・追加したい機能を入力欄に書いてから「直しはじめる」を押してください。",
                                   parent=self.root)
            self.external_notes.focus_set()
            return
        self._start_external_with_notes(notes)

    def _start_external_with_notes(self, notes):
        try:
            validate_local_endpoint(self.endpoint_var.get())
            args, required = self._build_args_and_required()
        except (ValueError, SystemExit) as exc:
            if isinstance(exc, SystemExit):
                return
            messagebox.showerror("設定を確認してください", str(exc), parent=self.root)
            return
        if not self._confirm_big_models(required):
            return
        self._begin_external(notes, required, args)

    def _begin_external(self, notes, required, args):
        self.busy = True
        self.run_mode = "external"
        self.cancel_event = threading.Event()
        self.output_ready = False
        self.active_models = required
        self._apply_phase_names("external")
        self.current_phase = 0
        self.activity_name = ""
        self.qa_round = 0
        self.run_started_at = time.monotonic()
        self.last_activity_at = self.run_started_at
        for key, _e, _n, _r in DEPARTMENTS:
            _icon, model_line, _state, _name = self.phase_widgets[key]
            model_line.configure(text=f"担当モデル: {required.get(key, '')}")
            self._set_phase(key, "wait")
        self.progress_bar.configure(value=0)
        self.run_status.configure(text="既存のアプリと依頼を確認しています…")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.append_log(f"既存アプリの改修を受け付けました。対象: {self.external_folder}")
        self.show_page("run")
        threading.Thread(target=self._external_worker,
                         args=(notes, required, args), daemon=True).start()

    def _external_worker(self, notes, required, args):
        try:
            output_root = app_builder.resolve_output_root("runs")
            code, new_run = app_builder.run_external_revision(
                self.external_folder, notes, args, output_root,
                progress_callback=lambda line: self.events.put(("progress", line)),
                token_callback=lambda stage, chars: self.events.put(("tokens", (stage, chars))),
                cancel_event=self.cancel_event)
            self.events.put(("output", str(new_run)))
            self.events.put(("finished", (code, str(new_run))))
        except Exception as exc:
            self.events.put(("worker_error", str(exc)))

    # ---- page: publish ----    # ---- page: publish ----
    def _page_publish(self, page):
        nav = self._nav_row(page)
        self._big_button(nav, "← 結果画面にもどる", lambda: self.show_page("done")).pack(side="left")
        self.publish_button = self._big_button(nav, "GitHubへアップロード", self.start_publication, primary=True)
        self.publish_button.configure(state="disabled")
        self.publish_button.pack(side="right")
        self._page_title(page, "GitHubへ公開する",
                         "完成したアプリ（index.html）を、あなたの公開リポジトリへアップロードします。公開すると、誰でも見られるようになります。")
        body = tk.Frame(page, bg=BG)
        body.pack(fill="x", padx=26)
        tk.Label(body, text="公開先リポジトリのURL", font=self.F(14, True), fg=INK, bg=BG,
                 anchor="w").pack(fill="x")
        self.repository_url_entry = tk.Entry(body, textvariable=self.repository_url_var,
                                             font=self.F(14))
        self.repository_url_entry.pack(fill="x", ipady=8, pady=(4, 4))
        tk.Label(body, text="例: https://github.com/あなたの名前/アプリ名　（空の公開リポジトリを先に作ってください）",
                 font=self.F(11), fg=MUTED, bg=BG, anchor="w").pack(fill="x")
        self.publication_review_check = tk.Checkbutton(
            body, text="アプリの内容を確認しました。公開リポジトリで誰でも見られることを理解しています。",
            variable=self.publication_reviewed_var, command=self.update_publish_button,
            font=self.F(13), bg=BG, fg=INK, activebackground=BG, wraplength=920,
            justify="left", anchor="w")
        self.publication_review_check.pack(fill="x", pady=(16, 6))
        self.publication_status_var = tk.StringVar(
            value="このアプリはリポジトリを作成せず、Pagesの有効化もしません（アップロードだけを行います）。")
        tk.Label(body, textvariable=self.publication_status_var, font=self.F(12),
                 fg=MUTED, bg=BG, anchor="w", wraplength=920, justify="left").pack(fill="x")
        self.repository_url_var.trace_add("write", lambda *_: self.update_publish_button())

    def update_publish_button(self):
        if not hasattr(self, "publish_button"):
            return
        ready = (not self.busy and self.output_ready
                 and bool(self.repository_url_var.get().strip())
                 and self.publication_reviewed_var.get())
        self.publish_button.configure(state="normal" if ready else "disabled")

    # ---- page: settings ----
    def _page_settings(self, page):
        nav = self._nav_row(page)
        self._big_button(nav, "← もどる", lambda: self.show_page("welcome"), primary=True).pack(side="left")
        self._page_title(page, "設定",
                         "通常は変更しなくて大丈夫です。動作が重いときは、コンテキストを 2048 に下げると軽くなります。")
        body = self._card(page)
        body.pack(fill="x", padx=26)
        fields = (
            ("Ollamaの接続先", self.endpoint_var, None),
            ("単一モードのモデル", self.model_var, "「単一モデル」モードと、手動モードで空欄の部署に使います"),
            ("コンテキスト（大きいほど賢いが重い）", self.context_var, "2048 / 4096 / 8192 から選びます。実装の部署だけ自動で最大8192まで広がります"),
            ("タイムアウト（秒）", self.timeout_var, "1回の生成を待つ上限です。遅いPCでは 900〜1200 が目安です"),
        )
        self._settings_entries = {}
        for label, var, note in fields:
            tk.Label(body, text=label, font=self.F(13, True), fg=INK, bg=CARD,
                     anchor="w").pack(fill="x", pady=(10, 2))
            if var is self.context_var:
                widget = ttk.Combobox(body, textvariable=var, values=("2048", "4096", "8192"),
                                      font=self.F(14), width=16, state="readonly")
                widget.pack(anchor="w", ipady=4)
            else:
                widget = tk.Entry(body, textvariable=var, font=self.F(14), width=52)
                widget.pack(anchor="w", fill="x", ipady=5)
            self._settings_entries[label] = widget
            if note:
                tk.Label(body, text=note, font=self.F(11), fg=MUTED, bg=CARD,
                         anchor="w").pack(fill="x")
        tk.Label(body, text="完成のお知らせ音", font=self.F(13, True), fg=INK, bg=CARD,
                 anchor="w").pack(fill="x", pady=(16, 2))
        sound_settings_row = tk.Frame(body, bg=CARD)
        sound_settings_row.pack(fill="x")
        tk.Checkbutton(sound_settings_row, text="焼き上がったら「チーン♪」と鳴らす",
                       variable=self.sound_var, font=self.F(13), bg=CARD, fg=INK,
                       activebackground=CARD, anchor="w").pack(side="left")
        tk.Button(sound_settings_row, text="試しに鳴らす",
                  command=lambda: play_chime("success"),
                  font=self.F(12), bg=BG, fg=PRIMARY_DARK, relief="flat",
                  padx=12, pady=5, cursor="hand2",
                  highlightbackground=LINE, highlightthickness=1).pack(side="left", padx=(12, 0))
        tk.Label(body, text="途中で止まったときは、完成と区別できるよう別の音が鳴ります。作成中の画面からも切り替えられます。",
                 font=self.F(11), fg=MUTED, bg=CARD, anchor="w").pack(fill="x")

    # ---- connection check ----
    def check_connection(self):
        if self.busy or self.checking:
            return
        self.checking = True
        self.conn_label.configure(text="● Ollamaを確認しています…", fg=MUTED)
        threading.Thread(target=self._check_worker,
                         args=(self.endpoint_var.get().strip(),), daemon=True).start()

    def _check_worker(self, endpoint):
        try:
            details = fetch_ollama_model_details(endpoint)
            self.events.put(("models", details))
        except Exception as exc:
            self.events.put(("check_error", str(exc)))

    # ---- run control ----
    def _build_args_and_required(self):
        stage_values = ({k: v.get() for k, v in self.stage_model_vars.items()}
                        if self.routing_var.get() == "manual" else {})
        args = build_run_args(
            self.endpoint_var.get().strip(), self.model_var.get(), stage_values,
            self.context_var.get(), self.routing_var.get(), self.timeout_var.get())
        required = models_for_mode(args.routing_mode, args.model, stage_values,
                                   installed_details=self.installed_details)
        return args, required

    def start_generation(self):
        if self.busy:
            return
        try:
            request = validate_request_text(self.request_text.get("1.0", "end"))
            validate_local_endpoint(self.endpoint_var.get())
            args, required = self._build_args_and_required()
            if self.routing_var.get() == "manual" and any(
                    not v.get().strip() for v in self.stage_model_vars.values()):
                messagebox.showinfo("部署別モデル設定",
                                    "空欄の部署には「単一モードのモデル」を使います。",
                                    parent=self.root)
        except (ValueError, SystemExit) as exc:
            if isinstance(exc, SystemExit):
                return
            messagebox.showerror("設定を確認してください", str(exc), parent=self.root)
            return
        if not self._confirm_big_models(required):
            return
        self._begin_run(request, required, args, resume_dir=None)

    def resume_generation(self, from_dir=None):
        if self.busy:
            return
        if from_dir is not None:
            run_dir = Path(from_dir)
        else:
            runs_root = Path(__file__).resolve().parent / "runs"
            chosen = filedialog.askdirectory(
                title="再開する実行フォルダーを選んでください（run.json のあるフォルダー）",
                initialdir=str(runs_root) if runs_root.exists() else str(Path(__file__).resolve().parent),
                parent=self.root)
            if not chosen:
                return
            run_dir = Path(chosen)
        manifest_path = run_dir / "run.json"
        if not manifest_path.is_file():
            messagebox.showerror("再開できません", "選んだフォルダーに run.json がありません。実行フォルダーそのものを選んでください。", parent=self.root)
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            request = str(manifest.get("request") or "").strip()
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("再開できません", f"run.json を読めません: {exc}", parent=self.root)
            return
        if not request:
            messagebox.showerror("再開できません", "元のアプリ案が run.json に残っていません。", parent=self.root)
            return
        # A finished work does not change by "resuming" (all stages are
        # reused as-is). Steer the user to the revision flow instead.
        if str(manifest.get("status") or "") in ("completed", "completed_unapproved") \
                and (run_dir / "index.html").is_file():
            go_revision = messagebox.askyesno(
                "この作品は完成しています",
                "「つづき」は同じ指示文で残りの工程だけを行うため、出来上がりは変わりません。\n"
                "直してほしい点がある場合は「ダメ出し」がおすすめです。\n\n"
                "ダメ出しに進みますか？（いいえ＝それでも「つづき」を実行します）",
                parent=self.root)
            if go_revision:
                self.goto_revision(run_dir, "welcome")
                return
        try:
            validate_local_endpoint(self.endpoint_var.get())
            args, required = self._build_args_and_required()
        except (ValueError, SystemExit) as exc:
            if isinstance(exc, SystemExit):
                return
            messagebox.showerror("設定を確認してください", str(exc), parent=self.root)
            return
        self.request_text.delete("1.0", "end")
        self.request_text.insert("1.0", request)
        self._begin_run(request, required, args, resume_dir=run_dir)

    def _begin_run(self, request, required, args, resume_dir):
        self.busy = True
        self.run_mode = "build"
        self.cancel_event = threading.Event()
        self.output_ready = False
        self.publication_reviewed_var.set(False)
        self.active_models = required
        self._apply_phase_names("build")
        self.current_phase = 0
        self.activity_name = ""
        self.qa_round = 0
        self.run_started_at = time.monotonic()
        self.last_activity_at = self.run_started_at
        for key, _e, _n, _r in DEPARTMENTS:
            _icon, model_line, _state, _name = self.phase_widgets[key]
            model_line.configure(text=f"担当モデル: {required.get(key, '')}")
            self._set_phase(key, "wait")
        self.progress_bar.configure(value=0)
        self.run_status.configure(text="Ollamaとモデルを確認しています…")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.append_log("入力を受け付けました。Ollama接続と必要モデルを確認します。")
        self.show_page("run")
        threading.Thread(target=self._generation_worker,
                         args=(request, required, args, resume_dir), daemon=True).start()

    def _generation_worker(self, request, required, args, resume_dir=None):
        try:
            if resume_dir is not None:
                run_dir = Path(resume_dir)
            else:
                if args.routing_mode != "smart":
                    installed = fetch_ollama_models(args.endpoint)
                    absent = missing_models(required, installed)
                    if absent:
                        self.events.put(("missing_models", (absent, installed)))
                        return
                output_root = app_builder.resolve_output_root("runs")
                run_dir = app_builder.create_run_directory(output_root)
            self.events.put(("output", str(run_dir)))
            exit_code = app_builder.run_pipeline(
                request, args, run_dir,
                progress_callback=lambda line: self.events.put(("progress", line)),
                token_callback=lambda stage, chars: self.events.put(("tokens", (stage, chars))),
                cancel_event=self.cancel_event,
                resume=resume_dir is not None)
            self.events.put(("finished", (exit_code, str(run_dir))))
        except Exception as exc:
            self.events.put(("worker_error", str(exc)))

    def stop_generation(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.run_status.configure(text="中止しています…（現在の文字の区切りで止まります）")
            self.append_log("中止ボタンが押されました。完了済みの工程は保存されています。")

    def preview_output(self):
        if not self.last_output:
            return
        html_path = Path(self.last_output) / "index.html"
        if not html_path.is_file():
            messagebox.showinfo("生成アプリ", "まだ index.html がありません。", parent=self.root)
            return
        webbrowser.open(html_path.resolve().as_uri())

    def open_output(self):
        if not self.last_output or not Path(self.last_output).exists():
            messagebox.showinfo("出力先", "まだ出力フォルダーは作成されていません。", parent=self.root)
            return
        try:
            if os.name == "nt":
                os.startfile(str(self.last_output))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.last_output)])
            else:
                subprocess.Popen(["xdg-open", str(self.last_output)])
        except OSError as exc:
            messagebox.showerror("フォルダーを開けません", error_guidance("出力フォルダーを開く", exc), parent=self.root)

    # ---- GitHub publication (unchanged flow, own screen) ----
    def start_publication(self):
        if self.busy:
            return
        if not self.output_ready or not self.last_output:
            messagebox.showwarning("生成HTMLがありません", "先にアプリを生成し、出力された index.html を確認してください。", parent=self.root)
            return
        if not self.publication_reviewed_var.get():
            messagebox.showwarning("公開前の確認が必要です", "HTMLの内容と、公開リポジトリでは誰でも閲覧できることを確認してください。", parent=self.root)
            return
        repository_url = self.repository_url_var.get().strip()
        try:
            github_publish.parse_repository_url(repository_url)
        except ValueError as exc:
            messagebox.showerror("リポジトリURLを確認してください", str(exc), parent=self.root)
            return
        output = self.last_output
        self.busy = True
        self.publish_button.configure(state="disabled")
        self.publication_status_var.set("GitHubログイン、公開設定、書き込み権限、Git接続を確認しています...")
        threading.Thread(target=self._publication_preflight_worker,
                         args=(repository_url, output), daemon=True).start()

    def _publication_preflight_worker(self, repository_url, output):
        try:
            target = github_publish.preflight(repository_url)
            self.events.put(("publish_preflight_ready", (target, output)))
        except Exception as exc:
            self.events.put(("publish_error", exc))

    def _confirm_publication(self, target, output):
        qa_status = "未承認" if self._run_is_unapproved(output) else "承認済み"
        branch_description = (
            "空のリポジトリに main ブランチを作成します。"
            if target.empty else
            f"既存の {target.default_branch} ブランチに index.html を追加します。")
        prompt = (
            f"公開先: {target.repo_url}\n"
            "リポジトリは公開状態で、ソースも誰でも閲覧できます。\n"
            f"アップロード対象: {output / 'index.html'}（index.htmlのみ）\n"
            f"AI QA: {qa_status}\n"
            f"{branch_description}\n"
            "既存の index.html がある場合や競合が起きた場合は上書きせず停止します。\n\n"
            f"アップロード後、GitHubのSettings > PagesでDeploy from a branchを選んでください。\n"
            f"想定URL: {target.pages_url}\n"
            "Pagesを有効化するまでは、このURLは利用できません。\n\n"
            "GitHubへアップロードしますか？")
        return messagebox.askyesno("公開内容を最終確認", prompt, parent=self.root)

    @staticmethod
    def _run_is_unapproved(output):
        try:
            manifest = json.loads((Path(output) / "run.json").read_text(encoding="utf-8"))
            return manifest.get("qa_approved") is not True
        except (OSError, json.JSONDecodeError):
            return True

    def _publication_worker(self, repository_url, output, target):
        try:
            result = github_publish.publish_public_pages(
                output, repository_url, target,
                progress_callback=lambda line: self.events.put(("publish_progress", line)))
            self.events.put(("publish_finished", result))
        except Exception as exc:
            self.events.put(("publish_error", exc))

    # ---- event loop ----
    def process_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "models":
                    self.checking = False
                    if data and isinstance(data[0], dict):
                        self.installed_details = data
                        self.installed_models = [item["name"] for item in data]
                    else:
                        self.installed_models = list(data)
                        self.installed_details = [{"name": n, "size": 0, "details": {}} for n in self.installed_models]
                    for _k, (_l, combo, _r) in self.assign_rows.items():
                        combo.configure(values=self.installed_models)
                    self.conn_label.configure(
                        text=f"● Ollama接続済み（モデル {len(self.installed_models)} 本）", fg=GREEN)
                    self.conn_detail.configure(text=" / ".join(self.installed_models[:4]))
                    self.refresh_models_display()
                elif kind == "check_error":
                    self.checking = False
                    self.conn_label.configure(text="● Ollamaに接続できません", fg=RED)
                    self.conn_detail.configure(text="Ollamaアプリを起動してから「再確認」を押してください")
                elif kind == "missing_models":
                    absent, installed = data
                    self.installed_models = installed
                    self.busy = False
                    advice = error_guidance("モデル確認", "指定モデルが未導入です。", absent)
                    messagebox.showerror("モデルが見つかりません", advice, parent=self.root)
                    self.show_page("models")
                elif kind == "tokens":
                    stage, chars = data
                    self.last_activity_at = time.monotonic()
                    if not (self.cancel_event is not None and self.cancel_event.is_set()):
                        activity = self.activity_name or {
                            "requirements": "PM・要件定義", "design": "デザイン設計",
                            "implementation": "実装", "qa": "QA・検査"}.get(stage, stage)
                        self.run_status.configure(text=f"{activity}が生成中… {chars}文字")
                        if stage in self.phase_widgets:
                            self._set_phase(stage, "active", f"生成中… {chars}文字")
                        expected = {"requirements": 1500, "design": 2500,
                                    "implementation": 9000, "qa": 1500}.get(stage, 4000)
                        # The bar tracks the CURRENT activity only, and never
                        # sits at 100 while work continues (95% cap).
                        self.progress_bar.configure(value=min(95, int(chars / expected * 100)))
                elif kind == "output":
                    self.last_output = Path(data)
                elif kind == "progress":
                    self.last_activity_at = time.monotonic()
                    self.append_log(data)
                    self._handle_progress_line(data)
                elif kind == "finished":
                    code, output = data
                    self.last_output = Path(output)
                    self.busy = False
                    if self.sound_var.get():
                        if code in (0, 2):
                            play_chime("success")
                        elif not (self.cancel_event is not None
                                  and self.cancel_event.is_set()):
                            play_chime("failed")
                    self.output_ready = code in (0, 2) and (self.last_output / "index.html").is_file()
                    for key, _e, _n, _r in DEPARTMENTS:
                        if code in (0, 2):
                            self._set_phase(key, "done")
                    self._show_done(code, output)
                    self.update_publish_button()
                elif kind == "publish_preflight_ready":
                    target, output = data
                    if self._confirm_publication(target, output):
                        self.publication_status_var.set("確認済み。index.htmlをアップロードしています...")
                        threading.Thread(target=self._publication_worker,
                                         args=(self.repository_url_var.get().strip(), output, target),
                                         daemon=True).start()
                    else:
                        self.busy = False
                        self.publication_status_var.set("アップロードはキャンセルされました。")
                        self.update_publish_button()
                elif kind == "publish_progress":
                    self.publication_status_var.set(data)
                elif kind == "publish_finished":
                    self.busy = False
                    self.output_ready = False
                    self.publication_status_var.set(f"HTMLをアップロードしました。Pagesは未有効: {data.pages_url}")
                    messagebox.showinfo(
                        "HTMLをアップロードしました",
                        f"{data.details}\n\nリポジトリ:\n{data.repo_url}\n\n"
                        f"Pages有効化後のURL:\n{data.pages_url}\n\n"
                        "URLはPagesを設定してデプロイが完了するまで利用できません。",
                        parent=self.root)
                    self.update_publish_button()
                elif kind == "publish_error":
                    self.busy = False
                    advice = publication_guidance(data)
                    self.publication_status_var.set("アップロードを中止しました。既存内容は上書きしていません。")
                    messagebox.showerror("GitHub公開を完了できませんでした", advice, parent=self.root)
                    self.update_publish_button()
                elif kind == "worker_error":
                    if self.sound_var.get():
                        play_chime("failed")
                    self.busy = False
                    messagebox.showerror("実行できませんでした", error_guidance("事前確認・実行", data), parent=self.root)
                    self._show_done(1, self.last_output or Path("."))
        except queue.Empty:
            pass
        self.root.after(150, self.process_events)

    def _set_activity(self, name):
        self.activity_name = name
        self.progress_bar.configure(value=0)
        self.run_status.configure(text=f"{name}…")

    def _handle_progress_line(self, line):
        index = progress_phase(line)
        if index is not None:
            self.current_phase = index
            keys = [k for k, _e, _n, _r in DEPARTMENTS]
            for pos, key in enumerate(keys):
                if pos < index:
                    self._set_phase(key, "done")
                elif pos == index:
                    self._set_phase(key, "active")
            if self.run_mode == "external":
                names = ("既存アプリを確認中", "改修の方針を立てています", "修正を実装中", "QAが再検査中")
            elif self.run_mode == "revision":
                names = ("元のアプリを確認中", "修正方針を立てています", "修正を実装中", "QAが再検査中")
            else:
                names = ("PM・要件定義が作成中", "デザイン設計が作成中", "実装が作成中", "QAが検査中")
            self._set_activity(names[index])
        review = re.search(r"\[QA\] レビュー (\d+)/(\d+)", line)
        if review:
            self.qa_round = int(review.group(1))
            self._set_activity(f"QAレビュー {self.qa_round}回目")
            self._set_phase("qa", "active")
        if "[QA] 指摘を実装部署で修正します" in line:
            self._set_activity(f"差し戻し{max(self.qa_round, 1)}回目：修正中")
            self._set_phase("implementation", "active", "差し戻しの修正中…")
        if line.startswith("[RENDER]"):
            self._set_activity("画面表示を自動確認中")
        if "[REPAIR]" in line:
            self._set_activity("自動修復中")
            self._set_phase("implementation", "active", "自動修復中…")
        if "[DONE]" in line:
            self.progress_bar.configure(value=100)

    def _tick_clock(self):
        if self.busy and self.run_started_at:
            elapsed = int(time.monotonic() - self.run_started_at)
            text = f"経過時間 {elapsed // 60:02d}:{elapsed % 60:02d}"
            if self.last_activity_at:
                idle = time.monotonic() - self.last_activity_at
                if idle >= 15:
                    text += f" ／ 最後の更新から {int(idle)} 秒（大きなモデルは考え込む時間があります。そのままお待ちください）"
            self.elapsed_label.configure(text=text)
        self.root.after(1000, self._tick_clock)

    def on_close(self):
        if self.busy:
            if self.cancel_event is not None:
                self.cancel_event.set()
            messagebox.showinfo(
                "中止して終了します",
                "生成を中止します。完了済みの工程は保存されており、次回「つづきから再開」できます。",
                parent=self.root)
        self.root.destroy()


def main():
    root = tk.Tk()
    AppBuilderWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
