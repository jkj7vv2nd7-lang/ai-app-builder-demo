#!/usr/bin/env python3
"""Tkinter desktop interface for the local Ollama app builder."""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from urllib.parse import urlparse

import app_builder
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


class AppBuilderWindow:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.busy = False
        self.last_output = None
        self.output_ready = False
        self.installed_models = None
        self.installed_details = None
        self.cancel_event = None
        self.active_models = {}
        self.input_controls = []

        root.title("AIアプリ開発エージェント v2")
        root.geometry("840x900")
        root.minsize(720, 740)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="AIアプリ開発エージェント v2", font=("", 18, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="アイデアを入力して開始します。部署（PM・設計・実装・QA）ごとに最適なモデルへ自動で切り替わり、生成は途中で止まらず進み具合が見えます。失敗しても自動で直し、途中成果はいつも保存されます。",
            wraplength=770,
        ).pack(anchor="w", pady=(4, 12))

        request_frame = ttk.LabelFrame(outer, text="1. 作りたいアプリ（必須）", padding=10)
        request_frame.pack(fill="x")
        self.request_text = scrolledtext.ScrolledText(request_frame, height=5, wrap="word")
        self.request_text.pack(fill="x")
        self.input_controls.append(self.request_text)
        ttk.Label(
            request_frame,
            text=f"入力例: {EXAMPLE_REQUEST}\n誰向けか・何をしたいかを書いてください。空欄では実行できません。",
            wraplength=750,
        ).pack(anchor="w", pady=(5, 0))

        settings = ttk.LabelFrame(outer, text="2. Ollama・モデル設定", padding=10)
        settings.pack(fill="x", pady=(10, 0))
        ttk.Label(settings, text="接続先").grid(row=0, column=0, sticky="w")
        self.endpoint_var = tk.StringVar(value=app_builder.DEFAULT_ENDPOINT)
        endpoint_entry = ttk.Entry(settings, textvariable=self.endpoint_var, width=48)
        endpoint_entry.grid(
            row=0, column=1, sticky="ew", padx=(8, 8)
        )
        self.input_controls.append(endpoint_entry)
        self.check_button = ttk.Button(settings, text="接続・モデルを確認", command=self.check_connection)
        self.check_button.grid(row=0, column=2, sticky="e")
        self.model_status = ttk.Label(settings, text="起動確認中...")
        self.model_status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 8))

        ttk.Label(settings, text="単一モードのモデル").grid(row=2, column=0, sticky="w")
        self.model_var = tk.StringVar(value=app_builder.DEFAULT_MODEL)
        self.model_entry = ttk.Entry(settings, textvariable=self.model_var, width=36)
        self.model_entry.grid(
            row=2, column=1, sticky="w", padx=(8, 0)
        )
        self.input_controls.append(self.model_entry)
        self.separate_var = tk.BooleanVar(value=False)
        ttk.Label(
            settings,
            text="部署ごとのモデル（「部署ごとに手動」を選んだときだけ有効。一覧から選べます）",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 4))

        self.stage_model_vars = {
            "requirements": tk.StringVar(),
            "design": tk.StringVar(),
            "implementation": tk.StringVar(),
            "qa": tk.StringVar(),
        }
        self.stage_model_entries = {}
        for row, (key, label) in enumerate(
            (("requirements", "PM・要件定義"), ("design", "デザイン設計"), ("implementation", "実装"), ("qa", "QA")),
            start=4,
        ):
            ttk.Label(settings, text=f"  {label}").grid(row=row, column=0, sticky="w")
            entry = ttk.Combobox(settings, textvariable=self.stage_model_vars[key], width=34, state="disabled")
            entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
            self.stage_model_entries[key] = entry
            self.input_controls.append(entry)
            ttk.Label(settings, text="空欄なら単一モデルを使用").grid(row=row, column=2, sticky="w")

        ttk.Label(settings, text="コンテキスト").grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.context_var = tk.StringVar(value="4096")
        context_combo = ttk.Combobox(
            settings,
            textvariable=self.context_var,
            values=("2048", "4096", "8192"),
            width=12,
            state="readonly",
        )
        context_combo.grid(row=8, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        self.input_controls.append(context_combo)
        ttk.Label(settings, text="タイムアウト（秒）").grid(row=9, column=0, sticky="w", pady=(6, 0))
        self.timeout_var = tk.StringVar(value="600")
        timeout_entry = ttk.Entry(settings, textvariable=self.timeout_var, width=12)
        timeout_entry.grid(row=9, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.input_controls.append(timeout_entry)
        settings.columnconfigure(1, weight=1)

        routing = ttk.LabelFrame(outer, text="3. モデルの使い方", padding=8)
        routing.pack(fill="x", pady=(8, 0))
        self.routing_var = tk.StringVar(value="smart")
        smart_radio = ttk.Radiobutton(
            routing,
            text="自動で部署ごとに切替（おすすめ）",
            value="smart",
            variable=self.routing_var,
            command=self.on_routing_changed,
        )
        smart_radio.pack(side="left")
        single_radio = ttk.Radiobutton(
            routing,
            text="単一モデル",
            value="single",
            variable=self.routing_var,
            command=self.on_routing_changed,
        )
        single_radio.pack(side="left", padx=(12, 0))
        manual_radio = ttk.Radiobutton(
            routing,
            text="部署ごとに手動で選ぶ",
            value="manual",
            variable=self.routing_var,
            command=self.on_routing_changed,
        )
        manual_radio.pack(side="left", padx=(12, 0))
        self.input_controls.extend((smart_radio, single_radio, manual_radio))
        self.routing_summary = tk.StringVar()
        ttk.Label(routing, textvariable=self.routing_summary, wraplength=770).pack(
            anchor="w", pady=(6, 0)
        )

        ttk.Label(
            outer,
            text="自動モードでは、導入済みのモデルの中から部署ごとに得意なものを選びます（PM・設計は日本語と推論が得意な汎用モデル、実装・QAはコード特化モデルを優先）。"
            "理想のモデルが無くても、今あるモデルの中で最適な割り当てで続行します。"
            "モデルは各工程の後にアンロードされるため、切替時に少しロード待ちがあります（16GBメモリを守るためです）。",
            wraplength=780,
        ).pack(anchor="w", pady=(8, 8))

        action_row = ttk.Frame(outer)
        action_row.pack(fill="x")
        self.start_button = ttk.Button(action_row, text="アプリを作成", command=self.start_generation)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(action_row, text="中止", command=self.stop_generation, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.preview_button = ttk.Button(action_row, text="生成アプリを開く", command=self.preview_output, state="disabled")
        self.preview_button.pack(side="left", padx=(8, 0))
        self.resume_button = ttk.Button(action_row, text="つづきから再開", command=self.resume_generation)
        self.resume_button.pack(side="left", padx=(8, 0))
        self.open_button = ttk.Button(action_row, text="出力フォルダーを開く", command=self.open_output, state="disabled")
        self.open_button.pack(side="left", padx=(8, 0))
        self.progress_bar = ttk.Progressbar(outer, mode="determinate", maximum=4, value=0)
        self.progress_bar.pack(fill="x", pady=(8, 0))
        self.status_var = tk.StringVar(value="準備中")
        ttk.Label(action_row, textvariable=self.status_var).pack(side="left", padx=(12, 0))

        publish_frame = ttk.LabelFrame(outer, text="GitHub Pages（公開リポジトリへHTMLをアップロード）", padding=8)
        publish_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(publish_frame, text="作成済みリポジトリURL").grid(row=0, column=0, sticky="w")
        self.repository_url_var = tk.StringVar()
        self.repository_url_entry = ttk.Entry(
            publish_frame,
            textvariable=self.repository_url_var,
            width=54,
        )
        self.repository_url_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.input_controls.append(self.repository_url_entry)
        ttk.Label(
            publish_frame,
            text="例: https://github.com/owner/my-app",
        ).grid(row=0, column=2, sticky="w")
        self.publication_reviewed_var = tk.BooleanVar(value=False)
        self.publication_review_check = ttk.Checkbutton(
            publish_frame,
            text="HTMLを確認し、公開リポジトリで誰でも見られることを理解しました。",
            variable=self.publication_reviewed_var,
            command=self.update_publish_button,
        )
        self.publication_review_check.grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))
        self.input_controls.append(self.publication_review_check)
        self.publish_button = ttk.Button(
            publish_frame,
            text="GitHubへアップロード",
            command=self.start_publication,
            state="disabled",
        )
        self.publish_button.grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.publication_status_var = tk.StringVar(
            value="このアプリはリポジトリを作成せず、Pagesも有効化しません。"
        )
        ttk.Label(
            publish_frame,
            textvariable=self.publication_status_var,
            wraplength=700,
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(5, 0))
        publish_frame.columnconfigure(1, weight=1)
        self.repository_url_var.trace_add("write", lambda *_: self.update_publish_button())

        progress_frame = ttk.LabelFrame(outer, text="進捗", padding=10)
        progress_frame.pack(fill="x", pady=(10, 0))
        self.phase_vars = {}
        for phase in PHASES:
            variable = tk.StringVar(value=f"○ {phase}")
            self.phase_vars[phase] = variable
            ttk.Label(progress_frame, textvariable=variable).pack(anchor="w")

        self.log_text = scrolledtext.ScrolledText(outer, height=9, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=(10, 0))
        self.model_var.trace_add("write", lambda *_: self.refresh_routing_summary())
        self.separate_var.trace_add("write", lambda *_: self.refresh_routing_summary())
        for variable in self.stage_model_vars.values():
            variable.trace_add("write", lambda *_: self.refresh_routing_summary())
        self.refresh_routing_summary()
        self.on_routing_changed()
        self.root.after(150, self.process_events)
        self.root.after(250, self.check_connection)

    def toggle_stage_models(self):
        state = "normal" if self.routing_var.get() == "manual" and not self.busy else "disabled"
        for entry in self.stage_model_entries.values():
            entry.configure(state=state)

    def on_routing_changed(self):
        is_auto = self.routing_var.get() in {"smart", "auto"}
        self.model_entry.configure(state="disabled" if is_auto or self.busy else "normal")
        for entry in self.stage_model_entries.values():
            enabled = self.routing_var.get() == "manual" and not self.busy
            entry.configure(state="normal" if enabled else "disabled")
        self.refresh_routing_summary()

    def refresh_routing_summary(self):
        stage_models = {key: value.get() for key, value in self.stage_model_vars.items()}
        try:
            models = models_for_mode(
                self.routing_var.get(),
                self.model_var.get(),
                stage_models,
                installed_details=self.installed_details,
            )
        except Exception:
            models = {key: "（未決定）" for key in STAGE_KEYS}
        self.routing_summary.set(
            f"開始前の割当 — PM・要件: {models['requirements']} / 設計: {models['design']} / "
            f"実装: {models['implementation']} / QA: {models['qa']}"
        )
        if self.installed_models is not None:
            if self.routing_var.get() == "smart":
                status = "自動モード: 導入済みモデルから部署ごとに割り当てます。"
            else:
                absent = missing_models(models, self.installed_models)
                status = ("必要モデル未導入: " + ", ".join(absent)) if absent else "選択中のモデルは利用可能です。"
            self.model_status.configure(text=f"Ollama接続済み。{status}")

    def append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_busy(self, busy, status=None):
        self.busy = busy
        self.start_button.configure(state="disabled" if busy else "normal")
        self.check_button.configure(state="disabled" if busy else "normal")
        if hasattr(self, "stop_button"):
            self.stop_button.configure(state="normal" if busy else "disabled")
        if hasattr(self, "resume_button"):
            self.resume_button.configure(state="disabled" if busy else "normal")
        self.publish_button.configure(state="disabled")
        for control in self.input_controls:
            if isinstance(control, scrolledtext.ScrolledText):
                control.configure(state="disabled" if busy else "normal")
            elif control in self.stage_model_entries.values():
                control.configure(state="disabled")
            elif isinstance(control, ttk.Combobox):
                control.configure(state="disabled" if busy else "readonly")
            elif control is self.model_entry:
                control.configure(state="disabled" if busy else "normal")
            else:
                control.configure(state="disabled" if busy else "normal")
        if status:
            self.status_var.set(status)
        if not busy:
            self.on_routing_changed()
            self.update_publish_button()

    def update_publish_button(self):
        if not hasattr(self, "publish_button"):
            return
        ready = (
            not self.busy
            and self.output_ready
            and bool(self.repository_url_var.get().strip())
            and self.publication_reviewed_var.get()
        )
        self.publish_button.configure(state="normal" if ready else "disabled")

    def check_connection(self):
        if self.busy:
            return
        endpoint = self.endpoint_var.get().strip()
        self.set_busy(True, "Ollamaとモデルを確認中...")
        self.model_status.configure(text="ローカルOllamaに接続しています...")
        threading.Thread(
            target=self._check_worker,
            args=(endpoint,),
            daemon=True,
        ).start()

    def _check_worker(self, endpoint):
        try:
            details = fetch_ollama_model_details(endpoint)
            self.events.put(("models", details))
        except Exception as exc:
            self.events.put(("check_error", str(exc)))

    def _build_args_and_required(self):
        stage_values = (
            {key: value.get() for key, value in self.stage_model_vars.items()}
            if self.routing_var.get() == "manual" else {}
        )
        args = build_run_args(
            self.endpoint_var.get().strip(),
            self.model_var.get(),
            stage_values,
            self.context_var.get(),
            self.routing_var.get(),
            self.timeout_var.get(),
        )
        required = models_for_mode(
            args.routing_mode,
            args.model,
            stage_values,
            installed_details=self.installed_details,
        )
        return args, required

    def start_generation(self):
        try:
            request = validate_request_text(self.request_text.get("1.0", "end"))
        except ValueError:
            messagebox.showwarning(
                "アプリ案を入力してください",
                "作りたいアプリの説明が空です。\n入力欄にアプリ案を書いてから「アプリを作成」を押してください。",
                parent=self.root,
            )
            self.request_text.focus_set()
            return
        try:
            validate_local_endpoint(self.endpoint_var.get())
            args, required = self._build_args_and_required()
            if self.routing_var.get() == "manual" and any(
                not value.get().strip() for value in self.stage_model_vars.values()
            ):
                messagebox.showinfo(
                    "部署別モデル設定",
                    "空欄の部署には単一モデル欄のモデルを使います。",
                    parent=self.root,
                )
        except (ValueError, SystemExit) as exc:
            if isinstance(exc, SystemExit):
                return
            messagebox.showerror("設定を確認してください", str(exc), parent=self.root)
            return
        self._begin_run(request, required, args, resume_dir=None)

    def resume_generation(self):
        if self.busy:
            return
        runs_root = Path(__file__).resolve().parent / "runs"
        chosen = filedialog.askdirectory(
            title="再開する実行フォルダーを選んでください（run.json のあるフォルダー）",
            initialdir=str(runs_root) if runs_root.exists() else str(Path(__file__).resolve().parent),
            parent=self.root,
        )
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
        self.append_log(f"再開します: {run_dir}")
        self._begin_run(request, required, args, resume_dir=run_dir)

    def _begin_run(self, request, required, args, resume_dir):
        self.cancel_event = threading.Event()
        self.set_busy(True, "接続とモデルを事前確認中...")
        self.output_ready = False
        self.publication_reviewed_var.set(False)
        self.publication_status_var.set("生成が完了し、HTMLを確認するとアップロードできます。")
        self.active_models = required
        self.model_status.configure(text="実行に必要なモデルを確認しています...")
        self.append_log("入力を受け付けました。Ollama接続と必要モデルを確認します。")
        self.progress_bar.configure(value=0)
        stage_keys = STAGE_KEYS
        for position, phase in enumerate(PHASES):
            model_name = required.get(stage_keys[position], "")
            self.phase_vars[phase].set(f"○ {phase}" + (f" — {model_name}" if model_name and "自動決定" not in model_name else ""))
        threading.Thread(
            target=self._generation_worker,
            args=(request, required, args, resume_dir),
            daemon=True,
        ).start()

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
                request,
                args,
                run_dir,
                progress_callback=lambda line: self.events.put(("progress", line)),
                token_callback=lambda stage, chars: self.events.put(("tokens", (stage, chars))),
                cancel_event=self.cancel_event,
                resume=resume_dir is not None,
            )
            self.events.put(("finished", (exit_code, str(run_dir))))
        except Exception as exc:
            self.events.put(("worker_error", str(exc)))

    def stop_generation(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.status_var.set("中止しています…（現在の文字の区切りで止まります）")
            self.append_log("中止ボタンが押されました。完了済みの工程は保存されています。")

    def preview_output(self):
        if not self.last_output:
            return
        html_path = Path(self.last_output) / "index.html"
        if not html_path.is_file():
            messagebox.showinfo("生成アプリ", "まだ index.html がありません。", parent=self.root)
            return
        webbrowser.open(html_path.resolve().as_uri())

    def start_publication(self):
        if self.busy:
            return
        if not self.output_ready or not self.last_output:
            messagebox.showwarning(
                "生成HTMLがありません",
                "先にアプリを生成し、出力された index.html を確認してください。",
                parent=self.root,
            )
            return
        if not self.publication_reviewed_var.get():
            messagebox.showwarning(
                "公開前の確認が必要です",
                "HTMLの内容と、公開リポジトリでは誰でも閲覧できることを確認してください。",
                parent=self.root,
            )
            return
        repository_url = self.repository_url_var.get().strip()
        try:
            github_publish.parse_repository_url(repository_url)
        except ValueError as exc:
            messagebox.showerror("リポジトリURLを確認してください", str(exc), parent=self.root)
            return

        output = self.last_output
        self.set_busy(True, "GitHubの認証とリポジトリを確認中...")
        self.publication_status_var.set("GitHubログイン、公開設定、書き込み権限、Git接続を確認しています...")
        self.append_log(f"GitHub公開の事前確認を開始: {repository_url}")
        threading.Thread(
            target=self._publication_preflight_worker,
            args=(repository_url, output),
            daemon=True,
        ).start()

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
            if target.empty
            else f"既存の {target.default_branch} ブランチに index.html を追加します。"
        )
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
            "GitHubへアップロードしますか？"
        )
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
                output,
                repository_url,
                target,
                progress_callback=lambda line: self.events.put(("publish_progress", line)),
            )
            self.events.put(("publish_finished", result))
        except Exception as exc:
            self.events.put(("publish_error", exc))

    def process_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "models":
                    if data and isinstance(data[0], dict):
                        self.installed_details = data
                        self.installed_models = [item["name"] for item in data]
                    else:
                        self.installed_models = list(data)
                        self.installed_details = [{"name": name, "size": 0, "details": {}} for name in self.installed_models]
                    names = self.installed_models
                    for combo in self.stage_model_entries.values():
                        combo.configure(values=names)
                    self.refresh_routing_summary()
                    if self.routing_var.get() == "smart":
                        absent = [] if names else ["（モデルが1つも導入されていません）"]
                    else:
                        absent = missing_models(
                            models_for_mode(
                                self.routing_var.get(),
                                self.model_var.get(),
                                {key: value.get() for key, value in self.stage_model_vars.items()},
                                installed_details=self.installed_details,
                            ),
                            names,
                        )
                    self.set_busy(False, "準備完了" if not absent else "モデルを導入してください")
                    self.append_log("Ollama接続を確認しました。")
                    if absent:
                        self.append_log(
                            error_guidance(
                                "モデル確認",
                                "指定モデルが未導入です。",
                                absent,
                            )
                        )
                elif kind == "check_error":
                    self.model_status.configure(text="Ollamaを確認できませんでした。")
                    self.set_busy(False, "接続を確認してください")
                    self.append_log(error_guidance("接続確認", data))
                elif kind == "missing_models":
                    absent, installed = data
                    self.installed_models = installed
                    advice = error_guidance("モデル確認", "指定モデルが未導入です。", absent)
                    self.model_status.configure(
                        text="未導入モデル: " + ", ".join(absent)
                    )
                    self.append_log(advice)
                    messagebox.showerror("モデルが見つかりません", advice, parent=self.root)
                    self.set_busy(False, "モデルを導入後に再実行してください")
                elif kind == "tokens":
                    stage, chars = data
                    label = {"requirements": "PM・要件定義", "design": "デザイン設計", "implementation": "実装", "qa": "QA"}.get(stage, stage)
                    if not (self.cancel_event is not None and self.cancel_event.is_set()):
                        self.status_var.set(f"{label}を生成中… {chars}文字")
                elif kind == "output":
                    self.last_output = Path(data)
                    self.open_button.configure(state="normal")
                    self.status_var.set("生成中...")
                elif kind == "progress":
                    self.update_progress(data)
                    self.append_log(data)
                elif kind == "finished":
                    code, output = data
                    self.last_output = Path(output)
                    self.open_button.configure(state="normal")
                    self.preview_button.configure(state="normal" if (self.last_output / "index.html").is_file() else "disabled")
                    self.output_ready = code in (0, 2) and (self.last_output / "index.html").is_file()
                    self.set_busy(False)
                    if code == 3:
                        self.status_var.set("中止しました（途中成果は保存済み・再開できます）")
                        self.progress_bar.configure(value=0)
                        self.append_log(f"中止しました。保存先: {output}。「つづきから再開」で続きを実行できます。")
                    elif code == 0:
                        self.status_var.set("完了（AI QA承認）")
                        self.append_log(f"完了しました。保存先: {output}")
                    elif code == 2:
                        self.status_var.set("未承認（HTMLは保存済み）")
                        self.append_log(f"QA未承認です。保存済みHTMLを確認してください: {output}")
                        messagebox.showwarning(
                            "QA未承認",
                            f"処理は完了しましたが、QAの承認は得られていません。\n"
                            f"HTMLとQA結果を確認してください。\n\n保存先:\n{output}",
                            parent=self.root,
                        )
                    else:
                        self.show_run_error(Path(output))
                    self.update_publish_button()
                elif kind == "publish_preflight_ready":
                    target, output = data
                    if self._confirm_publication(target, output):
                        self.publication_status_var.set("確認済み。index.htmlをアップロードしています...")
                        self.append_log(f"公開先を確認しました: {target.repo_url}")
                        threading.Thread(
                            target=self._publication_worker,
                            args=(self.repository_url_var.get().strip(), output, target),
                            daemon=True,
                        ).start()
                    else:
                        self.set_busy(False, "公開をキャンセルしました")
                        self.publication_status_var.set("アップロードはキャンセルされました。")
                elif kind == "publish_progress":
                    self.publication_status_var.set(data)
                    self.append_log(data)
                elif kind == "publish_finished":
                    self.output_ready = False
                    self.set_busy(False)
                    self.publication_status_var.set(
                        f"HTMLをアップロードしました。Pagesは未有効: {data.pages_url}"
                    )
                    self.status_var.set("アップロード完了・Pages未有効")
                    self.append_log(data.details)
                    self.append_log(f"GitHub Pagesの設定後に利用するURL: {data.pages_url}")
                    messagebox.showinfo(
                        "HTMLをアップロードしました",
                        f"{data.details}\n\n"
                        f"リポジトリ:\n{data.repo_url}\n\n"
                        f"Pages有効化後のURL:\n{data.pages_url}\n\n"
                        "URLはPagesを設定してデプロイが完了するまで利用できません。",
                        parent=self.root,
                    )
                elif kind == "publish_error":
                    self.set_busy(False, "GitHubアップロードを完了できませんでした")
                    advice = publication_guidance(data)
                    self.publication_status_var.set("アップロードを中止しました。既存内容は上書きしていません。")
                    self.append_log(advice)
                    messagebox.showerror("GitHub公開を完了できませんでした", advice, parent=self.root)
                elif kind == "worker_error":
                    self.set_busy(False, "処理を開始できませんでした")
                    self.append_log(error_guidance("事前確認・実行", data))
                    messagebox.showerror(
                        "実行できませんでした",
                        error_guidance("事前確認・実行", data),
                        parent=self.root,
                    )
        except queue.Empty:
            pass
        self.root.after(150, self.process_events)

    def update_progress(self, line):
        index = progress_phase(line)
        if index is not None:
            phase_model_key = ("requirements", "design", "implementation", "qa")[index]
            phase_model = self.active_models.get(phase_model_key, "")
            for position, phase in enumerate(PHASES):
                if position < index:
                    marker = "✓"
                elif position == index:
                    marker = "▶"
                else:
                    marker = "○"
                model_text = f" — {phase_model}" if position == index and phase_model else ""
                self.phase_vars[phase].set(f"{marker} {phase}{model_text}")
            self.status_var.set(
                f"{PHASES[index]}: {phase_model} をロード/実行中..."
                if phase_model
                else f"{PHASES[index]}を処理中..."
            )
        if index is not None:
            self.progress_bar.configure(value=index + 1)
        if "[DONE]" in line:
            self.progress_bar.configure(value=4)
            for phase in PHASES:
                self.phase_vars[phase].set(f"✓ {phase}")

    def show_run_error(self, run_dir):
        stage = "実行"
        error = "詳しいエラー情報を確認できませんでした。"
        try:
            manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            stage = manifest.get("failed_stage", stage)
            error = manifest.get("error", error)
        except (OSError, json.JSONDecodeError):
            pass
        self.status_var.set("途中で停止しました")
        advice = error_guidance(stage, error)
        self.append_log(advice)
        messagebox.showerror(
            "途中で停止しました",
            advice + f"\n\n途中成果物・ログの保存先:\n{run_dir}",
            parent=self.root,
        )

    def open_output(self):
        if not self.last_output or not self.last_output.exists():
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
            messagebox.showerror(
                "フォルダーを開けません",
                error_guidance("出力フォルダーを開く", exc),
                parent=self.root,
            )

    def on_close(self):
        if self.busy:
            if self.cancel_event is not None and not self.cancel_event.is_set():
                self.cancel_event.set()
                messagebox.showinfo(
                    "中止して終了します",
                    "生成を中止します。完了済みの工程は保存されており、次回「つづきから再開」できます。",
                    parent=self.root,
                )
            self.root.destroy()
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    AppBuilderWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
