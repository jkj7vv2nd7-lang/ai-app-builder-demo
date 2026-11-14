import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))
import app_builder  # noqa: E402
import app_gui  # noqa: E402


class HtmlExtractionTests(unittest.TestCase):
    def test_extracts_html_from_fenced_response_with_surrounding_text(self):
        response = "完成しました。\n```html\n<!doctype html><html></html>\n```\n以上です。"
        self.assertEqual(
            app_builder.extract_html(response),
            "<!doctype html><html></html>",
        )

    def test_does_not_treat_explanation_as_html(self):
        self.assertIsNone(app_builder.extract_html("HTMLを作成できませんでした。"))

    def test_checks_required_document_elements(self):
        checks = app_builder.validate_html(
            "<!doctype html><html><head><title>Test</title></head>"
            "<body><p>App</p></body></html>"
        )
        self.assertTrue(checks["passed"], checks["errors"])
        self.assertEqual(checks["external_dependencies"], [])
        self.assertEqual(checks["network_api_usage"], [])

    def test_reports_external_resources_and_network_api_usage(self):
        html = (
            '<html><head><script src="https://cdn.example/app.js"></script></head>'
            "<body><main>App</main><script>fetch('/data');</script></body></html>"
        )
        checks = app_builder.validate_html(html)
        self.assertTrue(checks["passed"], checks["errors"])
        self.assertTrue(any("cdn.example" in item for item in checks["external_dependencies"]))
        self.assertIn("fetch(", checks["network_api_usage"])

    def test_reports_missing_body(self):
        checks = app_builder.validate_html("<html><head></head></html>")
        self.assertFalse(checks["passed"])
        self.assertIn("missing <body> element", checks["errors"])

    def test_rejects_empty_body(self):
        checks = app_builder.validate_html("<html><head></head><body></body></html>")
        self.assertFalse(checks["passed"])
        self.assertIn("<body> has no content", checks["errors"])

    def test_qa_requires_explicit_verdict(self):
        self.assertEqual(
            app_builder.extract_qa_status("QA_STATUS: NEEDS_REVISION\nFix a label."),
            "NEEDS_REVISION",
        )
        self.assertIsNone(app_builder.extract_qa_status("Looks good to me."))


class OutputAndApiTests(unittest.TestCase):
    def test_output_path_must_stay_inside_tool_directory(self):
        with self.assertRaises(ValueError):
            app_builder.resolve_output_root(str(TOOL_DIR.parent / "outside"))

    @patch.object(app_builder.urllib.request, "urlopen", side_effect=URLError("offline"))
    def test_connection_errors_are_not_returned_as_empty_success(self, _urlopen):
        args = type(
            "Args",
            (),
            {
                "model": "test",
                "endpoint": "http://localhost:11434/api/generate",
                "temperature": 0.3,
                "num_predict": 128,
                "num_ctx": 2048,
                "timeout": 1,
            },
        )()
        with self.assertRaises(app_builder.StageError) as context:
            app_builder.call_ollama("prompt", "system", args, "design")
        self.assertEqual(context.exception.stage, "design")
        self.assertIn("could not connect", str(context.exception))

    def test_stage_model_falls_back_to_common_model(self):
        args = SimpleNamespace(
            model="common-model",
            planning_model="planning-model",
            design_model="design-model",
            implementation_model="implementation-model",
            qa_model=None,
        )
        self.assertEqual(app_builder.selected_model(args, "requirements"), "planning-model")
        self.assertEqual(app_builder.selected_model(args, "design"), "design-model")
        self.assertEqual(app_builder.selected_model(args, "implementation"), "implementation-model")
        self.assertEqual(app_builder.selected_model(args, "qa"), "common-model")

    @patch.object(app_builder.urllib.request, "urlopen")
    def test_api_uses_selected_stage_model_and_unloads_after_each_call(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = b'{"response":"ok"}'
        args = SimpleNamespace(
            model="common-model",
            planning_model=None,
            implementation_model="coder-model",
            qa_model=None,
            endpoint="http://localhost:11434/api/generate",
            temperature=0.3,
            num_predict=128,
            num_ctx=2048,
            timeout=10,
        )
        self.assertEqual(app_builder.call_ollama("prompt", "system", args, "implementation"), "ok")
        payload = app_builder.json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["model"], "coder-model")
        self.assertEqual(payload["keep_alive"], 0)

    def test_defaults_target_code_model_and_cpu_context(self):
        args = app_builder.build_parser().parse_args(["sample app"])
        self.assertEqual(args.model, "qwen2.5-coder:7b")
        self.assertEqual(args.num_ctx, 4096)
        self.assertFalse(args.browser_check)
        self.assertEqual(args.routing_mode, "smart")
        self.assertTrue(args.stream)
        self.assertEqual(args.max_impl_repairs, 2)

    def test_auto_routing_uses_fixed_role_models(self):
        args = app_builder.build_parser().parse_args(["--routing-mode", "auto", "sample app"])
        models = app_builder.models_for_run(args)
        self.assertEqual(models["requirements"], "qwen3.5:9b")
        self.assertEqual(models["design"], "qwen3.5:9b")
        self.assertEqual(models["implementation"], "qwen3.5:9b")
        self.assertEqual(models["qa"], "qwen3.5:9b")

    def test_smart_routing_assigns_best_installed_model_per_department(self):
        details = [
            {"name": "qwen3:8b", "size": 5_200_000_000, "details": {"parameter_size": "8B"}},
            {"name": "qwen2.5-coder:7b", "size": 4_700_000_000, "details": {"parameter_size": "7B"}},
            {"name": "llama3.1:8b", "size": 4_900_000_000, "details": {"parameter_size": "8B"}},
        ]
        assignments, reasons, suggestions = app_builder.smart_route_models(details)
        self.assertEqual(assignments["requirements"], "qwen3:8b")
        self.assertEqual(assignments["design"], "qwen3:8b")
        self.assertEqual(assignments["implementation"], "qwen2.5-coder:7b")
        self.assertEqual(assignments["qa"], "qwen2.5-coder:7b")
        self.assertEqual(suggestions, ["ollama pull qwen3.5:9b"])
        self.assertIn("コード特化", reasons["implementation"])

    def test_smart_routing_with_one_model_uses_it_everywhere_and_suggests_ideals(self):
        details = [{"name": "qwen2.5-coder:3b", "size": 1_900_000_000, "details": {"parameter_size": "3B"}}]
        assignments, _reasons, suggestions = app_builder.smart_route_models(details)
        self.assertTrue(all(name == "qwen2.5-coder:3b" for name in assignments.values()))
        self.assertIn("ollama pull qwen3.5:9b", suggestions)

    def test_smart_routing_requires_at_least_one_model(self):
        with self.assertRaises(app_builder.StageError):
            app_builder.smart_route_models([])

    def test_generation_options_give_implementation_a_bigger_budget(self):
        args = app_builder.build_parser().parse_args(["sample app"])
        impl = app_builder.generation_options(args, "implementation")
        qa = app_builder.generation_options(args, "qa")
        self.assertGreaterEqual(impl["num_predict"], 8192)
        self.assertGreaterEqual(impl["num_ctx"], 8192)
        self.assertLessEqual(qa["num_predict"], 2048)

    def test_auto_routing_requires_explicit_installation_instead_of_fallback(self):
        required = app_gui.models_for_mode("auto", "single-model")
        missing = app_gui.missing_models(required, ["single-model"])
        self.assertEqual(missing, ["qwen3.5:9b"])

    def test_smart_mode_in_gui_uses_installed_details(self):
        details = [
            {"name": "qwen3:8b", "size": 5_200_000_000, "details": {"parameter_size": "8B"}},
            {"name": "qwen2.5-coder:7b", "size": 4_700_000_000, "details": {"parameter_size": "7B"}},
        ]
        models = app_gui.models_for_mode("smart", "ignored", installed_details=details)
        self.assertEqual(models["requirements"], "qwen3:8b")
        self.assertEqual(models["implementation"], "qwen2.5-coder:7b")

    def test_auto_routing_cli_stops_with_pull_advice_when_model_is_missing(self):
        error_stream = io.StringIO()
        with patch.object(app_builder, "fetch_ollama_model_names", return_value=["qwen2.5-coder:7b"]), \
             patch.object(app_builder, "create_run_directory") as create_run, \
             contextlib.redirect_stderr(error_stream):
            code = app_builder.main(["--routing-mode", "auto", "sample app"])
        self.assertEqual(code, 1)
        self.assertIn("ollama pull qwen3.5:9b", error_stream.getvalue())
        self.assertIn("smart", error_stream.getvalue())
        create_run.assert_not_called()

    def test_smart_cli_resolves_models_from_installed_details(self):
        details = [{"name": "qwen2.5-coder:7b", "size": 4_700_000_000, "details": {"parameter_size": "7B"}}]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(app_builder, "fetch_ollama_model_details", return_value=details), \
                 patch.object(app_builder, "create_run_directory", return_value=run_dir), \
                 patch.object(app_builder, "run_pipeline", return_value=0) as run_pipeline:
                code = app_builder.main(["sample app"])
        self.assertEqual(code, 0)
        used_args = run_pipeline.call_args.args[1]
        self.assertEqual(used_args.resolved_models["implementation"], "qwen2.5-coder:7b")

    def test_auto_routing_cli_runs_only_after_required_model_is_present(self):
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(
                app_builder,
                "fetch_ollama_model_names",
                return_value=["qwen3.5:9b"],
            ), patch.object(app_builder, "create_run_directory", return_value=run_dir), patch.object(
                app_builder, "run_pipeline", return_value=0
            ) as run_pipeline:
                code = app_builder.main(["--routing-mode", "auto", "sample app"])
        self.assertEqual(code, 0)
        self.assertEqual(run_pipeline.call_args.args[1].routing_mode, "auto")
        self.assertEqual(run_pipeline.call_args.args[0], "sample app")

    @patch.object(app_builder.urllib.request, "urlopen")
    def test_local_model_check_reads_ollama_tags(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"models": [{"name": "qwen3:8b"}, {"name": "qwen2.5-coder:7b"}]}
        ).encode("utf-8")
        installed = app_gui.fetch_ollama_models("http://localhost:11434/api/generate")
        self.assertIn("qwen3:8b", installed)
        self.assertEqual(
            urlopen.call_args.args[0].full_url,
            "http://localhost:11434/api/tags",
        )

    def test_single_model_mode_remains_explicit_and_stage_models_are_optional(self):
        args = app_gui.build_run_args(
            "http://localhost:11434/api/generate",
            "one-model",
            {"requirements": "", "design": "", "implementation": "", "qa": ""},
            4096,
            routing_mode="single",
            timeout_seconds=1200,
        )
        self.assertEqual(app_builder.models_for_run(args), {
            "requirements": "one-model",
            "design": "one-model",
            "implementation": "one-model",
            "qa": "one-model",
        })
        self.assertEqual(args.timeout, 1200)

    def test_manual_mode_assigns_each_department_separately(self):
        args = app_gui.build_run_args(
            "http://localhost:11434/api/generate",
            "one-model",
            {"requirements": "pm-model", "design": "design-model", "implementation": "coder-model", "qa": "qa-model"},
            4096,
            routing_mode="manual",
        )
        self.assertEqual(app_builder.models_for_run(args), {
            "requirements": "pm-model",
            "design": "design-model",
            "implementation": "coder-model",
            "qa": "qa-model",
        })

    def test_auto_routing_disallows_hidden_manual_overrides(self):
        with self.assertRaises(ValueError):
            app_gui.build_run_args(
                "http://localhost:11434/api/generate",
                "one-model",
                {"requirements": "planner"},
                4096,
                routing_mode="smart",
            )

    def test_gui_reports_invalid_timeout_as_a_form_error(self):
        with self.assertRaisesRegex(ValueError, "タイムアウト"):
            app_gui.build_run_args(
                "http://localhost:11434/api/generate",
                "one-model",
                None,
                4096,
                timeout_seconds="not a number",
            )

    def test_gui_refuses_non_local_model_endpoints(self):
        with self.assertRaises(ValueError):
            app_gui.validate_local_endpoint("https://example.com/api/generate")

    def test_cli_endpoint_is_local_only(self):
        parser = app_builder.build_parser()
        args = parser.parse_args(["--endpoint", "https://example.com/api/generate", "app"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                app_builder.validate_args(parser, args)

    def test_error_guidance_suggests_memory_and_connection_recovery(self):
        memory_help = app_gui.error_guidance("実装", "out of memory")
        self.assertIn("2048", memory_help)
        self.assertIn("小さい", memory_help)
        connection_help = app_gui.error_guidance("接続確認", "Could not connect")
        self.assertIn("Ollamaアプリを起動", connection_help)
        self.assertIn("localhost", connection_help)
        model_help = app_gui.error_guidance("モデル確認", "指定モデルが未導入です。", ["qwen3:8b"])
        self.assertIn("ollama pull qwen3:8b", model_help)
        timeout_help = app_gui.error_guidance("QA", "request timed out")
        self.assertIn("タイムアウトを長く", timeout_help)

    def test_empty_app_idea_is_rejected_with_clear_guidance(self):
        with self.assertRaisesRegex(ValueError, "説明を入力"):
            app_gui.validate_request_text("  \n")
        self.assertEqual(app_gui.validate_request_text("  ToDo app  "), "ToDo app")

    def test_gui_progress_maps_pipeline_messages_to_phases(self):
        self.assertEqual(app_gui.progress_phase("[2/4] デザイン設計を作成中..."), 1)
        self.assertEqual(app_gui.progress_phase("[QA] レビュー 1/3"), 3)
        self.assertIsNone(app_gui.progress_phase("[INFO] 出力先"))

    def test_windows_launcher_resolves_gui_relative_to_its_own_folder(self):
        launcher = (TOOL_DIR / "launch_gui.bat").read_text(encoding="utf-8")
        self.assertIn('"%~dp0launch_gui.pyw"', launcher)
        self.assertIn("where pyw", launcher)
        self.assertIn("where pythonw", launcher)

    @patch.dict(sys.modules, {"playwright": None})
    def test_optional_browser_check_skips_when_playwright_is_missing(self):
        result = app_builder.run_browser_check("<html></html>")
        self.assertEqual(result["status"], "skipped")
        self.assertIn("未導入", result["reason"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model="test-model",
            planning_model=None,
            implementation_model=None,
            endpoint="http://localhost:11434/api/generate",
            temperature=0.3,
            num_predict=256,
            num_ctx=2048,
            timeout=10,
            max_qa_revisions=1,
            browser_check=False,
        )
        self.html = (
            "<!doctype html><html><head><title>Test</title></head>"
            "<body><main>Working app</main></body></html>"
        )

    def run_pipeline_with_responses(self, qa_response):
        responses = ["Requirements", "Design", self.html, qa_response]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = app_builder.run_pipeline("Test app", self.args, run_dir)
            manifest = app_builder.json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            saved_html = (run_dir / "index.html").read_text(encoding="utf-8")
            return exit_code, manifest, saved_html

    def test_explicit_approval_completes_pipeline(self):
        qa_response = f"QA_STATUS: APPROVED\n\n```html\n{self.html}\n```"
        exit_code, manifest, saved_html = self.run_pipeline_with_responses(qa_response)
        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(manifest["qa_approved"])
        self.assertEqual(manifest["models"]["requirements"], "test-model")
        self.assertEqual(manifest["browser_check"]["status"], "skipped")
        self.assertIn("<main>", saved_html)

    def test_qa_review_only_approval_completes_pipeline(self):
        # v2: QA returns a verdict only; approval no longer needs HTML back.
        exit_code, manifest, saved_html = self.run_pipeline_with_responses(
            "QA_STATUS: APPROVED\nDEFECTS:\n- なし"
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(manifest["qa_approved"])
        self.assertIn("<main>", saved_html)

    def test_qa_revision_is_repaired_by_implementation_then_approved(self):
        responses = [
            "Requirements", "Design", self.html,
            "QA_STATUS: NEEDS_REVISION\nDEFECTS:\n- 保存ボタンが動かない",
            self.html.replace("Working app", "Working app fixed"),
            "QA_STATUS: APPROVED\nDEFECTS:\n- なし",
        ]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = app_builder.run_pipeline("Test app", self.args, run_dir)
            saved_html = (run_dir / "index.html").read_text(encoding="utf-8")
        self.assertEqual(exit_code, 0)
        self.assertIn("fixed", saved_html)

    def test_broken_first_implementation_is_repaired_instead_of_aborting(self):
        broken = "<html><head></head><body></body></html>"
        responses = [
            "Requirements", "Design", broken, self.html,
            "QA_STATUS: APPROVED\nDEFECTS:\n- なし",
        ]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = app_builder.run_pipeline("Test app", self.args, run_dir)
            saved_html = (run_dir / "index.html").read_text(encoding="utf-8")
        self.assertEqual(exit_code, 0)
        self.assertIn("<main>", saved_html)

    def test_resume_skips_completed_stages(self):
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "01_requirements.md").write_text("Saved requirements\n", encoding="utf-8")
            (run_dir / "02_design.md").write_text("Saved design\n", encoding="utf-8")
            (run_dir / "run.json").write_text("{\"request\": \"Test app\"}", encoding="utf-8")
            responses = [self.html, "QA_STATUS: APPROVED\nDEFECTS:\n- なし"]
            with patch.object(app_builder, "call_ollama", side_effect=responses) as call_mock:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = app_builder.run_pipeline("Test app", self.args, run_dir, resume=True)
            self.assertEqual(exit_code, 0)
            self.assertEqual(call_mock.call_count, 2)


class PromptRuleTests(unittest.TestCase):
    def test_qa_prompt_requires_verifiable_defects_only(self):
        self.assertIn("コードを見て確認できる具体的な不具合だけ", app_builder.QA_SYSTEM)
        self.assertIn("NOTES", app_builder.QA_SYSTEM)
        self.assertIn("未実装」と書かないでください", app_builder.QA_SYSTEM)

    def test_requirements_prompt_scopes_to_single_file_offline(self):
        self.assertIn("対象外（将来拡張）", app_builder.REQUIREMENTS_SYSTEM)
        self.assertIn("単一HTML", app_builder.REQUIREMENTS_SYSTEM)

    def test_cli_parser_accepts_revise_options(self):
        parser = app_builder.build_parser()
        args = parser.parse_args(["--revise", "runs/run_x", "--notes", "ボタンを大きく"])
        self.assertEqual(args.revise, "runs/run_x")
        self.assertEqual(args.notes, "ボタンを大きく")


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model="test-model",
            planning_model=None,
            design_model=None,
            implementation_model=None,
            qa_model=None,
            endpoint="http://localhost:11434/api/generate",
            temperature=0.3,
            num_predict=256,
            num_ctx=2048,
            timeout=10,
            max_qa_revisions=1,
            max_impl_repairs=2,
            browser_check=False,
        )
        self.html = (
            "<!doctype html><html><head><title>Test</title></head>"
            "<body><main>Working app</main></body></html>"
        )

    def make_source(self, root):
        source = Path(root) / "source_run"
        source.mkdir()
        (source / "index.html").write_text(self.html, encoding="utf-8")
        (source / "run.json").write_text(
            app_builder.json.dumps({"request": "メモ帳アプリ", "status": "completed"}),
            encoding="utf-8")
        (source / "01_requirements.md").write_text("メモを保存できること", encoding="utf-8")
        return source

    def test_revision_produces_new_run_and_keeps_source(self):
        revised = self.html.replace("Working app", "Working app revised")
        responses = ["修正方針: ボタンを大きくする", revised,
                     "QA_STATUS: APPROVED\nDEFECTS:\n- なし"]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            root = Path(temporary_dir)
            source = self.make_source(root)
            out_root = root / "out"
            out_root.mkdir()
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    code, new_run = app_builder.run_revision(
                        source, "ボタンを大きくして", self.args, out_root)
            self.assertEqual(code, 0)
            self.assertNotEqual(new_run, source)
            self.assertIn("revised", (new_run / "index.html").read_text(encoding="utf-8"))
            self.assertIn("Working app</main>", (source / "index.html").read_text(encoding="utf-8"))
            manifest = app_builder.json.loads((new_run / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "revision")
            self.assertEqual(manifest["revision_notes"], "ボタンを大きくして")
            self.assertTrue(manifest["qa_approved"])
            self.assertIsNone(manifest["error"])

    def test_revision_unapproved_records_final_defects(self):
        responses = ["方針", self.html,
                     "QA_STATUS: NEEDS_REVISION\nDEFECTS:\n- まだ小さい",
                     self.html,
                     "QA_STATUS: NEEDS_REVISION\nDEFECTS:\n- まだ小さい"]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            root = Path(temporary_dir)
            source = self.make_source(root)
            out_root = root / "out"
            out_root.mkdir()
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    code, new_run = app_builder.run_revision(
                        source, "ボタンを大きくして", self.args, out_root)
            self.assertEqual(code, 2)
            manifest = app_builder.json.loads((new_run / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_unapproved")
            self.assertIn("まだ小さい", manifest["qa_defects_final"])

    def test_revision_requires_index_html(self):
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            root = Path(temporary_dir)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(app_builder.StageError):
                app_builder.run_revision(empty, "直して", self.args, root)



class ChangeHonestyTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model="test-model",
            planning_model=None, design_model=None,
            implementation_model=None, qa_model=None,
            endpoint="http://localhost:11434/api/generate",
            temperature=0.3, num_predict=256, num_ctx=2048, timeout=10,
            max_qa_revisions=1, max_impl_repairs=2, browser_check=False,
        )
        self.html = (
            "<!doctype html><html><head><title>Test</title></head>"
            "<body><main>Working app</main></body></html>"
        )

    def make_source(self, root):
        source = Path(root) / "source_run"
        source.mkdir()
        (source / "index.html").write_text(self.html, encoding="utf-8")
        (source / "run.json").write_text(
            app_builder.json.dumps({"request": "メモ帳アプリ", "status": "completed"}),
            encoding="utf-8")
        return source

    def test_pipeline_logs_the_request_text(self):
        lines = []
        responses = ["Requirements", "Design", self.html,
                     "QA_STATUS: APPROVED\nDEFECTS:\n- なし"]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    app_builder.run_pipeline(
                        "時計アプリを作って", self.args, run_dir,
                        progress_callback=lines.append)
        self.assertTrue(any("今回の指示文" in line and "時計アプリを作って" in line
                            for line in lines))

    def test_revision_changed_flag_false_when_identical(self):
        responses = ["方針", self.html, "QA_STATUS: APPROVED\nDEFECTS:\n- なし"]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            root = Path(temporary_dir)
            source = self.make_source(root)
            out_root = root / "out"; out_root.mkdir()
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    code, new_run = app_builder.run_revision(
                        source, "大きくして", self.args, out_root)
            manifest = app_builder.json.loads((new_run / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertIs(manifest["changed"], False)

    def test_revision_changed_flag_true_when_modified(self):
        revised = self.html.replace("Working app", "Working app v2")
        responses = ["方針", revised, "QA_STATUS: APPROVED\nDEFECTS:\n- なし"]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            root = Path(temporary_dir)
            source = self.make_source(root)
            out_root = root / "out"; out_root.mkdir()
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    code, new_run = app_builder.run_revision(
                        source, "文言を変えて", self.args, out_root)
            manifest = app_builder.json.loads((new_run / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertIs(manifest["changed"], True)



class RenderCheckTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model="test-model",
            planning_model=None, design_model=None,
            implementation_model=None, qa_model=None,
            endpoint="http://localhost:11434/api/generate",
            temperature=0.3, num_predict=256, num_ctx=2048, timeout=10,
            max_qa_revisions=0, max_impl_repairs=2, browser_check=False,
        )
        self.html = (
            "<!doctype html><html><head><title>Test</title></head>"
            "<body><main>Working app</main></body></html>"
        )

    def test_dom_stats_count_visible_content(self):
        dom = ("<html><body><h1>時計</h1><div>12:00:00</div>"
               "<script>var x = '見えない文字';</script></body></html>")
        stats = app_builder._render_dom_stats(dom)
        self.assertGreaterEqual(stats["visible_text_chars"], 9)
        self.assertGreaterEqual(stats["element_count"], 2)

    def test_render_unavailable_without_browser(self):
        with patch.object(app_builder, "find_browser", return_value=None):
            result = app_builder.check_render(self.html)
        self.assertFalse(result["available"])
        self.assertIsNone(app_builder.render_defect_text(result))

    def test_render_defect_text_for_empty_render(self):
        result = {"available": True, "ok": False,
                  "visible_text_chars": 0, "element_count": 1}
        self.assertIn("画面表示確認", app_builder.render_defect_text(result))

    def test_pipeline_render_failure_blocks_approval(self):
        bad_render = {"available": True, "ok": False, "reason": "test",
                      "visible_text_chars": 0, "element_count": 1,
                      "media_count": 0, "screenshot": None}
        responses = ["Requirements", "Design", self.html,
                     "QA_STATUS: APPROVED\nDEFECTS:\n- なし"]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(app_builder, "call_ollama", side_effect=responses), \
                    patch.object(app_builder, "check_render", return_value=dict(bad_render)):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = app_builder.run_pipeline("時計アプリ", self.args, run_dir)
            manifest = app_builder.json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            guide_exists = (run_dir / "はじめにお読みください.txt").is_file()
        self.assertEqual(code, 2)
        self.assertTrue(any("画面表示確認" in d for d in manifest["qa_defects_final"]))
        self.assertFalse(manifest["render_check"]["ok"])
        self.assertTrue(guide_exists)

    def test_pipeline_render_ok_allows_approval(self):
        good_render = {"available": True, "ok": True, "reason": "test",
                       "visible_text_chars": 20, "element_count": 5,
                       "media_count": 0, "screenshot": None}
        responses = ["Requirements", "Design", self.html,
                     "QA_STATUS: APPROVED\nDEFECTS:\n- なし"]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(app_builder, "call_ollama", side_effect=responses), \
                    patch.object(app_builder, "check_render", return_value=dict(good_render)):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = app_builder.run_pipeline("時計アプリ", self.args, run_dir)
        self.assertEqual(code, 0)



class ValidationRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model="test-model",
            planning_model=None, design_model=None,
            implementation_model=None, qa_model=None,
            endpoint="http://localhost:11434/api/generate",
            temperature=0.3, num_predict=256, num_ctx=2048, timeout=10,
            max_qa_revisions=0, max_impl_repairs=2, browser_check=False,
        )
        self.html = (
            "<!doctype html><html><head><title>Test</title></head>"
            "<body><main>Working app</main>"
            "<script>var ok = 1;</script></body></html>"
        )

    def test_node_timeout_is_not_a_validation_error(self):
        import subprocess as sp
        with patch.object(app_builder.subprocess, "run",
                          side_effect=sp.TimeoutExpired(cmd="node", timeout=60)):
            checks = app_builder.validate_html(self.html)
        self.assertTrue(checks["passed"])
        self.assertEqual(checks["javascript"]["status"], "unavailable")

    def test_identical_problems_abort_repairs_early(self):
        broken = (
            "<!doctype html><html><head><title>Test</title></head>"
            "<body><main>Broken</main>"
            "<script>var x = ;</script></body></html>"
        )
        lines = []
        responses = ["Requirements", "Design", broken, broken, broken, broken]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            run_dir = Path(temporary_dir)
            with patch.object(app_builder, "call_ollama", side_effect=responses) as call_mock:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = app_builder.run_pipeline(
                        "壊れるアプリ", self.args, run_dir,
                        progress_callback=lines.append)
        self.assertEqual(code, 1)
        # PM + design + implementation attempt 0 + one repair, then abort.
        self.assertEqual(call_mock.call_count, 4)
        self.assertTrue(any("打ち切ります" in line for line in lines))



class ExternalRevisionTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model="test-model",
            planning_model=None, design_model=None,
            implementation_model=None, qa_model=None,
            endpoint="http://localhost:11434/api/generate",
            temperature=0.3, num_predict=256, num_ctx=2048, timeout=10,
            max_qa_revisions=1, max_impl_repairs=2, browser_check=False,
        )
        self.html = (
            "<!doctype html><html><head><title>External</title></head>"
            "<body><main>External app</main></body></html>"
        )

    def make_folder(self, root, html=None, name="index.html"):
        folder = Path(root) / "my_app"
        folder.mkdir()
        (folder / name).write_text(self.html if html is None else html, encoding="utf-8")
        return folder

    def test_inspect_accepts_single_file_app(self):
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            folder = self.make_folder(temporary_dir)
            info = app_builder.inspect_external_target(folder)
        self.assertTrue(info["ok"])
        self.assertEqual(info["name"], "index.html")

    def test_inspect_rejects_multi_file_app(self):
        html = self.html.replace("</body>", "<script src=\"app.js\"></script></body>")
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            folder = self.make_folder(temporary_dir, html=html)
            info = app_builder.inspect_external_target(folder)
        self.assertFalse(info["ok"])
        self.assertIn("対象外", info["reason"])

    def test_inspect_rejects_folder_without_html(self):
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            info = app_builder.inspect_external_target(Path(temporary_dir))
        self.assertFalse(info["ok"])

    def test_external_revision_keeps_original_and_records_summary(self):
        revised = self.html.replace("External app", "External app improved")
        responses = ["改修方針", revised, "QA_STATUS: APPROVED\nDEFECTS:\n- なし"]
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            root = Path(temporary_dir)
            folder = self.make_folder(root)
            out_root = root / "out"; out_root.mkdir()
            with patch.object(app_builder, "call_ollama", side_effect=responses):
                with contextlib.redirect_stdout(io.StringIO()):
                    code, new_run = app_builder.run_external_revision(
                        folder, "文言を改善して", self.args, out_root)
            manifest = app_builder.json.loads((new_run / "run.json").read_text(encoding="utf-8"))
            original_now = (folder / "index.html").read_text(encoding="utf-8")
            backup = (new_run / "00_original_backup.html").read_text(encoding="utf-8")
            revised_now = (new_run / "index.html").read_text(encoding="utf-8")
            staging_left = [p.name for p in out_root.iterdir() if p.name.startswith(".external_src")]
        self.assertEqual(code, 0)
        self.assertEqual(original_now, self.html)
        self.assertEqual(backup, self.html)
        self.assertEqual(manifest["mode"], "external_revision")
        self.assertEqual(manifest["source_folder"], str(folder))
        self.assertIn("External app improved", revised_now)
        self.assertIn("文言を改善して", manifest["change_summary"])
        self.assertEqual(staging_left, [])

    def test_external_revision_rejects_out_of_scope(self):
        html = self.html.replace("</body>", "<script src=\"app.js\"></script></body>")
        with tempfile.TemporaryDirectory(dir=TOOL_DIR) as temporary_dir:
            root = Path(temporary_dir)
            folder = self.make_folder(root, html=html)
            out_root = root / "out"; out_root.mkdir()
            with self.assertRaises(app_builder.StageError):
                app_builder.run_external_revision(folder, "直して", self.args, out_root)



class RoutingV37Tests(unittest.TestCase):
    """v3.7: 16GB向けに、実装/QAを現行9B級中心へ寄せた振り分け。"""

    @staticmethod
    def _details(*specs):
        return [{"name": name, "size": 5_000_000_000,
                 "details": {"parameter_size": size}} for name, size in specs]

    def test_users_real_set_keeps_light_coder_for_code_roles(self):
        details = self._details(("qwen3.5:9b", "9B"), ("qwen2.5-coder:7b", "7B"))
        assignments, _reasons, _s = app_builder.smart_route_models(details)
        self.assertEqual(assignments["requirements"], "qwen3.5:9b")
        self.assertEqual(assignments["design"], "qwen3.5:9b")
        self.assertEqual(assignments["implementation"], "qwen2.5-coder:7b")
        self.assertEqual(assignments["qa"], "qwen2.5-coder:7b")

    def test_oversized_coder_is_not_auto_assigned(self):
        details = self._details(("qwen3.5:9b", "9B"), ("qwen2.5-coder:32b", "32B"))
        assignments, _reasons, _s = app_builder.smart_route_models(details)
        self.assertEqual(assignments["implementation"], "qwen3.5:9b")
        self.assertEqual(assignments["qa"], "qwen3.5:9b")
        self.assertEqual(assignments["requirements"], "qwen3.5:9b")

    def test_current_9b_coder_tops_implementation(self):
        details = self._details(
            ("qwen3.5-coder:9b", "9B"), ("qwen2.5-coder:7b", "7B"), ("qwen3.5:9b", "9B"))
        assignments, reasons, _s = app_builder.smart_route_models(details)
        self.assertEqual(assignments["implementation"], "qwen3.5-coder:9b")
        self.assertEqual(assignments["qa"], "qwen3.5-coder:9b")
        self.assertIn("9B級", reasons["implementation"])

    def test_only_big_model_installed_falls_back_with_note(self):
        details = self._details(("qwen3.5:27b", "27B"))
        assignments, reasons, _s = app_builder.smart_route_models(details)
        self.assertTrue(all(n == "qwen3.5:27b" for n in assignments.values()))
        self.assertIn("代用", reasons["implementation"])



class DefectNotesTextTests(unittest.TestCase):
    def test_empty_when_no_defects(self):
        self.assertEqual(app_builder.defect_notes_text({}), "")
        self.assertEqual(app_builder.defect_notes_text({"qa_defects_final": []}), "")
        self.assertEqual(app_builder.defect_notes_text(None), "")

    def test_deduplicates_and_keeps_order(self):
        manifest = {"qa_defects_final": ["ボタンAが動かない", "ボタンBが動かない", "ボタンAが動かない", "  "]}
        text = app_builder.defect_notes_text(manifest)
        self.assertIn("QAの検査で", text)
        self.assertEqual(text.count("ボタンAが動かない"), 1)
        self.assertLess(text.index("ボタンA"), text.index("ボタンB"))

    def test_render_failure_adds_a_line(self):
        manifest = {"qa_defects_final": [], "render_check": {"ok": False}}
        text = app_builder.defect_notes_text(manifest)
        self.assertIn("画面表示の自動確認", text)

    def test_render_ok_adds_nothing(self):
        manifest = {"render_check": {"ok": True}}
        self.assertEqual(app_builder.defect_notes_text(manifest), "")



if __name__ == "__main__":
    unittest.main()
