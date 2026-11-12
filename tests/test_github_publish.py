import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))
import app_gui  # noqa: E402
import github_publish  # noqa: E402


def completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class RepositoryValidationTests(unittest.TestCase):
    def test_accepts_repository_url_with_optional_git_suffix(self):
        self.assertEqual(
            github_publish.parse_repository_url("https://github.com/example-user/my-app.git"),
            ("example-user", "my-app"),
        )

    def test_rejects_non_github_or_ambiguous_urls(self):
        for value in (
            "http://github.com/example/my-app",
            "https://github.com/example/my-app/tree/main",
            "https://user:secret@github.com/example/my-app",
            "https://github.example.com/example/my-app",
            "https://github.com/example/my-app?private=true",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                github_publish.parse_repository_url(value)

    @patch.object(github_publish.shutil, "which", side_effect=lambda name: name)
    @patch.object(github_publish, "_run")
    def test_preflight_gives_login_advice_without_api_access_when_logged_out(self, run, _which):
        run.side_effect = [
            completed(["git", "--version"]),
            completed(["gh", "auth"], returncode=1),
        ]
        with self.assertRaises(github_publish.PublishError) as context:
            github_publish.preflight("https://github.com/example/my-app")
        self.assertEqual(context.exception.stage, "GitHub認証確認")
        self.assertIn("gh auth login", str(context.exception))
        self.assertEqual(run.call_count, 2)

    @patch.object(github_publish.shutil, "which", side_effect=lambda name: name)
    @patch.object(github_publish, "_run")
    def test_preflight_rejects_private_or_unwritable_repositories(self, run, _which):
        run.side_effect = [
            completed(["git", "--version"]),
            completed(["gh", "auth", "status"]),
            completed(
                ["gh", "api"],
                stdout='{"private": false, "default_branch": "main", "permissions": {"push": false}}',
            ),
        ]
        with self.assertRaises(github_publish.PublishError) as context:
            github_publish.preflight("https://github.com/example/my-app")
        self.assertEqual(context.exception.stage, "書き込み権限確認")
        self.assertEqual(run.call_count, 3)

    @patch.object(github_publish.shutil, "which", side_effect=lambda name: name)
    @patch.object(github_publish, "_run")
    def test_preflight_checks_git_transport_and_accepts_empty_public_repo(self, run, _which):
        run.side_effect = [
            completed(["git", "--version"]),
            completed(["gh", "auth", "status"]),
            completed(
                ["gh", "api"],
                stdout='{"private": false, "default_branch": null, "permissions": {"push": true}}',
            ),
            completed(["git", "ls-remote"], stdout=""),
        ]
        target = github_publish.preflight("https://github.com/example/my-app")
        self.assertTrue(target.empty)
        self.assertEqual(target.default_branch, "main")
        self.assertEqual(target.pages_url, "https://example.github.io/my-app/")

    @patch.object(github_publish.shutil, "which", side_effect=lambda name: name)
    @patch.object(github_publish, "_run")
    def test_preflight_advises_git_auth_setup_when_git_transport_fails(self, run, _which):
        run.side_effect = [
            completed(["git", "--version"]),
            completed(["gh", "auth", "status"]),
            completed(
                ["gh", "api"],
                stdout='{"private": false, "default_branch": "main", "permissions": {"push": true}}',
            ),
            completed(["git", "ls-remote"], returncode=128, stderr="authentication failed"),
        ]
        with self.assertRaises(github_publish.PublishError) as context:
            github_publish.preflight("https://github.com/example/my-app")
        self.assertEqual(context.exception.stage, "Git認証・ブランチ確認")
        self.assertIn("gh auth setup-git", str(context.exception))


class UploadSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=TOOL_DIR)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "run"
        self.output.mkdir()
        (self.output / "index.html").write_text("<html>reviewed app</html>", encoding="utf-8")
        (self.output / "run.json").write_text('{"qa_approved": true}', encoding="utf-8")
        (self.output / "private-notes.txt").write_text("must not be uploaded", encoding="utf-8")
        self.target = github_publish.RepositoryTarget(
            owner="example",
            name="my-app",
            repo_url="https://github.com/example/my-app",
            git_url="https://github.com/example/my-app.git",
            pages_url="https://example.github.io/my-app/",
            default_branch="main",
            empty=False,
        )

    @patch.object(github_publish.shutil, "which", side_effect=lambda name: name)
    @patch.object(github_publish, "preflight")
    @patch.object(github_publish, "_run")
    def test_upload_adds_only_index_and_does_not_create_repo_or_enable_pages(
        self, run, preflight, _which
    ):
        preflight.return_value = self.target
        commands = []

        def fake_run(command, **kwargs):
            commands.append((command, kwargs.get("cwd")))
            if command[1:4] == ["diff", "--cached", "--name-only"]:
                return completed(command, stdout="index.html\n")
            return completed(command)

        run.side_effect = fake_run
        result = github_publish.publish_public_pages(
            self.output,
            "https://github.com/example/my-app",
            self.target,
        )
        self.assertEqual(result.status, "uploaded")
        self.assertIn("まだ有効化していません", result.details)
        add_commands = [command for command, _cwd in commands if command[1:3] == ["add", "--"]]
        self.assertEqual(len(add_commands), 1)
        self.assertEqual(add_commands[0][-1], "index.html")
        self.assertFalse(any(command[1:3] == ["repo", "create"] for command, _ in commands))
        self.assertFalse(any(command[1:2] == ["api"] for command, _ in commands))
        push = [command for command, _ in commands if command[1:2] == ["push"]]
        self.assertEqual(push, [["git", "push", "--set-upstream", "origin", "main"]])

    @patch.object(github_publish.shutil, "which", side_effect=lambda name: name)
    @patch.object(github_publish, "preflight")
    @patch.object(github_publish, "_run")
    def test_existing_index_html_collision_stops_before_push(self, run, preflight, _which):
        preflight.return_value = self.target
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if command[1:4] == ["clone", "--single-branch", "--branch"]:
                (Path(kwargs["cwd"]) / "INDEX.HTML").write_text("existing", encoding="utf-8")
            return completed(command)

        run.side_effect = fake_run
        with self.assertRaises(github_publish.PublishError) as context:
            github_publish.publish_public_pages(
                self.output,
                "https://github.com/example/my-app",
                self.target,
            )
        self.assertEqual(context.exception.stage, "既存ファイル確認")
        self.assertFalse(any(command[1:2] == ["push"] for command in commands))

    @patch.object(github_publish.shutil, "which", side_effect=lambda name: name)
    @patch.object(github_publish, "preflight")
    @patch.object(github_publish, "_run")
    def test_empty_repository_gets_main_branch_without_force_push(self, run, preflight, _which):
        empty_target = github_publish.RepositoryTarget(
            **{**self.target.__dict__, "empty": True}
        )
        preflight.return_value = empty_target
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if command[1:4] == ["diff", "--cached", "--name-only"]:
                return completed(command, stdout="index.html\n")
            return completed(command)

        run.side_effect = fake_run
        result = github_publish.publish_public_pages(
            self.output,
            "https://github.com/example/my-app",
            empty_target,
        )
        self.assertEqual(result.status, "uploaded")
        self.assertIn(["git", "branch", "-M", "main"], commands)
        self.assertIn(["git", "remote", "add", "origin", self.target.git_url], commands)
        self.assertIn(["git", "push", "--set-upstream", "origin", "main"], commands)
        self.assertFalse(any("--force" in command for command in commands))

    def test_missing_html_is_rejected_without_git_operations(self):
        (self.output / "index.html").unlink()
        with patch.object(github_publish, "preflight", return_value=self.target):
            with self.assertRaises(github_publish.PublishError) as context:
                github_publish.publish_public_pages(
                    self.output,
                    "https://github.com/example/my-app",
                    self.target,
                )
        self.assertEqual(context.exception.stage, "公開内容の確認")

    def test_gui_publication_error_advice_is_actionable(self):
        auth = app_gui.publication_guidance(
            github_publish.PublishError("GitHub認証確認", "not logged in")
        )
        self.assertIn("gh auth login", auth)
        git_auth = app_gui.publication_guidance(
            github_publish.PublishError("Git認証・ブランチ確認", "git auth failed")
        )
        self.assertIn("gh auth setup-git", git_auth)
        collision = app_gui.publication_guidance(
            github_publish.PublishError("既存ファイル確認", "index exists")
        )
        self.assertIn("上書き", collision)

    def test_gui_requires_a_finished_output_and_explicit_review_before_upload(self):
        window = object.__new__(app_gui.AppBuilderWindow)
        states = []
        window.busy = False
        window.output_ready = True
        window.repository_url_var = SimpleNamespace(get=lambda: "https://github.com/example/my-app")
        window.publication_reviewed_var = SimpleNamespace(get=lambda: False)
        window.publish_button = SimpleNamespace(configure=lambda **kwargs: states.append(kwargs["state"]))
        window.update_publish_button()
        self.assertEqual(states[-1], "disabled")
        window.publication_reviewed_var = SimpleNamespace(get=lambda: True)
        window.update_publish_button()
        self.assertEqual(states[-1], "normal")
        window.busy = True
        window.update_publish_button()
        self.assertEqual(states[-1], "disabled")

    @patch.object(app_gui.messagebox, "askyesno", return_value=True)
    def test_gui_final_confirmation_discloses_public_access_and_pages_delay(self, askyesno):
        window = object.__new__(app_gui.AppBuilderWindow)
        window.root = object()
        window._run_is_unapproved = lambda _output: False
        output = Path("runs") / "reviewed-app"
        self.assertTrue(window._confirm_publication(self.target, output))
        prompt = askyesno.call_args.args[1]
        self.assertIn("誰でも閲覧できます", prompt)
        self.assertIn("index.htmlのみ", prompt)
        self.assertIn("Pagesを有効化するまでは", prompt)
        self.assertIn(self.target.pages_url, prompt)


if __name__ == "__main__":
    unittest.main()
