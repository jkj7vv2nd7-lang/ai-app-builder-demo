"""Safely upload a reviewed HTML app to a user-created GitHub repository."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


REPOSITORY_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,100}\Z")
OWNER_PATTERN = re.compile(r"[A-Za-z0-9-]{1,39}\Z")
TOKEN_PATTERN = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)


class PublishError(RuntimeError):
    def __init__(self, stage, message, repo_url=None, pages_url=None):
        super().__init__(message)
        self.stage = stage
        self.repo_url = repo_url
        self.pages_url = pages_url


@dataclass(frozen=True)
class RepositoryTarget:
    owner: str
    name: str
    repo_url: str
    git_url: str
    pages_url: str
    default_branch: str
    empty: bool


@dataclass(frozen=True)
class PublishResult:
    repository: str
    repo_url: str
    pages_url: str
    status: str
    details: str


def validate_repository_name(name):
    if not isinstance(name, str) or not REPOSITORY_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "GitHubのリポジトリURLを確認してください。所有者名とリポジトリ名を含む"
            " https://github.com/owner/repository 形式が必要です。"
        )
    if name in {".", ".."} or name.startswith(".") or name.endswith("."):
        raise ValueError("リポジトリ名の先頭・末尾にピリオドは使えません。")
    return name


def parse_repository_url(repository_url):
    try:
        parsed = urlsplit(repository_url.strip())
        path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    except (AttributeError, ValueError) as exc:
        raise ValueError("有効なGitHubリポジトリURLを入力してください。") from exc
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() != "github.com"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or len(path_parts) != 2
    ):
        raise ValueError(
            "URLは https://github.com/owner/repository の形式にしてください。"
            "GitHub以外のホストや追加パスは使用できません。"
        )
    owner, name = path_parts
    if (
        not OWNER_PATTERN.fullmatch(owner)
        or owner.startswith("-")
        or owner.endswith("-")
    ):
        raise ValueError("GitHub URLのowner部分が正しくありません。")
    if name.casefold().endswith(".git"):
        name = name[:-4]
    validate_repository_name(name)
    return owner, name


def _gh_environment():
    environment = os.environ.copy()
    environment["GH_HOST"] = "github.com"
    return environment


def _run(command, *, cwd=None, timeout=60, stage):
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=_gh_environment(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PublishError(
            stage,
            f"{stage}がタイムアウトしました。ネット接続とGitHub CLIの状態を確認してください。",
        ) from exc
    except OSError as exc:
        raise PublishError(stage, f"{stage}を起動できませんでした: {exc}") from exc


def _command_detail(result):
    detail = (result.stderr or result.stdout or "").strip()
    detail = TOKEN_PATTERN.sub("[認証情報を伏せました]", detail)
    return detail[:600]


def _failure_message(stage, result):
    detail = _command_detail(result).lower()
    if any(term in detail for term in ("already exists", "already taken", "already exists")):
        return f"{stage}で競合が見つかりました。既存の内容は上書きしていません。"
    if any(term in detail for term in ("permission", "403", "scope", "not authorized", "forbidden")):
        return (
            f"{stage}の権限がありません。gh auth status でログイン状態を確認し、"
            "リポジトリへの書き込み権限があるアカウントを使ってください。"
        )
    if any(term in detail for term in ("network", "connection", "timed out", "could not resolve")):
        return f"{stage}でGitHubに接続できませんでした。ネット接続を確認して再試行してください。"
    if "non-fast-forward" in detail or "fetch first" in detail or "rejected" in detail:
        return (
            f"{stage}でリモート更新との競合が見つかりました。履歴を書き換えず停止しました。"
            "GitHub上の変更を確認してから再試行してください。"
        )
    summary = _command_detail(result)
    if summary:
        return f"{stage}に失敗しました（終了コード {result.returncode}）。詳細: {summary}"
    return f"{stage}に失敗しました（終了コード {result.returncode}）。状態と権限を確認してください。"


def _run_checked(command, *, cwd=None, timeout=60, stage, repo_url=None, pages_url=None):
    result = _run(command, cwd=cwd, timeout=timeout, stage=stage)
    if result.returncode:
        raise PublishError(
            stage,
            _failure_message(stage, result),
            repo_url=repo_url,
            pages_url=pages_url,
        )
    return result.stdout.strip()


def _inspect_target(owner, name):
    gh_path = shutil.which("gh")
    git_path = shutil.which("git")
    if not git_path:
        raise PublishError(
            "事前確認",
            "Gitが見つかりません。Git for Windowsをインストールし、PATHを確認してください。",
        )
    if not gh_path:
        raise PublishError(
            "事前確認",
            "GitHub CLI（gh）が見つかりません。GitHub CLIをインストールし、PowerShellを開き直してください。",
        )

    git_check = _run([git_path, "--version"], stage="Git確認")
    if git_check.returncode:
        raise PublishError("Git確認", "Gitを実行できません。Git for Windowsのインストールを確認してください。")
    auth_check = _run(
        [gh_path, "auth", "status", "--hostname", "github.com"],
        stage="GitHub認証確認",
        timeout=30,
    )
    if auth_check.returncode:
        raise PublishError(
            "GitHub認証確認",
            "GitHub.comにログインしていません。PowerShellで gh auth login を実行し、"
            "ブラウザー認証を完了してください。このアプリはトークンを保存しません。",
        )

    response = _run(
        [gh_path, "api", "--hostname", "github.com", f"repos/{owner}/{name}"],
        stage="リポジトリ確認",
        timeout=30,
    )
    if response.returncode:
        raise PublishError(
            "リポジトリ確認",
            "リポジトリを読み取れませんでした。URL、GitHubログイン、リポジトリへのアクセス権を確認してください。",
        )
    try:
        repository = json.loads(response.stdout)
    except json.JSONDecodeError as exc:
        raise PublishError("リポジトリ確認", "GitHubからのリポジトリ情報を読み取れませんでした。") from exc
    if repository.get("private") is not False:
        raise PublishError(
            "公開設定確認",
            "対象リポジトリが公開状態ではありません。公開リポジトリを作成してから再試行してください。",
        )
    if (repository.get("permissions") or {}).get("push") is not True:
        raise PublishError(
            "書き込み権限確認",
            "このアカウントに対象リポジトリへの書き込み権限がありません。"
            "リポジトリ所有者に権限を確認してください。",
        )

    git_url = f"https://github.com/{owner}/{name}.git"
    refs = _run(
        [git_path, "ls-remote", "--heads", git_url],
        stage="Git認証・ブランチ確認",
        timeout=30,
    )
    if refs.returncode:
        raise PublishError(
            "Git認証・ブランチ確認",
            "Gitでリポジトリを読み取れませんでした。PowerShellで gh auth setup-git を実行してGit認証を設定し、"
            "再試行してください。アプリは認証情報を保存・変更しません。",
        )
    head_names = {
        line.split("\t", 1)[1]
        for line in refs.stdout.splitlines()
        if "\t" in line and line.split("\t", 1)[1].startswith("refs/heads/")
    }
    empty = not head_names
    default_branch = repository.get("default_branch") or "main"
    if head_names and f"refs/heads/{default_branch}" not in head_names:
        raise PublishError(
            "ブランチ確認",
            "既定ブランチを安全に特定できませんでした。GitHubのSettingsで既定ブランチを確認してください。",
        )

    return RepositoryTarget(
        owner=owner,
        name=name,
        repo_url=f"https://github.com/{owner}/{name}",
        git_url=git_url,
        pages_url=f"https://{owner}.github.io/{name}/",
        default_branch=default_branch,
        empty=empty,
    )


def preflight(repository_url):
    """Check GitHub auth, repository visibility/write access, and Git transport access."""
    owner, name = parse_repository_url(repository_url)
    return _inspect_target(owner, name)


def _git(git_path, args, *, cwd, stage, repo_url=None, pages_url=None):
    return _run_checked(
        [git_path, *args],
        cwd=cwd,
        timeout=900,
        stage=stage,
        repo_url=repo_url,
        pages_url=pages_url,
    )


def _prepare_checkout(target, git_path, directory):
    if target.empty:
        _git(git_path, ["init"], cwd=directory, stage="公開用Git準備")
        _git(git_path, ["branch", "-M", "main"], cwd=directory, stage="公開用Git準備")
        _git(
            git_path,
            ["remote", "add", "origin", target.git_url],
            cwd=directory,
            stage="公開用Git準備",
        )
        return "main"
    _git(
        git_path,
        ["clone", "--single-branch", "--branch", target.default_branch, target.git_url, "."],
        cwd=directory,
        stage="既存リポジトリの安全な取得",
        repo_url=target.repo_url,
        pages_url=target.pages_url,
    )
    return target.default_branch


def publish_public_pages(
    output_directory,
    repository_url,
    expected_target,
    *,
    progress_callback=lambda _message: None,
):
    """Add only index.html to a new or clean existing public repository branch."""
    if not isinstance(expected_target, RepositoryTarget):
        raise ValueError("公開前確認の情報がありません。もう一度公開操作を開始してください。")
    output_directory = Path(output_directory)
    html_path = output_directory / "index.html"
    if html_path.is_symlink() or not html_path.is_file():
        raise PublishError("公開内容の確認", "実行フォルダーに公開可能な index.html がありません。")

    target = preflight(repository_url)
    if (
        target.owner.casefold() != expected_target.owner.casefold()
        or target.name.casefold() != expected_target.name.casefold()
        or target.default_branch != expected_target.default_branch
        or target.empty != expected_target.empty
    ):
        raise PublishError(
            "公開前の再確認",
            "確認後にリポジトリの状態が変わりました。内容を確認するため公開を中止しました。",
            repo_url=target.repo_url,
            pages_url=target.pages_url,
        )

    git_path = shutil.which("git")
    staging_parent = Path(__file__).resolve().parent / "github_publish_staging"
    staging_parent.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="publish-", dir=staging_parent) as temporary_directory:
        directory = Path(temporary_directory) / "repo"
        directory.mkdir()
        branch = _prepare_checkout(target, git_path, directory)

        if any(path.name.casefold() == "index.html" for path in directory.iterdir()):
            raise PublishError(
                "既存ファイル確認",
                "対象ブランチに index.html（または同名ファイル）が既にあります。"
                "無断上書きを避けるため停止しました。既存内容を確認して整理してください。",
                repo_url=target.repo_url,
                pages_url=target.pages_url,
            )
        status = _git(
            git_path,
            ["status", "--porcelain"],
            cwd=directory,
            stage="既存内容確認",
            repo_url=target.repo_url,
            pages_url=target.pages_url,
        )
        if status:
            raise PublishError(
                "既存内容確認",
                "取得した作業フォルダーがクリーンではありません。変更を加えず停止しました。",
                repo_url=target.repo_url,
                pages_url=target.pages_url,
            )

        shutil.copyfile(html_path, directory / "index.html")
        progress_callback("公開する index.html を準備しています...")
        _git(
            git_path,
            ["config", "--local", "user.name", "AI App Builder"],
            cwd=directory,
            stage="公開用Git準備",
        )
        _git(
            git_path,
            ["config", "--local", "user.email", "ai-app-builder@users.noreply.github.com"],
            cwd=directory,
            stage="公開用Git準備",
        )
        _git(
            git_path,
            ["add", "--", "index.html"],
            cwd=directory,
            stage="公開内容の登録",
            repo_url=target.repo_url,
            pages_url=target.pages_url,
        )
        staged = _git(
            git_path,
            ["diff", "--cached", "--name-only"],
            cwd=directory,
            stage="公開内容の確認",
            repo_url=target.repo_url,
            pages_url=target.pages_url,
        ).splitlines()
        if staged != ["index.html"]:
            raise PublishError(
                "公開内容の確認",
                "公開対象が index.html だけではありません。pushせずに停止しました。",
                repo_url=target.repo_url,
                pages_url=target.pages_url,
            )
        _git(
            git_path,
            ["commit", "-m", "Add generated app"],
            cwd=directory,
            stage="公開用コミット",
            repo_url=target.repo_url,
            pages_url=target.pages_url,
        )

        progress_callback(f"{target.repo_url} に index.html のみをアップロードしています...")
        push_args = ["push", "--set-upstream", "origin", branch]
        _git(
            git_path,
            push_args,
            cwd=directory,
            stage="GitHubへのアップロード",
            repo_url=target.repo_url,
            pages_url=target.pages_url,
        )

    return PublishResult(
        repository=f"{target.owner}/{target.name}",
        repo_url=target.repo_url,
        pages_url=target.pages_url,
        status="uploaded",
        details=(
            "index.html をアップロードしました。GitHub Pagesはまだ有効化していません。"
            "GitHubのSettings > PagesでDeploy from a branch、ブランチ、/(root)を選んで保存してください。"
        ),
    )
