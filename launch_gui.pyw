"""Windowed launcher that explains startup errors without a console window."""

import ctypes
import sys
from pathlib import Path


def main():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from app_gui import main as run_gui

        run_gui()
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ModuleNotFoundError) and exc.name == "tkinter":
            advice = (
                "このPythonにはTkinterが含まれていません。\n"
                "Windows用のPython 3をインストールし、インストーラーのTcl/Tk項目を有効にしてください。"
            )
        else:
            advice = (
                "Python 3とTkinterの状態を確認してください。\n"
                "詳しいエラー:\n" + details
            )
        ctypes.windll.user32.MessageBoxW(
            None,
            "AIアプリ開発GUIを起動できません。\n\n" + advice,
            "起動エラー",
            0x10,
        )


if __name__ == "__main__":
    main()
