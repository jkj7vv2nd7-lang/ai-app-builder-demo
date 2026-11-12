@echo off
setlocal
where pythonw >nul 2>nul
if not errorlevel 1 (
    start "" pythonw "%~dp0launch_gui.pyw"
    exit /b 0
)
where pyw >nul 2>nul
if not errorlevel 1 (
    start "" pyw -3 "%~dp0launch_gui.pyw"
    exit /b 0
)
echo Python の pythonw.exe / pyw.exe が見つかりません。
echo Python 3 for Windows をインストールし、ランチャーを PATH に追加してください。
pause
exit /b 1
