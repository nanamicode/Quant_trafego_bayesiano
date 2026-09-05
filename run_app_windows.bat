@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo uv nao foi encontrado.
    echo Instale pelo WinGet:
    echo winget install --id=astral-sh.uv -e
    echo.
    exit /b 1
)

uv sync --python 3.12
if errorlevel 1 exit /b 1

uv run --python 3.12 streamlit run streamlit_app.py --server.address localhost
