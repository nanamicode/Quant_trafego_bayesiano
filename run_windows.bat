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

uv sync --python 3.12 --locked
if errorlevel 1 exit /b 1

echo.
echo Ambiente pronto com Python 3.12.
echo Exemplos:
echo uv run quant-trafego --input "C:\caminho\planilha.xlsx" --output output
echo uv run quant-trafego-backtest --input "C:\caminho\planilha.xlsx" --output backtest_output
echo.
cmd /k
