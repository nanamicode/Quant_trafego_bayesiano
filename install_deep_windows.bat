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

uv sync --python 3.12 --locked --extra deep
if errorlevel 1 exit /b 1

echo.
echo Dependencias profundas instaladas no ambiente Python 3.12.
echo Use:
echo uv run quant-trafego-mcmc --input "C:\caminho\planilha.xlsx" --output output_mcmc
echo.
cmd /k
