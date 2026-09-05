@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv" (
    py -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .

echo.
echo Uso:
echo quant-trafego --input "C:\caminho\planilha.xlsx" --output output --draws 30000
echo.
cmd /k
