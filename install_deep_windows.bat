@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv" (
    py -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[deep]"

echo.
echo Dependencias profundas instaladas.
echo Agora voce pode usar:
echo quant-trafego-mcmc --input "C:\caminho\planilha.xlsx" --output output_mcmc
echo.
cmd /k
