@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv" (
    py -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .

python -m streamlit run streamlit_app.py --server.address localhost
