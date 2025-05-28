pip install uv
git clone ....
cd repositories root
uv sync
source .venv/bin/activate
export PYTHONPATH=$(pwd) && python3 backend/app/main.py
