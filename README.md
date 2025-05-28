## new venv install

``` bash
pip install uv 
git clone ....
cd repositories root
uv sync
source .venv/bin/activate
export PYTHONPATH=$(pwd) && python3 backend/app/main.py
```


## system python install

``` bash
pip install uv
git clone ....
cd repositories root
uv pip install -r pyproject.toml --system
export PYTHONPATH=$(pwd) && python3 backend/app/main.py
```

