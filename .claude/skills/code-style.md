# 開發規範
- 環境管理：嚴禁直接使用 python 或 pip。所有指令必須前綴 uv run 或 uv。
代碼質量：
- 修改 Python 文件後，必須立即執行 uv run ruff check --fix <file> ；uv run ty check <file> 
- 修改 JS/TS 文件後，必須立即執行 npx eslint --fix <file>。

禁止提交未經過 Lint 修復的代碼。