# 技能評估與活化規範 (Skill Force Evaluation)

當你收到任務時，**禁止**直接調用 `Bash` 或 `Write`，必須先執行以下思維流程：

1. **掃描技能庫**：檢查當前可用的 `sc:` 技能（如 sc:analyze, sc:design, sc:implement, sc:test）。
2. **匹配任務階段**：
   - 需求不明確？ -> 調用 `sc:analyze`
   - 準備寫代碼？ -> 調用 `sc:design` 先產出方案
   - 正在修 Bug？ -> 調用 `sc:troubleshoot`
   - 需要確認環境？ -> 調用 `sc:research`
3. **強制宣告**：在執行任何實質性修改前，必須先調用 `sc:select-tool` 或 `sc:workflow` 宣告你將使用的技能路徑。