# /dev 命令：標準工程化開發流程

當用戶輸入 `/dev [需求]` 時，請嚴格執行以下步驟：

1. **分析 (Analyze)**: 調用 `sc:analyze` 評估需求影響。
2. **設計 (Design)**: 調用 `sc:design` 產出技術實作方案。
3. **實作 (Implement)**: 調用 `sc:implement` 開始寫代碼。
   - *注意：* 必須遵守 `.claude/skills/global_rules.md` 中的 `uv` 規範。
4. **測試 (Test)**: 代碼完成後，調用 `sc:test` 進行功能驗證。
5. **清理與校驗 (Cleanup)**: 調用 `sc:cleanup` 移除臨時文件。
   - *自動觸發：* 此時會觸發我們的 PostToolUse Hook 執行 `ruff/eslint`。
6. **文件化 (Document)**: 調用 `sc:document` 更新相關文檔。
7. **反思 (Reflect)**: 調用 `sc:reflect` 確認是否有遺漏的邊界情況。