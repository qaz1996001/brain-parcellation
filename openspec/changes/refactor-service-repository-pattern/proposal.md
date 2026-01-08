# Change: Refactor Service Repository Pattern

## Why

`DCOPEventDicomService` 違反 advanced-alchemy Repository-Service 分層原則，導致：

- **直接 Session 操作**: 8 處 `session.add()/commit()/refresh()` 繞過 Repository 抽象層
- **原始 SQL 散落**: 6 處 `text("SELECT...")` 失去型別安全和可維護性
- **N+1 效能問題**: 迴圈內逐次 `commit()`，100 筆資料 = 100 次 DB 往返
- **重複查詢**: 相同 SQL 模式重複 4 次，違反 DRY 原則
- **不一致抽象**: 3 種 Session 管理方式混用

遵循 **Linus 原則**「良好的資料結構消除特殊情況」和 **advanced-alchemy** 最佳實踐，需將資料庫操作封裝到 Repository 層。

## What Changes

### 1. **Repository 層增強**
   - 將原始 SQL 查詢封裝為 Repository 方法
   - 使用 `list()`, `create()`, `create_many()` 等標準方法
   - 支援複雜查詢的自訂 Repository 方法

### 2. **Service 層清理**
   - 消除直接 `session.add()/commit()` 調用
   - 改用 `await self.create(data=..., auto_commit=True)`
   - 批次操作取代迴圈內 commit

### 3. **查詢整合**
   - 4 個重複 SQL 模式統一到 1 個 Repository 方法
   - 使用 `list_and_count()` 取代雙重查詢
   - 參數化查詢確保 SQL 注入防護

### 4. **Session 管理一致化**
   - 統一使用 `session_manager.get_session()` 模式
   - 移除 Session 作為參數傳遞的模式
   - Repository 管理 Session 生命週期

## Impact

**受影響的 specs:**
- `sync-service` (MODIFIED) - DCOPEventDicomService 資料庫操作模式

**受影響的程式碼:**
- `backend/app/sync/service.py` - 主要重構目標
- `backend/app/services/sync_v2.py` - V2 服務對應調整
- `backend/app/service.py` - BaseRepositoryService 可能增強

**量化改進:**
- 直接 Session 操作: 8 處 → 0 處 (減少 100%)
- 原始 SQL text(): 6 處 → 0 處 (減少 100%)
- 重複查詢模式: 4 個 → 1 個 (減少 75%)
- DB 往返 (100 筆): 100 次 → 1 次 (減少 99%)

**設計原則:**
- **Linus (資料結構)**: Repository 方法 = 良好資料結構，消除 SQL 字串散落
- **DRY**: 單一查詢方法，多處重用
- **SOLID**: Repository 單一職責 (資料存取)，Service 單一職責 (業務邏輯)
- **advanced-alchemy**: 遵循官方 Repository-Service 分層模式

**向後相容:**
- ✅ 外部 API 介面不變
- ✅ 內部方法簽名維持
- ✅ 行為語義等價 (契約測試驗證)
