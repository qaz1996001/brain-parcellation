# Design: Service Repository Pattern Refactoring

## Context

**背景**: `DCOPEventDicomService` 繼承 `BaseRepositoryService[DCOPEventModel]`，但未正確使用 advanced-alchemy 提供的 Repository 方法，而是直接操作 SQLAlchemy Session。

**利害關係人**:
- 後端開發人員 (維護性)
- DevOps (效能監控)
- QA (可測試性)

**約束**:
- 必須維持 API 向後相容
- 不改變業務邏輯語義
- 漸進式重構，可分批部署

## Goals / Non-Goals

### Goals
1. 消除 Service 層直接 Session 操作
2. 將複雜 SQL 封裝到 Repository 層
3. 使用批次操作優化效能
4. 統一 Session 管理模式
5. 提高程式碼可測試性

### Non-Goals
1. 不改變資料庫 Schema
2. 不新增 API 端點
3. 不修改儲存過程 (stored procedures)
4. 不遷移到其他 ORM

## Decisions

### Decision 1: Repository 方法封裝策略

**選擇**: 將複雜 SQL 查詢封裝為 Repository 自訂方法

**理由**:
- advanced-alchemy 支援自訂 Repository 方法
- 保持 SQL 邏輯集中、可測試
- Service 層只調用語義化方法

**替代方案**:
- ❌ 保留 `text()` SQL: 失去型別安全，難以維護
- ❌ 使用 SQLAlchemy ORM 重寫所有查詢: 儲存過程 (`get_stydy_series_ope_no_status`) 無法用 ORM 表達
- ✅ 混合策略: 簡單查詢用 ORM，複雜查詢封裝到 Repository

```python
# ✅ 選擇的模式
class Repo(SQLAlchemyAsyncRepository[DCOPEventModel]):
    model_type = DCOPEventModel

    async def get_series_by_status(
        self, status: str, study_uid: Optional[str] = None
    ) -> List[Row]:
        """封裝 stored procedure 調用"""
        sql = text("SELECT * FROM public.get_stydy_series_ope_no_status(:status)")
        params = {"status": status}
        if study_uid:
            sql = text(sql.text + " WHERE study_uid = :study_uid")
            params["study_uid"] = study_uid
        result = await self.session.execute(sql, params)
        return result.all()
```

### Decision 2: 批次操作模式

**選擇**: 使用 `create_many()` 和 `add_many()` 取代迴圈 commit

**理由**:
- advanced-alchemy 提供 dialect-specific 優化 (PostgreSQL COPY, MySQL bulk insert)
- 單次 DB 往返，效能提升 99%+
- 事務原子性保證

**實作**:
```python
# ❌ 目前: N 次往返
for event in data:
    session.add(obj)
    await session.commit()  # N 次

# ✅ 新模式: 1 次往返
records = [self._build_record(event) for event in data]
await self.create_many(data=records, auto_commit=True)
```

### Decision 3: Session 管理統一

**選擇**: 統一使用 `self.session_manager.get_session()` context manager

**理由**:
- Repository 基類已管理 Session 生命週期
- 避免 Session 作為參數傳遞造成的混淆
- 符合 advanced-alchemy 設計意圖

**不再使用**:
- ❌ `session: AsyncSession` 作為方法參數
- ❌ `self.repository.session` 直接存取

### Decision 4: 狀態轉換資料結構化

**選擇**: 使用 Dict 映射取代 match-case 串聯

**理由 (Linus 原則)**: 「良好的資料結構使程式碼自然簡單」

```python
# ❌ 目前: 程序化邏輯
async def get_check_url_by_ope_no(self, ope_no: str) -> Optional[str]:
    match ope_no:
        case DCOPStatus.STUDY_TRANSFER_COMPLETE.value:
            url = f"{api_url}{SYNC_PROT_STUDY_TRANSFER_COMPLETE}"
        case DCOPStatus.STUDY_CONVERSION_COMPLETE.value:
            url = f"{api_url}{SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID}"
        # ... 更多 case

# ✅ 新模式: 資料結構驅動
STATUS_URL_MAP = {
    DCOPStatus.STUDY_TRANSFER_COMPLETE: SYNC_PROT_STUDY_TRANSFER_COMPLETE,
    DCOPStatus.STUDY_CONVERSION_COMPLETE: SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID,
    DCOPStatus.SERIES_TRANSFER_COMPLETE: SYNC_PROT_STUDY_TRANSFER_COMPLETE,
    DCOPStatus.SERIES_CONVERSION_COMPLETE: SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID,
}

def get_check_url_by_ope_no(self, ope_no: str) -> Optional[str]:
    endpoint = self.STATUS_URL_MAP.get(DCOPStatus(ope_no))
    return f"{self.config.api.upload_data_url}{endpoint}" if endpoint else None
```

## Risks / Trade-offs

### Risk 1: Stored Procedure 依賴
- **風險**: 系統依賴 PostgreSQL stored procedures，無法完全 ORM 化
- **緩解**: 在 Repository 層封裝 stored procedure 調用，提供語義化介面

### Risk 2: 效能回歸
- **風險**: 批次操作在某些邊緣情況可能有不同行為
- **緩解**: 契約測試驗證行為等價，效能基準測試

### Risk 3: 遷移期間的並存
- **風險**: V1 和 V2 服務並存可能造成混淆
- **緩解**: 明確標記 V2 為重構版本，V1 標記為 deprecated

### Trade-off: 封裝 vs 透明度
- 封裝 SQL 到 Repository 降低透明度
- 但提高可維護性和可測試性
- **決定**: 接受封裝，因長期維護價值更高

## Migration Plan

### Phase 1: Repository 增強 (無破壞性)
1. 新增 Repository 自訂方法
2. 保持現有 Service 程式碼不變
3. 驗證 Repository 方法與原始 SQL 等價

### Phase 2: Service 漸進重構
1. 逐一替換直接 Session 操作
2. 每個方法單獨測試和部署
3. 使用 feature flag 控制 (可選)

### Phase 3: 清理和優化
1. 移除重複程式碼
2. 統一錯誤處理
3. 效能基準測試和優化

### Rollback Plan
- **Layer 1**: Git revert 單一 commit (< 5 分鐘)
- **Layer 2**: Feature flag 回滾 (若實施)
- **Layer 3**: 保留 V1 服務作為備份

## Open Questions

1. **Q**: 是否需要為 stored procedures 建立 Python wrapper 類型?
   - **初步答案**: 使用 TypedDict 或 Pydantic model 表示返回結果

2. **Q**: `sync_v2.py` 是否應與 `service.py` 合併?
   - **初步答案**: Phase 3 考慮，目前保持分離以降低風險

3. **Q**: 是否需要遷移 `nifti_tool_get_series_info` 的 session 參數模式?
   - **初步答案**: 是，應改為從 Repository 獲取 session
