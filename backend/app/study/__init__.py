"""
研究 (Study) 模組 - 醫學影像 DICOM 研究的生命週期管理。

此模組負責管理醫學影像研究（Study）的完整生命週期，包括：

1. **傳輸階段** (Transfer)
   - Study 到達系統
   - Series 從 Orthanc DICOM 伺服器傳輸

2. **轉檔階段** (Conversion)
   - DICOM 檔案轉換為 NIFTI 格式
   - 準備推理

3. **推理階段** (Inference)
   - 執行 AI 模型推理
   - 生成分析結果

4. **完成階段** (Result)
   - 結果報告發送

模組結構
--------
- **routers.py**: HTTP API 端點定義
- **service.py**: 業務邏輯實現
- **model.py**: 資料庫模型（在 sync 模組中）
- **schemas.py**: 資料序列化方案
- **urls.py**: API 路由路徑
- **deps.py**: 依賴注入配置

核心特性
--------
✨ **事件驅動架構**
   - 每個狀態變化都產生一條事件紀錄
   - 完整的審計追蹤

✨ **異步任務隊列**
   - 非同步處理耗時操作
   - 與外部系統無縫集成

✨ **複雜查詢支援**
   - 多欄位搜索
   - 集合過濾
   - 日期範圍過濾
   - 排序和分頁

✨ **Good Taste 設計**
   - 消除特殊情況
   - 資料結構驅動
   - 鬆散耦合

API 端點
--------
GET /study/list
    查詢研究事件，支援多種過濾、搜索和排序選項

Examples
--------
查詢所有研究事件：

    >>> import httpx
    >>> async with httpx.AsyncClient() as client:
    ...     response = await client.get("/study/list")
    ...     events = response.json()

複雜查詢範例：

    >>> # 查詢特定工具創建的事件
    >>> response = await client.get("/study/list?tool_id=DICOM_TOOL")
    >>> 
    >>> # 按日期範圍查詢
    >>> response = await client.get(
    ...     "/study/list"
    ...     "?createTimeAfter=2024-01-01"
    ...     "&createTimeBefore=2024-12-31"
    ... )
    >>> 
    >>> # 複合查詢
    >>> response = await client.get(
    ...     "/study/list"
    ...     "?searchString=stroke"
    ...     "&tool_id=NIFTI_TOOL,INFERENCE_TOOL"
    ...     "&limit=20&offset=0"
    ... )

See Also
--------
backend.app.sync : 同步模組的核心實現
backend.app.rerun : 重新執行模組
backend.app.inference : 推理結果管理

Notes
-----
此模組遵循以下設計原則：
- 分離關注點：routers/service/model/schemas
- 依賴注入：所有外部依賴通過 DI 系統提供
- 非同步優先：使用 async/await 實現高效率
- 類型安全：完整的 Pydantic 驗證和類型提示
"""

from .routers import router

__all__ = ["router"]
