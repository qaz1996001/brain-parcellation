# DWI0/DWI1000 根本原因分析

## 問題現象
正則表達式匹配失敗，導致已完成的 Study 無法進入推理佇列。

**日誌顯示：**
```
pattern: (100.025),(100.055),(100.095),(200.155),(200.195|200.190)
test_str: 100.025,100.055,200.155,200.195
match_result: None
```

**資料庫實際數據：**
```
Series 1: {100.025,100.055,100.095,200.155,200.195} ✅ T1FLAIR_AXI
Series 2: {100.025,100.055,100.095,200.155,200.195} ✅ DWI1000
Series 3: {100.025,100.055,100.095,200.155,200.190} ⚠️ MRAVR_BRAIN (SKIP)
```

## 根本原因

### 1. 正則表達式語法錯誤

**當前代碼：**
```python
pattern_str = '({}),({}),({}),({}),({}|{})'.format(
    DCOPStatus.SERIES_NEW.value,                  # 100.025
    DCOPStatus.SERIES_TRANSFERRING.value,         # 100.055
    DCOPStatus.SERIES_TRANSFER_COMPLETE.value,    # 100.095
    DCOPStatus.SERIES_CONVERTING.value,           # 200.155
    DCOPStatus.SERIES_CONVERSION_COMPLETE.value,  # 200.195
    DCOPStatus.SERIES_CONVERSION_SKIP.value       # 200.190
)
can_inference_pattern = re.compile(pattern_str)
```

**問題點：**
1. ❌ **括號沒有轉義**：`(100.025)` 被當作正則捕獲組，而非字面括號
2. ❌ **點號沒有轉義**：`.` 匹配任意字符，`100.025` 可能匹配 `100X025`
3. ❌ **使用 `match()` 方法**：只匹配字符串開頭，需要完整匹配
4. ❌ **要求特定順序**：PostgreSQL 的 `array_agg(DISTINCT ope_no)` 不保證順序

### 2. PostgreSQL 陣列順序不確定

```sql
array_agg(DISTINCT dcop_event_bt.ope_no) as ope_no
```

`array_agg(DISTINCT ...)` 的返回順序是**不確定的**，可能是：
- `{100.025,100.055,100.095,200.155,200.195}` ✅
- `{200.195,200.155,100.095,100.055,100.025}` ❌
- 任何其他排列組合

### 3. 為什麼有時候缺少 100.095？

**可能原因：**
- **時間窗口問題**：查詢執行時，某些事件還在寫入中
- **資料庫事務隔離**：讀取到的是部分提交的數據
- **Python 字符串轉換問題**：`','.join(list(result.ope_no))` 可能漏掉某些元素

## 解決方案

### 方案 A：使用集合檢查（推薦）

```python
async def query_studies_pending_completion(self, study_uid=None):
    """Query studies that are pending completion."""
    # 定義必須包含的狀態
    REQUIRED_STATUSES = {
        DCOPStatus.SERIES_NEW.value,                  # 100.025
        DCOPStatus.SERIES_TRANSFERRING.value,         # 100.055
        DCOPStatus.SERIES_TRANSFER_COMPLETE.value,    # 100.095
        DCOPStatus.SERIES_CONVERTING.value,           # 200.155
    }
    
    COMPLETE_STATUSES = {
        DCOPStatus.SERIES_CONVERSION_COMPLETE.value,  # 200.195
        DCOPStatus.SERIES_CONVERSION_SKIP.value,      # 200.190
    }
    
    async with self.session_manager.get_session() as session:
        # ... SQL 查詢邏輯 ...
        results = execute.all()
    
    can_inference_dict = {}
    wait_inference_dict = {}
    
    for result in results:
        ope_no_set = set(result.ope_no)  # 直接使用陣列，不需要轉換
        
        # 檢查是否包含所有必要狀態
        has_required = REQUIRED_STATUSES.issubset(ope_no_set)
        has_complete = bool(COMPLETE_STATUSES & ope_no_set)
        
        logger.info(f'Series {result.series_uid}: ope_no={ope_no_set}, '
                   f'has_required={has_required}, has_complete={has_complete}')
        
        if has_required and has_complete:
            can_inference_dict.update({
                result.series_uid: (result.study_uid, result.study_id)
            })
        else:
            wait_inference_dict.update({
                result.series_uid: (result.study_uid, result.study_id)
            })
    
    # ... 後續邏輯 ...
    return result_set
```

### 方案 B：修正正則表達式（次選）

```python
# 如果堅持使用正則表達式，需要：
# 1. 轉義點號
# 2. 使用集合匹配，不限定順序
# 3. 使用 search 而非 match

import re

# 建立正則表達式檢查是否包含所有必要元素
required_patterns = [
    r'\b100\.025\b',
    r'\b100\.055\b',
    r'\b100\.095\b',
    r'\b200\.155\b',
    r'\b(200\.195|200\.190)\b'
]

test_str = ','.join(sorted(result.ope_no))  # 排序以便除錯

# 檢查每個必要模式
all_match = all(re.search(pattern, test_str) for pattern in required_patterns)

if all_match:
    can_inference_dict.update(...)
```

### 方案 C：改善 SQL 查詢（最佳）

在 SQL 層面直接過濾：

```sql
CREATE OR REPLACE FUNCTION get_stydy_series_ope_no_status_v2(p_ope_no character varying)
RETURNS TABLE(...) AS $$
BEGIN
    RETURN QUERY
    WITH series_ope_no_status AS (
        SELECT
            study_uid,
            series_uid,
            MIN(DISTINCT study_id)::varchar as study_id,
            array_agg(DISTINCT ope_no ORDER BY ope_no) as ope_no,  -- 加入 ORDER BY
            array_agg(result_data) as result_data,
            array_agg(params_data) as params_data
        FROM dcop_event_bt
        WHERE series_uid IS NOT NULL
        GROUP BY study_uid, series_uid
    )
    SELECT * FROM series_ope_no_status
    WHERE
        p_ope_no::NUMERIC > ALL (ope_no::NUMERIC[])
        AND '100.025' = ANY(ope_no)  -- 明確檢查必要狀態
        AND '100.055' = ANY(ope_no)
        AND '100.095' = ANY(ope_no)
        AND '200.155' = ANY(ope_no)
        AND ('200.195' = ANY(ope_no) OR '200.190' = ANY(ope_no))
        AND EXISTS (
            SELECT 1 FROM unnest(result_data) AS pd
            WHERE pd IS NOT NULL
        );
END;
$$;
```

## 建議實施順序

1. **立即修復**：採用方案 A（集合檢查），風險最低
2. **中期優化**：實施方案 C（SQL 改進），提升查詢效率
3. **移除舊代碼**：刪除錯誤的正則表達式邏輯

## 測試驗證

```python
# 測試案例
test_cases = [
    {
        'ope_no': ['100.025', '100.055', '100.095', '200.155', '200.195'],
        'expected': True,
        'description': '正常完成流程'
    },
    {
        'ope_no': ['100.025', '100.055', '100.095', '200.155', '200.190'],
        'expected': True,
        'description': '跳過轉換（SKIP）'
    },
    {
        'ope_no': ['100.025', '100.055', '200.155', '200.195'],
        'expected': False,
        'description': '缺少 100.095 (TRANSFER_COMPLETE)'
    },
    {
        'ope_no': ['200.195', '200.155', '100.095', '100.055', '100.025'],
        'expected': True,
        'description': '亂序但完整'
    }
]
```

## 結論

核心問題是**正則表達式使用不當**和**PostgreSQL 陣列順序不確定性**。建議使用**集合操作**替代正則表達式，這樣：
- ✅ 不受順序影響
- ✅ 邏輯清晰易懂
- ✅ 性能更好
- ✅ 容易測試和維護
