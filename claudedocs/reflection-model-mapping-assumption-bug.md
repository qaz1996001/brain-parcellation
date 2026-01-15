# Bug 反思：model_mapping[0] 假設錯誤

> 基於 Linus Torvalds 與 Donald Knuth 程式設計哲學的反思

---

## Bug 摘要

**問題**：`_reorder_series_by_config` 函數取 `model_mapping[0]`（第一個配置），假設「通常只有一個配置」

**實際情況**：CMB 模型有兩種有效的 series 組合：
- `["SWAN", "T1BRAVO_AXI"]`
- `["SWAN", "T1FLAIR_AXI"]`

**後果**：當輸入是 `["T1FLAIR_AXI", "SWAN"]` 時，無法正確匹配並排序

---

## 違反的 Linus 原則

### 1. 第二原則：資料結構優先

> "Bad programmers worry about the code. Good programmers worry about data structures and their relationships."

**違反**：
- 沒有先理解 `model_mapping` 的資料結構（是 `List[List[str]]`，不是 `List[str]`）
- 直接假設「取第一個就好」，而非理解資料的完整形態

**教訓**：
```python
# ❌ 錯誤：假設單一配置
config_series = model_mapping[0]

# ✅ 正確：先理解資料結構，遍歷所有配置
for config_series in model_mapping:
    if _is_match(config_series, target_labels):
        return sorted_by_config(config_series)
```

### 2. 第三原則：Good Taste（好品味）- 消除特殊情況

> "當你看到 if/else 處理「特殊情況」，想想有沒有辦法讓它變成「正常情況」"

**違反**：
- 把「第一個配置」當作「正常情況」，其他配置變成「特殊情況」（被忽略）
- 好的設計應該讓**所有配置**都是「正常情況」

**教訓**：
- 當你發現自己在寫「通常」、「大部分情況」的註解時，停下來思考
- 如果有多種情況，應該統一處理，而非偷懶只處理一種

### 3. 第六原則：演化優於設計 - 不要過度簡化

> "不要試圖一次設計出完美的系統"

**諷刺**：這次的問題是**過度簡化**，不是過度設計

**教訓**：
- 簡化必須基於對問題的完整理解
- 「取第一個」看似簡單，實際上是**錯誤**的簡化

---

## 違反的 Knuth 原則

### 1. 第三原則：數學嚴謹性

> "不要猜測——要分析、要證明"

**違反**：
- 註解寫「通常只有一個配置」是**猜測**，不是**驗證**
- 沒有查看 `config.yaml` 確認實際有多少配置

**教訓**：
```python
# ❌ 猜測式註解
# 通常只有一個配置，取第一個就好
config_series = model_mapping[0]

# ✅ 驗證式註解
# config.yaml 定義：CMB 有 2 個配置，Infarct 有 1 個配置
# 必須遍歷所有配置找到匹配的
for config_series in model_mapping:
    ...
```

### 2. 第三原則：精確性

> "每一個變數、每一個邊界條件都要精確定義"

**違反**：
- `model_mapping` 的類型是 `List[List[str]]`
- 取 `[0]` 後變成 `List[str]`
- 沒有考慮「如果第一個配置不匹配怎麼辦」

**教訓**：
- 處理 Collection 時，總是問自己：
  - 是否可能為空？
  - 是否只有一個元素？
  - **是否需要遍歷全部？**

### 3. 第六原則：對錯誤的態度

> "記錄每一個錯誤，分類錯誤"

**錯誤分類**：假設錯誤（Assumption Error）
- **類型**：未驗證的假設
- **根因**：懶惰（只測試 T1BRAVO_AXI，沒測 T1FLAIR_AXI）
- **模式**：Happy Path 測試

---

## 防止類似問題的檢查清單

### 設計階段

- [ ] 我完整理解了輸入資料的結構嗎？
- [ ] 如果是 Collection，我知道它可能有多少元素嗎？
- [ ] 我的「簡化」是基於驗證還是猜測？

### 實作階段

- [ ] 當我寫「通常」、「大部分」時，我驗證過嗎？
- [ ] 取 `[0]` 之前，我考慮過其他元素嗎？
- [ ] 我的 for loop 是否應該遍歷 **全部** 而非 **第一個**？

### 測試階段

- [ ] 我測試了所有可能的輸入組合嗎？
- [ ] 我測試了「非第一個」的配置嗎？
- [ ] config.yaml 中定義了幾種變體？我全部測過了嗎？

---

## 總結

> "Talk is cheap. Show me the code." — Linus

這個 bug 的本質是：

```
假設 > 驗證
懶惰 > 嚴謹
Happy Path > 完整覆蓋
```

正確的做法是：

```
驗證 > 假設（讀完 config.yaml）
嚴謹 > 懶惰（遍歷所有配置）
完整覆蓋 > Happy Path（測試所有變體）
```

---

## 附錄：修復程式碼

```python
# ❌ 原始程式碼（錯誤）
config_series = model_mapping[0] if model_mapping else []

# ✅ 修復程式碼（正確）
for config_series in model_mapping:
    config_labels = {_extract_label(s) for s in config_series}
    if config_labels == target_labels:
        # 找到匹配的配置，按此順序排序
        sorted_paths = _sort_by_config_order(config_series, paths)
        return sorted_paths
```

---

*反思日期：2026-01-13*
*錯誤類型：Assumption Error / Happy Path Testing*
*嚴重程度：Medium（影響 CMB 模型的 T1FLAIR_AXI 變體）*
