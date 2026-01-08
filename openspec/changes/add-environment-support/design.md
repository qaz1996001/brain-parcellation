# Design: Environment Support Architecture

## Context

**問題域:**
Brain parcellation 推理系統需要在 production 和 testing 環境間切換，但當前架構缺乏環境感知能力。

**技術棧:**
- Python 3.10
- FastAPI backend
- TensorFlow GPU 推理
- Docker Compose 部署
- pipelinecore 推理框架

**約束條件:**
- 不能破壞現有 API 介面
- 需保持簡單性（遵循 Ken Thompson "最簡單的方案"原則）
- 必須可在腦中理解整個機制（符合 Ken Thompson 可理解性要求）

## Goals / Non-Goals

### Goals
1. 支援 `production` 和 `testing` 兩個明確定義的環境
2. 透過環境變數實現零編譯切換
3. 環境配置清晰可見、可驗證（資料優先設計）
4. 向後相容，預設行為不變

### Non-Goals
1. ❌ 不支援 development, staging 等其他環境（YAGNI）
2. ❌ 不建立複雜的配置管理框架（保持簡單）
3. ❌ 不實作動態環境切換（執行時固定）
4. ❌ 不修改 pipelinecore 核心（影響範圍最小化）

## Decisions

### Decision 1: 環境變數作為單一真相來源

**選擇:** 使用 `ENV` 環境變數控制所有環境相關行為

**理由:**
- **Twelve-Factor App 原則**: 配置儲存於環境變數
- **Ken Thompson 簡單性**: 單一控制點，機制清晰
- **容器友好**: Docker/K8s 原生支援
- **測試友好**: `ENV=testing pytest` 即可

**替代方案:**
- ~~配置檔案切換~~ → 需要檔案管理，增加複雜性
- ~~命令列參數~~ → 不符合 Twelve-Factor，容器不友好

### Decision 2: 三層配置策略

**資料結構設計 (Linus: 資料優先):**

```python
# Layer 1: 環境變數 (最高優先級)
ENV = os.getenv("ENV", "production")

# Layer 2: 環境特定配置映射
ENVIRONMENT_CONFIGS = {
    "production": {
        "data_root": "/data/production",
        "log_level": "INFO",
        "model_path": "/models/production"
    },
    "testing": {
        "data_root": "/data/testing",
        "log_level": "DEBUG",
        "model_path": "/models/testing"
    }
}

# Layer 3: 執行時配置解析
config = ENVIRONMENT_CONFIGS[ENV]
```

**理由:**
- **Linus 資料結構原則**: 先設計資料（配置映射），程式碼自然簡單
- **Knuth 精確性**: 每個環境的邊界清晰定義
- **可測試性**: 配置是資料，可直接檢視和驗證

### Decision 3: 最小影響範圍

**修改策略 (Martin Fowler 演化式設計):**

**Phase 1: 基礎設施 (核心配置)**
- 新增 `back/config/environments.py` - 環境配置定義
- 不修改現有模組，僅添加新模組

**Phase 2: Backend 整合**
- `back/main.py` - 讀取環境配置
- `back/database.py` - 環境感知資料庫路徑
- 漸進式遷移，保持向後相容

**Phase 3: 部署整合**
- 更新啟動腳本支援 `ENV` 參數
- Docker Compose 環境變數注入

**理由:**
- **Martin Fowler 小步驟重構**: 每階段獨立可測試
- **Linus 務實主義**: 先讓基礎動起來，再擴展
- **可回滾性**: 任何階段失敗可快速復原

## Risks / Trade-offs

### Risk 1: 環境變數遺漏導致錯誤環境執行

**風險:** 啟動 production 時忘記設定 `ENV`，使用錯誤配置

**緩解:**
- 預設值為 `production`（最保守選擇）
- 啟動時記錄環境配置到日誌（可審計）
- 添加環境驗證函數：`assert_environment(expected="production")`

```python
def validate_environment(expected: str):
    """Knuth 式驗證：明確、可證明正確"""
    actual = os.getenv("ENV", "production")
    if actual != expected:
        raise EnvironmentMismatchError(
            f"Expected ENV={expected}, got ENV={actual}"
        )
```

### Risk 2: 配置漂移 (Configuration Drift)

**風險:** production 和 testing 配置隨時間產生不預期的差異

**緩解:**
- 配置作為程式碼管理（存於 Git）
- 定義配置 schema，自動驗證（Knuth 精確性）
- 差異項目最小化，僅限必要項（YAGNI）

### Trade-off: 簡單性 vs 靈活性

**選擇:** 優先簡單性

- ✅ 只支援兩個環境
- ✅ 配置寫死在程式碼中（避免複雜的外部配置檔案）
- ✅ 執行時不可變（啟動時決定環境）

**後果:** 若未來需要更多環境，需修改程式碼而非僅改配置檔案

**為何可接受:**
- **Martin Fowler 演化式設計**: 當需求出現時再演化
- **Ken Thompson**: "等需要時再做"比"現在猜測未來"更聰明

## Migration Plan

### Step 1: 準備環境配置模組 (無風險)
```bash
# 新增檔案，不影響現有系統
create back/config/environments.py
create tests/test_environments.py
```

### Step 2: Backend 整合 (可測試)
```bash
# 修改 main.py，添加環境載入
# 添加環境驗證測試
ENV=testing pytest tests/test_backend_env.py
```

### Step 3: 啟動腳本更新 (向後相容)
```bash
# 更新 brain-parcellation-start.sh
# 保持無參數呼叫時的預設行為
./brain-parcellation-start.sh  # 預設 production
./brain-parcellation-start.sh testing  # 明確 testing
```

### Rollback Plan

**任何階段可回滾:**
```bash
git revert <commit-hash>  # 配置模組獨立，移除即回滾
```

**驗證步驟:**
```bash
# 每個階段完成後執行
ENV=production pytest tests/
ENV=testing pytest tests/
```

## Open Questions

~~1. pipelinecore 是否需要環境感知？~~
   - **決策:** 當前 No，透過 config.yaml 路徑外部控制即可（YAGNI）
   - 若未來 pipelinecore 內部需要環境邏輯，再重新評估

~~2. 需要環境切換的審計日誌嗎？~~
   - **決策:** Yes，啟動時記錄 `INFO: Environment=production` 到日誌

3. **待確認:** Docker Compose 是否需要多個 compose 檔案？
   - 選項 A: `docker-compose.yml` + 環境變數覆蓋（推薦）
   - 選項 B: `docker-compose.prod.yml` + `docker-compose.test.yml`
   - **建議:** 選項 A，符合 Twelve-Factor 和簡單性原則

## References

### 設計哲學來源
- **Ken Thompson**: 簡單至上、資料結構優先、由下而上思考
- **Linus Torvalds**: 資料優先設計、務實主義、Good Taste (消除特殊情況)
- **Martin Fowler**: YAGNI、演化式設計、小步驟重構
- **Donald Knuth**: 精確性、可證明正確、文學式程式設計

### 外部標準
- [Twelve-Factor App - Config](https://12factor.net/config)
- [Twelve-Factor App - Dev/Prod Parity](https://12factor.net/dev-prod-parity)
