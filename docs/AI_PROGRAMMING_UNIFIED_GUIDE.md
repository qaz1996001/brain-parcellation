# Martin Fowler 軟體開發哲學 × 設計模式 - AI 程式設計統一指引

> 整合 GoF 設計模式與 Martin Fowler 軟體工程原則，為 AI 輔助程式設計提供統一指導框架

---

## 一、核心哲學基礎

### 1.1 Martin Fowler 五大支柱

| 支柱 | 原則 | AI 程式設計應用 |
|------|------|----------------|
| **敏捷開發** | 彈性規劃、開放協作 | AI 應支援迭代式開發，而非一次性產出完美代碼 |
| **架構模組化** | 清晰代碼、模組分離 | 生成的代碼應遵循單一職責，便於理解和維護 |
| **重構優先** | 小步驟改變、保持行為 | AI 修改代碼時應遵循重構原則，不改變外部行為 |
| **測試自動化** | 自動化測試、持續交付 | 每次代碼生成應考慮可測試性 |
| **資料管理** | 遷移策略、資料一致性 | 資料層修改需考慮向後相容 |

### 1.2 設計模式核心理念

> "設計模式是針對軟體設計中常見問題的典型解決方案" — Refactoring Guru

**三大分類原則**：
- **創建型 (Creational)**: 控制物件創建，提升靈活性與重用性
- **結構型 (Structural)**: 組織物件組合，形成更大的結構
- **行為型 (Behavioral)**: 管理演算法與物件間的責任分配

---

## 二、創建型模式 → AI 程式設計指引

### 2.1 模式概覽

| 模式 | 意圖 | AI 應用場景 |
|------|------|-------------|
| **Factory Method** | 子類別決定實例化的具體類別 | 依據配置動態選擇實作 |
| **Abstract Factory** | 創建相關物件家族 | 跨平台 UI 元件、資料庫抽象層 |
| **Builder** | 步驟式建構複雜物件 | 複雜配置物件、查詢建構器 |
| **Prototype** | 複製現有物件 | 效能敏感的物件創建 |
| **Singleton** | 確保單一實例 | 配置管理、連線池 |

### 2.2 Fowler 整合原則

```
┌─────────────────────────────────────────────────────────────┐
│  創建邏輯分離原則                                            │
│  ═══════════════                                            │
│  • 使用 Factory 隔離 new 關鍵字                              │
│  • Builder 用於 >3 個建構參數的物件                          │
│  • Singleton 需謹慎：考慮依賴注入替代                        │
│  • 測試友善：所有創建模式應支援 mock 注入                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 AI 代碼生成規則

```python
# ✅ 推薦：Factory Method 範例
class NotificationFactory:
    @staticmethod
    def create(channel: str) -> Notification:
        factories = {
            "email": EmailNotification,
            "sms": SMSNotification,
            "push": PushNotification,
        }
        return factories.get(channel, EmailNotification)()

# ❌ 避免：硬編碼創建邏輯
def send_notification(channel, message):
    if channel == "email":
        EmailNotification().send(message)  # 緊耦合
    elif channel == "sms":
        SMSNotification().send(message)
```

---

## 三、結構型模式 → AI 程式設計指引

### 3.1 模式概覽

| 模式 | 意圖 | AI 應用場景 |
|------|------|-------------|
| **Adapter** | 讓不相容介面協作 | 第三方 API 整合、Legacy 系統遷移 |
| **Bridge** | 分離抽象與實作 | 跨平台渲染、資料庫驅動切換 |
| **Composite** | 樹狀結構統一處理 | 檔案系統、組織架構、UI 元件樹 |
| **Decorator** | 動態附加行為 | 日誌、快取、權限檢查 |
| **Facade** | 簡化複雜子系統 | SDK 封裝、微服務 Gateway |
| **Flyweight** | 共享狀態節省記憶體 | 大量相似物件（文字編輯器字元） |
| **Proxy** | 控制物件存取 | 延遲載入、存取控制、請求日誌 |

### 3.2 Fowler 整合原則

```
┌─────────────────────────────────────────────────────────────┐
│  介面隔離與組合原則                                          │
│  ═══════════════════                                        │
│  • Adapter 處理外部整合邊界                                  │
│  • Facade 隱藏子系統複雜度（符合 Fowler 模組化原則）          │
│  • Decorator 優於繼承（組合優於繼承）                        │
│  • Proxy 實現橫切關注點（AOP 思維）                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 AI 代碼生成規則

```python
# ✅ 推薦：Decorator 實現橫切關注點
def with_retry(max_attempts: int = 3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(2 ** attempt)
            return wrapper
    return decorator

@with_retry(max_attempts=3)
def fetch_data(url: str) -> dict:
    return requests.get(url).json()

# ❌ 避免：在每個函數中重複重試邏輯
```

---

## 四、行為型模式 → AI 程式設計指引

### 4.1 模式概覽

| 模式 | 意圖 | AI 應用場景 |
|------|------|-------------|
| **Chain of Responsibility** | 請求沿鏈傳遞 | 中間件、驗證管道、事件處理 |
| **Command** | 請求封裝為物件 | 撤銷/重做、任務佇列、巨集錄製 |
| **Iterator** | 遍歷集合不暴露內部 | 自訂資料結構遍歷 |
| **Mediator** | 減少物件間直接依賴 | 聊天室、表單元件協調 |
| **Memento** | 保存/恢復物件狀態 | 編輯器歷史、遊戲存檔 |
| **Observer** | 訂閱機制通知變更 | 事件系統、響應式 UI |
| **State** | 狀態改變行為 | 訂單狀態機、工作流程引擎 |
| **Strategy** | 演算法族可互換 | 支付方式、排序策略、壓縮演算法 |
| **Template Method** | 定義演算法骨架 | 框架擴展點、資料處理管道 |
| **Visitor** | 分離演算法與物件結構 | AST 處理、報表生成 |

### 4.2 Fowler 整合原則

```
┌─────────────────────────────────────────────────────────────┐
│  職責分離與事件驅動原則                                      │
│  ═════════════════════                                      │
│  • Strategy 實現「開放封閉原則」                              │
│  • Observer 支援鬆耦合的事件驅動架構                         │
│  • State 管理有限狀態機（明確狀態轉換）                      │
│  • Command 支援撤銷與審計追蹤（符合 Fowler 可追溯原則）      │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 AI 代碼生成規則

```python
# ✅ 推薦：Strategy 模式實現可擴展演算法
from abc import ABC, abstractmethod
from typing import Protocol

class PaymentStrategy(Protocol):
    def pay(self, amount: float) -> bool: ...

class CreditCardPayment:
    def pay(self, amount: float) -> bool:
        # 信用卡支付邏輯
        return True

class PayPalPayment:
    def pay(self, amount: float) -> bool:
        # PayPal 支付邏輯
        return True

class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self._strategy = strategy

    def process(self, amount: float) -> bool:
        return self._strategy.pay(amount)

# 使用：輕鬆切換支付策略
processor = PaymentProcessor(CreditCardPayment())
processor.process(100.0)
```

---

## 五、企業應用架構模式 (Fowler PoEAA)

### 5.1 六大架構關注點

| 關注點 | 核心模式 | AI 設計考量 |
|--------|----------|-------------|
| **分層架構** | Presentation → Domain → Data Source | 生成代碼時維持層次邊界 |
| **領域邏輯** | Transaction Script, Domain Model, Table Module | 根據複雜度選擇適當模式 |
| **資料庫映射** | Active Record, Data Mapper, Repository | ORM 選型與資料存取抽象 |
| **Web 呈現** | MVC, Front Controller, Application Controller | 前後端分離架構 |
| **分散式設計** | Remote Facade, DTO | API 設計與資料傳輸最佳化 |
| **離線併發** | Optimistic/Pessimistic Locking, Unit of Work | 併發控制策略 |

### 5.2 Fowler 分層原則

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│        (Controllers, Views, API Endpoints)                  │
├─────────────────────────────────────────────────────────────┤
│                     Domain Layer                            │
│     (Business Logic, Domain Models, Services)               │
├─────────────────────────────────────────────────────────────┤
│                   Data Source Layer                         │
│        (Repositories, Data Mappers, ORM)                    │
└─────────────────────────────────────────────────────────────┘

原則：
• 上層可依賴下層，反向禁止
• 跨層呼叫使用介面抽象
• 每層職責單一且明確
```

---

## 六、AI 程式設計統一規則

### 6.1 代碼生成核心原則

| # | 原則 | 說明 | 對應模式/理念 |
|---|------|------|--------------|
| 1 | **小步重構** | 每次修改保持行為不變，逐步改善結構 | Fowler Refactoring |
| 2 | **依賴注入** | 避免硬編碼依賴，使用介面和工廠 | Factory, DI |
| 3 | **組合優於繼承** | 使用 Decorator/Strategy 而非深層繼承 | GoF 原則 |
| 4 | **介面隔離** | 定義小而專注的介面 | Adapter, Facade |
| 5 | **單一職責** | 每個類別/函數只做一件事 | Fowler 模組化 |
| 6 | **開放封閉** | 對擴展開放，對修改封閉 | Strategy, Template |
| 7 | **可測試性** | 所有代碼應可獨立測試 | Fowler 測試優先 |
| 8 | **明確狀態** | 使用 State/Memento 管理複雜狀態 | 行為型模式 |

### 6.2 AI 輔助決策樹

```
需要創建物件？
├── 單一類型 → Factory Method
├── 物件家族 → Abstract Factory
├── 複雜建構 → Builder
├── 複製現有 → Prototype
└── 全域單例 → Singleton (謹慎使用)

需要組織結構？
├── 整合外部 → Adapter
├── 簡化複雜 → Facade
├── 動態擴展 → Decorator
├── 控制存取 → Proxy
└── 樹狀組合 → Composite

需要管理行為？
├── 演算法可換 → Strategy
├── 狀態驅動 → State
├── 請求管道 → Chain of Responsibility
├── 事件通知 → Observer
├── 撤銷支援 → Command + Memento
└── 遍歷集合 → Iterator
```

### 6.3 代碼品質檢查清單

```markdown
□ 是否遵循分層架構？
□ 創建邏輯是否與使用邏輯分離？
□ 是否使用介面而非具體類別？
□ 是否支援依賴注入進行測試？
□ 複雜狀態是否有明確的狀態機？
□ 是否避免了 God Class？
□ 橫切關注點是否使用 Decorator/Proxy？
□ 外部整合是否使用 Adapter 隔離？
```

---

## 七、AI 時代的 Fowler 觀點

### 7.1 LLM 與軟體設計 (2025-2026)

> "AI 正在改變軟體開發：從確定性到非確定性編程的轉變"

**Fowler 最新觀點**：
1. **LLM 輸出必須嚴格測試** — 非確定性代碼需要更強的驗證
2. **重構比以往更重要** — AI 生成的代碼需要人類重構以符合架構
3. **AI + 確定性技術結合** — 工程團隊需要的是混合方法

### 7.2 AI 輔助開發工作流程

```
┌──────────────────────────────────────────────────────────────┐
│  AI 輔助開發循環                                              │
│  ════════════════                                            │
│                                                              │
│  1. 理解需求 → 2. AI 生成初稿 → 3. 人類審查架構               │
│       ↑                              ↓                       │
│       │                        4. 重構對齊模式                │
│       │                              ↓                       │
│       └──────── 6. 迭代 ← 5. 自動化測試驗證                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 7.3 持久原則

> "軟體架構中的基本問題和解決方案實際上不會有太大變化"

**不變的核心**：
- 清晰的代碼勝過聰明的代碼
- 模組化設計支持演進
- 測試是變更的安全網
- 重構是持續的過程，不是一次性事件

---

## 八、快速參考卡

### 設計模式速查

| 類別 | 模式 | 一句話意圖 |
|------|------|-----------|
| 創建 | Factory Method | 子類決定創建什麼 |
| 創建 | Abstract Factory | 創建相關物件家族 |
| 創建 | Builder | 分步建構複雜物件 |
| 創建 | Prototype | 複製現有物件 |
| 創建 | Singleton | 確保單一實例 |
| 結構 | Adapter | 轉換不相容介面 |
| 結構 | Bridge | 分離抽象與實作 |
| 結構 | Composite | 樹狀結構統一處理 |
| 結構 | Decorator | 動態附加行為 |
| 結構 | Facade | 簡化複雜子系統 |
| 結構 | Flyweight | 共享狀態省記憶體 |
| 結構 | Proxy | 控制物件存取 |
| 行為 | Chain of Resp. | 請求沿鏈傳遞 |
| 行為 | Command | 請求封裝為物件 |
| 行為 | Iterator | 遍歷不暴露內部 |
| 行為 | Mediator | 減少直接依賴 |
| 行為 | Memento | 保存恢復狀態 |
| 行為 | Observer | 訂閱通知機制 |
| 行為 | State | 狀態改變行為 |
| 行為 | Strategy | 演算法可互換 |
| 行為 | Template | 定義演算法骨架 |
| 行為 | Visitor | 分離演算法與結構 |

### Fowler 原則速查

| 原則 | 核心思想 |
|------|----------|
| 重構 | 小步驟、保持行為、持續改善 |
| 模組化 | 清晰邊界、單一職責、可理解片段 |
| 測試 | 自動化、文檔化、安全網 |
| 敏捷 | 迭代、協作、適應變化 |
| 分層 | 職責分離、依賴方向、介面抽象 |

---

## 九、AI Coding 工作流程詳解

### 9.1 Prompt Engineering 三層指引

#### 9.1.1 基本原則層

> "好的 Prompt 是具體的、可驗證的、且包含足夠上下文"

| 原則 | 說明 | 範例 |
|------|------|------|
| **具體性 (Specific)** | 避免模糊描述，明確說明期望輸出 | ❌ "幫我優化程式碼" → ✅ "將這個 O(n²) 排序改為 O(n log n)" |
| **可驗證性 (Verifiable)** | 提供可判斷成功與否的標準 | ❌ "寫更好的程式碼" → ✅ "確保所有公開方法有 docstring" |
| **上下文完整 (Contextual)** | 提供相關背景、限制條件 | ❌ "新增登入功能" → ✅ "使用 JWT 新增登入功能，需相容現有 User model" |
| **範圍界定 (Scoped)** | 明確界定修改範圍 | ❌ "重構這個專案" → ✅ "重構 auth/ 目錄下的認證邏輯" |

```markdown
# 基本 Prompt 結構模板

## 任務描述
[明確說明要做什麼]

## 上下文
[相關背景、現有架構、限制條件]

## 期望輸出
[具體的輸出格式、品質標準]

## 成功標準
[如何判斷任務完成]
```

#### 9.1.2 進階技巧層

**Chain-of-Thought (思維鏈)**

引導 AI 逐步思考，適用於複雜邏輯問題：

```markdown
分析這個效能問題，請按以下步驟：

1. 首先，識別瓶頸位置（哪個函數耗時最長？）
2. 接著，分析該函數的時間複雜度
3. 然後，提出 2-3 個可能的優化方案
4. 最後，推薦最佳方案並說明理由
```

**Few-Shot Examples (少量範例)**

提供範例讓 AI 理解期望格式：

```markdown
請按照以下格式為函數添加 docstring：

範例輸入：
def add(a, b):
    return a + b

範例輸出：
def add(a: int, b: int) -> int:
    """
    計算兩數之和。

    Args:
        a: 第一個加數
        b: 第二個加數

    Returns:
        兩數相加的結果

    Example:
        >>> add(2, 3)
        5
    """
    return a + b

現在請處理以下函數：
[待處理的函數]
```

**角色設定 (Role Setting)**

賦予 AI 特定角色以獲得專業視角：

```markdown
# 安全審查角色
你是一位資深安全工程師，專注於 OWASP Top 10 漏洞。
請審查以下程式碼的安全性，特別關注：
- SQL 注入
- XSS 攻擊
- 認證繞過

# 架構師角色
你是一位熟悉 DDD 和微服務的架構師。
請評估這個設計方案的可擴展性和維護性。
```

#### 9.1.3 場景化策略矩陣

| 場景 | 推薦技巧 | 適用情況 |
|------|----------|----------|
| **Bug 修復** | 具體性 + 上下文 | 需要精確定位和修復 |
| **新功能開發** | Chain-of-Thought + 角色設定 | 需要設計思考 |
| **程式碼重構** | Few-Shot + 可驗證性 | 需要一致的風格轉換 |
| **程式碼審查** | 角色設定 + 檢查清單 | 需要多角度審視 |
| **效能優化** | Chain-of-Thought + 具體性 | 需要分析和比較 |
| **測試生成** | Few-Shot + 範圍界定 | 需要一致的測試風格 |

---

### 9.2 場景化 Prompt 模板

#### 9.2.1 Bug Fix 模板

```markdown
## Bug 修復請求

### 問題描述
- **症狀**: [具體錯誤訊息或異常行為]
- **重現步驟**: [1. 做了什麼 2. 發生了什麼]
- **期望行為**: [應該發生什麼]

### 相關程式碼位置
- 檔案: `[路徑/檔名]`
- 函數/類別: `[名稱]`
- 行號: [約略範圍]

### 已嘗試的方案
- [方案1]: [結果]
- [方案2]: [結果]

### 環境資訊
- Python/Node 版本: [版本]
- 相關套件版本: [套件列表]

### 限制條件
- [ ] 不能改變公開 API
- [ ] 需要向後相容
- [ ] 效能不能降低

### 成功標準
- [ ] 錯誤不再出現
- [ ] 現有測試通過
- [ ] 新增迴歸測試
```

#### 9.2.2 Feature Implementation 模板

```markdown
## 功能實作請求

### 需求摘要
[一句話描述這個功能要做什麼]

### 使用者故事
作為 [角色]
我想要 [功能]
以便 [價值/目的]

### 技術規格
- **輸入**: [資料格式、來源]
- **輸出**: [資料格式、目的地]
- **觸發條件**: [何時執行]

### 架構考量
- 應放置於: `[模組/目錄]`
- 依賴服務: `[列出依賴]`
- 資料模型: `[相關 Model]`

### 設計模式建議
根據需求特性，考慮使用：
- [ ] Strategy 模式（如有多種演算法）
- [ ] Factory 模式（如需動態創建物件）
- [ ] Observer 模式（如需事件通知）

### 非功能需求
- 效能目標: [回應時間/吞吐量]
- 安全要求: [認證/授權]
- 可擴展性: [預期規模]

### 驗收標準
- [ ] [標準1]
- [ ] [標準2]
- [ ] 單元測試覆蓋率 > 80%
```

#### 9.2.3 Code Refactoring 模板

```markdown
## 重構請求

### 重構目標
- **目標程式碼**: `[檔案路徑或模組名]`
- **重構類型**: [結構重構 / 效能優化 / 可讀性改善]

### 當前問題
```
[描述目前程式碼的問題]
例如：
- 函數超過 50 行
- 重複程式碼出現 3 次以上
- 循環依賴
- 違反單一職責原則
```

### 期望改善
```
[描述重構後的預期狀態]
例如：
- 每個函數不超過 20 行
- 抽取共用邏輯到 utility 模組
- 解除循環依賴
- 分離讀取和寫入職責
```

### 重構約束
- [ ] 不改變外部 API 簽名
- [ ] 不改變現有行為（純重構）
- [ ] 所有現有測試必須通過
- [ ] 逐步提交，每個 commit 可獨立 review

### 參考的重構手法
- [ ] Extract Method
- [ ] Move Method
- [ ] Replace Conditional with Polymorphism
- [ ] Introduce Parameter Object

### 驗證方式
- [ ] 重構前後測試結果一致
- [ ] 效能基準測試無退化
- [ ] 程式碼複雜度降低（Cyclomatic Complexity）
```

#### 9.2.4 Code Review Request 模板

```markdown
## 程式碼審查請求

### 審查範圍
- **檔案**: `[檔案列表]`
- **變更類型**: [新功能 / Bug 修復 / 重構 / 配置變更]
- **變更規模**: [~XX 行]

### 變更摘要
[簡述這次變更做了什麼]

### 審查重點
請特別關注以下面向：
- [ ] **正確性**: 邏輯是否正確處理所有情況
- [ ] **安全性**: 是否有潛在安全風險
- [ ] **效能**: 是否有效能問題
- [ ] **可維護性**: 程式碼是否易讀易改
- [ ] **測試**: 測試是否充分

### 已知風險或權衡
[說明你已知的限制或做的權衡決定]

### 設計決策說明
[說明為何選擇這個方案而非其他方案]

### 期望回饋
- [ ] 架構層面建議
- [ ] 實作細節改進
- [ ] 測試覆蓋建議
- [ ] 文件補充建議
```

#### 9.2.5 Test Generation 模板

```markdown
## 測試生成請求

### 待測試目標
- **檔案**: `[檔案路徑]`
- **函數/類別**: `[名稱列表]`
- **測試框架**: [pytest / unittest / jest]

### 測試策略
- [ ] 單元測試（隔離測試）
- [ ] 整合測試（跨模組）
- [ ] 端對端測試（完整流程）

### 測試案例類型
請為每個函數產生以下類型的測試：

1. **Happy Path** - 正常情況
2. **Edge Cases** - 邊界條件
   - 空值/None
   - 空集合
   - 最大/最小值
3. **Error Cases** - 錯誤處理
   - 無效輸入
   - 例外情況
4. **Regression** - 迴歸測試（如有 bug）

### Mock 需求
需要 Mock 的外部依賴：
- [ ] 資料庫操作
- [ ] 外部 API 呼叫
- [ ] 檔案系統
- [ ] 時間相關函數

### 測試風格範例
```python
# 請按照此風格撰寫測試
class TestUserService:
    """使用者服務測試"""

    def test_create_user_success(self):
        """成功建立使用者"""
        # Arrange
        user_data = {"name": "test", "email": "test@example.com"}

        # Act
        result = user_service.create(user_data)

        # Assert
        assert result.id is not None
        assert result.name == "test"
```

### 覆蓋率目標
- 行覆蓋率: >= 80%
- 分支覆蓋率: >= 70%
```

---

### 9.3 AI 代碼審查清單

#### 9.3.1 正確性檢查 (Correctness)

| # | 檢查項目 | 說明 | 嚴重度 |
|---|----------|------|--------|
| C1 | **邏輯正確性** | 程式碼是否正確實作需求？ | Critical |
| C2 | **邊界條件處理** | null/空值/0/負數/最大值是否正確處理？ | High |
| C3 | **錯誤處理完整性** | 所有異常情況是否有適當處理？ | High |
| C4 | **競態條件** | 多執行緒/非同步情況是否安全？ | High |
| C5 | **資源釋放** | 檔案、連線、記憶體是否正確釋放？ | Medium |

```python
# ❌ 缺少邊界條件處理
def divide(a, b):
    return a / b  # b=0 會拋出異常

# ✅ 完整的邊界條件處理
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("除數不能為零")
    return a / b
```

#### 9.3.2 安全性檢查 (Security)

| # | 檢查項目 | 說明 | 嚴重度 |
|---|----------|------|--------|
| S1 | **SQL 注入** | 是否使用參數化查詢？ | Critical |
| S2 | **XSS 防護** | 使用者輸入是否正確跳脫？ | Critical |
| S3 | **硬編碼機密** | 密碼/API Key 是否寫死在程式碼中？ | Critical |
| S4 | **權限檢查** | 敏感操作是否驗證權限？ | High |
| S5 | **輸入驗證** | 所有外部輸入是否驗證？ | High |

```python
# ❌ SQL 注入風險
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ 參數化查詢
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

#### 9.3.3 可維護性檢查 (Maintainability)

| # | 檢查項目 | 說明 | 嚴重度 |
|---|----------|------|--------|
| M1 | **命名清晰度** | 變數/函數名稱是否表達意圖？ | Medium |
| M2 | **函數長度** | 單一函數是否超過 30 行？ | Medium |
| M3 | **循環複雜度** | 巢狀層數是否超過 4 層？ | Medium |
| M4 | **重複程式碼** | 是否有超過 5 行的重複區塊？ | Medium |
| M5 | **魔術數字** | 是否有未命名的常數？ | Low |
| M6 | **註解品質** | 複雜邏輯是否有適當註解？ | Low |

```python
# ❌ 魔術數字 + 命名不清
def calc(x):
    return x * 1.08 if x > 100 else x * 1.05

# ✅ 命名清晰 + 常數定義
PREMIUM_TAX_RATE = 1.08
STANDARD_TAX_RATE = 1.05
PREMIUM_THRESHOLD = 100

def calculate_price_with_tax(base_price: float) -> float:
    """計算含稅價格，超過門檻使用高稅率"""
    tax_rate = PREMIUM_TAX_RATE if base_price > PREMIUM_THRESHOLD else STANDARD_TAX_RATE
    return base_price * tax_rate
```

#### 9.3.4 效能檢查 (Performance)

| # | 檢查項目 | 說明 | 嚴重度 |
|---|----------|------|--------|
| P1 | **演算法複雜度** | 是否有不必要的 O(n²) 或更高複雜度？ | High |
| P2 | **N+1 查詢** | 是否在迴圈中執行資料庫查詢？ | High |
| P3 | **記憶體使用** | 大資料集是否使用串流/分批處理？ | Medium |
| P4 | **快取使用** | 重複計算是否可以快取？ | Medium |
| P5 | **非同步處理** | I/O 操作是否適當非同步化？ | Medium |

```python
# ❌ N+1 查詢問題
orders = Order.objects.all()
for order in orders:
    print(order.customer.name)  # 每次迴圈都查詢 customer

# ✅ 使用 select_related 預載入
orders = Order.objects.select_related('customer').all()
for order in orders:
    print(order.customer.name)  # 已預載入，無額外查詢
```

#### 9.3.5 風格一致性檢查 (Consistency)

| # | 檢查項目 | 說明 | 嚴重度 |
|---|----------|------|--------|
| T1 | **格式規範** | 是否符合專案的 linting 規則？ | Low |
| T2 | **命名慣例** | 是否遵循專案的命名規範？ | Low |
| T3 | **Import 順序** | 是否按標準順序排列？ | Low |
| T4 | **文件結構** | 是否符合專案的目錄結構慣例？ | Low |

#### 9.3.6 完整審查清單摘要

```markdown
## AI 代碼審查檢查清單 (Quick Reference)

### Critical (必須修正)
□ C1 邏輯正確性
□ S1 SQL 注入防護
□ S2 XSS 防護
□ S3 無硬編碼機密

### High (建議修正)
□ C2 邊界條件處理
□ C3 錯誤處理完整性
□ C4 競態條件安全
□ S4 權限檢查
□ S5 輸入驗證
□ P1 演算法複雜度合理
□ P2 無 N+1 查詢

### Medium (可選改善)
□ C5 資源正確釋放
□ M1 命名清晰
□ M2 函數長度 < 30 行
□ M3 巢狀 < 4 層
□ M4 無重複程式碼
□ P3 大資料分批處理
□ P4 適當使用快取
□ P5 I/O 非同步化

### Low (風格建議)
□ M5 無魔術數字
□ M6 複雜邏輯有註解
□ T1-T4 風格一致性
```

---

### 9.4 AI 輔助開發完整循環

#### 9.4.1 七階段開發循環圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AI 輔助開發完整循環                                   │
│═════════════════════════════════════════════════════════════════════════│
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ 1. 需求  │───▶│ 2. 設計  │───▶│ 3. 實作  │───▶│ 4. 審查  │          │
│  │   分析   │    │   規劃   │    │   生成   │    │   驗證   │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│       │              │              │              │                    │
│       │ 人機協作     │ 人機協作     │ AI 主導      │ 人類主導           │
│       ▼              ▼              ▼              ▼                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ 品質閘門 │    │ 品質閘門 │    │ 品質閘門 │    │ 品質閘門 │          │
│  │ QG-1     │    │ QG-2     │    │ QG-3     │    │ QG-4     │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                          │
│  │ 5. 測試  │───▶│ 6. 重構  │───▶│ 7. 文件  │──┐                       │
│  │   驗證   │    │   優化   │    │   完成   │  │                       │
│  └──────────┘    └──────────┘    └──────────┘  │                       │
│       │              │              │          │                       │
│       │ AI 輔助      │ 人機協作     │ AI 輔助  │                       │
│       ▼              ▼              ▼          ▼                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  迭代                    │
│  │ 品質閘門 │    │ 品質閘門 │    │ 品質閘門 │  回到                    │
│  │ QG-5     │    │ QG-6     │    │ QG-7     │  第一步                  │
│  └──────────┘    └──────────┘    └──────────┘                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 9.4.2 各階段詳解

**階段 1: 需求分析 (Requirement Analysis)**

| 項目 | 說明 |
|------|------|
| **協作模式** | 人機協作 - 人類主導意圖，AI 協助釐清 |
| **AI 職責** | 提問澄清、識別潛在需求、建議範圍界定 |
| **人類職責** | 提供業務背景、優先級判斷、最終決策 |
| **輸入** | 原始需求描述、使用者故事 |
| **輸出** | 結構化需求文件、驗收標準 |

```markdown
# 品質閘門 QG-1: 需求完整性檢查

□ 需求是否可測試？（有明確的驗收標準）
□ 範圍是否明確界定？（知道什麼在範圍內/外）
□ 依賴項是否識別？（需要什麼前置條件）
□ 非功能需求是否考慮？（效能、安全、可用性）
```

**階段 2: 設計規劃 (Design Planning)**

| 項目 | 說明 |
|------|------|
| **協作模式** | 人機協作 - AI 提案，人類審核 |
| **AI 職責** | 提供設計方案、識別設計模式、評估技術選型 |
| **人類職責** | 架構決策、風險評估、模式選擇確認 |
| **輸入** | 結構化需求、現有架構約束 |
| **輸出** | 技術設計文件、API 設計、資料模型 |

```markdown
# 品質閘門 QG-2: 設計合理性檢查

□ 是否符合現有架構風格？
□ 是否遵循 SOLID 原則？
□ 依賴方向是否正確？（高層不依賴低層）
□ 是否考慮了擴展性？
□ 是否有適當的抽象層次？
```

**階段 3: 實作生成 (Implementation)**

| 項目 | 說明 |
|------|------|
| **協作模式** | AI 主導 - AI 生成，人類監督 |
| **AI 職責** | 生成程式碼、遵循設計規範、套用設計模式 |
| **人類職責** | 監督品質、提供修正指導、確認關鍵邏輯 |
| **輸入** | 技術設計、場景化 Prompt 模板 |
| **輸出** | 實作程式碼、初步單元測試 |

```markdown
# 品質閘門 QG-3: 程式碼生成品質檢查

□ 程式碼是否可編譯/執行？
□ 是否符合專案程式碼風格？
□ 是否遵循設計文件？
□ 是否有適當的錯誤處理？
□ 是否通過靜態分析（lint）？
```

**階段 4: 審查驗證 (Review & Verification)**

| 項目 | 說明 |
|------|------|
| **協作模式** | 人類主導 - 人類審查，AI 輔助分析 |
| **AI 職責** | 自動化檢查、提供審查建議、識別潛在問題 |
| **人類職責** | 邏輯審查、安全審查、最終核准 |
| **輸入** | 生成的程式碼、AI 代碼審查清單 |
| **輸出** | 審查報告、修改建議 |

```markdown
# 品質閘門 QG-4: 審查通過標準

□ 所有 Critical 項目通過
□ 所有 High 項目通過或有例外說明
□ 至少 80% 的 Medium 項目通過
□ 安全性無紅旗警告
□ 程式碼審查已完成並核准
```

**階段 5: 測試驗證 (Testing)**

| 項目 | 說明 |
|------|------|
| **協作模式** | AI 輔助 - AI 生成測試，人類補充邊界案例 |
| **AI 職責** | 生成測試案例、識別邊界條件、執行測試 |
| **人類職責** | 定義測試策略、審查測試品質、處理複雜場景 |
| **輸入** | 實作程式碼、測試生成 Prompt 模板 |
| **輸出** | 測試程式碼、測試報告、覆蓋率報告 |

```markdown
# 品質閘門 QG-5: 測試覆蓋標準

□ 單元測試覆蓋率 >= 80%
□ 所有公開 API 有測試
□ 邊界條件有測試案例
□ 錯誤路徑有測試案例
□ 整合測試通過
```

**階段 6: 重構優化 (Refactoring)**

| 項目 | 說明 |
|------|------|
| **協作模式** | 人機協作 - AI 建議，人類決策 |
| **AI 職責** | 識別技術債、建議重構手法、執行重構 |
| **人類職責** | 判斷重構必要性、優先級排序、驗證行為不變 |
| **輸入** | 通過測試的程式碼、重構 Prompt 模板 |
| **輸出** | 重構後的程式碼（行為不變） |

```markdown
# 品質閘門 QG-6: 重構完成標準

□ 所有現有測試通過（行為不變）
□ 程式碼複雜度降低或維持
□ 無新增技術債
□ 效能基準測試無退化
□ 遵循小步重構原則（可追溯）
```

**階段 7: 文件完成 (Documentation)**

| 項目 | 說明 |
|------|------|
| **協作模式** | AI 輔助 - AI 生成，人類審核 |
| **AI 職責** | 生成 API 文件、更新 README、產生使用範例 |
| **人類職責** | 審核準確性、補充業務知識、最終發布 |
| **輸入** | 完成的程式碼、文件模板 |
| **輸出** | API 文件、使用說明、變更日誌 |

```markdown
# 品質閘門 QG-7: 文件完整性檢查

□ API 文件與程式碼一致
□ 使用範例可執行
□ 變更日誌已更新
□ README 反映最新狀態
□ 內部 docstring 完整
```

#### 9.4.3 人機協作矩陣

```
┌─────────────────────────────────────────────────────────────────┐
│                    人機協作責任分配矩陣                           │
├──────────────┬──────────────────────┬──────────────────────────┤
│    階段      │      AI 職責         │      人類職責            │
├──────────────┼──────────────────────┼──────────────────────────┤
│ 1. 需求分析  │ ○ 輔助釐清           │ ● 主導決策               │
│ 2. 設計規劃  │ ◐ 提供方案           │ ◐ 審核決策               │
│ 3. 實作生成  │ ● 主導生成           │ ○ 監督品質               │
│ 4. 審查驗證  │ ○ 自動檢查           │ ● 人工審核               │
│ 5. 測試驗證  │ ◐ 生成測試           │ ◐ 補充案例               │
│ 6. 重構優化  │ ◐ 建議執行           │ ◐ 決策驗證               │
│ 7. 文件完成  │ ◐ 生成初稿           │ ○ 審核發布               │
├──────────────┴──────────────────────┴──────────────────────────┤
│ ● 主導 (>70%)  ◐ 協作 (30-70%)  ○ 輔助 (<30%)                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 9.4.4 常見反模式與對策

| 反模式 | 問題描述 | 正確做法 |
|--------|----------|----------|
| **跳過設計** | 需求直接進入實作，無設計階段 | 遵循完整循環，設計是必要階段 |
| **過度信任 AI** | 不審查 AI 生成的程式碼 | 人類審查是品質閘門，不可跳過 |
| **無測試部署** | 未經測試就部署 | 測試覆蓋率是硬性標準 |
| **忽略重構** | 功能完成後不進行重構 | 重構是維護長期程式碼健康的必要步驟 |
| **文件滯後** | 程式碼完成但文件未更新 | 文件與程式碼同步更新 |

#### 9.4.5 迭代與持續改善

```
                    ┌────────────────────────────┐
                    │     持續改善循環           │
                    └────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   收集回饋 ──→ 分析問題 ──→ 調整流程 ──→ 驗證效果   │
    │       ↑                                      │       │
    │       └──────────────────────────────────────┘       │
    │                                                      │
    └──────────────────────────────────────────────────────┘

回饋來源：
• 程式碼審查紀錄
• 測試失敗模式
• 重構頻率分析
• 生產環境事故
```

---

## 十、測試策略指引

本章節提供 AI 輔助開發情境下的測試策略指導，涵蓋測試金字塔配置、AI 生成測試的驗證方法，以及測試驅動開發的工作流程。

---

## 10.1 測試金字塔（按專案類型）

### 核心原則

測試金字塔的配置應依據專案特性調整。底層單元測試執行快速、維護成本低；頂層端對端測試接近真實使用情境但成本高昂。

```
        /\
       /  \     E2E（使用者流程驗證）
      /----\
     /      \   Integration（模組互動驗證）
    /--------\
   /          \  Unit（單一函式/類別驗證）
  /____________\
```

### 專案類型配置矩陣

| 專案類型 | Unit | Integration | E2E | 關鍵考量 |
|---------|------|-------------|-----|---------|
| 微服務 API | 60% | 30% | 10% | API 契約穩定性優先 |
| 前端 SPA | 40% | 40% | 20% | 使用者互動路徑覆蓋 |
| CLI 工具 | 70% | 20% | 10% | 參數組合與輸出格式 |
| 資料處理 Pipeline | 50% | 35% | 15% | 資料轉換正確性 |

### 10.1.1 微服務 API（Unit 60% / Integration 30% / E2E 10%）

**適用情境**：REST/GraphQL API、gRPC 服務、後端業務邏輯

**測試重點分布**：

```yaml
unit_tests:
  coverage_target: 60%
  focus:
    - 業務邏輯函式
    - 資料驗證規則
    - 錯誤處理路徑
    - 工具函式與轉換器

integration_tests:
  coverage_target: 30%
  focus:
    - API 端點行為
    - 資料庫操作（使用測試容器）
    - 外部服務互動（使用 mock server）
    - 認證授權流程

e2e_tests:
  coverage_target: 10%
  focus:
    - 關鍵業務流程
    - 跨服務調用鏈
    - 部署驗證（smoke test）
```

**範例結構**：

```
tests/
├── unit/
│   ├── services/
│   │   └── test_order_service.py      # 業務邏輯
│   ├── validators/
│   │   └── test_order_validator.py    # 驗證規則
│   └── utils/
│       └── test_price_calculator.py   # 工具函式
├── integration/
│   ├── api/
│   │   └── test_order_endpoints.py    # API 行為
│   └── repositories/
│       └── test_order_repository.py   # 資料庫操作
└── e2e/
    └── test_order_flow.py             # 完整訂單流程
```

### 10.1.2 前端 SPA（Unit 40% / Integration 40% / E2E 20%）

**適用情境**：React、Vue、Angular 應用程式

**測試重點分布**：

```yaml
unit_tests:
  coverage_target: 40%
  focus:
    - 純函式與工具
    - 狀態管理邏輯（reducers、selectors）
    - 自定義 hooks
    - 格式化與轉換函式

integration_tests:
  coverage_target: 40%
  focus:
    - 元件互動行為
    - 表單提交流程
    - API 整合（使用 MSW）
    - 路由切換

e2e_tests:
  coverage_target: 20%
  focus:
    - 使用者核心流程
    - 跨頁面操作
    - 認證流程
    - 關鍵錯誤處理
```

**範例：React Testing Library 整合測試**：

```typescript
// tests/integration/OrderForm.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { server } from '../mocks/server';
import { rest } from 'msw';
import { OrderForm } from '@/components/OrderForm';

describe('OrderForm', () => {
  it('submits order and shows confirmation', async () => {
    const user = userEvent.setup();
    render(<OrderForm />);

    // 填寫表單
    await user.type(screen.getByLabelText('商品名稱'), 'Test Product');
    await user.type(screen.getByLabelText('數量'), '5');

    // 提交
    await user.click(screen.getByRole('button', { name: '送出訂單' }));

    // 驗證結果
    await waitFor(() => {
      expect(screen.getByText('訂單已成功建立')).toBeInTheDocument();
    });
  });

  it('displays validation errors for invalid input', async () => {
    const user = userEvent.setup();
    render(<OrderForm />);

    // 直接提交空表單
    await user.click(screen.getByRole('button', { name: '送出訂單' }));

    // 驗證錯誤訊息
    expect(screen.getByText('商品名稱為必填')).toBeInTheDocument();
    expect(screen.getByText('數量必須大於 0')).toBeInTheDocument();
  });
});
```

### 10.1.3 CLI 工具（Unit 70% / Integration 20% / E2E 10%）

**適用情境**：命令列工具、開發者工具、自動化腳本

**測試重點分布**：

```yaml
unit_tests:
  coverage_target: 70%
  focus:
    - 參數解析邏輯
    - 輸出格式化
    - 核心處理函式
    - 設定檔解析

integration_tests:
  coverage_target: 20%
  focus:
    - 子命令執行
    - 檔案系統操作
    - 標準輸入輸出
    - 錯誤碼回傳

e2e_tests:
  coverage_target: 10%
  focus:
    - 完整命令執行
    - 管道組合
    - 安裝與全域執行
```

**範例：CLI 整合測試**：

```python
# tests/integration/test_cli_commands.py
import subprocess
import tempfile
from pathlib import Path

class TestCLICommands:
    def test_init_command_creates_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ['mytool', 'init', '--path', tmpdir],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0
            assert (Path(tmpdir) / '.mytool.yaml').exists()
            assert 'Initialized successfully' in result.stdout

    def test_process_command_with_invalid_input(self):
        result = subprocess.run(
            ['mytool', 'process', '--input', '/nonexistent/file'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1
        assert 'Error: Input file not found' in result.stderr

    def test_version_flag(self):
        result = subprocess.run(
            ['mytool', '--version'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert 'mytool version' in result.stdout
```

### 10.1.4 資料處理 Pipeline（Unit 50% / Integration 35% / E2E 15%）

**適用情境**：ETL 流程、資料分析、ML Pipeline

**測試重點分布**：

```yaml
unit_tests:
  coverage_target: 50%
  focus:
    - 資料轉換函式
    - 驗證規則
    - 聚合邏輯
    - 異常資料處理

integration_tests:
  coverage_target: 35%
  focus:
    - 階段間資料流
    - 資料來源連接
    - 輸出格式驗證
    - 效能基準

e2e_tests:
  coverage_target: 15%
  focus:
    - 完整 Pipeline 執行
    - 資料品質驗證
    - 錯誤恢復機制
```

**範例：Pipeline 階段測試**：

```python
# tests/integration/test_data_pipeline.py
import pandas as pd
import pytest
from pipeline.stages import CleaningStage, TransformStage, AggregationStage

class TestPipelineStages:
    @pytest.fixture
    def sample_raw_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, None],
            'value': ['100', '200', 'invalid', '300'],
            'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
        })

    def test_cleaning_stage_removes_nulls(self, sample_raw_data):
        stage = CleaningStage(null_strategy='drop')
        result = stage.process(sample_raw_data)

        assert len(result) == 3
        assert result['id'].isna().sum() == 0

    def test_transform_stage_converts_types(self, sample_raw_data):
        cleaned = CleaningStage(null_strategy='drop').process(sample_raw_data)
        stage = TransformStage(type_mappings={'value': 'numeric'})
        result = stage.process(cleaned)

        # 'invalid' 應被轉換為 NaN
        assert result['value'].dtype == 'float64'
        assert result['value'].isna().sum() == 1

    def test_pipeline_stages_compose_correctly(self, sample_raw_data):
        pipeline = [
            CleaningStage(null_strategy='drop'),
            TransformStage(type_mappings={'value': 'numeric'}),
            AggregationStage(group_by='timestamp', agg={'value': 'sum'})
        ]

        data = sample_raw_data
        for stage in pipeline:
            data = stage.process(data)

        assert len(data) > 0
        assert 'value' in data.columns
```

---

## 10.2 AI 生成測試的陷阱與驗證

### 常見陷阱概覽

| 陷阱類型 | 症狀 | 風險等級 | 偵測難度 |
|---------|------|---------|---------|
| 過度 Mock | 測試全過但實際行為錯誤 | 🔴 高 | 中 |
| Happy Path 偏好 | 邊界/錯誤情境未覆蓋 | 🔴 高 | 低 |
| 測試實現耦合 | 重構導致測試大量失敗 | 🟡 中 | 中 |
| 無效斷言 | 測試永遠通過 | 🔴 高 | 高 |
| 重複測試 | 同樣行為多次驗證 | 🟢 低 | 低 |

### 10.2.1 過度 Mock（Mock Overuse）

**問題描述**：AI 傾向為所有依賴建立 mock，導致測試驗證的是 mock 行為而非實際邏輯。

**錯誤範例**：

```python
# ❌ 過度 mock：幾乎沒有測試到真實邏輯
def test_order_service_creates_order():
    mock_repo = Mock()
    mock_validator = Mock()
    mock_calculator = Mock()
    mock_notifier = Mock()

    mock_validator.validate.return_value = True
    mock_calculator.calculate_total.return_value = 100
    mock_repo.save.return_value = Order(id=1)

    service = OrderService(mock_repo, mock_validator, mock_calculator, mock_notifier)
    result = service.create_order({'items': []})

    # 這只驗證了 mock 被呼叫，沒有驗證實際業務邏輯
    mock_repo.save.assert_called_once()
    assert result.id == 1
```

**正確範例**：

```python
# ✅ 適度 mock：只 mock 外部依賴，保留核心邏輯
def test_order_service_creates_order():
    # 只 mock 真正的外部依賴（資料庫、通知服務）
    mock_repo = Mock()
    mock_notifier = Mock()
    mock_repo.save.side_effect = lambda order: setattr(order, 'id', 1) or order

    # 使用真實的驗證器和計算器
    validator = OrderValidator()
    calculator = PriceCalculator(tax_rate=0.1)

    service = OrderService(mock_repo, validator, calculator, mock_notifier)
    result = service.create_order({
        'items': [{'product_id': 'A', 'quantity': 2, 'price': 50}]
    })

    # 驗證真實業務邏輯
    assert result.total == 110  # 100 + 10% tax
    assert result.status == 'pending'
```

**Mock 使用決策指南**：

```yaml
should_mock:
  - 資料庫連線
  - 外部 API 呼叫
  - 檔案系統（大量 I/O）
  - 時間相關函式
  - 隨機數生成
  - 第三方服務（付款、通知）

should_not_mock:
  - 核心業務邏輯類別
  - 驗證器與計算器
  - 資料轉換函式
  - 內部工具類別
  - 設定解析
```

### 10.2.2 Happy Path 偏好（Happy Path Bias）

**問題描述**：AI 生成的測試傾向驗證正常流程，忽略邊界條件與錯誤處理。

**偵測方法**：檢查測試是否包含以下類型

```python
# 測試完整性檢查清單
test_coverage_checklist = {
    'happy_path': {
        'normal_input': True,      # ✅ AI 通常會產生
        'valid_edge_values': False, # ⚠️ 常被忽略
    },
    'boundary_conditions': {
        'empty_input': False,       # ⚠️ 常被忽略
        'max_values': False,        # ⚠️ 常被忽略
        'min_values': False,        # ⚠️ 常被忽略
        'off_by_one': False,        # ⚠️ 常被忽略
    },
    'error_handling': {
        'invalid_input': False,     # ⚠️ 常被忽略
        'null_handling': False,     # ⚠️ 常被忽略
        'exception_paths': False,   # ⚠️ 常被忽略
        'timeout_handling': False,  # ⚠️ 常被忽略
    },
    'concurrency': {
        'race_conditions': False,   # ❌ 很少產生
        'deadlock_scenarios': False, # ❌ 很少產生
    }
}
```

**補強策略**：

```python
# 針對函式 process_order(order: Order) -> Result
# AI 生成基礎測試後，手動補充邊界測試

class TestProcessOrderBoundaries:
    """邊界條件測試 - 補強 AI 生成測試"""

    def test_empty_items_list(self):
        order = Order(items=[])
        with pytest.raises(ValidationError, match='至少需要一個商品'):
            process_order(order)

    def test_quantity_at_maximum(self):
        order = Order(items=[Item(quantity=MAX_QUANTITY)])
        result = process_order(order)
        assert result.status == 'success'

    def test_quantity_exceeds_maximum(self):
        order = Order(items=[Item(quantity=MAX_QUANTITY + 1)])
        with pytest.raises(ValidationError, match='數量超過上限'):
            process_order(order)

    def test_price_precision_edge_case(self):
        # 浮點數精度邊界
        order = Order(items=[Item(price=0.1 + 0.2)])  # 0.30000000000000004
        result = process_order(order)
        assert result.total == Decimal('0.30')  # 應正確處理精度

    def test_concurrent_order_processing(self):
        # 併發處理測試
        orders = [Order(items=[Item(product_id='A')]) for _ in range(10)]
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_order, orders))

        assert all(r.status == 'success' for r in results)
```

### 10.2.3 測試與實現耦合過緊

**問題描述**：測試驗證的是實現細節而非行為，導致重構時測試大量失敗。

**錯誤範例**：

```python
# ❌ 耦合過緊：驗證內部實現
def test_user_service_fetches_user():
    service = UserService()

    with patch.object(service, '_cache') as mock_cache:
        with patch.object(service, '_repository') as mock_repo:
            mock_cache.get.return_value = None
            mock_repo.find_by_id.return_value = User(id=1, name='Test')

            result = service.get_user(1)

            # 驗證內部呼叫順序 - 耦合過緊
            mock_cache.get.assert_called_once_with('user:1')
            mock_repo.find_by_id.assert_called_once_with(1)
            mock_cache.set.assert_called_once()
```

**正確範例**：

```python
# ✅ 行為驅動：驗證可觀察的結果
def test_user_service_returns_user_by_id():
    # 使用測試替身而非直接 patch 內部
    repository = InMemoryUserRepository()
    repository.save(User(id=1, name='Test'))

    service = UserService(repository=repository)
    result = service.get_user(1)

    # 只驗證行為結果
    assert result is not None
    assert result.id == 1
    assert result.name == 'Test'

def test_user_service_returns_none_for_nonexistent_user():
    repository = InMemoryUserRepository()
    service = UserService(repository=repository)

    result = service.get_user(999)

    assert result is None
```

### 10.2.4 無效斷言（Tautological Assertions）

**問題描述**：測試包含永遠為真的斷言，無法偵測實際錯誤。

**常見模式**：

```python
# ❌ 無效斷言範例集
def test_invalid_assertions():
    result = some_function()

    # 永遠為真
    assert result is not None or result is None  # 恆真
    assert isinstance(result, object)            # 所有物件都是 object
    assert len([]) == 0                          # 測試常數而非變數

    # 測試 mock 的回傳值
    mock = Mock(return_value=42)
    assert mock() == 42  # 只測試了 mock 設定

    # 無意義的型別檢查
    data = {'key': 'value'}
    assert isinstance(data, dict)  # 測試字面值型別
```

**偵測與修正**：

```python
# ✅ 有效斷言
def test_valid_assertions():
    result = calculate_total(items=[
        {'price': 100, 'quantity': 2},
        {'price': 50, 'quantity': 1}
    ])

    # 具體數值驗證
    assert result == 250

    # 結構驗證
    order = create_order(user_id=1, items=['A', 'B'])
    assert order.user_id == 1
    assert len(order.items) == 2
    assert order.status == 'pending'

    # 行為驗證
    with pytest.raises(ValueError) as exc_info:
        calculate_total(items=[])
    assert '至少需要一個商品' in str(exc_info.value)
```

### 驗證方法

#### Mutation Testing（突變測試）

突變測試透過修改程式碼來驗證測試是否能偵測到這些變更。

```bash
# Python - 使用 mutmut
pip install mutmut
mutmut run --paths-to-mutate=src/

# 檢視存活的突變體（測試未能偵測的變更）
mutmut results
mutmut show <id>

# JavaScript - 使用 Stryker
npx stryker run
```

**突變測試報告解讀**：

```yaml
mutation_score_threshold:
  excellent: ">= 80%"    # 測試品質優秀
  good: "60% - 79%"      # 可接受
  needs_improvement: "< 60%"  # 需要補強測試

common_surviving_mutants:
  - 邊界條件變更: "< 改為 <="
  - 運算子替換: "+ 改為 -"
  - 常數變更: "0 改為 1"
  - 條件反轉: "if x 改為 if not x"
```

#### Coverage + Manual Review

覆蓋率數值需配合人工審查才有意義。

```python
# pytest 覆蓋率配置
# pyproject.toml
[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
fail_under = 80
show_missing = true
```

**人工審查檢查點**：

```markdown
## 測試審查清單

### 覆蓋率品質（非數量）
- [ ] 核心業務邏輯是否有專屬測試？
- [ ] 錯誤處理路徑是否被執行？
- [ ] 分支條件是否都有測試案例？

### 測試設計
- [ ] 測試名稱是否清楚描述行為？
- [ ] 每個測試是否只驗證一件事？
- [ ] 測試之間是否獨立（無順序依賴）？

### 斷言品質
- [ ] 斷言是否具體而非泛化？
- [ ] 是否驗證了正確的屬性？
- [ ] 錯誤訊息是否有被驗證？
```

#### 測試有效性檢查清單

```yaml
# AI 生成測試驗證清單
pre_commit_checks:
  - name: "斷言存在性"
    check: "每個測試至少一個 assert"
    command: "grep -L 'assert' tests/**/*.py"

  - name: "避免空測試"
    check: "測試函式體不為空"
    pattern: "def test_.*:\n\\s*pass"

  - name: "Mock 比例"
    check: "mock 數量不超過實際物件的 50%"
    manual: true

  - name: "邊界測試存在"
    check: "包含 empty、null、max、min 關鍵字"
    pattern: "(empty|null|none|max|min|boundary|edge)"

post_generation_review:
  - "執行測試並確認失敗情境真的會失敗"
  - "故意破壞程式碼，確認測試會偵測到"
  - "檢查測試是否可讀、可維護"
  - "確認無重複測試相同行為"
```

---

## 10.3 TDD with AI 工作流程

### 工作流程選擇矩陣

| 情境 | 建議模式 | 原因 |
|-----|---------|------|
| 需求明確、介面已定義 | AI 寫測試 + AI 實現 | 效率最高 |
| 複雜業務邏輯 | 人寫測試 + AI 實現 | 確保測試品質 |
| 探索性開發 | AI 寫測試 + 人審查 + AI 實現 | 平衡效率與品質 |
| 重構既有程式 | 人寫特徵測試 + AI 重構 | 保護既有行為 |
| 修復 Bug | 人寫失敗測試 + AI 修復 | 確保 Bug 不復發 |

### 10.3.1 AI 先寫測試模式

**適用情境**：介面明確、行為規格清楚

**Prompt 策略**：

```markdown
## 測試生成 Prompt 模板

請為以下函式規格生成測試：

**函式簽名**：
```python
def calculate_shipping(
    weight: float,
    destination: str,
    express: bool = False
) -> Decimal:
    """
    計算運費

    規則：
    - 基本運費：每公斤 $10
    - 離島加收 $50
    - 快遞加收 50%
    - 滿 $500 免運費

    Raises:
        ValueError: 重量為負數或零
        ValueError: 目的地無效
    """
```

**測試要求**：
1. 涵蓋所有正常情境
2. 涵蓋所有邊界條件（0、負數、最大值）
3. 涵蓋所有錯誤情境
4. 使用 pytest 參數化減少重複
5. 測試名稱採用 `test_<行為>_when_<條件>` 格式
```

**生成結果驗證**：

```python
# AI 生成後，人工確認以下項目：

# ✅ 正常情境
def test_calculate_shipping_returns_base_rate_for_standard_delivery(): ...
def test_calculate_shipping_adds_island_surcharge(): ...
def test_calculate_shipping_adds_express_surcharge(): ...
def test_calculate_shipping_returns_zero_when_over_threshold(): ...

# ✅ 邊界條件
def test_calculate_shipping_at_exact_free_threshold(): ...
def test_calculate_shipping_at_minimum_weight(): ...

# ✅ 錯誤情境
def test_calculate_shipping_raises_for_negative_weight(): ...
def test_calculate_shipping_raises_for_zero_weight(): ...
def test_calculate_shipping_raises_for_invalid_destination(): ...

# ⚠️ 可能需補充
def test_calculate_shipping_handles_float_precision(): ...
def test_calculate_shipping_with_combined_surcharges(): ...
```

### 10.3.2 人寫測試 AI 實現模式

**適用情境**：複雜業務邏輯、需要精確控制測試

**工作流程**：

```markdown
## 步驟 1：人工撰寫測試

```python
# tests/test_discount_engine.py
class TestDiscountEngine:
    def test_applies_percentage_discount(self):
        engine = DiscountEngine()
        result = engine.apply(
            subtotal=Decimal('100'),
            discount=PercentageDiscount(10)
        )
        assert result == Decimal('90')

    def test_applies_fixed_discount(self):
        engine = DiscountEngine()
        result = engine.apply(
            subtotal=Decimal('100'),
            discount=FixedDiscount(Decimal('15'))
        )
        assert result == Decimal('85')

    def test_discount_cannot_exceed_subtotal(self):
        engine = DiscountEngine()
        result = engine.apply(
            subtotal=Decimal('10'),
            discount=FixedDiscount(Decimal('20'))
        )
        assert result == Decimal('0')  # 不能為負

    def test_stacks_multiple_discounts(self):
        engine = DiscountEngine()
        result = engine.apply(
            subtotal=Decimal('100'),
            discounts=[
                PercentageDiscount(10),  # -10 = 90
                FixedDiscount(Decimal('5'))  # -5 = 85
            ]
        )
        assert result == Decimal('85')
```

## 步驟 2：AI 實現 Prompt

請實現 `DiscountEngine` 類別以通過以下測試：

[貼上測試程式碼]

實現要求：
1. 使用 Decimal 確保金額精度
2. 遵循單一職責原則
3. 折扣策略使用策略模式
4. 處理所有邊界條件
```

### 10.3.3 紅-綠-重構循環中的 AI 角色

**傳統 TDD 循環**：

```
🔴 Red    → 寫一個失敗的測試
🟢 Green  → 寫最少程式碼讓測試通過
🔵 Refactor → 改善程式碼結構，保持測試通過
```

**AI 輔助 TDD 循環**：

```yaml
red_phase:
  human_tasks:
    - 定義下一個行為需求
    - 撰寫測試（或審查 AI 生成的測試）
    - 確認測試確實失敗
  ai_tasks:
    - 根據規格生成測試草稿
    - 建議測試邊界條件

green_phase:
  human_tasks:
    - 審查 AI 實現
    - 確認符合設計意圖
    - 調整必要的細節
  ai_tasks:
    - 生成通過測試的最小實現
    - 提供多種實現方案選擇

refactor_phase:
  human_tasks:
    - 決定重構方向
    - 審查重構結果
    - 確認測試仍然通過
  ai_tasks:
    - 識別程式碼異味
    - 建議重構策略
    - 執行重構並保持測試通過
```

**實際操作範例**：

```markdown
## 🔴 Red Phase

**人**：我需要一個函式驗證台灣手機號碼格式

**AI**：建議測試如下
```python
def test_validates_taiwan_mobile_format():
    assert is_valid_taiwan_mobile('0912345678') is True
    assert is_valid_taiwan_mobile('0912-345-678') is True
    assert is_valid_taiwan_mobile('+886912345678') is True

def test_rejects_invalid_formats():
    assert is_valid_taiwan_mobile('12345678') is False    # 缺少 09
    assert is_valid_taiwan_mobile('0812345678') is False  # 不是 09 開頭
    assert is_valid_taiwan_mobile('091234567') is False   # 長度不足
```

**人**：加入市話格式支援

**AI**：補充測試
```python
def test_validates_taiwan_landline():
    assert is_valid_taiwan_phone('02-12345678') is True   # 台北
    assert is_valid_taiwan_phone('04-12345678') is True   # 台中
```

---

## 🟢 Green Phase

**人**：實現這些測試

**AI**：
```python
import re

def is_valid_taiwan_mobile(phone: str) -> bool:
    # 移除常見分隔符號
    cleaned = re.sub(r'[-\s]', '', phone)

    # +886 格式轉換
    if cleaned.startswith('+886'):
        cleaned = '0' + cleaned[4:]

    # 驗證格式：09 開頭，共 10 碼
    return bool(re.match(r'^09\d{8}$', cleaned))
```

---

## 🔵 Refactor Phase

**人**：請重構，支援市話和手機統一處理

**AI**：
```python
from dataclasses import dataclass
from enum import Enum
import re

class PhoneType(Enum):
    MOBILE = 'mobile'
    LANDLINE = 'landline'
    UNKNOWN = 'unknown'

@dataclass
class PhoneValidationResult:
    is_valid: bool
    phone_type: PhoneType
    normalized: str | None

class TaiwanPhoneValidator:
    MOBILE_PATTERN = re.compile(r'^09\d{8}$')
    LANDLINE_PATTERN = re.compile(r'^0[2-8]\d{7,8}$')

    def validate(self, phone: str) -> PhoneValidationResult:
        normalized = self._normalize(phone)

        if self.MOBILE_PATTERN.match(normalized):
            return PhoneValidationResult(True, PhoneType.MOBILE, normalized)

        if self.LANDLINE_PATTERN.match(normalized):
            return PhoneValidationResult(True, PhoneType.LANDLINE, normalized)

        return PhoneValidationResult(False, PhoneType.UNKNOWN, None)

    def _normalize(self, phone: str) -> str:
        cleaned = re.sub(r'[-\s()]', '', phone)
        if cleaned.startswith('+886'):
            cleaned = '0' + cleaned[4:]
        return cleaned
```

**人**：確認所有測試通過 ✅
```

### 10.3.4 測試驅動 Prompt 策略

**策略一：規格優先**

```markdown
## Prompt：規格驅動測試生成

根據以下 API 規格，生成完整測試套件：

**端點**：POST /api/orders
**請求格式**：
```json
{
  "user_id": "string (required)",
  "items": [
    {
      "product_id": "string",
      "quantity": "integer (1-100)"
    }
  ]
}
```

**回應**：
- 201: 訂單建立成功
- 400: 請求格式錯誤
- 404: 商品不存在
- 409: 庫存不足

**測試要求**：
1. 每個回應碼至少一個測試
2. 驗證回應結構
3. 使用 parametrize 減少重複
```

**策略二：範例驅動**

```markdown
## Prompt：從範例推導測試

以下是函式的使用範例，請推導完整測試：

```python
# 範例 1：基本使用
result = parse_date('2024-01-15')
# result = datetime(2024, 1, 15)

# 範例 2：不同格式
result = parse_date('15/01/2024', format='DD/MM/YYYY')
# result = datetime(2024, 1, 15)

# 範例 3：相對日期
result = parse_date('yesterday')
# result = datetime.now() - timedelta(days=1)

# 範例 4：錯誤處理
parse_date('invalid')
# raises ParseError
```

請生成涵蓋所有範例的測試，並補充邊界條件。
```

**策略三：行為清單驅動**

```markdown
## Prompt：行為清單測試生成

為購物車模組生成測試，需涵蓋以下行為：

**基本操作**
- [ ] 新增商品到購物車
- [ ] 更新商品數量
- [ ] 移除商品
- [ ] 清空購物車

**業務規則**
- [ ] 相同商品合併數量
- [ ] 數量不可超過庫存
- [ ] 數量不可為零或負數

**計算邏輯**
- [ ] 小計計算
- [ ] 折扣應用
- [ ] 運費計算
- [ ] 總計計算

每個行為生成一個測試函式，遵循 Given-When-Then 結構。
```

---

## 本章總結

### 測試策略決策樹

```
專案開始
    │
    ├─ 確定專案類型 → 選擇測試比例配置
    │
    ├─ 選擇 TDD 模式
    │   ├─ 介面明確 → AI 先寫測試
    │   ├─ 邏輯複雜 → 人寫測試
    │   └─ 探索階段 → 混合模式
    │
    ├─ AI 生成測試後驗證
    │   ├─ 執行突變測試
    │   ├─ 人工審查斷言品質
    │   └─ 補充邊界測試
    │
    └─ 持續維護
        ├─ 定期審查測試有效性
        ├─ 重構時確保測試獨立性
        └─ 監控覆蓋率趨勢
```

### 關鍵原則

| 原則 | 說明 |
|-----|------|
| **測試行為而非實現** | 驗證可觀察的結果，而非內部細節 |
| **AI 生成需人審查** | AI 測試傾向 happy path，需補充邊界 |
| **Mock 最小化** | 只 mock 真正的外部依賴 |
| **持續驗證** | 使用突變測試確保測試有效 |
| **保持獨立** | 測試之間無順序依賴 |

---

## 十一、反模式警示

> **核心原則**：最好的代碼是最簡單的代碼。設計模式是工具，不是目標。

---

## 11.1 設計模式濫用表

### Singleton 模式

| 維度 | 說明 |
|------|------|
| ✅ **適用場景** | 硬體資源存取（印表機、GPU）、全域配置載入器、日誌系統核心 |
| ❌ **濫用場景** | 資料庫連線池（應用依賴注入）、用戶 session（應用 context）、「方便」的全域變數替代 |
| ⚠️ **警示信號** | 單元測試難以隔離、多執行緒競爭條件、「到處都在呼叫 getInstance()」 |

```python
# ❌ 反模式：將資料庫連線做成 Singleton
class DatabaseConnection:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._create_connection()
        return cls._instance

# 問題：測試時無法替換、無法支援多資料庫、連線池管理困難

# ✅ 正確做法：依賴注入
class UserRepository:
    def __init__(self, db_connection):  # 注入依賴
        self.db = db_connection

    def find_by_id(self, user_id):
        return self.db.query("SELECT * FROM users WHERE id = ?", user_id)

# 使用時注入
repo = UserRepository(create_connection(config))
```

---

### Factory 模式

| 維度 | 說明 |
|------|------|
| ✅ **適用場景** | 物件建立邏輯複雜、需根據配置動態選擇實作、跨平台元件建立 |
| ❌ **濫用場景** | 簡單物件實例化、只有一種實作、建構邏輯固定不變 |
| ⚠️ **警示信號** | Factory 只回傳一種類型、Factory 方法只有一行 `return new X()` |

```python
# ❌ 反模式：不必要的 Factory
class UserFactory:
    @staticmethod
    def create_user(name, email):
        return User(name, email)  # 毫無意義的包裝

user = UserFactory.create_user("Alice", "alice@example.com")

# ✅ 正確做法：直接建構
user = User(name="Alice", email="alice@example.com")

# ✅ Factory 的正確使用時機：建立邏輯複雜
class NotificationFactory:
    @staticmethod
    def create(channel: str, config: dict) -> Notification:
        if channel == "email":
            return EmailNotification(smtp_server=config["smtp"])
        elif channel == "sms":
            return SMSNotification(api_key=config["twilio_key"])
        elif channel == "push":
            return PushNotification(fcm_token=config["fcm_token"])
        raise ValueError(f"Unknown channel: {channel}")
```

---

### Strategy 模式

| 維度 | 說明 |
|------|------|
| ✅ **適用場景** | 演算法可互換、執行時期需切換行為、多種變體共存 |
| ❌ **濫用場景** | 只有一種策略、策略永遠不會改變、簡單的 if-else 足夠 |
| ⚠️ **警示信號** | 策略類別只有一個、Context 類比策略本身更複雜 |

```python
# ❌ 反模式：過度設計的 Strategy
class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, price): pass

class NoDiscountStrategy(DiscountStrategy):
    def calculate(self, price):
        return price

class TenPercentStrategy(DiscountStrategy):
    def calculate(self, price):
        return price * 0.9

class DiscountContext:
    def __init__(self, strategy: DiscountStrategy):
        self.strategy = strategy

    def apply(self, price):
        return self.strategy.calculate(price)

# 使用時：5 個類別只為了做一個乘法
context = DiscountContext(TenPercentStrategy())
final_price = context.apply(100)

# ✅ 正確做法：函式即可
def apply_discount(price: float, discount_rate: float = 0.0) -> float:
    return price * (1 - discount_rate)

final_price = apply_discount(100, discount_rate=0.1)
```

---

### Observer 模式

| 維度 | 說明 |
|------|------|
| ✅ **適用場景** | 事件驅動系統、UI 狀態更新、發布/訂閱架構 |
| ❌ **濫用場景** | 同步的直接呼叫足夠、觀察者只有一個、事件鏈過深 |
| ⚠️ **警示信號** | 記憶體持續增長（未解除訂閱）、循環通知、除錯時追蹤困難 |

```python
# ❌ 反模式：記憶體洩漏的 Observer
class EventEmitter:
    def __init__(self):
        self.listeners = []  # 永遠增長，從不清理

    def subscribe(self, callback):
        self.listeners.append(callback)

    def emit(self, data):
        for listener in self.listeners:
            listener(data)  # 如果 listener 已被銷毀？

# 使用時：忘記 unsubscribe
emitter.subscribe(lambda x: print(x))  # 匿名函式無法移除

# ✅ 正確做法：弱引用 + 明確生命週期
import weakref

class SafeEventEmitter:
    def __init__(self):
        self.listeners = weakref.WeakSet()

    def subscribe(self, callback):
        # 使用可追蹤的物件
        self.listeners.add(callback)
        return lambda: self.listeners.discard(callback)  # 回傳 unsubscribe 函式

    def emit(self, data):
        for listener in list(self.listeners):
            try:
                listener(data)
            except ReferenceError:
                pass  # 物件已被回收

# 使用時：明確管理生命週期
unsubscribe = emitter.subscribe(handler)
# ... 使用完畢
unsubscribe()
```

---

### 模式濫用速查表

| 模式 | 類別數量警示 | 程式碼行數警示 | 替代方案 |
|------|-------------|---------------|---------|
| Singleton | 測試需要 mock 它 | 全域狀態 >50 行 | 依賴注入 |
| Factory | 只生產 1 種物件 | Factory 比產品簡單 | 直接 `new` |
| Strategy | <3 種策略 | Context 類 >20 行 | 函式參數 |
| Observer | 訂閱者 <3 個 | 事件鏈 >3 層 | 直接呼叫 |
| Decorator | <2 層裝飾 | 裝飾器比核心複雜 | 繼承或組合 |
| Builder | 參數 <5 個 | Builder 比物件複雜 | 建構子參數 |

---

## 11.2 AI 特有反模式

### 11.2.1 過度抽象症候群

**症狀描述**：
AI 傾向產生「教科書風格」的代碼，即使問題只需要 3 行解決，也會包裝成 5 層架構。這源於訓練資料中大量的教學範例和設計模式文章。

**警示信號**：
- 程式碼檔案數量 > 實際功能數量
- 抽象層級 > 3 層
- 介面定義比實作更長
- 「為了未來擴展」但沒有具體需求

```python
# ❌ 反模式：3 行功能包 5 層設計
# file: interfaces/data_processor.py
class IDataProcessor(ABC):
    @abstractmethod
    def process(self, data): pass

# file: processors/base_processor.py
class BaseProcessor(IDataProcessor):
    def __init__(self, config):
        self.config = config

# file: processors/string_processor.py
class StringProcessor(BaseProcessor):
    def process(self, data):
        return data.upper()

# file: factories/processor_factory.py
class ProcessorFactory:
    @staticmethod
    def create(processor_type):
        if processor_type == "string":
            return StringProcessor(Config())

# file: services/processing_service.py
class ProcessingService:
    def __init__(self, factory):
        self.factory = factory

    def execute(self, data, processor_type):
        processor = self.factory.create(processor_type)
        return processor.process(data)

# 使用：5 個檔案、5 個類別，只為了轉大寫
service = ProcessingService(ProcessorFactory())
result = service.execute("hello", "string")

# ✅ 正確做法：一行解決
result = "hello".upper()

# 如果確實需要可配置的處理器
def process_text(text: str, transform: str = "upper") -> str:
    transforms = {"upper": str.upper, "lower": str.lower, "title": str.title}
    return transforms.get(transform, str.upper)(text)
```

**如何避免**：
1. 先寫最簡單的實作
2. 只在第三次重複時才抽象
3. 質疑每一層抽象：「這層解決什麼問題？」

---

### 11.2.2 幻覺依賴

**症狀描述**：
AI 可能引用不存在的 API、過時的函式庫版本，或混淆不同框架的語法。這是語言模型的固有限制，尤其在較新或較冷門的技術上更常見。

**警示信號**：
- `ModuleNotFoundError` 或 `ImportError`
- 函式簽章與文件不符
- 套件名稱「看起來合理」但 pip 找不到
- API 路徑不存在（404）

```python
# ❌ 反模式：引用不存在的 API
from tensorflow.keras.optimizers import AdamOptimizer  # 舊版 API，已移除
from sklearn.neural_network import DeepNeuralNetwork  # 不存在的類別
import fastapi.security.jwt as jwt  # 虛構的模組路徑

# AI 可能生成的「合理但錯誤」的程式碼
response = requests.get(url, verify_ssl=True)  # 正確參數是 verify
df.to_sql(table, conn, method="multi_insert")  # 正確是 method="multi"

# ✅ 正確做法：驗證後再使用
# 1. 檢查官方文件
# 2. 在 REPL 中測試 import
# 3. 確認版本相容性

from tensorflow.keras.optimizers import Adam  # 正確的類別名
from sklearn.neural_network import MLPClassifier  # 實際存在的類別
from fastapi.security import OAuth2PasswordBearer  # 正確的模組路徑

# 驗證腳本
import importlib
def verify_import(module_path: str, class_name: str) -> bool:
    try:
        module = importlib.import_module(module_path)
        return hasattr(module, class_name)
    except ImportError:
        return False

assert verify_import("tensorflow.keras.optimizers", "Adam")
```

**如何避免**：
1. 對 AI 生成的 import 保持懷疑
2. 使用 IDE 的自動補全驗證
3. 維護專案的 requirements.txt 並 pin 版本
4. 建立常用套件的正確 import 參考表

---

### 11.2.3 上下文遺忘

**症狀描述**：
AI 在處理跨檔案任務時，可能遺忘先前建立的變數命名、型別定義或架構決策，導致不一致的程式碼。

**警示信號**：
- 同一概念在不同檔案有不同名稱（`userId` vs `user_id` vs `uid`）
- 型別定義重複且略有差異
- 匯入路徑不一致
- 相同邏輯的不同實作散落各處

```python
# ❌ 反模式：跨檔案不一致

# file: models/user.py (AI 第一次生成)
class User:
    def __init__(self, user_id: int, user_name: str):
        self.user_id = user_id
        self.user_name = user_name

# file: services/auth.py (AI 第二次生成，忘記之前的命名)
def authenticate(userId: int, userName: str):  # 駝峰命名
    return {"uid": userId, "name": userName}  # 又一個新名稱

# file: api/routes.py (AI 第三次生成)
@router.get("/users/{id}")  # 現在是 id
def get_user(id: int):
    user = db.get_user(id=id)
    return {"user_id": user.userId}  # 混用各種命名

# ✅ 正確做法：建立命名規範文件
# file: .claude/NAMING.md
"""
## 命名規範
- 資料庫欄位：snake_case (user_id)
- Python 變數：snake_case (user_id)
- API 參數：snake_case (user_id)
- TypeScript：camelCase (userId)
- 常數：SCREAMING_SNAKE_CASE (MAX_USER_ID)
"""

# file: models/user.py (統一後)
@dataclass
class User:
    user_id: int  # 統一使用 snake_case
    user_name: str

    def to_api_response(self) -> dict:
        return {"user_id": self.user_id, "user_name": self.user_name}
```

**如何避免**：
1. 在 `.claude/` 中建立命名規範文件
2. 提供 AI 現有程式碼的上下文
3. 使用型別檢查工具（mypy、TypeScript）
4. 建立共用的型別定義檔

---

### 11.2.4 樣板膨脹

**症狀描述**：
AI 傾向生成完整的「生產級」樣板，即使只是原型或簡單腳本，包含大量註解、完整的錯誤處理、詳盡的 docstring，導致核心邏輯被淹沒。

**警示信號**：
- 註解行數 > 程式碼行數
- 每個函式都有完整的 docstring（包含 Raises、Examples）
- 錯誤處理比正常邏輯更複雜
- 配置和常數宣告佔檔案 50% 以上

```python
# ❌ 反模式：樣板膨脹
class DataProcessor:
    """
    A comprehensive data processing utility class.

    This class provides methods for processing, transforming,
    and validating data according to business requirements.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (Logger): Logger instance for debugging

    Examples:
        >>> processor = DataProcessor({'mode': 'strict'})
        >>> processor.process([1, 2, 3])
        [2, 4, 6]

    Raises:
        ValueError: If data is None or empty
        TypeError: If data is not a list
        ConfigurationError: If config is invalid
    """

    DEFAULT_CONFIG = {
        'mode': 'normal',
        'strict_validation': True,
        'max_retries': 3,
        'timeout': 30,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the DataProcessor instance.

        Args:
            config: Optional configuration dictionary
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config, dict):
            raise TypeError("Config must be a dictionary")
        # ... 20 more lines of validation

    def process(self, data: List[int]) -> List[int]:
        """
        Process the input data.

        Args:
            data: List of integers to process

        Returns:
            Processed list of integers

        Raises:
            ValueError: If data is empty
        """
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        self.logger.debug(f"Processing {len(data)} items")

        try:
            result = [x * 2 for x in data]  # 實際邏輯只有這一行
            self.logger.info(f"Successfully processed {len(result)} items")
            return result
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise

# ✅ 正確做法：簡潔優先
def double_values(data: list[int]) -> list[int]:
    """將列表中的每個數值加倍。"""
    return [x * 2 for x in data]

# 需要時再加入錯誤處理
def double_values_safe(data: list[int]) -> list[int]:
    """將列表中的每個數值加倍，包含基本驗證。"""
    if not data:
        return []
    return [x * 2 for x in data]
```

**如何避免**：
1. 明確告知 AI 程式碼用途（原型 vs 生產）
2. 要求「最小可行實作」
3. 之後再逐步加入必要的樣板

---

### 11.2.5 過早最佳化

**症狀描述**：
AI 可能在沒有效能資料的情況下，引入複雜的快取機制、非同步處理、批次操作等，增加程式碼複雜度但沒有實際效益。

**警示信號**：
- 使用快取但沒有快取失效策略
- 引入非同步但只處理一個 I/O 操作
- 使用物件池但物件建立成本很低
- 「為了效能」的註解但沒有基準測試

```python
# ❌ 反模式：過早最佳化
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import asyncio

class UserService:
    _instance = None
    _cache = {}
    _lock = threading.Lock()
    _executor = ThreadPoolExecutor(max_workers=10)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @lru_cache(maxsize=1000)
    def _cached_get_user(self, user_id: int):
        # 快取 + Singleton + 執行緒池，只為了讀一個用戶
        return self._fetch_user_sync(user_id)

    async def get_user(self, user_id: int):
        with self._lock:
            if user_id in self._cache:
                return self._cache[user_id]

        loop = asyncio.get_event_loop()
        user = await loop.run_in_executor(
            self._executor,
            self._cached_get_user,
            user_id
        )

        with self._lock:
            self._cache[user_id] = user

        return user

# ✅ 正確做法：先簡單，有資料再最佳化
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int) -> User:
        """取得用戶資料。效能問題時再加快取。"""
        return self.db.query(User).filter_by(id=user_id).first()

# 確認有效能問題後，有針對性地最佳化
class UserService:
    def __init__(self, db, cache: Redis):
        self.db = db
        self.cache = cache

    def get_user(self, user_id: int) -> User:
        # 有監控資料顯示此查詢是瓶頸後才加入
        cached = self.cache.get(f"user:{user_id}")
        if cached:
            return User.from_json(cached)

        user = self.db.query(User).filter_by(id=user_id).first()
        self.cache.setex(f"user:{user_id}", 3600, user.to_json())
        return user
```

**如何避免**：
1. 先讓程式正確運作
2. 使用 profiler 找出真正的瓶頸
3. 只最佳化有測量資料支持的熱點
4. 記錄最佳化前後的效能數據

---

### AI 反模式檢查清單

| 反模式 | 檢測問題 | 行動 |
|--------|---------|------|
| 過度抽象 | 「這個抽象解決什麼問題？」 | 刪除無用的層級 |
| 幻覺依賴 | 「這個 import 能執行嗎？」 | 驗證每個新依賴 |
| 上下文遺忘 | 「命名是否與現有程式碼一致？」 | 提供規範文件 |
| 樣板膨脹 | 「核心邏輯佔比多少？」 | 要求精簡版 |
| 過早最佳化 | 「有效能數據支持嗎？」 | 先簡單，後優化 |

---

## 11.3 決策指引：何時不該使用模式

### 問題複雜度閾值

```
問題複雜度評估：

┌─────────────────────────────────────────────────────────────┐
│ 複雜度 1-2：直接編碼                                          │
│ • 單一責任、明確輸入輸出                                       │
│ • 例：字串處理、簡單計算、單次 API 呼叫                         │
│ → 不需要設計模式                                              │
├─────────────────────────────────────────────────────────────┤
│ 複雜度 3-4：函式抽象                                          │
│ • 邏輯重複 2-3 次                                             │
│ • 例：表單驗證、資料轉換、條件分支                              │
│ → 抽取函式，考慮參數化                                         │
├─────────────────────────────────────────────────────────────┤
│ 複雜度 5-6：類別封裝                                          │
│ • 狀態管理需求、相關函式群組                                    │
│ • 例：用戶會話、資料庫連線、檔案處理                            │
│ → 使用類別，但避免過度設計                                      │
├─────────────────────────────────────────────────────────────┤
│ 複雜度 7-8：設計模式                                          │
│ • 多個變體、執行期切換、複雜建立邏輯                            │
│ • 例：支付處理、外掛系統、工作流程引擎                          │
│ → 謹慎選擇適合的模式                                           │
├─────────────────────────────────────────────────────────────┤
│ 複雜度 9-10：架構模式                                         │
│ • 系統級關注點、跨服務通訊                                     │
│ • 例：微服務架構、事件驅動系統、CQRS                           │
│ → 需要完整的架構設計                                           │
└─────────────────────────────────────────────────────────────┘
```

### YAGNI 原則應用

**You Ain't Gonna Need It（你不會需要它）**

| 情境 | ❌ YAGNI 違反 | ✅ YAGNI 遵守 |
|------|--------------|--------------|
| 資料庫選擇 | 建立抽象層支援 5 種資料庫 | 直接使用 PostgreSQL |
| 用戶類型 | 預先設計 10 種角色權限系統 | 實作目前需要的 2 種角色 |
| API 版本 | 第一版就建立完整版本控制 | 先實作 v1，有需要再擴展 |
| 配置系統 | 支援 YAML/JSON/TOML/ENV | 使用環境變數 |
| 快取策略 | 三級快取架構 | 無快取或簡單記憶體快取 |

```python
# ❌ YAGNI 違反：預先建立不需要的抽象
class DatabaseAdapter(ABC):
    @abstractmethod
    def connect(self): pass
    @abstractmethod
    def query(self, sql): pass
    @abstractmethod
    def close(self): pass

class PostgreSQLAdapter(DatabaseAdapter): ...
class MySQLAdapter(DatabaseAdapter): ...      # 目前不需要
class MongoDBAdapter(DatabaseAdapter): ...    # 目前不需要
class SQLiteAdapter(DatabaseAdapter): ...     # 目前不需要

class DatabaseFactory:
    @staticmethod
    def create(db_type: str) -> DatabaseAdapter:
        adapters = {
            'postgresql': PostgreSQLAdapter,
            'mysql': MySQLAdapter,        # 目前不需要
            'mongodb': MongoDBAdapter,    # 目前不需要
            'sqlite': SQLiteAdapter,      # 目前不需要
        }
        return adapters[db_type]()

# ✅ YAGNI 遵守：只實作需要的
import psycopg2

def get_db_connection():
    return psycopg2.connect(os.environ['DATABASE_URL'])

def query(sql: str, params: tuple = None):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
```

### 重構時機 vs 預先設計

```
判斷流程圖：

「我需要這個模式嗎？」
         │
         ▼
┌─────────────────────┐
│ 問題是否已經存在？   │
└─────────────────────┘
         │
    ┌────┴────┐
   Yes        No
    │          │
    ▼          ▼
「解決它」   「等待信號」
    │          │
    │     ┌────┴─────────────────────┐
    │     │ 信號出現時（重構觸發點）：  │
    │     │ • 相同程式碼第 3 次出現     │
    │     │ • 修改一處需改多處          │
    │     │ • 新需求與現有結構衝突      │
    │     │ • 測試變得困難              │
    │     └────┬─────────────────────┘
    │          │
    ▼          ▼
┌─────────────────────┐
│ 選擇最簡單的解決方案 │
└─────────────────────┘
```

**三次法則（Rule of Three）**：

```python
# 第 1 次：直接寫
def send_welcome_email(user):
    send_email(user.email, "Welcome!", render_welcome(user))

# 第 2 次：注意到相似，但不急
def send_password_reset_email(user):
    send_email(user.email, "Reset Password", render_reset(user))

# 第 3 次：現在該抽象了
def send_order_confirmation_email(user, order):
    send_email(user.email, "Order Confirmed", render_order(user, order))

# ✅ 重構：通用化
def send_templated_email(
    user: User,
    template_name: str,
    subject: str,
    **context
):
    content = render_template(template_name, user=user, **context)
    send_email(user.email, subject, content)

# 使用
send_templated_email(user, "welcome", "Welcome!")
send_templated_email(user, "reset", "Reset Password")
send_templated_email(user, "order", "Order Confirmed", order=order)
```

---

### 決策矩陣：模式 vs 簡單方案

| 場景 | 模式方案 | 簡單方案 | 選擇依據 |
|------|---------|---------|---------|
| 2 種支付方式 | Strategy 模式 | `if-else` | 選簡單：變體少 |
| 10 種報表格式 | Factory + Strategy | `if-else` | 選模式：變體多、可能增加 |
| 單一通知管道 | Observer 模式 | 直接呼叫 | 選簡單：只有一個觀察者 |
| 多元件事件系統 | Observer 模式 | 直接呼叫 | 選模式：解耦需求明確 |
| 配置物件建立 | Builder 模式 | 建構子參數 | 參數 <5 選簡單，>7 選模式 |

---

## 本章總結

### 核心原則

1. **簡單優先**：最好的代碼是沒有代碼
2. **驗證依賴**：AI 生成的 import 必須驗證
3. **一致性**：建立並維護命名規範
4. **精簡樣板**：核心邏輯應該顯而易見
5. **資料驅動**：最佳化前先測量

### 檢查口訣

```
模式三問：
1. 這個問題值得用模式嗎？（複雜度閾值）
2. 現在就需要這個靈活性嗎？（YAGNI）
3. 有更簡單的方案嗎？（簡單優先）

AI 程式碼五驗：
1. Import 能執行嗎？（幻覺檢測）
2. 命名與現有一致嗎？（上下文一致）
3. 核心邏輯在哪裡？（樣板識別）
4. 有效能數據嗎？（最佳化驗證）
5. 抽象解決什麼問題？（必要性確認）
```

---

## 十二、Python 框架整合指引

## 12.1 FastAPI 設計模式實踐

FastAPI 的設計哲學強調類型安全與依賴注入，這與多種設計模式高度契合。

### 12.1.1 依賴注入系統與 Factory 模式

FastAPI 的 `Depends` 機制是 Factory 模式的最佳實踐場域。

```python
# ============================================================
# 依賴注入與 Factory 模式
# ============================================================

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from typing import Generator, Protocol
from functools import lru_cache

# --- 抽象工廠：資料庫連線 ---
class DatabaseFactory(Protocol):
    """資料庫工廠協議"""
    def create_session(self) -> Session: ...

class PostgresFactory:
    """PostgreSQL 工廠實作"""
    def __init__(self, url: str):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        self.engine = create_engine(url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_session(self) -> Session:
        return self.SessionLocal()

class SQLiteFactory:
    """SQLite 工廠實作（測試用）"""
    def __init__(self, url: str = "sqlite:///./test.db"):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        self.engine = create_engine(url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_session(self) -> Session:
        return self.SessionLocal()

# --- 工廠註冊器 ---
@lru_cache()
def get_db_factory() -> DatabaseFactory:
    """根據環境選擇工廠"""
    import os
    env = os.getenv("ENV", "development")
    if env == "production":
        return PostgresFactory(os.getenv("DATABASE_URL"))
    return SQLiteFactory()

# --- 依賴注入函數 ---
def get_db() -> Generator[Session, None, None]:
    """資料庫 Session 依賴"""
    factory = get_db_factory()
    db = factory.create_session()
    try:
        yield db
    finally:
        db.close()

# --- 服務層工廠 ---
class UserService:
    def __init__(self, db: Session):
        self.db = db

    def get_user(self, user_id: int):
        return self.db.query(User).filter(User.id == user_id).first()

def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """UserService 工廠依賴"""
    return UserService(db)

# --- 路由使用 ---
app = FastAPI()

@app.get("/users/{user_id}")
def read_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
):
    return service.get_user(user_id)
```

**設計要點**：
| 概念 | FastAPI 實作 | 設計模式 |
|------|-------------|----------|
| `Depends()` | 自動注入依賴 | Dependency Injection |
| `@lru_cache()` | 單例工廠 | Singleton + Factory |
| `Generator` | 生命週期管理 | Context Manager |
| `Protocol` | 抽象介面 | Abstract Factory |

### 12.1.2 Pydantic 與 Builder 模式

Pydantic 模型結合 Builder 模式，實現複雜物件的逐步建構。

```python
# ============================================================
# Pydantic Builder 模式
# ============================================================

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Self
from datetime import datetime
from enum import Enum

class OrderStatus(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"

# --- 基礎模型 ---
class OrderItem(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0)

    @property
    def subtotal(self) -> float:
        return self.quantity * self.unit_price

class ShippingInfo(BaseModel):
    address: str
    city: str
    postal_code: str
    country: str = "Taiwan"

class Order(BaseModel):
    """訂單模型 - 支援 Builder 模式"""
    id: Optional[int] = None
    customer_id: int
    items: list[OrderItem] = Field(default_factory=list)
    shipping: Optional[ShippingInfo] = None
    status: OrderStatus = OrderStatus.DRAFT
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def total(self) -> float:
        return sum(item.subtotal for item in self.items)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.status == OrderStatus.CONFIRMED and not self.shipping:
            raise ValueError("確認訂單必須包含運送資訊")
        if self.status != OrderStatus.DRAFT and not self.items:
            raise ValueError("非草稿訂單必須包含商品")
        return self

# --- Builder 實作 ---
class OrderBuilder:
    """訂單建構器"""

    def __init__(self, customer_id: int):
        self._data = {"customer_id": customer_id, "items": []}

    def add_item(
        self,
        product_id: int,
        quantity: int,
        unit_price: float
    ) -> Self:
        """添加商品"""
        self._data["items"].append(
            OrderItem(
                product_id=product_id,
                quantity=quantity,
                unit_price=unit_price
            )
        )
        return self

    def with_shipping(
        self,
        address: str,
        city: str,
        postal_code: str,
        country: str = "Taiwan"
    ) -> Self:
        """設定運送資訊"""
        self._data["shipping"] = ShippingInfo(
            address=address,
            city=city,
            postal_code=postal_code,
            country=country
        )
        return self

    def with_notes(self, notes: str) -> Self:
        """添加備註"""
        self._data["notes"] = notes
        return self

    def as_draft(self) -> Self:
        """設為草稿"""
        self._data["status"] = OrderStatus.DRAFT
        return self

    def as_confirmed(self) -> Self:
        """設為已確認"""
        self._data["status"] = OrderStatus.CONFIRMED
        return self

    def build(self) -> Order:
        """建構訂單"""
        return Order(**self._data)

# --- 使用範例 ---
order = (
    OrderBuilder(customer_id=123)
    .add_item(product_id=1, quantity=2, unit_price=99.0)
    .add_item(product_id=2, quantity=1, unit_price=199.0)
    .with_shipping(
        address="信義路五段7號",
        city="台北市",
        postal_code="110"
    )
    .with_notes("請在週末配送")
    .as_confirmed()
    .build()
)
```

### 12.1.3 中間件與 Chain of Responsibility

FastAPI 中間件實現責任鏈模式，處理橫切關注點。

```python
# ============================================================
# 中間件責任鏈模式
# ============================================================

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from typing import Callable, Awaitable
import time
import logging
from abc import ABC, abstractmethod

# --- 抽象處理器 ---
class RequestHandler(ABC):
    """請求處理器抽象類"""

    def __init__(self):
        self._next_handler: Optional[RequestHandler] = None

    def set_next(self, handler: "RequestHandler") -> "RequestHandler":
        self._next_handler = handler
        return handler

    @abstractmethod
    async def handle(self, request: Request) -> Optional[Response]:
        pass

    async def pass_to_next(self, request: Request) -> Optional[Response]:
        if self._next_handler:
            return await self._next_handler.handle(request)
        return None

# --- 具體處理器 ---
class AuthenticationHandler(RequestHandler):
    """認證處理器"""

    async def handle(self, request: Request) -> Optional[Response]:
        auth_header = request.headers.get("Authorization")
        if request.url.path.startswith("/public"):
            return await self.pass_to_next(request)

        if not auth_header:
            return Response(
                content='{"error": "未提供認證"}',
                status_code=401,
                media_type="application/json"
            )

        if not auth_header.startswith("Bearer "):
            return Response(
                content='{"error": "認證格式錯誤"}',
                status_code=401,
                media_type="application/json"
            )

        request.state.user_token = auth_header[7:]
        return await self.pass_to_next(request)

class RateLimitHandler(RequestHandler):
    """速率限制處理器"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        super().__init__()
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    async def handle(self, request: Request) -> Optional[Response]:
        client_ip = request.client.host
        now = time.time()

        if client_ip in self._requests:
            self._requests[client_ip] = [
                ts for ts in self._requests[client_ip]
                if now - ts < self.window_seconds
            ]
        else:
            self._requests[client_ip] = []

        if len(self._requests[client_ip]) >= self.max_requests:
            return Response(
                content='{"error": "請求過於頻繁"}',
                status_code=429,
                media_type="application/json"
            )

        self._requests[client_ip].append(now)
        return await self.pass_to_next(request)

# --- 組裝責任鏈 ---
def create_request_chain() -> RequestHandler:
    """建立請求處理責任鏈"""
    logging_handler = LoggingHandler()
    rate_limit_handler = RateLimitHandler(max_requests=100)
    auth_handler = AuthenticationHandler()

    logging_handler.set_next(rate_limit_handler).set_next(auth_handler)

    return logging_handler
```

### 12.1.4 路由組織與 Facade 模式

使用 APIRouter 實現 Facade 模式，簡化複雜的路由結構。

```python
# ============================================================
# 路由 Facade 模式
# ============================================================

from fastapi import APIRouter, FastAPI, Depends, HTTPException
from typing import Optional
from pydantic import BaseModel

# === 領域模型 ===
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

# === 子系統服務 ===
class UserSubsystem:
    """使用者子系統"""

    def create_user(self, data: UserCreate) -> UserResponse:
        return UserResponse(id=1, username=data.username, email=data.email)

    def get_user(self, user_id: int) -> Optional[UserResponse]:
        return UserResponse(id=user_id, username="test", email="test@example.com")

# === Facade 路由器 ===
class UserRouterFacade:
    """使用者路由 Facade"""

    def __init__(self):
        self.router = APIRouter(prefix="/users", tags=["users"])
        self.subsystem = UserSubsystem()
        self._register_routes()

    def _register_routes(self):
        @self.router.post("/", response_model=UserResponse)
        def create_user(data: UserCreate):
            return self.subsystem.create_user(data)

        @self.router.get("/{user_id}", response_model=UserResponse)
        def get_user(user_id: int):
            user = self.subsystem.get_user(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return user

# === API Facade（頂層入口）===
class APIFacade:
    """API 統一入口 Facade"""

    def __init__(self, app: FastAPI):
        self.app = app
        self._setup_routes()

    def _setup_routes(self):
        user_facade = UserRouterFacade()
        api_v1 = APIRouter(prefix="/api/v1")
        api_v1.include_router(user_facade.router)
        self.app.include_router(api_v1)

app = FastAPI(title="E-Commerce API", version="1.0.0")
api = APIFacade(app)
```

---

## 12.2 Django 設計模式實踐

Django 的「約定優於配置」哲學內建了多種設計模式，理解這些模式能讓開發更高效。

### 12.2.1 ORM 與 Active Record / Repository 模式

Django ORM 預設使用 Active Record，但可透過 Repository 模式實現更好的關注點分離。

```python
# ============================================================
# Django ORM 模式選擇
# ============================================================

from django.db import models
from django.db.models import Q, F, Count, Avg
from typing import Optional, Protocol
from abc import abstractmethod
from dataclasses import dataclass

# === Active Record 模式（Django 預設）===
class Product(models.Model):
    """產品模型 - Active Record 風格"""
    name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField(default=0)
    category = models.ForeignKey('Category', on_delete=models.CASCADE)

    # Active Record: 業務邏輯直接在模型上
    def apply_discount(self, percentage: float) -> None:
        """應用折扣"""
        self.price = self.price * (1 - percentage / 100)
        self.save(update_fields=['price', 'updated_at'])

    def reserve_stock(self, quantity: int) -> bool:
        """預留庫存"""
        if self.stock >= quantity:
            self.stock = F('stock') - quantity
            self.save(update_fields=['stock'])
            self.refresh_from_db()
            return True
        return False

# === Repository 模式 ===
@dataclass
class ProductDTO:
    """產品資料傳輸物件"""
    id: Optional[int]
    name: str
    price: float
    stock: int
    category_id: int
    is_active: bool = True

class ProductRepositoryProtocol(Protocol):
    """產品倉儲協議"""

    @abstractmethod
    def get_by_id(self, product_id: int) -> Optional[ProductDTO]: ...

    @abstractmethod
    def find_by_category(self, category_id: int) -> list[ProductDTO]: ...

class DjangoProductRepository:
    """Django ORM 實作的產品倉儲"""

    def _to_dto(self, model: Product) -> ProductDTO:
        return ProductDTO(
            id=model.id,
            name=model.name,
            price=float(model.price),
            stock=model.stock,
            category_id=model.category_id,
            is_active=model.is_active
        )

    def get_by_id(self, product_id: int) -> Optional[ProductDTO]:
        try:
            product = Product.objects.get(id=product_id)
            return self._to_dto(product)
        except Product.DoesNotExist:
            return None
```

### 12.2.2 Class-Based Views 與 Template Method 模式

Django CBV 是 Template Method 模式的典型應用。

```python
# ============================================================
# Class-Based Views Template Method 模式
# ============================================================

from django.views.generic import ListView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import QuerySet, Q

class FilterableMixin:
    """可篩選 Mixin"""

    filter_fields: list[str] = []
    search_fields: list[str] = []

    def get_filter_kwargs(self) -> dict:
        kwargs = {}
        for field in self.filter_fields:
            value = self.request.GET.get(field)
            if value:
                kwargs[field] = value
        return kwargs

class BaseListView(LoginRequiredMixin, FilterableMixin, ListView):
    """基礎列表視圖 - Template Method 模式"""

    def get(self, request, *args, **kwargs):
        self.pre_list_hook()
        self.object_list = self.get_queryset()
        self.object_list = self.apply_filters(self.object_list)
        self.post_list_hook()
        context = self.get_context_data()
        return self.render_to_response(context)

    def pre_list_hook(self) -> None:
        """前置處理鉤子 - 子類可覆寫"""
        pass

    def post_list_hook(self) -> None:
        """後置處理鉤子 - 子類可覆寫"""
        pass

    def apply_filters(self, queryset: QuerySet) -> QuerySet:
        filter_kwargs = self.get_filter_kwargs()
        if filter_kwargs:
            queryset = queryset.filter(**filter_kwargs)
        return queryset
```

### 12.2.3 Signals 與 Observer 模式

Django Signals 實現了 Observer 模式，用於解耦事件處理。

```python
# ============================================================
# Django Signals Observer 模式
# ============================================================

from django.db.models.signals import post_save, pre_delete
from django.dispatch import Signal, receiver
import logging

logger = logging.getLogger(__name__)

# === 自訂信號（自訂事件）===
order_created = Signal()
order_confirmed = Signal()
stock_low = Signal()

# === 信號接收器（Observer）===
@receiver(order_created)
def on_order_created(sender, order, user, **kwargs):
    """訂單建立時的處理"""
    logger.info(f"訂單 {order.id} 已建立 by {user}")

    # 發送確認郵件
    from django.core.mail import send_mail
    send_mail(
        subject=f'訂單確認 - #{order.id}',
        message=f'感謝您的訂單',
        from_email='orders@example.com',
        recipient_list=[user.email],
        fail_silently=True
    )

@receiver(stock_low)
def on_stock_low(sender, product, current_stock, **kwargs):
    """庫存不足時的處理"""
    logger.warning(f"產品 {product.name} 庫存不足: {current_stock}")

    # 通知採購團隊
    from django.core.mail import send_mail
    send_mail(
        subject=f'[警告] 產品庫存不足: {product.name}',
        message=f'目前庫存僅剩 {current_stock} 件',
        from_email='system@example.com',
        recipient_list=['purchasing@example.com'],
        fail_silently=True
    )
```

### 12.2.4 Manager 與 Strategy 模式

Django Manager 可實現 Strategy 模式，封裝不同的查詢策略。

```python
# ============================================================
# Manager Strategy 模式
# ============================================================

from django.db import models
from django.db.models import QuerySet, Q, F, Count
from typing import Protocol
from abc import abstractmethod

# === 查詢策略協議 ===
class ProductQueryStrategy(Protocol):
    @abstractmethod
    def apply(self, queryset: QuerySet) -> QuerySet: ...

class ActiveProductsStrategy:
    def apply(self, queryset: QuerySet) -> QuerySet:
        return queryset.filter(is_active=True, stock__gt=0)

class BestsellerStrategy:
    def __init__(self, days: int = 30, min_orders: int = 10):
        self.days = days
        self.min_orders = min_orders

    def apply(self, queryset: QuerySet) -> QuerySet:
        from django.utils import timezone
        from datetime import timedelta
        since = timezone.now() - timedelta(days=self.days)
        return queryset.filter(
            is_active=True,
            orderitem__order__created_at__gte=since
        ).annotate(
            order_count=Count('orderitem')
        ).filter(
            order_count__gte=self.min_orders
        ).order_by('-order_count')

# === 策略 Manager ===
class ProductQuerySet(QuerySet):
    def apply_strategy(self, strategy: ProductQueryStrategy) -> 'ProductQuerySet':
        return strategy.apply(self)

    def active(self) -> 'ProductQuerySet':
        return self.apply_strategy(ActiveProductsStrategy())

    def bestsellers(self, days: int = 30) -> 'ProductQuerySet':
        return self.apply_strategy(BestsellerStrategy(days=days))

class ProductManager(models.Manager):
    def get_queryset(self) -> ProductQuerySet:
        return ProductQuerySet(self.model, using=self._db)

    def active(self):
        return self.get_queryset().active()

    def bestsellers(self, days: int = 30):
        return self.get_queryset().bestsellers(days)
```

---

## 12.3 框架選型決策樹

### 12.3.1 快速決策指南

```
需要什麼？
│
├── 高效能 API 服務
│   ├── 需要 WebSocket/即時通訊？
│   │   ├── 是 → FastAPI（原生支援 WebSocket）
│   │   └── 否 → FastAPI（async + 高效能）
│   │
│   └── 微服務架構？
│       ├── 是 → FastAPI + Pydantic
│       └── 否 → 見下方評估
│
├── 全功能 Web 應用
│   ├── 需要內建 Admin？
│   │   ├── 是 → Django
│   │   └── 否 → 見下方評估
│   │
│   └── 需要 ORM + Migration？
│       ├── 是 → Django
│       └── 否 → FastAPI + SQLAlchemy
│
├── 快速原型/MVP
│   └── 開發時間限制？
│       ├── 極短（1-2週）→ Django（約定優於配置）
│       └── 正常 → 依需求選擇
│
└── 團隊考量
    ├── 團隊經驗？
    │   ├── Django 熟悉 → Django
    │   ├── 現代 Python/async → FastAPI
    │   └── 無偏好 → 見效能需求
    │
    └── 學習曲線接受度？
        ├── 需快速上手 → Django
        └── 願意學習 → FastAPI
```

### 12.3.2 詳細比較矩陣

| 面向 | FastAPI | Django |
|------|---------|--------|
| **效能** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **開發速度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **學習曲線** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **生態系統** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **文件支援** | ⭐⭐⭐⭐⭐（自動生成）| ⭐⭐⭐ |
| **Admin 介面** | ⭐⭐（需第三方）| ⭐⭐⭐⭐⭐ |
| **ORM** | ⭐⭐⭐（SQLAlchemy）| ⭐⭐⭐⭐⭐ |
| **非同步** | ⭐⭐⭐⭐⭐（原生）| ⭐⭐⭐（Django 4.0+）|
| **類型安全** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 12.3.3 場景決策表

| 場景 | 推薦框架 | 理由 |
|------|----------|------|
| RESTful API | FastAPI | 自動文件、類型驗證、高效能 |
| 電商網站 | Django | Admin、認證、ORM、成熟生態 |
| 內容管理系統 | Django | Admin、使用者權限、模板 |
| 微服務 | FastAPI | 輕量、快速啟動、容器友好 |
| 即時應用 | FastAPI | WebSocket、async 支援 |
| 企業後台 | Django | Admin、權限系統、審計日誌 |

---

## 12.4 設計模式對照速查表

| 設計模式 | FastAPI 實作 | Django 實作 |
|----------|-------------|-------------|
| **Factory** | `Depends()` + 工廠函數 | `Manager.create()` |
| **Builder** | Pydantic + Builder 類 | Form + ModelForm |
| **Singleton** | `@lru_cache()` 依賴 | `apps.get_app_config()` |
| **Adapter** | Pydantic 轉換器 | Serializer |
| **Facade** | APIRouter 組合 | Admin + View 聚合 |
| **Strategy** | 依賴注入策略類 | Manager + QuerySet |
| **Observer** | 事件系統 / WebSocket | Signals |
| **Chain of Responsibility** | Middleware 串接 | Middleware + View 裝飾器 |
| **Template Method** | 抽象依賴 + 覆寫 | CBV 方法覆寫 |
| **Repository** | 自訂 Repository 類 | Manager + QuerySet |

---

## 本章重點摘要

```
FastAPI 核心模式：
├── Depends() → 依賴注入 + Factory
├── Pydantic → 驗證 + Builder
├── Middleware → Chain of Responsibility
└── APIRouter → Facade

Django 核心模式：
├── ORM → Active Record / Repository
├── CBV → Template Method
├── Signals → Observer
└── Manager → Strategy

選型決策：
├── 高效能 API → FastAPI
├── 全功能 Web → Django
├── 微服務 → FastAPI
├── Admin 需求 → Django
└── 混合需求 → Django + FastAPI
```

---

## 十三、錯誤處理與韌性設計

> **核心原則**：錯誤不是例外，而是系統的必然組成部分。優秀的系統不是不會出錯，而是出錯後仍能優雅運作。

---

## 13.1 錯誤傳播策略

### 異常處理決策樹

```
捕獲異常後？
├── 可恢復 → 處理並繼續
│   └── 例：快取失效 → 重新載入
├── 需要上下文 → 包裝後重拋
│   └── 例：資料庫錯誤 → 加入查詢資訊
├── 跨層邊界 → 轉換為適當類型
│   └── 例：SQLError → ServiceError
└── 無法處理 → 原樣重拋
    └── 例：記憶體耗盡、系統級錯誤
```

### 13.1.1 何時包裝異常（Wrap）

**使用場景**：原始異常缺乏足夠的上下文資訊

```python
# ❌ 不良：遺失重要上下文
def get_user(user_id: str) -> User:
    try:
        return self.db.query(User).filter_by(id=user_id).one()
    except NoResultFound:
        raise  # 呼叫者不知道是哪個 user_id 找不到

# ✅ 良好：包裝並添加上下文
class UserNotFoundError(Exception):
    def __init__(self, user_id: str, original: Exception | None = None):
        self.user_id = user_id
        self.original = original
        super().__init__(f"User not found: {user_id}")

def get_user(user_id: str) -> User:
    try:
        return self.db.query(User).filter_by(id=user_id).one()
    except NoResultFound as e:
        raise UserNotFoundError(user_id, original=e) from e
```

### 13.1.2 異常層級設計

```
AppError（應用程式基礎）
├── ServiceError（服務層 - 對外 API）
│   ├── ValidationError     # 輸入驗證失敗
│   ├── AuthenticationError # 身份驗證失敗
│   ├── AuthorizationError  # 權限不足
│   ├── NotFoundError       # 資源不存在
│   └── ConflictError       # 狀態衝突
│
├── InfrastructureError（基礎設施層 - 內部）
│   ├── DatabaseError       # 資料庫相關
│   ├── CacheError          # 快取相關
│   ├── QueueError          # 訊息佇列相關
│   └── ExternalAPIError    # 外部 API 相關
│
└── DomainError（領域層 - 業務規則）
    ├── BusinessRuleViolation   # 業務規則違反
    ├── InvalidStateTransition  # 無效狀態轉換
    └── InvariantViolation      # 不變量違反
```

---

## 13.2 韌性模式實現

### 韌性模式選擇決策樹

```
面對什麼問題？
├── 暫時性失敗（網路抖動）
│   └── Retry + Exponential Backoff
├── 響應時間不確定
│   └── Timeout
├── 依賴服務不穩定
│   ├── 頻繁失敗 → Circuit Breaker
│   └── 有備選方案 → Fallback
├── 資源競爭
│   └── Rate Limiter + Bulkhead
└── 複合問題
    └── 組合多個模式
```

### 13.2.1 Retry：重試模式

**使用場景**：暫時性故障、網路抖動、資源競爭

```python
import asyncio
from functools import wraps
from dataclasses import dataclass

@dataclass
class RetryConfig:
    """重試配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)

def async_retry(config: RetryConfig | None = None):
    """非同步重試裝飾器"""
    cfg = config or RetryConfig()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(cfg.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_exception = e
                    if attempt < cfg.max_attempts - 1:
                        delay = calculate_delay(attempt, cfg)
                        await asyncio.sleep(delay)

            raise RetryExhaustedError(cfg.max_attempts, last_exception)

        return wrapper
    return decorator

# 使用 tenacity 函式庫
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
)
async def robust_api_call(endpoint: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(endpoint)
        response.raise_for_status()
        return response.json()
```

### 13.2.2 Circuit Breaker：熔斷器模式

**使用場景**：防止對不健康服務的持續請求

```
狀態轉換圖：

   CLOSED ──失敗達閾值──→ OPEN
      ↑                      │
      │                      │ 等待超時
      │                      ↓
      └──成功──── HALF_OPEN ←┘
                     │
                     └──失敗──→ OPEN
```

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    half_open_max_calls: int = 3

class CircuitBreaker:
    """熔斷器實現"""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()

    async def execute(self, func, *args, **kwargs):
        if not self._can_execute():
            raise CircuitBreakerOpenError(self.name, self._get_retry_after())

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

### 13.2.3 組合模式：構建韌性服務

```python
def resilient(
    circuit_breaker: CircuitBreaker | None = None,
    bulkhead: Bulkhead | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
    fallback: Callable | None = None,
):
    """組合多個韌性模式的裝飾器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 應用各層韌性模式（由外到內）
            # 1. 限流 → 2. 艙壁 → 3. 熔斷 → 4. 超時 → 5. 重試 → 6. 降級
            ...
        return wrapper
    return decorator

# 使用範例
@resilient(
    circuit_breaker=CircuitBreaker("payment"),
    bulkhead=Bulkhead("payment", BulkheadConfig(max_concurrent=10)),
    rate_limiter=TokenBucketRateLimiter(RateLimitConfig(requests_per_second=50)),
    retry_config=RetryConfig(max_attempts=3),
    timeout=30.0,
    fallback=lambda order_id, amount: PaymentResult(status="pending")
)
async def process_payment(order_id: str, amount: Decimal) -> PaymentResult:
    return await payment_gateway.charge(order_id, amount)
```

---

## 13.3 Graceful Degradation 範例

### 服務降級決策樹

```
服務出現問題？
├── 主服務完全不可用
│   ├── 有備援服務 → 切換至備援
│   ├── 有快取資料 → 提供快取結果（標記為過期）
│   └── 都沒有 → 提供預設回應 + 通知用戶
│
├── 主服務部分功能受損
│   ├── 非核心功能 → 隱藏該功能
│   └── 核心功能 → 提供簡化版本
│
└── 主服務響應緩慢
    ├── 可接受延遲 → 顯示載入狀態
    └── 不可接受 → 超時後返回快取/預設值
```

### 電商產品服務降級範例

```python
class ProductService:
    """產品服務 - 具備完整降級策略"""

    async def get_product_page(
        self,
        product_id: str,
        user_id: str | None = None
    ) -> DegradedProduct:
        """
        取得產品頁面資訊
        各功能獨立降級，確保頁面始終可顯示
        """
        features = {
            "basic_info": False,
            "real_time_stock": False,
            "dynamic_pricing": False,
            "recommendations": False,
        }

        # 1. 基本產品資訊（必須）
        product = await self._get_basic_info(product_id)
        if product:
            features["basic_info"] = True

        # 2. 即時庫存（可降級）
        stock = await self._get_stock_with_fallback(product_id, product.stock)

        # 3. 動態定價（可降級）
        dynamic_price = await self._get_dynamic_price(product_id, user_id)

        # 4. 推薦商品（完全可選）
        recommendations = await self._get_recommendations(product_id, user_id)

        return DegradedProduct(
            product=product,
            features_available=features,
            user_message=self._generate_user_message(features)
        )
```

---

## 總結

### 韌性設計清單

| 模式 | 使用場景 | 關鍵配置 |
|------|----------|----------|
| **Retry** | 暫時性故障 | 最大次數、退避策略、可重試異常 |
| **Timeout** | 防止無限等待 | 合理超時值、操作分類 |
| **Fallback** | 需要備選方案 | 降級邏輯、預設值 |
| **Circuit Breaker** | 依賴不穩定 | 失敗閾值、恢復時間、半開測試 |
| **Bulkhead** | 資源隔離 | 並發上限、等待佇列 |
| **Rate Limiter** | 過載保護 | 速率、突發容量 |

### 組合使用原則

```
請求流入
    ↓
[Rate Limiter] ← 第一道防線：限制流量
    ↓
[Bulkhead] ← 隔離資源：防止連鎖故障
    ↓
[Circuit Breaker] ← 快速失敗：保護下游
    ↓
[Timeout] ← 控制延遲：防止無限等待
    ↓
[Retry] ← 自動恢復：處理暫時性故障
    ↓
[Fallback] ← 優雅降級：確保可用性
    ↓
回應返回
```

---

## 十四、併發與非同步模式

> 將經典設計模式與 Python asyncio 整合，建構高效能、可維護的非同步系統

### 14.1 async/await 與設計模式整合

#### 14.1.1 Strategy 模式的非同步版本

**同步版本 vs 非同步版本對照**：

```python
# ═══════════════════════════════════════════════════════════════
# 同步版本：Strategy 模式
# ═══════════════════════════════════════════════════════════════
from abc import ABC, abstractmethod
from typing import Protocol

class DataFetcher(Protocol):
    """同步資料獲取策略介面"""
    def fetch(self, url: str) -> dict: ...

class HttpFetcher:
    def fetch(self, url: str) -> dict:
        import requests
        return requests.get(url).json()

class CachedFetcher:
    def __init__(self, cache: dict):
        self._cache = cache

    def fetch(self, url: str) -> dict:
        if url in self._cache:
            return self._cache[url]
        # 回退到 HTTP 獲取
        result = HttpFetcher().fetch(url)
        self._cache[url] = result
        return result

# ═══════════════════════════════════════════════════════════════
# 非同步版本：Async Strategy 模式
# ═══════════════════════════════════════════════════════════════
from typing import Protocol, runtime_checkable
import asyncio
import aiohttp

@runtime_checkable
class AsyncDataFetcher(Protocol):
    """非同步資料獲取策略介面"""
    async def fetch(self, url: str) -> dict: ...

class AsyncHttpFetcher:
    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def fetch(self, url: str) -> dict:
        session = self._session or aiohttp.ClientSession()
        try:
            async with session.get(url) as response:
                return await response.json()
        finally:
            if self._owns_session and session:
                await session.close()

class AsyncCachedFetcher:
    def __init__(self, cache: dict, fallback: AsyncDataFetcher):
        self._cache = cache
        self._fallback = fallback
        self._lock = asyncio.Lock()  # 防止快取競爭

    async def fetch(self, url: str) -> dict:
        # 快速路徑：無鎖檢查
        if url in self._cache:
            return self._cache[url]

        async with self._lock:
            # 雙重檢查鎖定
            if url in self._cache:
                return self._cache[url]
            result = await self._fallback.fetch(url)
            self._cache[url] = result
            return result

# ═══════════════════════════════════════════════════════════════
# 使用範例：策略注入
# ═══════════════════════════════════════════════════════════════
class DataProcessor:
    def __init__(self, fetcher: AsyncDataFetcher):
        self._fetcher = fetcher

    async def process(self, urls: list[str]) -> list[dict]:
        """並行處理多個 URL"""
        tasks = [self._fetcher.fetch(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# 使用方式
async def main():
    async with aiohttp.ClientSession() as session:
        fetcher = AsyncHttpFetcher(session)
        cached_fetcher = AsyncCachedFetcher({}, fetcher)
        processor = DataProcessor(cached_fetcher)
        results = await processor.process([
            "https://api.example.com/data1",
            "https://api.example.com/data2",
        ])
```

**注意事項**：

| 項目 | 同步版本 | 非同步版本 |
|------|----------|------------|
| **介面定義** | `def method()` | `async def method()` |
| **資源管理** | 可直接使用 | 需要 `async with` 或手動關閉 |
| **競爭條件** | 通常無需考慮 | 需使用 `asyncio.Lock` 保護共享狀態 |
| **錯誤處理** | `try/except` | `try/except` + 考慮取消處理 |
| **Session 管理** | 每次建立或連線池 | 共享 `aiohttp.ClientSession` |

---

#### 14.1.2 Command 模式與非同步執行

```python
# ═══════════════════════════════════════════════════════════════
# 同步版本：Command 模式
# ═══════════════════════════════════════════════════════════════
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

class Command(ABC):
    @abstractmethod
    def execute(self) -> Any: ...

    @abstractmethod
    def undo(self) -> None: ...

@dataclass
class CreateFileCommand(Command):
    path: str
    content: str
    _created: bool = False

    def execute(self) -> str:
        with open(self.path, 'w') as f:
            f.write(self.content)
        self._created = True
        return self.path

    def undo(self) -> None:
        if self._created:
            import os
            os.remove(self.path)
            self._created = False

# ═══════════════════════════════════════════════════════════════
# 非同步版本：Async Command 模式
# ═══════════════════════════════════════════════════════════════
import aiofiles
import aiofiles.os

class AsyncCommand(ABC):
    @abstractmethod
    async def execute(self) -> Any: ...

    @abstractmethod
    async def undo(self) -> None: ...

@dataclass
class AsyncCreateFileCommand(AsyncCommand):
    path: str
    content: str
    _created: bool = False

    async def execute(self) -> str:
        async with aiofiles.open(self.path, 'w') as f:
            await f.write(self.content)
        self._created = True
        return self.path

    async def undo(self) -> None:
        if self._created:
            await aiofiles.os.remove(self.path)
            self._created = False

# ═══════════════════════════════════════════════════════════════
# 非同步命令執行器（支援批次執行與撤銷）
# ═══════════════════════════════════════════════════════════════
class AsyncCommandExecutor:
    def __init__(self, max_concurrent: int = 10):
        self._history: list[AsyncCommand] = []
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(self, command: AsyncCommand) -> Any:
        """執行單一命令"""
        async with self._semaphore:
            result = await command.execute()
            self._history.append(command)
            return result

    async def execute_batch(
        self,
        commands: list[AsyncCommand],
        stop_on_error: bool = True
    ) -> list[Any]:
        """批次執行命令"""
        results = []
        for cmd in commands:
            try:
                result = await self.execute(cmd)
                results.append(result)
            except Exception as e:
                if stop_on_error:
                    # 回滾已執行的命令
                    await self.undo_all()
                    raise
                results.append(e)
        return results

    async def undo_last(self) -> None:
        """撤銷最後一個命令"""
        if self._history:
            command = self._history.pop()
            await command.undo()

    async def undo_all(self) -> None:
        """撤銷所有命令（反向順序）"""
        while self._history:
            await self.undo_last()

# 使用範例
async def batch_file_creation():
    executor = AsyncCommandExecutor(max_concurrent=5)
    commands = [
        AsyncCreateFileCommand(f"/tmp/file_{i}.txt", f"Content {i}")
        for i in range(10)
    ]
    try:
        results = await executor.execute_batch(commands)
        print(f"Created {len(results)} files")
    except Exception:
        print("Operation failed, all files rolled back")
```

**注意事項**：

```
┌─────────────────────────────────────────────────────────────┐
│  非同步 Command 模式注意事項                                 │
│  ═══════════════════════                                    │
│  • 使用 Semaphore 控制併發數量，避免資源耗盡               │
│  • 批次執行時考慮原子性：全部成功或全部回滾                │
│  • undo() 也必須是非同步的，且應處理部分失敗場景           │
│  • 考慮使用 asyncio.shield() 保護關鍵的撤銷操作            │
└─────────────────────────────────────────────────────────────┘
```

---

#### 14.1.3 Observer 模式的非同步事件處理

```python
# ═══════════════════════════════════════════════════════════════
# 同步版本：Observer 模式
# ═══════════════════════════════════════════════════════════════
from typing import Callable, Any

class EventEmitter:
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event: str, *args, **kwargs) -> None:
        for callback in self._listeners.get(event, []):
            callback(*args, **kwargs)  # 同步呼叫，阻塞

# ═══════════════════════════════════════════════════════════════
# 非同步版本：Async Observer 模式
# ═══════════════════════════════════════════════════════════════
from typing import Callable, Coroutine, Any
from enum import Enum, auto
import asyncio
import logging

logger = logging.getLogger(__name__)

class DispatchMode(Enum):
    """事件分發模式"""
    PARALLEL = auto()      # 並行執行所有監聽器
    SEQUENTIAL = auto()    # 順序執行（保證順序）
    FIRE_AND_FORGET = auto()  # 發射後不管（不等待完成）

class AsyncEventEmitter:
    def __init__(self, default_mode: DispatchMode = DispatchMode.PARALLEL):
        self._listeners: dict[str, list[Callable[..., Coroutine]]] = {}
        self._default_mode = default_mode

    def on(self, event: str, callback: Callable[..., Coroutine]) -> Callable:
        """註冊事件監聽器"""
        self._listeners.setdefault(event, []).append(callback)
        return callback  # 返回原函數，支援裝飾器用法

    def off(self, event: str, callback: Callable[..., Coroutine]) -> None:
        """移除事件監聽器"""
        if event in self._listeners:
            self._listeners[event] = [
                cb for cb in self._listeners[event] if cb != callback
            ]

    async def emit(
        self,
        event: str,
        *args,
        mode: DispatchMode | None = None,
        timeout: float | None = None,
        **kwargs
    ) -> list[Any]:
        """發射事件並收集結果"""
        callbacks = self._listeners.get(event, [])
        if not callbacks:
            return []

        dispatch_mode = mode or self._default_mode

        if dispatch_mode == DispatchMode.PARALLEL:
            return await self._emit_parallel(callbacks, args, kwargs, timeout)
        elif dispatch_mode == DispatchMode.SEQUENTIAL:
            return await self._emit_sequential(callbacks, args, kwargs, timeout)
        else:  # FIRE_AND_FORGET
            self._emit_fire_and_forget(callbacks, args, kwargs)
            return []

    async def _emit_parallel(
        self,
        callbacks: list,
        args: tuple,
        kwargs: dict,
        timeout: float | None
    ) -> list[Any]:
        """並行執行所有監聽器"""
        tasks = [
            asyncio.create_task(cb(*args, **kwargs))
            for cb in callbacks
        ]

        if timeout:
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            # 取消超時的任務
            for task in pending:
                task.cancel()
            return [t.result() for t in done if not t.cancelled()]

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _emit_sequential(
        self,
        callbacks: list,
        args: tuple,
        kwargs: dict,
        timeout: float | None
    ) -> list[Any]:
        """順序執行監聽器"""
        results = []
        for cb in callbacks:
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        cb(*args, **kwargs),
                        timeout=timeout
                    )
                else:
                    result = await cb(*args, **kwargs)
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning(f"Listener {cb.__name__} timed out")
                results.append(None)
            except Exception as e:
                logger.error(f"Listener {cb.__name__} failed: {e}")
                results.append(e)
        return results

    def _emit_fire_and_forget(
        self,
        callbacks: list,
        args: tuple,
        kwargs: dict
    ) -> None:
        """發射後不等待（背景執行）"""
        for cb in callbacks:
            asyncio.create_task(self._safe_call(cb, args, kwargs))

    async def _safe_call(
        self,
        callback: Callable,
        args: tuple,
        kwargs: dict
    ) -> None:
        """安全執行回調（捕獲所有例外）"""
        try:
            await callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background listener failed: {e}")

# ═══════════════════════════════════════════════════════════════
# 使用範例：訂單事件系統
# ═══════════════════════════════════════════════════════════════
class OrderEventSystem:
    def __init__(self):
        self.emitter = AsyncEventEmitter(DispatchMode.PARALLEL)

    def setup_listeners(self):
        @self.emitter.on("order.created")
        async def send_confirmation_email(order: dict):
            await asyncio.sleep(0.1)  # 模擬發送郵件
            print(f"Email sent for order {order['id']}")
            return "email_sent"

        @self.emitter.on("order.created")
        async def update_inventory(order: dict):
            await asyncio.sleep(0.05)  # 模擬更新庫存
            print(f"Inventory updated for order {order['id']}")
            return "inventory_updated"

        @self.emitter.on("order.created")
        async def notify_warehouse(order: dict):
            await asyncio.sleep(0.02)  # 模擬通知倉庫
            print(f"Warehouse notified for order {order['id']}")
            return "warehouse_notified"

async def demo_order_system():
    system = OrderEventSystem()
    system.setup_listeners()

    order = {"id": "ORD-001", "items": ["item1", "item2"]}

    # 並行執行所有監聽器
    results = await system.emitter.emit(
        "order.created",
        order,
        timeout=1.0
    )
    print(f"Results: {results}")
```

**事件分發模式比較**：

| 模式 | 特點 | 適用場景 |
|------|------|----------|
| **PARALLEL** | 最快，同時執行所有監聽器 | 監聽器間無依賴關係 |
| **SEQUENTIAL** | 保證順序，可中斷 | 需要順序執行或優先級 |
| **FIRE_AND_FORGET** | 不阻塞發射者 | 日誌、非關鍵通知 |

---

#### 14.1.4 Factory 模式創建非同步資源

```python
# ═══════════════════════════════════════════════════════════════
# 同步版本：Factory 創建資源
# ═══════════════════════════════════════════════════════════════
import psycopg2
from typing import Protocol

class DatabaseConnection(Protocol):
    def execute(self, query: str) -> list: ...
    def close(self) -> None: ...

class PostgresConnection:
    def __init__(self, dsn: str):
        self._conn = psycopg2.connect(dsn)

    def execute(self, query: str) -> list:
        cursor = self._conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()

    def close(self) -> None:
        self._conn.close()

class DatabaseFactory:
    @staticmethod
    def create(db_type: str, dsn: str) -> DatabaseConnection:
        if db_type == "postgres":
            return PostgresConnection(dsn)
        raise ValueError(f"Unknown database type: {db_type}")

# ═══════════════════════════════════════════════════════════════
# 非同步版本：Async Factory 創建非同步資源
# ═══════════════════════════════════════════════════════════════
import asyncpg
from typing import Protocol, runtime_checkable
from contextlib import asynccontextmanager

@runtime_checkable
class AsyncDatabaseConnection(Protocol):
    async def execute(self, query: str) -> list: ...
    async def close(self) -> None: ...

class AsyncPostgresConnection:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def execute(self, query: str) -> list:
        async with self._pool.acquire() as conn:
            return await conn.fetch(query)

    async def close(self) -> None:
        await self._pool.close()

class AsyncDatabaseFactory:
    """非同步資料庫工廠 - 使用 async classmethod 創建"""

    _pools: dict[str, asyncpg.Pool] = {}  # 連線池快取

    @classmethod
    async def create(
        cls,
        db_type: str,
        dsn: str,
        min_size: int = 5,
        max_size: int = 20
    ) -> AsyncDatabaseConnection:
        """創建非同步資料庫連線"""
        if db_type == "postgres":
            # 重用現有連線池
            if dsn not in cls._pools:
                cls._pools[dsn] = await asyncpg.create_pool(
                    dsn,
                    min_size=min_size,
                    max_size=max_size
                )
            return AsyncPostgresConnection(cls._pools[dsn])
        raise ValueError(f"Unknown database type: {db_type}")

    @classmethod
    async def close_all(cls) -> None:
        """關閉所有連線池"""
        for pool in cls._pools.values():
            await pool.close()
        cls._pools.clear()

    @classmethod
    @asynccontextmanager
    async def connection(cls, db_type: str, dsn: str):
        """上下文管理器版本"""
        conn = await cls.create(db_type, dsn)
        try:
            yield conn
        finally:
            # 不關閉連線池，只釋放連線
            pass

# ═══════════════════════════════════════════════════════════════
# 進階：Async Abstract Factory（創建相關資源家族）
# ═══════════════════════════════════════════════════════════════
from abc import ABC, abstractmethod

class AsyncInfrastructureFactory(ABC):
    """非同步基礎設施抽象工廠"""

    @abstractmethod
    async def create_database(self) -> AsyncDatabaseConnection: ...

    @abstractmethod
    async def create_cache(self) -> "AsyncCache": ...

    @abstractmethod
    async def create_queue(self) -> "AsyncQueue": ...

    @abstractmethod
    async def close_all(self) -> None: ...

class ProductionInfrastructureFactory(AsyncInfrastructureFactory):
    def __init__(self, config: dict):
        self._config = config
        self._resources: list = []

    async def create_database(self) -> AsyncDatabaseConnection:
        db = await AsyncDatabaseFactory.create(
            "postgres",
            self._config["database_url"]
        )
        self._resources.append(db)
        return db

    async def create_cache(self) -> "AsyncCache":
        import aioredis
        redis = await aioredis.from_url(self._config["redis_url"])
        self._resources.append(redis)
        return redis

    async def create_queue(self) -> "AsyncQueue":
        # 創建訊息佇列連線
        ...

    async def close_all(self) -> None:
        for resource in reversed(self._resources):
            await resource.close()
        self._resources.clear()

# 使用範例
async def main():
    config = {
        "database_url": "postgresql://localhost/mydb",
        "redis_url": "redis://localhost",
    }

    factory = ProductionInfrastructureFactory(config)
    try:
        db = await factory.create_database()
        cache = await factory.create_cache()

        # 使用資源...
        result = await db.execute("SELECT * FROM users")

    finally:
        await factory.close_all()
```

**非同步 Factory 模式注意事項**：

```
┌─────────────────────────────────────────────────────────────┐
│  非同步資源創建的關鍵考量                                    │
│  ═════════════════════════                                  │
│  1. 使用 async classmethod 或 async __init__ 替代方案       │
│     (Python 不支援 async __init__，需用 Factory 或 create) │
│  2. 實現連線池重用，避免重複創建昂貴資源                    │
│  3. 使用 asynccontextmanager 確保資源正確釋放               │
│  4. 追蹤已創建資源，提供統一的 close_all() 方法             │
│  5. 關閉順序應與創建順序相反（後進先出）                    │
└─────────────────────────────────────────────────────────────┘
```

---

### 14.2 Producer-Consumer 實現

#### 14.2.1 asyncio.Queue 基礎實現

```python
import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Coroutine
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """任務資料結構"""
    id: str
    payload: Any
    created_at: datetime = None
    priority: int = 0  # 數字越小優先級越高

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

# ═══════════════════════════════════════════════════════════════
# 基礎 Producer-Consumer 實現
# ═══════════════════════════════════════════════════════════════

class SimpleProducerConsumer:
    """簡單的 Producer-Consumer 實現"""

    def __init__(self, queue_size: int = 100):
        self.queue: asyncio.Queue[Task] = asyncio.Queue(maxsize=queue_size)
        self._running = False

    async def producer(
        self,
        task_generator: Callable[[], Coroutine[Any, Any, Task | None]],
        name: str = "producer"
    ) -> None:
        """生產者：持續生成任務放入佇列"""
        logger.info(f"{name} started")
        while self._running:
            try:
                task = await task_generator()
                if task is None:
                    break
                await self.queue.put(task)
                logger.debug(f"{name} produced task {task.id}")
            except asyncio.CancelledError:
                logger.info(f"{name} cancelled")
                break
            except Exception as e:
                logger.error(f"{name} error: {e}")
        logger.info(f"{name} stopped")

    async def consumer(
        self,
        handler: Callable[[Task], Coroutine[Any, Any, None]],
        name: str = "consumer"
    ) -> None:
        """消費者：從佇列取出任務並處理"""
        logger.info(f"{name} started")
        while self._running or not self.queue.empty():
            try:
                # 使用 timeout 避免無限等待
                task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                try:
                    await handler(task)
                    logger.debug(f"{name} processed task {task.id}")
                finally:
                    self.queue.task_done()
            except asyncio.TimeoutError:
                continue  # 繼續等待
            except asyncio.CancelledError:
                logger.info(f"{name} cancelled")
                break
            except Exception as e:
                logger.error(f"{name} handler error: {e}")
        logger.info(f"{name} stopped")

    async def run(
        self,
        producers: list[Callable],
        handler: Callable[[Task], Coroutine],
        num_consumers: int = 3
    ) -> None:
        """啟動 Producer-Consumer 系統"""
        self._running = True

        # 啟動生產者
        producer_tasks = [
            asyncio.create_task(
                self.producer(gen, f"producer-{i}")
            )
            for i, gen in enumerate(producers)
        ]

        # 啟動消費者
        consumer_tasks = [
            asyncio.create_task(
                self.consumer(handler, f"consumer-{i}")
            )
            for i in range(num_consumers)
        ]

        # 等待所有生產者完成
        await asyncio.gather(*producer_tasks)

        # 等待佇列清空
        await self.queue.join()

        # 停止消費者
        self._running = False
        await asyncio.gather(*consumer_tasks, return_exceptions=True)

# 使用範例
async def example_basic():
    system = SimpleProducerConsumer(queue_size=50)

    # 定義任務生成器
    counter = 0
    async def generate_task() -> Task | None:
        nonlocal counter
        if counter >= 100:
            return None
        counter += 1
        await asyncio.sleep(0.01)  # 模擬生成延遲
        return Task(id=f"task-{counter}", payload={"value": counter})

    # 定義任務處理器
    async def handle_task(task: Task) -> None:
        await asyncio.sleep(0.05)  # 模擬處理時間
        print(f"Processed: {task.id}")

    await system.run(
        producers=[generate_task, generate_task],  # 2 個生產者
        handler=handle_task,
        num_consumers=5
    )
```

---

#### 14.2.2 多 Producer / 多 Consumer 進階實現

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Generic, TypeVar
from datetime import datetime
from enum import Enum, auto
import uuid

T = TypeVar("T")

class TaskStatus(Enum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

@dataclass
class TrackedTask(Generic[T]):
    """可追蹤狀態的任務"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    payload: T = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: Exception | None = None
    retries: int = 0
    max_retries: int = 3

class MultiProducerConsumer(Generic[T]):
    """
    多生產者/多消費者系統

    特點：
    - 支援多個獨立的生產者
    - 可配置的消費者數量
    - 任務狀態追蹤
    - 重試機制
    - 優雅關閉
    """

    def __init__(
        self,
        queue_size: int = 1000,
        num_consumers: int = 5,
        retry_delay: float = 1.0
    ):
        self.queue: asyncio.Queue[TrackedTask[T]] = asyncio.Queue(maxsize=queue_size)
        self.num_consumers = num_consumers
        self.retry_delay = retry_delay

        self._running = False
        self._producer_count = 0
        self._producer_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # 統計資訊
        self._stats = {
            "produced": 0,
            "consumed": 0,
            "failed": 0,
            "retried": 0,
        }

    async def register_producer(self) -> None:
        """註冊生產者"""
        async with self._producer_lock:
            self._producer_count += 1

    async def unregister_producer(self) -> None:
        """取消註冊生產者"""
        async with self._producer_lock:
            self._producer_count -= 1
            if self._producer_count == 0:
                self._shutdown_event.set()

    async def produce(self, payload: T) -> TrackedTask[T]:
        """提交任務到佇列"""
        task = TrackedTask(payload=payload)
        await self.queue.put(task)
        self._stats["produced"] += 1
        return task

    async def producer_wrapper(
        self,
        generator: Callable[[], Coroutine[Any, Any, T | None]],
        name: str
    ) -> None:
        """生產者包裝器"""
        await self.register_producer()
        try:
            while self._running:
                try:
                    payload = await generator()
                    if payload is None:
                        break
                    await self.produce(payload)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"{name} error: {e}")
        finally:
            await self.unregister_producer()

    async def consumer_wrapper(
        self,
        handler: Callable[[T], Coroutine[Any, Any, Any]],
        name: str
    ) -> None:
        """消費者包裝器"""
        while True:
            try:
                # 使用 wait_for 結合 shutdown_event
                task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                if self._shutdown_event.is_set() and self.queue.empty():
                    break
                continue
            except asyncio.CancelledError:
                break

            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()

            try:
                task.result = await handler(task.payload)
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                self._stats["consumed"] += 1

            except Exception as e:
                task.error = e
                task.retries += 1

                if task.retries < task.max_retries:
                    # 重試：重新放回佇列
                    task.status = TaskStatus.PENDING
                    await asyncio.sleep(self.retry_delay * task.retries)
                    await self.queue.put(task)
                    self._stats["retried"] += 1
                else:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    self._stats["failed"] += 1
                    logger.error(f"{name} task {task.id} failed after {task.retries} retries: {e}")

            finally:
                self.queue.task_done()

    async def run(
        self,
        producers: list[Callable[[], Coroutine[Any, Any, T | None]]],
        handler: Callable[[T], Coroutine[Any, Any, Any]]
    ) -> dict:
        """
        啟動系統

        Returns:
            統計資訊字典
        """
        self._running = True
        self._shutdown_event.clear()

        # 啟動消費者
        consumer_tasks = [
            asyncio.create_task(
                self.consumer_wrapper(handler, f"consumer-{i}")
            )
            for i in range(self.num_consumers)
        ]

        # 啟動生產者
        producer_tasks = [
            asyncio.create_task(
                self.producer_wrapper(gen, f"producer-{i}")
            )
            for i, gen in enumerate(producers)
        ]

        # 等待所有生產者完成
        await asyncio.gather(*producer_tasks, return_exceptions=True)

        # 等待佇列處理完成
        await self.queue.join()

        # 停止系統
        self._running = False

        # 取消消費者
        for task in consumer_tasks:
            task.cancel()
        await asyncio.gather(*consumer_tasks, return_exceptions=True)

        return self._stats.copy()
```

---

#### 14.2.3 背壓（Backpressure）處理

```python
import asyncio
from dataclasses import dataclass
from typing import Callable, Coroutine, Any
from enum import Enum, auto
import time

class BackpressureStrategy(Enum):
    """背壓處理策略"""
    BLOCK = auto()       # 阻塞等待（預設）
    DROP_OLDEST = auto() # 丟棄最舊的
    DROP_NEWEST = auto() # 丟棄最新的（不入隊）
    ADAPTIVE = auto()    # 動態調整生產速率

@dataclass
class BackpressureMetrics:
    """背壓監控指標"""
    queue_size: int
    queue_capacity: int
    utilization: float
    drop_count: int
    block_count: int
    avg_wait_time: float

class BackpressureQueue:
    """
    支援背壓處理的非同步佇列

    當佇列滿時，根據策略處理新任務：
    - BLOCK: 阻塞直到有空間（預設 asyncio.Queue 行為）
    - DROP_OLDEST: 丟棄最舊的任務，放入新任務
    - DROP_NEWEST: 直接丟棄新任務
    - ADAPTIVE: 動態減慢生產者速度
    """

    def __init__(
        self,
        maxsize: int = 1000,
        strategy: BackpressureStrategy = BackpressureStrategy.BLOCK,
        high_watermark: float = 0.8,
        low_watermark: float = 0.3
    ):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._strategy = strategy
        self._maxsize = maxsize
        self._high_watermark = high_watermark
        self._low_watermark = low_watermark

        # 統計
        self._drop_count = 0
        self._block_count = 0
        self._total_wait_time = 0.0
        self._wait_count = 0

        # 自適應背壓
        self._throttle_factor = 1.0  # 1.0 = 正常速度
        self._throttle_lock = asyncio.Lock()

    @property
    def utilization(self) -> float:
        """佇列使用率"""
        return self._queue.qsize() / self._maxsize

    @property
    def metrics(self) -> BackpressureMetrics:
        """取得監控指標"""
        return BackpressureMetrics(
            queue_size=self._queue.qsize(),
            queue_capacity=self._maxsize,
            utilization=self.utilization,
            drop_count=self._drop_count,
            block_count=self._block_count,
            avg_wait_time=(
                self._total_wait_time / self._wait_count
                if self._wait_count > 0 else 0
            )
        )

    async def put(self, item: Any, timeout: float | None = None) -> bool:
        """
        放入任務，根據策略處理背壓

        Returns:
            True 如果成功放入，False 如果被丟棄
        """
        if self._strategy == BackpressureStrategy.ADAPTIVE:
            await self._adaptive_throttle()

        if self._queue.full():
            if self._strategy == BackpressureStrategy.DROP_NEWEST:
                self._drop_count += 1
                return False

            elif self._strategy == BackpressureStrategy.DROP_OLDEST:
                try:
                    self._queue.get_nowait()
                    self._drop_count += 1
                except asyncio.QueueEmpty:
                    pass

        start_time = time.monotonic()
        try:
            if timeout:
                await asyncio.wait_for(self._queue.put(item), timeout)
            else:
                await self._queue.put(item)

            wait_time = time.monotonic() - start_time
            if wait_time > 0.001:  # 超過 1ms 視為等待
                self._block_count += 1
                self._total_wait_time += wait_time
                self._wait_count += 1

            return True

        except asyncio.TimeoutError:
            self._drop_count += 1
            return False

    async def get(self) -> Any:
        """取出任務"""
        item = await self._queue.get()

        # 更新自適應節流因子
        if self._strategy == BackpressureStrategy.ADAPTIVE:
            await self._update_throttle()

        return item

    def task_done(self) -> None:
        """標記任務完成"""
        self._queue.task_done()

    async def join(self) -> None:
        """等待所有任務完成"""
        await self._queue.join()

    def empty(self) -> bool:
        """佇列是否為空"""
        return self._queue.empty()

    async def _adaptive_throttle(self) -> None:
        """自適應節流：根據佇列使用率調整等待時間"""
        async with self._throttle_lock:
            if self.utilization > self._high_watermark:
                # 超過高水位，增加節流
                self._throttle_factor = min(
                    self._throttle_factor * 1.5,
                    10.0  # 最大 10 倍減速
                )
            elif self.utilization < self._low_watermark:
                # 低於低水位，減少節流
                self._throttle_factor = max(
                    self._throttle_factor * 0.8,
                    1.0  # 最小正常速度
                )

        if self._throttle_factor > 1.0:
            await asyncio.sleep(0.01 * self._throttle_factor)

    async def _update_throttle(self) -> None:
        """消費時更新節流因子"""
        async with self._throttle_lock:
            if self.utilization < self._low_watermark:
                self._throttle_factor = 1.0
```

**背壓策略比較**：

| 策略 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| **BLOCK** | 不丟失資料 | 可能導致生產者阻塞過久 | 資料不可丟失的場景 |
| **DROP_OLDEST** | 保持最新資料 | 丟失歷史資料 | 即時性優先（監控、日誌） |
| **DROP_NEWEST** | 實現簡單 | 可能丟失重要資料 | 突發流量緩衝 |
| **ADAPTIVE** | 自動調節，平衡性能 | 實現複雜 | 生產環境推薦 |

---

#### 14.2.4 優雅關閉

```python
import asyncio
import signal
from typing import Callable, Coroutine, Any, Set
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class GracefulShutdownManager:
    """
    優雅關閉管理器

    確保：
    1. 停止接受新任務
    2. 等待進行中的任務完成
    3. 設定超時強制終止
    4. 按順序清理資源
    """

    def __init__(self, shutdown_timeout: float = 30.0):
        self.shutdown_timeout = shutdown_timeout
        self._shutdown_event = asyncio.Event()
        self._active_tasks: Set[asyncio.Task] = set()
        self._cleanup_handlers: list[Callable[[], Coroutine]] = []
        self._is_shutting_down = False

    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down

    def register_cleanup(self, handler: Callable[[], Coroutine]) -> None:
        """註冊清理處理器"""
        self._cleanup_handlers.append(handler)

    def track_task(self, task: asyncio.Task) -> asyncio.Task:
        """追蹤任務"""
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return task

    async def shutdown(self) -> None:
        """執行優雅關閉"""
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        logger.info("Initiating graceful shutdown...")

        # 1. 發送關閉信號
        self._shutdown_event.set()

        # 2. 等待活動任務完成（有超時）
        if self._active_tasks:
            logger.info(f"Waiting for {len(self._active_tasks)} active tasks...")

            done, pending = await asyncio.wait(
                self._active_tasks,
                timeout=self.shutdown_timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            # 3. 強制取消超時任務
            if pending:
                logger.warning(f"Force cancelling {len(pending)} tasks")
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        # 4. 執行清理處理器（反向順序）
        for handler in reversed(self._cleanup_handlers):
            try:
                await handler()
            except Exception as e:
                logger.error(f"Cleanup handler failed: {e}")

        logger.info("Shutdown complete")

    async def wait_for_shutdown(self) -> None:
        """等待關閉信號"""
        await self._shutdown_event.wait()

# ═══════════════════════════════════════════════════════════════
# 完整的 Producer-Consumer 系統（含優雅關閉）
# ═══════════════════════════════════════════════════════════════

class RobustProducerConsumer:
    """
    生產級 Producer-Consumer 實現

    特點：
    - 優雅關閉
    - 信號處理（SIGINT, SIGTERM）
    - 健康檢查
    - 錯誤恢復
    """

    def __init__(
        self,
        queue_size: int = 1000,
        num_consumers: int = 5,
        shutdown_timeout: float = 30.0
    ):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.num_consumers = num_consumers
        self.shutdown_manager = GracefulShutdownManager(shutdown_timeout)

        self._running = False
        self._producers_done = asyncio.Event()

    def _setup_signal_handlers(self) -> None:
        """設定信號處理"""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self._handle_signal(sig))
            )

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """處理關閉信號"""
        logger.info(f"Received signal {sig.name}")
        self._running = False
        await self.shutdown_manager.shutdown()

    @asynccontextmanager
    async def running(self):
        """上下文管理器：自動處理啟動和關閉"""
        self._running = True
        self._setup_signal_handlers()

        try:
            yield self
        finally:
            self._running = False
            await self.shutdown_manager.shutdown()

    async def run(
        self,
        producers: list[Callable[[], Coroutine]],
        handler: Callable[[Any], Coroutine]
    ) -> None:
        """運行系統"""
        async with self.running():
            # 啟動消費者和生產者（略，完整版請見上文）
            pass
```

---

### 14.3 Worker Pool 最佳實踐

#### 14.3.1 asyncio.Semaphore 控制併發

```python
import asyncio
from typing import TypeVar, Callable, Coroutine, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
import time

T = TypeVar("T")
R = TypeVar("R")

@dataclass
class PoolStats:
    """Worker Pool 統計"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_workers: int
    max_workers: int
    avg_task_time: float

class AsyncWorkerPool:
    """
    非同步 Worker Pool

    使用 Semaphore 限制併發數量，適用於 I/O bound 任務
    """

    def __init__(self, max_workers: int = 10):
        self._semaphore = asyncio.Semaphore(max_workers)
        self._max_workers = max_workers
        self._active_count = 0
        self._lock = asyncio.Lock()

        # 統計
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._total_time = 0.0

    @property
    def stats(self) -> PoolStats:
        return PoolStats(
            total_tasks=self._total_tasks,
            completed_tasks=self._completed_tasks,
            failed_tasks=self._failed_tasks,
            active_workers=self._active_count,
            max_workers=self._max_workers,
            avg_task_time=(
                self._total_time / self._completed_tasks
                if self._completed_tasks > 0 else 0
            )
        )

    @asynccontextmanager
    async def acquire(self):
        """取得 worker slot"""
        await self._semaphore.acquire()
        async with self._lock:
            self._active_count += 1
        try:
            yield
        finally:
            async with self._lock:
                self._active_count -= 1
            self._semaphore.release()

    async def submit(self, coro: Coroutine[Any, Any, R]) -> R:
        """提交單一任務"""
        self._total_tasks += 1
        start_time = time.monotonic()

        async with self.acquire():
            try:
                result = await coro
                self._completed_tasks += 1
                return result
            except Exception:
                self._failed_tasks += 1
                raise
            finally:
                self._total_time += time.monotonic() - start_time

    async def map(
        self,
        func: Callable[[T], Coroutine[Any, Any, R]],
        items: list[T],
        return_exceptions: bool = False
    ) -> list[R]:
        """並行處理多個項目"""
        async def wrapped(item: T) -> R:
            return await self.submit(func(item))

        tasks = [asyncio.create_task(wrapped(item)) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
```

---

#### 14.3.2 ProcessPoolExecutor 用於 CPU 密集

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar, Callable, Any
import multiprocessing as mp

T = TypeVar("T")
R = TypeVar("R")

# CPU 密集型任務必須定義在模組級別（可被 pickle）
def cpu_intensive_task(n: int) -> int:
    """CPU 密集型任務：計算質數"""
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
    return sum(1 for i in range(2, n) if is_prime(i))

class ProcessPool:
    """
    進程池封裝

    適用於 CPU 密集型任務，如：
    - 影像處理
    - 數據壓縮
    - 密碼雜湊
    - 科學計算
    """

    def __init__(self, max_workers: int | None = None):
        self._max_workers = max_workers or mp.cpu_count()
        self._executor: ProcessPoolExecutor | None = None

    async def __aenter__(self):
        self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def run(self, func: Callable[..., R], *args: Any) -> R:
        """在進程池中執行函數"""
        if not self._executor:
            raise RuntimeError("ProcessPool not initialized. Use 'async with'.")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    async def map(self, func: Callable[[T], R], items: list[T]) -> list[R]:
        """並行處理多個項目"""
        tasks = [self.run(func, item) for item in items]
        return await asyncio.gather(*tasks)
```

---

#### 14.3.3 ThreadPoolExecutor 用於阻塞 I/O

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Callable, Any
import functools

T = TypeVar("T")
R = TypeVar("R")

class ThreadPool:
    """
    執行緒池封裝

    適用於阻塞 I/O 操作，如：
    - 同步資料庫驅動
    - 檔案系統操作（非 aiofiles）
    - 第三方同步 SDK
    - DNS 查詢
    """

    def __init__(self, max_workers: int = 10, thread_name_prefix: str = "worker"):
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix
        self._executor: ThreadPoolExecutor | None = None

    async def __aenter__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=self._thread_name_prefix
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def run(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """在執行緒池中執行阻塞函數"""
        if not self._executor:
            raise RuntimeError("ThreadPool not initialized. Use 'async with'.")
        loop = asyncio.get_running_loop()
        if kwargs:
            func = functools.partial(func, **kwargs)
        return await loop.run_in_executor(self._executor, func, *args)

    async def map(self, func: Callable[[T], R], items: list[T]) -> list[R]:
        """並行處理多個項目"""
        tasks = [self.run(func, item) for item in items]
        return await asyncio.gather(*tasks)
```

---

#### 14.3.4 混合策略

```python
from enum import Enum, auto
from dataclasses import dataclass

class TaskType(Enum):
    """任務類型"""
    IO_ASYNC = auto()      # 非同步 I/O（使用 asyncio）
    IO_BLOCKING = auto()   # 阻塞 I/O（使用 ThreadPool）
    CPU_BOUND = auto()     # CPU 密集（使用 ProcessPool）

@dataclass
class PoolConfig:
    """池配置"""
    async_semaphore: int = 100    # 非同步併發限制
    thread_workers: int = 20       # 執行緒數
    process_workers: int = None    # 進程數（None = CPU 核心數）

class HybridExecutor:
    """
    混合執行器

    智能選擇執行策略：
    - IO_ASYNC: 直接使用 asyncio（最高效）
    - IO_BLOCKING: ThreadPoolExecutor
    - CPU_BOUND: ProcessPoolExecutor
    """

    def __init__(self, config: PoolConfig | None = None):
        self._config = config or PoolConfig()
        self._async_semaphore = asyncio.Semaphore(self._config.async_semaphore)
        self._thread_pool: ThreadPoolExecutor | None = None
        self._process_pool: ProcessPoolExecutor | None = None

    async def __aenter__(self):
        import multiprocessing as mp
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self._config.thread_workers
        )
        self._process_pool = ProcessPoolExecutor(
            max_workers=self._config.process_workers or mp.cpu_count()
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)

    async def run(
        self,
        func: Callable[..., R],
        *args: Any,
        task_type: TaskType = TaskType.IO_ASYNC,
        **kwargs: Any
    ) -> R:
        """執行任務，自動選擇執行器"""
        loop = asyncio.get_running_loop()

        if task_type == TaskType.IO_ASYNC:
            async with self._async_semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

        elif task_type == TaskType.IO_BLOCKING:
            if kwargs:
                func = functools.partial(func, **kwargs)
            return await loop.run_in_executor(self._thread_pool, func, *args)

        elif task_type == TaskType.CPU_BOUND:
            if kwargs:
                func = functools.partial(func, **kwargs)
            return await loop.run_in_executor(self._process_pool, func, *args)
```

---

#### 14.3.5 I/O bound vs CPU bound 選擇指引

| 特徵 | I/O Bound | CPU Bound |
|------|-----------|-----------|
| **瓶頸** | 等待外部資源（網路、磁碟） | 計算處理能力 |
| **特點** | 大部分時間在等待 | 大部分時間在計算 |
| **最佳方案** | `asyncio` + `aiohttp`/`aiofiles` | `ProcessPoolExecutor` |
| **次選方案** | `ThreadPoolExecutor` | 多進程 + 共享記憶體 |
| **併發數** | 可以很高（100-1000） | 受限於 CPU 核心數 |
| **GIL 影響** | 無（等待時釋放 GIL） | 嚴重（需多進程繞過） |

**任務類型判斷流程**：

```
┌─────────────────────────────────────────────────────────────┐
│  任務類型判斷流程                                            │
│  ═══════════════                                            │
│                                                              │
│  Q: 任務主要在做什麼？                                       │
│  ├── 等待網路/磁碟 → I/O Bound                              │
│  │   ├── 有 async 版本？ → asyncio + Semaphore             │
│  │   └── 只有同步版本？ → ThreadPoolExecutor               │
│  │                                                          │
│  └── 大量計算 → CPU Bound                                   │
│      ├── 可 pickle？ → ProcessPoolExecutor                 │
│      └── 不可 pickle？ → 重構或使用共享記憶體               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

#### 14.3.6 併發數量調優原則

```python
import multiprocessing as mp
from dataclasses import dataclass

@dataclass
class ConcurrencyConfig:
    """併發配置建議"""

    @staticmethod
    def for_io_bound(
        target_throughput: int = 1000,  # 每秒請求數
        avg_latency_ms: float = 100     # 平均延遲（毫秒）
    ) -> int:
        """
        I/O bound 併發數計算

        Little's Law: L = λW
        併發數 = 吞吐量 × 延遲
        """
        concurrency = int(target_throughput * (avg_latency_ms / 1000))
        return int(concurrency * 1.2)  # 加上 20% 緩衝

    @staticmethod
    def for_cpu_bound() -> int:
        """CPU bound 併發數 = CPU 核心數"""
        return mp.cpu_count()

    @staticmethod
    def for_mixed(io_ratio: float = 0.7, cpu_ratio: float = 0.3) -> dict:
        """混合負載配置"""
        cpu_count = mp.cpu_count()
        return {
            "async_semaphore": 100,
            "thread_workers": cpu_count * 4,
            "process_workers": cpu_count,
        }
```

**調優檢查清單**：

```
┌─────────────────────────────────────────────────────────────┐
│  併發調優檢查清單                                            │
│  ═══════════════                                            │
│                                                              │
│  □ 測量實際延遲和吞吐量基準                                 │
│  □ 監控記憶體使用（每個連線/任務的開銷）                    │
│  □ 檢查外部服務限制（API rate limit）                       │
│  □ 設定合理的超時時間                                       │
│  □ 使用漸進式增加併發數測試                                 │
│  □ 監控錯誤率和延遲變化                                     │
│  □ 考慮背壓和熔斷機制                                       │
│  □ 區分峰值和平均負載配置                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

#### 14.3.7 資源清理最佳實踐

```python
from contextlib import asynccontextmanager, AsyncExitStack
from typing import AsyncGenerator, Any

class ResourceManager:
    """資源管理器 - 確保所有資源正確清理"""

    def __init__(self):
        self._exit_stack = AsyncExitStack()
        self._resources: list[tuple[str, Any]] = []

    async def __aenter__(self):
        await self._exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 反向清理所有資源
        for name, resource in reversed(self._resources):
            try:
                if hasattr(resource, 'close'):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
            except Exception as e:
                logger.error(f"Failed to cleanup {name}: {e}")
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def register(self, name: str, resource: Any) -> Any:
        """註冊資源"""
        self._resources.append((name, resource))
        return resource
```

---

### 快速參考

#### 併發模式速查表

| 場景 | 推薦方案 | 併發控制 | 注意事項 |
|------|----------|----------|----------|
| HTTP API 呼叫 | `aiohttp` + `asyncio.Semaphore` | 100-500 | 尊重 rate limit |
| 資料庫查詢 | `asyncpg` + 連線池 | 10-50 | 池大小 = 預期併發 |
| 檔案讀寫 | `aiofiles` 或 `ThreadPool` | 10-50 | 注意磁碟 I/O 限制 |
| 影像處理 | `ProcessPoolExecutor` | CPU 核心數 | 資料序列化開銷 |
| 混合負載 | `HybridExecutor` | 按類型配置 | 合理劃分任務類型 |

#### 設計模式非同步整合速查

| 模式 | 同步關鍵字 | 非同步關鍵字 | 額外考量 |
|------|------------|--------------|----------|
| Strategy | `def method()` | `async def method()` | 使用 `Protocol` 定義介面 |
| Command | `execute()` | `async execute()` | `Semaphore` 控制批次 |
| Observer | `emit()` | `await emit()` | 選擇分發模式 |
| Factory | `create()` | `async create()` | 連線池重用 |

---

## 十五、演進與遷移策略

> 「重構是持續的小步改進，而非一次性的大爆炸重寫。」

本章提供系統化的遷移方法論，協助 AI 工具在理解現有代碼結構後，規劃安全、可驗證的演進路徑。

---

## 15.1 模式遷移步驟

### 核心原則

```
安全遷移三原則：
1. 小步前進：每次變更可獨立測試與回滾
2. 測試先行：遷移前確保現有行為有測試覆蓋
3. 並行運行：新舊實現共存，逐步切換
```

### 場景一：硬編碼 → Factory Pattern

**遷移前代碼**：

```python
# ❌ 硬編碼：新增類型需修改多處
class ReportGenerator:
    def generate(self, report_type: str, data: dict) -> str:
        if report_type == "pdf":
            # PDF 生成邏輯
            ...
        elif report_type == "excel":
            # Excel 生成邏輯
            ...
        else:
            raise ValueError(f"Unknown report type: {report_type}")
```

**遷移步驟**：

```
步驟 1: 提取介面
    ↓
步驟 2: 為每個分支創建具體類
    ↓
步驟 3: 實現 Factory 邏輯
    ↓
步驟 4: 替換原始調用
    ↓
步驟 5: 移除舊代碼
```

**遷移後代碼**：

```python
# ✅ Factory 模式
class ReportRenderer(Protocol):
    def render(self, data: dict) -> str: ...

class RendererFactory:
    _renderers: dict[str, Type[ReportRenderer]] = {}

    @classmethod
    def register(cls, report_type: str, renderer_class: Type[ReportRenderer]):
        cls._renderers[report_type] = renderer_class

    @classmethod
    def create(cls, report_type: str) -> ReportRenderer:
        if renderer_class := cls._renderers.get(report_type):
            return renderer_class()
        raise ValueError(f"Unknown: {report_type}")
```

### 場景二：條件分支 → Strategy Pattern

**遷移前**：多個 if-elif 處理不同定價策略

**遷移後**：

```python
class PricingStrategy(ABC):
    @abstractmethod
    def calculate_discount(self, context: PricingContext) -> float: ...

class VipPricing(PricingStrategy):
    def calculate_discount(self, context: PricingContext) -> float:
        discount = 0.2
        if context.quantity >= 10:
            discount += (1 - discount) * 0.05
        return discount
```

### 場景三：緊耦合 → Observer Pattern

**遷移前**：Order 直接呼叫 EmailService、SmsService、InventoryService

**遷移後**：

```python
# 事件定義
class OrderCompletedEvent(DomainEvent):
    order_id: str
    user_id: str
    items: list[dict]

# 發布者
class Order:
    def complete(self, order_data: dict):
        self._save(order_data)
        self._publisher.publish(OrderCompletedEvent(...))

# 訂閱者
@receiver(order_created)
def on_order_created(sender, order, **kwargs):
    send_confirmation_email(order)
```

---

## 15.2 技術債優先級框架

### 四象限評估模型

```
                        高影響
                           │
        ┌──────────────────┼──────────────────┐
        │   🔥 緊急        │   📋 重要        │
        │  (立即處理)      │  (規劃排期)      │
        │                  │                  │
高緊迫 ─┼──────────────────┼──────────────────┼─ 低緊迫
        │                  │                  │
        │   🕳️ 填坑        │   📝 規劃        │
        │  (順手修復)      │  (長期改善)      │
        └──────────────────┼──────────────────┘
                           │
                        低影響
```

### 評估維度

| 維度 | 評分標準 | 權重 |
|------|----------|------|
| **業務影響** | 影響核心流程 5 分，無影響 1 分 | 2x |
| **修復成本** | XS=1, S=2, M=3, L=4, XL=5 | 1x |
| **風險程度** | 可能導致生產事故 5 分 | 1.5x |
| **依賴關係** | 阻擋其他任務 5 分 | 1x |

### 優先級計算

```python
def calculate_priority(item: TechDebtItem) -> Priority:
    urgency = (item.business_impact * 2 + item.risk_level * 1.5) / 3.5
    impact = (item.business_impact + item.dependency_score) / 2

    if urgency >= 4 and impact >= 4:
        return Priority.CRITICAL
    elif impact >= 4:
        return Priority.HIGH
    elif urgency >= 4:
        return Priority.MEDIUM
    else:
        return Priority.LOW
```

---

## 15.3 API 向後相容策略

### 版本化策略比較

| 策略 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| **URL 路徑** `/v1/` | 直觀、便於路由 | URL 污染 | 主版本升級 |
| **請求頭** `Accept: vnd.api+v1` | URL 穩定 | 測試複雜 | 漸進演進 |
| **查詢參數** `?version=1` | 簡單 | 不夠優雅 | 快速原型 |

### 棄用流程

```
Week 1-2        Week 3-4        Week 5-6        Week 7-8
─────────────────────────────────────────────────────→

[發布 v2]       [雙寫期]        [v1 棄用通知]    [v1 關閉]
```

**棄用響應頭**：

```python
response.headers["Deprecation"] = "true"
response.headers["Sunset"] = "2024-06-01"
response.headers["Link"] = '<https://api.example.com/docs/v2>; rel="successor-version"'
```

### 雙寫期處理

```python
class DualWriteRepository:
    def __init__(self, legacy_repo, new_repo, phase: MigrationPhase):
        self._legacy = legacy_repo
        self._new = new_repo
        self._phase = phase

    async def save(self, entity):
        match self._phase:
            case MigrationPhase.SHADOW:
                result = await self._legacy.save(entity)
                await self._new.save(entity)  # 影子寫入
                return result
            case MigrationPhase.DUAL_WRITE:
                # 兩邊都寫，任一失敗則回滾
                ...
            case MigrationPhase.NEW_PRIMARY:
                # 新系統為主
                ...
```

---

## 章節總結

| 模式遷移 | 技術債管理 | API 演進 |
|----------|------------|----------|
| 小步重構 | 四象限評估 | 版本化策略 |
| 測試先行 | ROI 排序 | 棄用流程 |
| 並行運行 | 持續追蹤 | 雙寫期處理 |

**AI 工具實踐**：
- 使用 `find_referencing_symbols` 識別影響範圍
- 使用 `write_memory` 追蹤遷移進度
- 每個階段設置檢查點

---

## 十六、AI 特有挑戰

本章節針對 AI 輔助程式設計中的獨特挑戰，提供系統化的解決方案與最佳實踐。

---

## 16.1 安全審查清單（OWASP 對照）

### 基礎檢查

```markdown
### 憑證與密鑰
- [ ] 無硬編碼密鑰（API keys、passwords、tokens）
- [ ] 敏感配置使用環境變數
- [ ] .gitignore 包含所有敏感檔案

### SQL 注入防護
- [ ] 使用參數化查詢
- [ ] ORM 操作避免原始 SQL 拼接
- [ ] 動態表名經過白名單驗證

### 路徑遍歷防護
- [ ] 使用者輸入的檔案路徑經過正規化
- [ ] 禁止 ../ 等相對路徑操作
- [ ] 檔案操作限制在指定目錄內
```

### 進階檢查

```markdown
### Prompt Injection 防護
- [ ] 使用者輸入與系統指令明確分離
- [ ] 輸入長度限制與格式驗證
- [ ] 敏感操作需要額外確認機制

### 輸出消毒
- [ ] HTML 輸出經過跳脫處理（XSS 防護）
- [ ] 錯誤訊息不暴露系統細節
- [ ] 日誌記錄遮蔽敏感資訊
```

### OWASP Top 10 對照表

| OWASP 項目 | AI 生成代碼常見問題 | 檢查要點 |
|------------|---------------------|----------|
| **A01 存取控制失效** | 缺少權限檢查 | 每個端點驗證權限 |
| **A02 加密失效** | 使用弱加密、明文儲存 | bcrypt/argon2 + TLS |
| **A03 注入攻擊** | 字串拼接 SQL | 參數化查詢 |
| **A04 不安全設計** | 缺乏威脅建模 | 設計階段納入安全 |
| **A05 安全配置錯誤** | 預設密碼、debug 模式 | 環境分離配置 |
| **A06 易受攻擊元件** | 使用過時依賴 | 啟用 Dependabot |

---

## 16.2 上下文長度管理策略

### 檔案拆分原則

| 指標 | 閾值 | 行動 |
|------|------|------|
| 檔案行數 | > 300 行 | 依職責拆分模組 |
| 函式數量 | > 10 個 | 依功能分組拆分 |
| 循環複雜度 | > 10 | 重構為更小單元 |
| 匯入數量 | > 15 個 | 檢視職責是否過多 |

### 跨檔案引用的 Prompt 技巧

```markdown
## 有效的上下文提供

❌ 錯誤：貼上整個檔案（500 行）

✅ 正確：提供相關摘要
「修改 UserService.create_user，相關上下文：

介面定義：
- UserRepository.save(user: User) -> User
- EmailService.send_welcome(email: str) -> None

目前實作簽章：
- async def create_user(self, data: UserCreate) -> User

需求：增加重複 Email 檢查」
```

### 大型重構策略

```markdown
## 分階段執行

第一階段：分析與規劃
- [ ] 識別影響範圍
- [ ] 建立依賴圖
- [ ] 設置回滾計畫

第二階段：基礎設施
- [ ] 建立新的抽象層
- [ ] 實作轉接器

第三階段：漸進遷移
- [ ] 依優先級遷移
- [ ] 每個模組獨立驗證

第四階段：清理
- [ ] 移除舊代碼
- [ ] 更新文件
```

---

## 16.3 AI 輔助 Code Review 流程

### 完整流程

```
1. AI 初審           2. 人工覆審          3. AI 修正          4. 最終驗證
┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
│ 自動化  │   →     │ 判斷與  │   →     │ 根據    │   →     │ 合併    │
│ 檢查    │         │ 決策    │         │ 回饋    │         │ 確認    │
└─────────┘         └─────────┘         └─────────┘         └─────────┘
```

### AI 初審 Prompt 模板

```markdown
請對以下代碼進行審查：

## 審查維度（按優先級）
1. **安全性**：注入風險、敏感資訊
2. **正確性**：邏輯錯誤、邊界條件
3. **效能**：演算法複雜度、N+1 查詢
4. **可維護性**：代碼清晰度、測試覆蓋

## 輸出格式
- 位置：[檔案:行號]
- 嚴重度：[🚨 Critical / ⚠️ Warning / 💡 Suggestion]
- 描述：[問題說明]
- 建議：[修正方式]
```

### 人工覆審重點

| 面向 | AI 能力 | 人工必須審查 |
|------|---------|--------------|
| 業務邏輯 | ❌ 無法判斷 | 邏輯是否符合需求 |
| 架構決策 | ❌ 缺乏全局 | 是否符合設計 |
| 權衡取捨 | ❌ 難以評估 | 技術債是否可接受 |
| 團隊慣例 | ❌ 不知道 | 是否符合約定 |

### 常見問題清單

| 類別 | 問題 | 頻率 |
|------|------|------|
| 安全 | 缺少輸入驗證 | 高 |
| 安全 | 硬編碼敏感資訊 | 高 |
| 正確性 | 邊界條件忽略 | 高 |
| 正確性 | 錯誤處理不完整 | 高 |
| 效能 | N+1 查詢 | 高 |
| 可維護性 | 命名不當 | 高 |

---

## 本章摘要

| 主題 | 核心要點 | 實踐建議 |
|------|----------|----------|
| **安全審查** | 多層檢查 + OWASP 對照 | 自動化安全掃描 |
| **上下文管理** | 檔案拆分 + 任務分解 | 300 行上限 |
| **Code Review** | 人機協作四階段 | AI 初審 + 人工決策 |

### 關鍵原則

1. **AI 不可信任安全性**：所有安全代碼需人工審查
2. **小上下文大成效**：精準的小上下文優於模糊的大上下文
3. **人機互補**：AI 處理重複檢查，人工處理判斷決策
4. **持續優化**：記錄問題模式，定期優化 Prompt

---

## 參考資源

- [Refactoring Guru - Design Patterns](https://refactoring.guru/design-patterns)
- [Martin Fowler 官方網站](https://martinfowler.com/)
- [Patterns of Enterprise Application Architecture](https://martinfowler.com/books/eaa.html)
- [Refactoring: Improving the Design of Existing Code](https://www.goodreads.com/book/show/44936.Refactoring)
- [The Pragmatic Engineer - Martin Fowler](https://newsletter.pragmaticengineer.com/p/martin-fowler)
- [Python asyncio 官方文件](https://docs.python.org/3/library/asyncio.html)
- [Real Python - Async IO in Python](https://realpython.com/async-io-python/)

---

*文件版本: 1.3 | 最後更新: 2026-01-26*
*整合來源: GoF Design Patterns + Martin Fowler Software Engineering Principles + AI Coding Workflow Best Practices + Python Concurrency Patterns + Testing Strategies + Resilience Patterns + Security Best Practices*
