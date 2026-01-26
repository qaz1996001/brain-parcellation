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

## 參考資源

- [Refactoring Guru - Design Patterns](https://refactoring.guru/design-patterns)
- [Martin Fowler 官方網站](https://martinfowler.com/)
- [Patterns of Enterprise Application Architecture](https://martinfowler.com/books/eaa.html)
- [Refactoring: Improving the Design of Existing Code](https://www.goodreads.com/book/show/44936.Refactoring)
- [The Pragmatic Engineer - Martin Fowler](https://newsletter.pragmaticengineer.com/p/martin-fowler)

---

*文件版本: 1.0 | 最後更新: 2026-01-26*
*整合來源: GoF Design Patterns + Martin Fowler Software Engineering Principles*
