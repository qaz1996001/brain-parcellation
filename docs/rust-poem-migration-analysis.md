# Rust Poem 替代 Backend 方案分析

## 1. 執行摘要

本文件分析使用 Rust Poem 框架替代現有 Python FastAPI backend 的可行性、優缺點及實施方案。

### 結論摘要

| 面向 | 評估 |
|------|------|
| 性能提升 | ⭐⭐⭐⭐⭐ 顯著提升 (3-10x) |
| 開發成本 | ⭐⭐ 高（需重寫全部程式碼）|
| 維護難度 | ⭐⭐⭐ 中等 |
| 生態系統 | ⭐⭐⭐ DICOM 相關庫較少 |
| 建議 | **部分採用** - 建議混合架構 |

---

## 2. 現有 Backend 技術棧分析

### 2.1 核心技術

| 組件 | 技術 | 版本 |
|------|------|------|
| Web Framework | FastAPI | >=0.115.12 |
| ASGI Server | Uvicorn | >=0.34.2 |
| Database | PostgreSQL | - |
| ORM | SQLAlchemy + Advanced Alchemy | 2.0.38 |
| Task Queue | Celery + funboost | 5.4.0 |
| Cache | Redis | 5.2.1 |
| HTTP Client | httpx | 0.28.1 |

### 2.2 DICOM 專用庫

| 庫 | 用途 |
|----|------|
| pydicom | DICOM 檔案讀寫 |
| pyorthanc | Orthanc 伺服器 API |
| nibabel | NIfTI 檔案處理 |
| SimpleITK | 影像處理與轉換 |

### 2.3 API 端點統計

| 模組 | 端點數量 | 複雜度 |
|------|----------|--------|
| Sync | 7 | 高 |
| Series | 4 | 中 |
| Rerun | 2 | 中 |
| Find | 3 | 低 |
| **總計** | **16** | - |

---

## 3. Rust Poem 框架分析

### 3.1 框架特性

```rust
// Poem 的簡潔 API 範例
use poem::{get, handler, listener::TcpListener, Route, Server};

#[handler]
fn hello() -> String {
    "Hello, World!".to_string()
}

#[tokio::main]
async fn main() {
    let app = Route::new().at("/", get(hello));
    Server::new(TcpListener::bind("0.0.0.0:3000"))
        .run(app)
        .await
        .unwrap();
}
```

### 3.2 核心優勢

| 特性 | 說明 |
|------|------|
| **100% Safe Rust** | 使用 `#![forbid(unsafe_code)]` |
| **OpenAPI 原生支援** | `poem-openapi` 自動生成文檔 |
| **極低延遲** | 無 GC，零成本抽象 |
| **型別安全** | 編譯時錯誤檢查 |
| **Tokio 生態** | 與 Rust 異步生態完全整合 |

### 3.3 性能比較

基於社群 Benchmark 數據：

```
Requests/sec (Hello World):
┌─────────────┬─────────────┬────────────┐
│ Framework   │ RPS         │ Latency    │
├─────────────┼─────────────┼────────────┤
│ Actix Web   │ 180,000+    │ 0.5ms      │
│ Axum        │ 165,000+    │ 0.6ms      │
│ Poem        │ 155,000+    │ 0.7ms      │
│ FastAPI     │ 15,000-20,000│ 5-10ms    │
└─────────────┴─────────────┴────────────┘
```

**預估性能提升：7-10 倍**

---

## 4. Rust 生態對應分析

### 4.1 直接對應的 Crates

| Python | Rust Crate | 成熟度 |
|--------|------------|--------|
| FastAPI | poem + poem-openapi | ⭐⭐⭐⭐⭐ |
| SQLAlchemy | sqlx / sea-orm / diesel | ⭐⭐⭐⭐⭐ |
| httpx | reqwest | ⭐⭐⭐⭐⭐ |
| Redis | redis-rs / deadpool-redis | ⭐⭐⭐⭐⭐ |
| Celery | - (需自建或用替代方案) | ⭐⭐ |
| Pydantic | serde + validator | ⭐⭐⭐⭐⭐ |

### 4.2 DICOM 相關 (關鍵挑戰)

| Python | Rust 替代方案 | 狀態 |
|--------|---------------|------|
| pydicom | dicom-rs | ⭐⭐⭐ 基本功能可用 |
| pyorthanc | 自行封裝 REST API | 需開發 |
| nibabel | nifti-rs | ⭐⭐⭐ 讀取可用 |
| SimpleITK | 無直接替代 | ❌ 重大挑戰 |

### 4.3 關鍵缺口分析

```
❌ SimpleITK - 無 Rust 原生替代
   影響: Series 分析模組、影像方向判斷
   解決方案:
   1. FFI 調用 C++ ITK
   2. 保留 Python 微服務
   3. WebAssembly 橋接

❌ Celery - 無直接等效
   影響: 背景任務處理
   解決方案:
   1. Tokio tasks + Redis 隊列
   2. 使用 RabbitMQ + lapin
   3. 自建任務調度器
```

---

## 5. 架構方案比較

### 方案 A: 完全重寫 (Full Rewrite)

```
┌─────────────────────────────────────────┐
│              Rust Poem Backend          │
├─────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  Sync   │ │ Series  │ │  Find   │   │
│  │ Module  │ │ Module  │ │ Module  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │         │
│  ┌────┴───────────┴───────────┴────┐   │
│  │        SQLx / Sea-ORM           │   │
│  └─────────────────┬───────────────┘   │
│                    │                    │
└────────────────────┼────────────────────┘
                     │
              ┌──────┴──────┐
              │  PostgreSQL │
              └─────────────┘
```

| 優點 | 缺點 |
|------|------|
| 最佳性能 | 開發週期長 (3-6 個月) |
| 統一技術棧 | DICOM 庫不成熟 |
| 長期維護簡單 | 學習曲線陡峭 |

**風險等級: 高**

---

### 方案 B: 混合架構 (推薦)

```
┌───────────────────────────────────────────────────────┐
│                    API Gateway                        │
│                   (Rust Poem)                         │
└───────────────┬───────────────────────┬───────────────┘
                │                       │
    ┌───────────┴───────────┐ ┌─────────┴─────────┐
    │    High Performance   │ │  DICOM Processing │
    │    Services (Rust)    │ │  Service (Python) │
    │                       │ │                   │
    │  • Sync API           │ │  • Series Analyze │
    │  • Find API           │ │  • Image Convert  │
    │  • Rerun API          │ │  • NIfTI Tools    │
    │  • Auth/Rate Limit    │ │                   │
    └───────────┬───────────┘ └─────────┬─────────┘
                │                       │
                └───────────┬───────────┘
                            │
                     ┌──────┴──────┐
                     │  PostgreSQL │
                     └─────────────┘
```

| 優點 | 缺點 |
|------|------|
| 漸進式遷移 | 系統複雜度增加 |
| 保留 DICOM 處理能力 | 需維護兩套程式碼 |
| 風險可控 | 服務間通訊開銷 |
| 快速獲得性能提升 | - |

**風險等級: 中**

---

### 方案 C: API Gateway + 現有後端

```
┌─────────────────────────────────────────┐
│         Rust Poem API Gateway           │
│  • Rate Limiting                        │
│  • Authentication                       │
│  • Request Validation                   │
│  • Response Caching                     │
│  • Load Balancing                       │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      Existing FastAPI Backend           │
│         (Unchanged)                     │
└─────────────────────────────────────────┘
```

| 優點 | 缺點 |
|------|------|
| 最低風險 | 性能提升有限 |
| 快速部署 | 未解決根本問題 |
| 零業務邏輯改動 | 增加架構層次 |

**風險等級: 低**

---

## 6. 詳細實施計畫 (方案 B)

### Phase 1: 基礎設施 (2-3 週)

```rust
// 專案結構
brain-parcellation-api/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── config.rs
│   ├── db/
│   │   ├── mod.rs
│   │   ├── models.rs
│   │   └── pool.rs
│   ├── api/
│   │   ├── mod.rs
│   │   ├── sync.rs
│   │   ├── find.rs
│   │   └── rerun.rs
│   └── services/
│       ├── mod.rs
│       └── dicom_proxy.rs
└── tests/
```

**Cargo.toml 依賴:**

```toml
[dependencies]
poem = { version = "3.0", features = ["openapi"] }
poem-openapi = "5.0"
tokio = { version = "1", features = ["full"] }
sqlx = { version = "0.8", features = ["runtime-tokio", "postgres", "json", "chrono"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.12", features = ["json"] }
deadpool-redis = "0.18"
tracing = "0.1"
tracing-subscriber = "0.3"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
```

### Phase 2: 核心 API 遷移 (3-4 週)

#### 2.1 資料模型轉換

```rust
// src/db/models.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct DcopEvent {
    #[sqlx(rename = "VsPrimaryKey")]
    pub vs_primary_key: String,
    pub tool_id: Option<String>,
    pub study_uid: Option<String>,
    pub series_uid: Option<String>,
    pub study_id: Option<String>,
    pub event_cate: Option<i32>,
    pub code_name: Option<String>,
    pub code_desc: Option<String>,
    pub params_data: Option<JsonValue>,
    pub result_data: Option<JsonValue>,
    pub ope_no: Option<String>,
    pub ope_name: Option<String>,
    pub claim_time: Option<DateTime<Utc>>,
    pub rec_time: Option<DateTime<Utc>>,
    pub create_time: Option<DateTime<Utc>>,
    pub update_time: Option<DateTime<Utc>>,
}
```

#### 2.2 API 端點實現

```rust
// src/api/sync.rs
use poem_openapi::{param::Query, payload::Json, Object, OpenApi};

#[derive(Object, Debug, Clone)]
pub struct StudyRequest {
    pub study_uid: String,
    pub series_uid: Option<String>,
}

#[derive(Object, Debug, Clone)]
pub struct StudyResponse {
    pub success: bool,
    pub data: Option<DcopEventDto>,
    pub message: Option<String>,
}

pub struct SyncApi;

#[OpenApi]
impl SyncApi {
    /// Get or create study by UUID
    #[oai(path = "/sync/study", method = "post")]
    async fn add_study(
        &self,
        pool: Data<&PgPool>,
        body: Json<StudyRequest>,
    ) -> Result<Json<StudyResponse>> {
        let result = sync_service::add_study_new(&pool, &body.study_uid).await?;
        Ok(Json(StudyResponse {
            success: true,
            data: Some(result.into()),
            message: None,
        }))
    }

    /// Check transfer complete
    #[oai(path = "/sync/study/transfer/complete", method = "post")]
    async fn check_transfer_complete(
        &self,
        pool: Data<&PgPool>,
        body: Json<TransferCheckRequest>,
    ) -> Result<Json<TransferCheckResponse>> {
        // Implementation
    }
}
```

### Phase 3: DICOM 代理服務 (2 週)

```rust
// src/services/dicom_proxy.rs
use reqwest::Client;

pub struct DicomProxyService {
    client: Client,
    python_service_url: String,
}

impl DicomProxyService {
    pub async fn analyze_series(&self, file_paths: Vec<String>) -> Result<Vec<SeriesAnalysis>> {
        let response = self.client
            .post(format!("{}/series/dicom/analyze/by-path", self.python_service_url))
            .json(&AnalyzeRequest { file_paths })
            .send()
            .await?;

        Ok(response.json().await?)
    }
}
```

### Phase 4: 整合測試與部署 (2 週)

```yaml
# docker-compose.yml
version: '3.8'
services:
  api-gateway:
    build: ./rust-api
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://...
      - PYTHON_SERVICE_URL=http://dicom-service:8000
    depends_on:
      - postgres
      - redis
      - dicom-service

  dicom-service:
    build: ./python-dicom
    ports:
      - "8000:8000"
    volumes:
      - /data/dicom:/data/dicom

  postgres:
    image: postgres:16

  redis:
    image: redis:7-alpine
```

---

## 7. 性能預估

### 7.1 API 響應時間比較

| 端點 | FastAPI (現況) | Rust Poem (預估) | 改善 |
|------|----------------|------------------|------|
| GET /sync/study | ~15ms | ~2ms | 7.5x |
| POST /sync/study | ~25ms | ~5ms | 5x |
| GET /find/study/pending | ~50ms | ~8ms | 6x |
| POST /series/analyze | ~200ms | ~180ms* | 1.1x |

*Series 分析仍需調用 Python 服務處理 DICOM

### 7.2 資源使用預估

```
                    FastAPI      Rust Poem
────────────────────────────────────────────
Memory (idle)       150-200 MB   10-15 MB
Memory (load)       500-800 MB   50-100 MB
CPU (idle)          2-5%         <1%
CPU (load)          60-80%       30-40%
Cold start          3-5s         <100ms
```

---

## 8. 風險與緩解措施

### 8.1 技術風險

| 風險 | 等級 | 緩解措施 |
|------|------|----------|
| DICOM 處理能力不足 | 高 | 保留 Python 微服務 |
| 團隊 Rust 經驗不足 | 中 | 漸進式遷移、培訓 |
| 第三方庫不穩定 | 中 | 選用成熟 crates |
| 調試困難 | 中 | 完善日誌、追蹤 |

### 8.2 業務風險

| 風險 | 等級 | 緩解措施 |
|------|------|----------|
| 遷移期間服務中斷 | 中 | 藍綠部署 |
| 功能回歸 | 中 | 完整測試套件 |
| 延誤時程 | 中 | 分階段交付 |

---

## 9. 成本效益分析

### 9.1 開發成本

| 項目 | 人月 | 說明 |
|------|------|------|
| 基礎設施搭建 | 1 | 專案架構、CI/CD |
| 核心 API 遷移 | 2 | Sync, Find, Rerun |
| DICOM 代理整合 | 1 | Python 服務介接 |
| 測試與調優 | 1 | 壓測、效能優化 |
| **總計** | **5** | - |

### 9.2 長期收益

| 收益項目 | 預估節省 |
|----------|----------|
| 伺服器成本 | 40-60% (資源減少) |
| 響應時間 | 提升 5-10x |
| 維護成本 | 長期降低 (型別安全) |
| 擴展能力 | 更好的並發處理 |

---

## 10. 建議與結論

### 10.1 最終建議

**採用方案 B: 混合架構**

理由：
1. **平衡性能與風險** - 核心 API 獲得顯著性能提升，DICOM 處理保留成熟方案
2. **漸進式遷移** - 可分階段交付，降低專案風險
3. **保留專業能力** - Python 生態在醫學影像處理領域仍具優勢
4. **快速回報** - 2-3 個月可見初步成效

### 10.2 實施優先級

```
Phase 1 (必要): API Gateway + 基礎設施
Phase 2 (高優先): Sync 和 Find 模組遷移
Phase 3 (中優先): Rerun 模組遷移
Phase 4 (可選): 探索純 Rust DICOM 處理
```

### 10.3 不建議完全重寫的原因

1. **SimpleITK 無替代** - 影像處理核心功能無法用 Rust 實現
2. **開發週期過長** - 完全重寫需 6+ 個月
3. **風險過高** - DICOM 處理出錯可能影響醫療診斷

---

## 附錄 A: 參考資源

### Rust Poem 相關
- [Poem GitHub](https://github.com/poem-web/poem)
- [Poem Documentation](https://docs.rs/poem/latest/poem/)
- [Poem OpenAPI](https://docs.rs/poem-openapi/latest/poem_openapi/)

### 性能比較
- [Rust Web Framework Benchmark](https://github.com/randiekas/rust-web-framework-benchmark)
- [Actix vs Axum vs Rocket Comparison](https://dev.to/leapcell/rust-web-frameworks-compared-actix-vs-axum-vs-rocket-4bad)

### 醫學影像處理
- [dicom-rs](https://github.com/Enet4/dicom-rs)
- [nifti-rs](https://github.com/Enet4/nifti-rs)

---

## 附錄 B: 快速開始範例

```rust
// main.rs - Minimal Poem API example
use poem::{listener::TcpListener, Route, Server};
use poem_openapi::{payload::Json, Object, OpenApi, OpenApiService};

#[derive(Object)]
struct HealthResponse {
    status: String,
    version: String,
}

struct Api;

#[OpenApi]
impl Api {
    #[oai(path = "/health", method = "get")]
    async fn health(&self) -> Json<HealthResponse> {
        Json(HealthResponse {
            status: "ok".to_string(),
            version: "1.0.0".to_string(),
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    let api_service = OpenApiService::new(Api, "Brain Parcellation API", "1.0")
        .server("http://localhost:3000/api/v1");

    let ui = api_service.swagger_ui();

    let app = Route::new()
        .nest("/api/v1", api_service)
        .nest("/docs", ui);

    Server::new(TcpListener::bind("0.0.0.0:3000"))
        .run(app)
        .await
}
```

---

*文件版本: 1.0*
*建立日期: 2026-01-30*
*作者: Claude Code Analysis*
