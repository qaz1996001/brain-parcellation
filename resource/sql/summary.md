Let me break down this data collection system's database schema for you, explaining each table's purpose and interactions.

Database Overview: MES OMI (Manufacturing Execution System - Operations Management Interface)

The database appears to be designed for collecting and managing manufacturing process data, with multiple tables serving different roles in data tracking and analysis.

1. Tables with "collect" prefix (Data Collection Tables):
   
   a. `dcop_collect_bt` (Basic Table):
   - Purpose: Captures real-time data for each production second
   - Characteristics:
     * Extremely detailed, with 360 numeric fields (n01-n360)
     * 10 string fields (s01-s10)
     * Tracks machine parameters every second
   - Contains high-frequency, granular machine performance data
   - Unique key: VsPrimaryKey (seems to include tool ID, timestamp)

   b. `dcop_collect_bth` (Basic Table History):
   - Purpose: Historical archive of basic table data
   - Identical structure to `dcop_collect_bt`
   - Likely used for long-term storage and historical analysis
   - Currently empty in this schema

   c. `dcop_collect_ct` (Cycle Table):
   - Purpose: Captures data for each production cycle
   - Similar structure to basic table, but records data per production cycle
   - Contains less frequent, more summarized data
   - Useful for cycle-level performance analysis

   d. `dcop_collect_tt1` (Temporary Table):
   - Purpose: Temporary data storage
   - In this schema, contains a series of unique keys
   - Likely used for interim data processing or buffering

   e. `dcop_collect_cth` (Custom Table, possibly "Collected Table History"):
   - Most comprehensive table with 360 numeric and 10 string fields
   - Appears to be a more complex data collection table

2. Tables with "event" prefix (Event Tracking Tables):

   a. `dcop_event_bt` (Basic Event Table):
   - Purpose: Tracks machine events and status changes
   - Captures information like:
     * Event type (kind)
     * Event code
     * Event description
     * Timestamp
   - Helps in tracking machine states, errors, and significant events

   b. `dcop_event_bth` (Event History Table):
   - Similar to event basic table
   - Likely for long-term event log storage

   c. `dcop_event_code_bt` (Event Code Configuration):
   - Defines event codes and their meanings
   - Helps interpret event codes in other tables
   - Contains event descriptions, categories

3. Configuration and Parameter Tables:

   a. `dcop_conf_bt` (Basic Configuration Table):
   - Purpose: Stores configuration parameters for tools/machines
   - Tracks things like recipe configurations, measurement settings
   - Helps define how data should be collected and interpreted

   b. `dcop_param_tool_bt` (Tool Parameter Table):
   - Stores specific parameters for different tools
   - Tracks active configurations, rates, and other tool-specific settings

Interaction and Data Flow:
1. During production, `dcop_collect_bt` captures second-by-second data
2. Events are logged in `dcop_event_bt` with codes defined in `dcop_event_code_bt`
3. `dcop_collect_ct` summarizes data at production cycle level
4. `dcop_conf_bt` and `dcop_param_tool_bt` provide context and configuration for data collection
5. Historical tables (`*_bth`) serve as archives for long-term storage and analysis

Key Observations:
- Highly granular data collection system
- Supports multiple levels of data aggregation
- Flexible configuration for different tools and processes
- Built for detailed manufacturing performance tracking

Would you like me to elaborate on any specific aspect of this database schema?



讓我為您分解這個資料收集系統的資料庫模式，解釋每個資料表的用途和互動。

資料庫總覽： MES OMI (製造執行系統 - 作業管理介面)

這個資料庫似乎是專為收集和管理製造過程資料而設計，其中有多個資料表在資料追蹤和分析中扮演不同的角色。

1. 具有「collect」前綴的資料表（資料收集表）：

   a. dcop_collect_bt（基本表）： 
    - 目的：擷取每秒生產的即時資料 
    - 特性：非常詳細，
      - 有 360 個數值欄位 (n01-n360)
      - 10 個字串欄位 (s01-s10)
      - 每秒追蹤機器參數
      - 包含高頻率、細粒度的機器效能資料
      - 唯一鍵： VsPrimaryKey (似乎包含工具 ID、時間戳記)
    
    b. dcop_collect_bth（基本表歷史）： 
    - 目的：結構與 dcop_collect_bt 相同可能用於長期儲存和歷史分析目前在此模式中為空
    
    c. dcop_collect_ct（循環表）：
    - 目的：擷取每個生產週期的資料結構與基本表類似，但記錄每個生產週期的資料包含頻率較低、摘要較多的資料有助於週期層級的效能分析
    
    d. dcop_collect_tt1（臨時表）：
    - 目的：目的：臨時資料儲存在此模式中，包含一系列唯一鍵可能用於臨時資料處理或緩衝
   
    e. dcop_collect_cth （自訂表，可能是「收集表歷史」）：
    - 最全面的資料表，有 360 個數值欄位和 10 個字串欄位
    - 似乎是更複雜的資料收集表
   
2. 帶有「事件」前綴的表（事件追蹤表）：:

    a. dcop_event_bt （基本事件表）： 
    - 目的：追蹤機器事件和狀態變更 
    - 擷取下列資訊
      - 事件類型 (種類)
      - 事件代碼
      - 事件描述
      - 時間戳記
      - 有助於追蹤機器狀態、錯誤和重要事件
    
    b. dcop_event_bth（事件歷史表）： 
    - 目的：類似於事件基本表
    - 可能用於長期事件日誌儲存    
   
    c. dcop_event_code_bt（事件代碼組態）：
    - 定義事件代碼及其含義
    - 有助於詮釋其他資料表的事件代碼
    - 包含事件描述、類別配置和參數表
3. 配置和參數表：

    a. dcop_conf_bt（基本組態表）：
    - 目的：儲存工具/機器的組態參數
    - 追蹤配方配置、量測設定等資料
    - 有助於定義應該如何收集和解釋資料
   
    b. dcop_param_tool_bt（工具參數表）：：
    - 目的：儲存不同工具的特定參數
    - 追蹤作用中的組態、速率和其他工具特定設定
   
在生產過程中，dcop_collect_bt 擷取每秒的資料事件記錄在 dcop_event_bt 中，代碼定義在 dcop_event_code_bt 中
dcop_collect_ct 總結生產週期層級的資料dcop_conf_bt 和 dcop_param_tool_bt 為資料收集提供上下文和設定
歷史資料表 (*_bth) 可作為長期儲存和分析的檔案

主要觀察：

高度粒度化的資料收集系統
支援多層次的資料聚合
針對不同的工具和製程靈活配置
針對詳細的製造績效追蹤而建立
您需要我詳細說明這個資料庫架構的任何特定方面嗎？


---
```python
class StudyStatus(Enum):
    NEW = "new"                          # 新檢測到的study
    TRANSFERRING = "transferring"        # DICOM傳輸中
    TRANSFER_COMPLETE = "transfer_complete"  # DICOM傳輸完成
    CONVERTING = "converting"            # 轉換中
    CONVERSION_COMPLETE = "conversion_complete"  # 轉換完成
    INFERENCE_READY = "inference_ready"  # 準備推論
    INFERENCE_QUEUED = "inference_queued"  # 推論排隊中
    INFERENCE_RUNNING = "inference_running"  # 推論執行中
    INFERENCE_FAILED = "inference_failed"  # 推論失敗
    INFERENCE_COMPLETE = "inference_complete"  # 推論完成
    RESULTS_SENT = "results_sent"        # 結果已發送

class SeriesStatus(Enum):
    NEW = "new"                          # 新檢測到的series
    TRANSFERRING = "transferring"        # DICOM傳輸中
    TRANSFER_COMPLETE = "transfer_complete"  # DICOM傳輸完成
    CONVERTING = "converting"            # 轉換中
    CONVERSION_COMPLETE = "conversion_complete"  # 轉換完成

class InferenceTaskStatus(Enum):
    QUEUED = "queued"                    # 等待中
    RUNNING = "running"                  # 執行中
    FAILED = "failed"                    # 執行失敗
    COMPLETE = "complete"                # 執行完成
    SENT = "sent"                        # 結果已發送

```
---
