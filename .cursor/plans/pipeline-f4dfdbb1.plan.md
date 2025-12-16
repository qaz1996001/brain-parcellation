<!-- f4dfdbb1-44e8-4e7c-8c08-26fdb499a5b4 acc4c35b-cd5a-4b1c-a2b6-91d9da65fde0 -->
# TensorFlow Pipeline OOP 重構計畫

## 1. 架構盤點與需求整理

- 確認 `code_ai/pipeline/main.py`、`main5class.py`（現 synthseg CLI）與既有 `CmbPipeline` 共用哪些流程（CLI 參數解析、GPU 控制、輸出整理）。
- 梳理兩支 main 的特有行為（多任務 flag、輸出檔案命名、併行模式）以界定抽象層次。

## 2. 建立 SynthSeg 專用抽象層

- 以 `BasePipeline` 延伸 `SynthsegBasePipeline`（於 `code_ai/pipeline/core/synthseg_pipeline.py`），封裝共用步驟：輸入驗證、呼叫 `predict.py`/`main.py`、輸出路徑正規化。
- 針對多模型差異（標準版本、5class 版本）透過策略或 enum 指派適用的模型/參數組合。

## 3. 重構 `main.py`（一般 SynthSeg）

- 改寫為 `SynthsegPipelineApp`（類別或 CLI entry），負責解析參數並建立 `SynthsegPipeline` 實例。
- 讓 `main.py` 只保留 CLI + pipeline 實例化；具體執行邏輯移至新類別，整合 GPU 控制與錯誤處理。

## 4. 重構 `main5class.py`

- 依 (3) 做相同改寫，但使用 5-class 專屬子類（例如 `SynthsegFiveClassPipeline`）。
- 抽掉重複的檔案複製、日誌與輸出命名規則，沿用核心抽象層。

## 5. 驗證與回歸

- 以現有測試指令（或新增 smoke test）跑一次一般與 5-class 流程，確保 CLI 參數與輸出不變。
- 依序更新 TODO 狀態，並規劃後續 pipeline（aneurysm、wmh 等）套用相同架構。

### To-dos

- [ ] 盤點 synthseg main/main5class 流程
- [ ] 建立 SynthsegBasePipeline 抽象層
- [ ] 重構 main.py 為 pipeline 類別
- [ ] 重構 main5class.py 為 pipeline 類別
- [ ] 驗證兩管線與 CLI 相容性