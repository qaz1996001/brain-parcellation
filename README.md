## new venv install

``` bash
pip install uv 
git clone ....
cd ./brain-parcellation
uv sync
source .venv/bin/activate
export PYTHONPATH=$(pwd) && python3 backend/app/main.py
```


## system python install

``` bash
pip install uv
git clone ....
cd ./brain-parcellation
uv pip install -r pyproject.toml --system
export PYTHONPATH=$(pwd) && python3 backend/app/main.py

conda activate shhai
conda deactivate
```

```bash 
conda activate tf_2_14

cd /var/www/brain-parcellation && conda activate tf_2_14 && export PYTHONPATH=$(pwd) &&  python3 backend/app/main.py
cd /var/www/brain-parcellation && conda activate tf_2_14 && export PYTHONPATH=$(pwd) &&  python3 funboost_cli_user.py


conda activate tf_2_14

```

#### MRA_BRAIN 

```bash 
cd ./brain-parcellation

```

```bash 
export PYTHONPATH=$(pwd) &&  python3 code_ai/pipeline/pipeline_aneurysm_tensorflow.py \
 --ID 14914694_20220905_MR_21109050071 \
  --Inputs /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/MRA_BRAIN.nii.gz \
   --Output_folder /mnt/e/pipeline/sean/rename_nifti \
   --InputsDicomDir /mnt/e/pipeline/sean/rename_dicom/14914694_20220905_MR_21109050071/MRA_BRAIN
```
#### WMH

```bash 

 &&  python3 code_ai/pipeline/pipeline_synthseg_wmh_tensorflow.py \
 --ID 14914694_20220905_MR_21109050071 \
  --Inputs /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/T2FLAIR_AXI.nii.gz \
   --Output_folder /mnt/e/pipeline/sean/rename_nifti \
   --InputsDicomDir /mnt/e/pipeline/sean/rename_dicom/14914694_20220905_MR_21109050071/T2FLAIR_AXI
```
```bash 
export PYTHONPATH=$(pwd) &&  python3 code_ai/pipeline/pipeline_wmh_tensorflow.py \
 --ID 14914694_20220905_MR_21109050071 \
  --Inputs /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/T2FLAIR_AXI.nii.gz \
  /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/synthseg_T2FLAIR_AXI_original_WMH_PVS.nii.gz \
   /mnt/e/pipeline/sean/rename_nifti/14914694_20220905_MR_21109050071/synthseg_T2FLAIR_AXI_original_synthseg5.nii.gz \
   --Output_folder /mnt/e/pipeline/sean/rename_nifti \
   --InputsDicomDir /mnt/e/pipeline/sean/rename_dicom/14914694_20220905_MR_21109050071/T2FLAIR_AXI
```
```bash 
sudo apt install unzip
conda activate tf_2_14
uv pip install -r pyproject.toml --system

cd /mnt/d/00_Chen/Task04_git && conda activate tf_2_14 && export PYTHONPATH=$(pwd) &&  python3 backend/app/main.py
cd /mnt/d/00_Chen/Task04_git && conda activate tf_2_14 && export PYTHONPATH=$(pwd) &&  python3 funboost_cli_user.py


conda activate tf_2_14

cat <<EOF | xargs -I{} bash -c 'conda run -n tf_2_14 bash -c "
cd /var/www/brain-parcellation && \
&export PYTHONPATH=\$(pwd) & \
python3 code_ai/pipeline/raw_diom_to_nii_inference.py \
  --input_dicom \"{}\" \
  --output_dicom /data/4TB1/pipeline/sean/rename_dicom \
  --output_nifti /data/4TB1/pipeline/sean/rename_nifti" &' 
/data/4TB1/raw_dicom/ai_team/sync_orthanc/5cbf750a-90427da9-fca347a6-45ca1732-6893b6c6
/data/4TB1/raw_dicom/ai_team/sync_orthanc/8cfadfd6-fe04d6c7-2d1f2c12-9e852e40-ef6777c7
/data/4TB1/raw_dicom/ai_team/sync_orthanc/9c4a93b7-b06654b8-dfe4c530-607a39d1-b75f0672
/data/4TB1/raw_dicom/ai_team/sync_orthanc/45fd38e7-8822c36d-43d1302c-77d831de-d9dbc470
/data/4TB1/raw_dicom/ai_team/sync_orthanc/1823ce5c-7fc2efe7-fb7c042a-9fe3edea-0b1660d9
/data/4TB1/raw_dicom/ai_team/sync_orthanc/5069a078-75659a75-78bbdf0d-4adf4b42-d3981727
/data/4TB1/raw_dicom/ai_team/sync_orthanc/bb7d7abf-77552121-92b730aa-d874dfcb-1b026701
/data/4TB1/raw_dicom/ai_team/sync_orthanc/cb71cc54-a9d2d7c0-8e7b0b76-e222a8f5-50440dd4
/data/4TB1/raw_dicom/ai_team/sync_orthanc/d6c3de1b-4356ed70-8cd30c60-5751ea45-f52934c0

EOF

```

### 

1. raw dicom -> rename dicom 
   1. DCOPEventDicomService.get_series_info
   2. dicom_to_nii - > process_dir
   3. DCOPEventDicomService.post_ope_no_task 
   4. DCOPEventDicomService.check_study_series_transfer_complete

2. rename dicom -> rename nifti 
   1. DCOPEventDicomService.check_study_series_transfer_complete
   2. dicom_to_nii - > dicom_2_nii_file
   3. DCOPEventDicomService.study_series_nifti_tool 
   4. DCOPEventDicomService.check_study_series_conversion_complete
   
3. rename nifti -> pipeline_inference 
   1. DCOPEventDicomService.check_study_series_conversion_complete 
   2. 
   3. DCOPEventDicomService.study_series_inference_nifti_tool
   4. DCOPEventDicomService.check_study_series_inference_complete
4. pipeline_inference -> upload 
   1. DCOPEventDicomService.check_study_upload_complete 


/sc:design 請分析現有相關的功能@backend 、@code_ai 、@pyproject.toml 、@System_Design.md、@system_analysis.md、@README.md，重新規劃並設計，讓其可以 devops-architect (infrastructure), performance-engineer (optimization), security-engineer (compliance) --introspect --ultrathink --sequential	 

# 如何執行單元測試

本專案包含多個單元測試，以下說明不同的執行方式：

## 1. 執行所有測試

### 使用 UV（推薦，但需要所有依賴）
```bash
# 安裝所有依賴（包括開發依賴）
uv sync --dev

# 執行所有測試
uv run pytest -v

# 執行測試並生成覆蓋率報告
uv run pytest --cov=code_ai --cov=backend --cov-report=html
uv run pytest  --cov=backend --cov-report=html
```

### 直接使用 pytest
```bash
# 如果 pytest 已安裝在虛擬環境中
.venv\Scripts\pytest -v
```

## 2. 執行特定測試文件

```bash
# 執行單一測試文件
uv run pytest tests/code_ai/test_pipeline_base.py -v

# 執行多個測試文件
uv run pytest tests/code_ai/test_pipeline_base.py tests/code_ai/test_task_base.py -v
```

## 3. 執行特定測試函數

```bash
# 執行特定類別的測試
uv run pytest tests/code_ai/test_pipeline_base.py::TestPipelineInput -v

# 執行特定測試函數
uv run pytest tests/code_ai/test_pipeline_base.py::TestPipelineInput::test_pipeline_input_validation -v
```

## 4. 使用測試執行腳本（避開依賴問題）

我建立了一個 `run_tests.py` 腳本，可以更簡單地執行測試：

```bash
# 執行預設的安全測試（避開有依賴問題的測試）
python run_tests.py

# 執行特定測試文件
python run_tests.py tests/code_ai/test_pipeline_base.py

# 詳細輸出
python run_tests.py -v

# 執行所有測試
python run_tests.py --all
```

## 5. 測試選項說明

### 常用 pytest 參數：
- `-v` 或 `--verbose`：詳細輸出
- `-x`：第一個失敗就停止
- `-k EXPRESSION`：只執行符合表達式的測試
- `--tb=short`：簡短的錯誤追蹤
- `--no-cov`：不計算覆蓋率（加快執行速度）
- `-s`：顯示 print 輸出

### 範例：
```bash
# 執行名稱包含 "pipeline" 的測試
uv run pytest -k pipeline -v

# 執行測試但跳過慢速測試
uv run pytest -m "not slow" -v

# 顯示前 10 個最慢的測試
uv run pytest --durations=10
```

## 6. 測試結構說明

我們的測試結構如下：
```
tests/
├── conftest.py                          # 全域測試配置和 fixtures
├── code_ai/
│   ├── test_pipeline_base.py           # Pipeline 基礎功能測試
│   ├── test_pipeline_main_refactored.py # 重構後的 pipeline 測試
│   ├── test_task_base.py               # Task 基礎功能測試
│   └── test_utils_database.py          # 資料庫工具測試
├── backend/
│   └── test_server.py                   # FastAPI 伺服器測試
└── test_performance_comparison.py       # 效能比較測試
```

## 7. 解決常見問題

### 問題：ModuleNotFoundError
如果遇到模組找不到的錯誤，可能是因為：
1. 依賴未安裝：執行 `uv sync --dev`
2. 路徑問題：確保在專案根目錄執行測試

### 問題：測試執行太慢
使用以下方式加速：
```bash
# 不計算覆蓋率
uv run pytest --no-cov

# 並行執行測試（需要安裝 pytest-xdist）
uv run pytest -n auto
```

### 問題：特定測試失敗
可以單獨執行該測試來調試：
```bash
# 執行單一測試並顯示詳細輸出
uv run pytest path/to/test.py::test_function -vvs
```

## 8. 測試範例執行

讓我們執行一個簡單的測試來驗證環境：

```bash
# 執行 pipeline 基礎測試
python -m pytest tests/code_ai/test_pipeline_base.py::TestPipelineType -v
```

這個測試不依賴外部模組，應該可以順利執行。py

## 9. 生成測試報告

### HTML 覆蓋率報告
```bash
uv run pytest --cov=code_ai --cov=backend --cov-report=html
# 報告會生成在 htmlcov/index.html
```

### 終端機覆蓋率報告
```bash
uv run pytest --cov=code_ai --cov=backend --cov-report=term-missing
```

## 10. 持續整合（CI）

專案包含 GitHub Actions 配置（`.github/workflows/test.yml`），會在每次推送時自動執行測試。
