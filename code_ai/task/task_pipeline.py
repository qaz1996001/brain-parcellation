"""
推論任務管道模組 - 執行醫學影像推論任務。

此模組提供推論任務的執行框架，負責：
1. 推論命令的建立和管理
2. 推論任務的執行和監控
3. 狀態事件的發送和追蹤
4. 執行結果的收集和返回

主要函數
--------
task_pipeline_inference
    完整的推論流程函數，執行多步驟推論管道。
task_subprocess_inference
    單一命令執行函數，用於執行獨立的推論命令。

輔助函數
--------
_prepare_directories
    準備推論任務所需的目錄結構。
_build_inference_commands
    根據 NIFTI 和 DICOM 路徑建立推論命令。
_save_commands_to_file
    將推論命令儲存到 JSON 檔案。
_should_send_events
    判斷是否應該發送狀態事件。
_send_status_event
    發送狀態事件到同步 API。
_execute_inference_commands
    執行所有推論命令並收集結果。

工作流程
--------
推論任務的完整流程：

1. 參數驗證
   └─> InferenceTaskParams.model_validate()

2. 目錄準備
   └─> _prepare_directories()

3. 命令建立
   └─> _build_inference_commands()
   └─> 根據配置檔案分析需要的推論任務

4. 命令儲存
   └─> _save_commands_to_file()
   └─> 儲存到 JSON 檔案用於審計

5. 狀態通知（RUNNING）
   └─> _send_status_event(STUDY_INFERENCE_RUNNING)

6. 命令執行
   └─> _execute_inference_commands()
   └─> 依序執行所有推論命令

7. 狀態通知（COMPLETE）
   └─> _send_status_event(STUDY_INFERENCE_COMPLETE)
   └─> 包含執行結果

8. 結果返回
   └─> 返回序列化的執行結果

狀態事件
--------
推論任務會發送兩種狀態事件：

- STUDY_INFERENCE_RUNNING
  推論任務開始執行時發送，包含推論命令和任務參數。

- STUDY_INFERENCE_COMPLETE
  推論任務完成時發送，包含執行結果。

事件發送條件：
- 必須提供 study_uid 和 study_id
- 如果任一為 None，則不發送事件

依賴配置
--------
此模組依賴以下環境變數：

- PATH_PROCESS: 處理檔案根目錄
- PATH_JSON: JSON 檔案目錄
- PATH_LOG: 日誌檔案目錄
- UPLOAD_DATA_API_URL: 同步 API 基礎 URL

See Also
--------
code_ai.task.schema.intput_params : 任務參數模型定義
code_ai.utils.inference : 推論命令建立工具
backend.app.sync.schemas : 狀態事件資料結構

Notes
-----
此模組已完成階段一和階段二的重構：
- ✅ 類型安全和輸入驗證（階段一）
- ✅ 函數拆分和重構（階段二）
- ⏳ 錯誤處理和日誌（階段三，待處理）

參考：docs/code_ai/task/TASK_PIPELINE_IMPROVEMENT_PLAN.md
"""

import json
import os
import pathlib
import subprocess
import nb_log

from typing import Dict, Any, List, Tuple, Optional
from funboost import Booster, fct
from funboost.core.serialization import Serialization
from backend.app.sync.schemas import DCOPStatus, DCOPEventRequest
from backend.app.sync.urls import SYNC_PROT_OPE_NO
from code_ai.task.params import BoosterParamsMyAI, BoosterParamsMyRABBITMQ
from code_ai.task.schema.intput_params import InferenceTaskParams, SubprocessTaskParams
from code_ai.utils.inference import build_inference_cmd, InferenceCmd
from code_ai.utils.database import save_result_status_to_sqlalchemy

# 設置日誌記錄器
logger = nb_log.LogManager("task_pipeline_inference_queue").get_logger_and_add_handlers(
    log_filename="task_pipeline_inference_queue.log"
)


# ============================================================================
# 輔助函數 - 階段二：函數拆分和重構
# ============================================================================

def _prepare_directories() -> Dict[str, str]:
    """
    準備推論任務所需的目錄結構。

    此函數從環境變數讀取目錄路徑配置，並確保所有必要的目錄存在。
    如果目錄不存在，會自動創建。此函數在推論任務開始前調用，確保
    所有輸出目錄都已準備就緒。

    Returns
    -------
    Dict[str, str]
        包含以下鍵的字典：
        - 'json': JSON 檔案輸出目錄路徑，用於儲存推論結果的 JSON 檔案
        - 'log': 日誌檔案輸出目錄路徑，用於儲存推論任務的日誌檔案
        - 'cmd_tools': 推論命令檔案輸出目錄路徑，用於儲存推論命令的 JSON 檔案

        如果環境變數未設置，對應的值可能為 None。但函數會跳過 None 值，
        不會嘗試創建不存在的路徑。

    Notes
    -----
    此函數依賴以下環境變數：
    - PATH_PROCESS: 處理檔案根目錄，必須設置
    - PATH_JSON: JSON 檔案目錄，必須設置
    - PATH_LOG: 日誌檔案目錄，必須設置

    cmd_tools 目錄會自動建立在 PATH_PROCESS/Deep_cmd_tools。

    目錄創建行為：
    - 使用 os.makedirs(path, exist_ok=True)，如果目錄已存在不會報錯
    - 如果環境變數為 None，該路徑會被跳過，不會創建
    - 此函數不會驗證目錄創建是否成功，假設環境變數配置正確

    See Also
    --------
    os.makedirs : Python 標準庫函數，用於創建目錄
    os.getenv : 從環境變數讀取配置

    Examples
    --------
    基本使用：

    >>> directories = _prepare_directories()
    >>> print(directories['cmd_tools'])
    /path/to/process/Deep_cmd_tools
    >>> print(directories['json'])
    /path/to/json
    >>> print(directories['log'])
    /path/to/log

    檢查目錄是否存在：

    >>> directories = _prepare_directories()
    >>> import os
    >>> os.path.exists(directories['cmd_tools'])
    True
    """
    path_process = os.getenv("PATH_PROCESS")
    path_json = os.getenv("PATH_JSON")
    path_log = os.getenv("PATH_LOG")
    path_cmd_tools = os.path.join(path_process, "Deep_cmd_tools")
    
    directories = {
        'json': path_json,
        'log': path_log,
        'cmd_tools': path_cmd_tools,
    }
    
    # 確保所有目錄存在
    for path in directories.values():
        if path:
            os.makedirs(path, exist_ok=True)
    
    return directories


def _build_inference_commands(
    nifti_study_path: str,
    dicom_study_path: str
) -> InferenceCmd:
    """
    建立推論命令列表。

    根據 NIFTI 和 DICOM Study 路徑，分析需要執行的推論任務，
    並生成對應的命令字串列表。此函數是推論任務規劃的核心，
    決定哪些推論模型需要執行。

    Parameters
    ----------
    nifti_study_path : str
        NIFTI Study 目錄路徑。此路徑應包含已轉換的 NIFTI 檔案。
        路徑必須存在且可讀取，否則推論命令建立可能失敗。
        格式範例："/path/to/nifti/study_id"
    dicom_study_path : str
        DICOM Study 目錄路徑。此路徑應包含原始 DICOM 檔案。
        用於推論任務的元資料提取和結果對應。
        格式範例："/path/to/dicom/study_id"

    Returns
    -------
    InferenceCmd
        推論命令物件，包含以下屬性：
        - cmd_items: List[InferenceCmdItem]，每個項目包含：
          - study_id: Study 識別碼，格式為 "patient_id_study_date_modality_accession"
          - name: 推論任務名稱（如 'SynthSeg', 'Aneurysm', 'SeriesClassification'）
          - cmd_str: 要執行的完整命令字串，可直接用於 subprocess 執行
          - input_list: 輸入檔案路徑列表，包含所有需要的 NIFTI 檔案
          - output_list: 輸出檔案路徑列表，包含預期的輸出檔案位置
          - input_dicom_dir: DICOM 輸入目錄，用於後處理和結果對應

        如果 Study 中沒有符合條件的序列，cmd_items 將為空列表。

    Notes
    -----
    此函數會掃描 NIFTI Study 目錄，根據配置檔案判斷需要執行哪些推論任務。
    推論任務的選擇基於：
    1. Study 中可用的序列類型（T1, T2, DWI, FLAIR 等）
    2. 配置檔案中定義的推論任務映射規則
    3. 序列的完整性檢查（確保所有必要的檔案都存在）

    推論任務類型：
    - SynthSeg: 腦部組織分割，需要 T1 或 T2 序列
    - Aneurysm: 動脈瘤檢測，需要特定的血管序列
    - SeriesClassification: 序列分類，用於驗證序列類型

    如果 Study 中沒有符合條件的序列，cmd_items 將為空列表，但不會報錯。
    這種情況下，推論任務會正常完成，只是不執行任何推論命令。

    See Also
    --------
    build_inference_cmd : 實際執行推論命令建立的函數
    InferenceCmd : 推論命令資料結構
    InferenceCmdItem : 單個推論命令項目的資料結構

    Examples
    --------
    基本使用：

    >>> inference_cmd = _build_inference_commands(
    ...     "/path/to/nifti/study",
    ...     "/path/to/dicom/study"
    ... )
    >>> print(len(inference_cmd.cmd_items))
    3
    >>> print(inference_cmd.cmd_items[0].name)
    SynthSeg

    檢查推論命令內容：

    >>> inference_cmd = _build_inference_commands(
    ...     "/path/to/nifti/study",
    ...     "/path/to/dicom/study"
    ... )
    >>> for item in inference_cmd.cmd_items:
    ...     print(f"Task: {item.name}")
    ...     print(f"Command: {item.cmd_str}")
    ...     print(f"Inputs: {item.input_list}")
    ...     print(f"Outputs: {item.output_list}")
    Task: SynthSeg
    Command: python /path/to/synthseg.py --input /path/to/input.nii.gz
    Inputs: ['/path/to/T1.nii.gz']
    Outputs: ['/path/to/synthseg_output.nii.gz']

    處理空結果的情況：

    >>> inference_cmd = _build_inference_commands(
    ...     "/path/to/empty/study",
    ...     "/path/to/empty/dicom"
    ... )
    >>> print(len(inference_cmd.cmd_items))
    0
    >>> # 空列表表示沒有需要執行的推論任務
    """
    return build_inference_cmd(
        pathlib.Path(nifti_study_path),
        pathlib.Path(dicom_study_path)
    )


def _save_commands_to_file(
    inference_cmd: InferenceCmd,
    cmd_tools_path: str,
    dicom_study_path: str
) -> str:
    """
    儲存推論命令到 JSON 檔案。

    將推論命令列表序列化為 JSON 格式並寫入檔案，用於：
    1. 審計追蹤：記錄執行的推論命令，便於後續審計和追蹤
    2. 除錯：可以查看實際執行的命令，快速定位問題
    3. 重跑：可以根據 JSON 檔案重新執行命令，無需重新分析

    Parameters
    ----------
    inference_cmd : InferenceCmd
        推論命令物件，包含要執行的命令列表。如果 cmd_items 為空，
        將使用 dicom_study_path 的目錄名稱作為檔案名稱。
    cmd_tools_path : str
        命令檔案輸出目錄路徑。此目錄必須存在或可創建。
        格式範例："/path/to/process/Deep_cmd_tools"
    dicom_study_path : str
        DICOM Study 路徑，當 inference_cmd.cmd_items 為空時，
        使用此路徑的目錄名稱作為檔案名稱。
        格式範例："/path/to/dicom/study_id"

    Returns
    -------
    str
        寫入的 JSON 檔案完整路徑。路徑格式為：
        {cmd_tools_path}/{study_id}_cmd.json 或
        {cmd_tools_path}/{directory_name}_cmd.json

    Notes
    -----
    檔案命名規則：
    - 如果 cmd_items 不為空：使用第一個命令的 study_id
      格式：{study_id}_cmd.json
      範例：10089413_20210201_MR_21002010079_cmd.json
    - 如果 cmd_items 為空：使用 dicom_study_path 的目錄名稱
      格式：{directory_name}_cmd.json
      範例：study_id_cmd.json

    JSON 檔案內容為 cmd_items 列表的序列化結果，每個項目包含：
    - study_id: Study 識別碼
    - name: 推論任務名稱
    - cmd_str: 命令字串
    - input_list: 輸入檔案路徑列表
    - output_list: 輸出檔案路徑列表
    - input_dicom_dir: DICOM 輸入目錄

    檔案寫入行為：
    - 使用 'w' 模式打開檔案，如果檔案已存在會被覆蓋
    - JSON 序列化使用 json.dumps()，不包含縮排（緊湊格式）
    - 如果目錄不存在，會先創建目錄（由調用方確保）

    See Also
    --------
    json.dumps : Python 標準庫函數，用於 JSON 序列化
    InferenceCmd.model_dump : Pydantic 模型序列化方法

    Examples
    --------
    基本使用：

    >>> inference_cmd = InferenceCmd(cmd_items=[...])
    >>> file_path = _save_commands_to_file(
    ...     inference_cmd,
    ...     "/path/to/cmd_tools",
    ...     "/path/to/dicom/study"
    ... )
    >>> print(file_path)
    /path/to/cmd_tools/10089413_20210201_MR_21002010079_cmd.json

    處理空命令列表的情況：

    >>> inference_cmd = InferenceCmd(cmd_items=[])
    >>> file_path = _save_commands_to_file(
    ...     inference_cmd,
    ...     "/path/to/cmd_tools",
    ...     "/path/to/dicom/study_id"
    ... )
    >>> print(file_path)
    /path/to/cmd_tools/study_id_cmd.json

    讀取儲存的命令檔案：

    >>> file_path = _save_commands_to_file(inference_cmd, ...)
    >>> import json
    >>> with open(file_path, 'r') as f:
    ...     saved_commands = json.load(f)
    >>> print(len(saved_commands))
    3
    """
    # 決定檔案名稱：優先使用 study_id，否則使用目錄名稱
    if inference_cmd.cmd_items:
        study_id = inference_cmd.cmd_items[0].study_id
        cmd_output_path = os.path.join(cmd_tools_path, f"{study_id}_cmd.json")
    else:
        temp_id = os.path.basename(dicom_study_path)
        cmd_output_path = os.path.join(cmd_tools_path, f"{temp_id}_cmd.json")
    
    # 將命令列表序列化為 JSON 並寫入檔案
    with open(cmd_output_path, "w") as f:
        f.write(json.dumps(inference_cmd.model_dump()["cmd_items"]))
    
    return cmd_output_path


def _should_send_events(study_uid: Optional[str], study_id: Optional[str]) -> bool:
    """
    判斷是否應該發送狀態事件到 API。

    此函數用於統一判斷邏輯，避免重複的條件檢查。
    只有當 study_uid 和 study_id 都不為 None 時，才應該發送事件。
    此函數確保狀態事件的完整性和可追蹤性。

    Parameters
    ----------
    study_uid : Optional[str]
        Study UID，來自 DICOM 標籤 (0020,000D)。
        格式範例："1.2.840.113619.2.44.5554020.7707121.19025.1612063861.703"
        如果為 None，表示此推論任務不屬於任何 DICOM Study。
    study_id : Optional[str]
        Study ID，系統內部使用的識別碼。
        格式範例："10089413_20210201_MR_21002010079"
        如果為 None，表示此推論任務沒有對應的 Study ID。

    Returns
    -------
    bool
        如果兩個參數都不為 None，返回 True；否則返回 False。
        True 表示應該發送狀態事件，False 表示不應該發送。

    Notes
    -----
    此函數消除了重複的條件檢查，符合 DRY 原則。
    用於統一判斷是否應該發送 RUNNING 和 COMPLETE 狀態事件。

    事件發送條件：
    - 兩個參數都必須不為 None
    - 如果任一參數為 None，則不發送事件
    - 空字串會被視為有效值（不為 None），但通常不應該出現

    使用場景：
    - 在發送 RUNNING 事件前調用
    - 在發送 COMPLETE 事件前調用
    - 確保狀態事件的完整性和可追蹤性

    See Also
    --------
    _send_status_event : 實際發送狀態事件的函數
    DCOPStatus : 狀態枚舉定義

    Examples
    --------
    兩個參數都不為 None（應該發送事件）：

    >>> _should_send_events("1.2.3.4", "study-123")
    True

    study_uid 為 None（不應該發送事件）：

    >>> _should_send_events(None, "study-123")
    False

    study_id 為 None（不應該發送事件）：

    >>> _should_send_events("1.2.3.4", None)
    False

    兩個參數都為 None（不應該發送事件）：

    >>> _should_send_events(None, None)
    False

    在條件判斷中使用：

    >>> study_uid = "1.2.3.4"
    >>> study_id = "study-123"
    >>> if _should_send_events(study_uid, study_id):
    ...     _send_status_event(study_uid, study_id, ...)
    ... else:
    ...     print("跳過事件發送")
    """
    return study_uid is not None and study_id is not None


def _send_status_event(
    study_uid: str,
    study_id: str,
    status: DCOPStatus,
    inference_cmd: InferenceCmd,
    params: InferenceTaskParams,
    result_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    發送推論任務狀態事件到同步 API。

    此函數統一處理狀態事件的建立和發送，消除了重複的事件建立邏輯。
    用於通知系統推論任務的狀態變化（RUNNING 或 COMPLETE）。

    Parameters
    ----------
    study_uid : str
        Study UID，來自 DICOM 標籤 (0020,000D)。
    study_id : str
        Study ID，系統內部使用的識別碼。
    status : DCOPStatus
        狀態枚舉值，應為以下之一：
        - DCOPStatus.STUDY_INFERENCE_RUNNING: 推論任務開始執行
        - DCOPStatus.STUDY_INFERENCE_COMPLETE: 推論任務完成
    inference_cmd : InferenceCmd
        推論命令物件，包含要執行的命令列表。
    params : InferenceTaskParams
        推論任務參數，包含輸入路徑和識別資訊。
    result_data : Optional[Dict[str, Any]], default None
        可選的結果資料。通常用於 COMPLETE 狀態時包含執行結果。
        格式：{'result': <序列化的結果字串>}

    Returns
    -------
    None
        此函數不返回值，事件通過異步隊列發送。

    Notes
    -----
    事件發送機制：
    - 使用 funboost 任務隊列異步發送 HTTP POST 請求
    - 請求發送到 UPLOAD_DATA_API_URL + SYNC_PROT_OPE_NO 端點
    - 事件資料包含完整的推論命令和任務參數，用於審計追蹤
    - 事件發送是異步的，不會阻塞主流程

    params_data 包含：
    - inference_item_cmd: 推論命令列表，包含所有要執行的命令
    - func_params: 任務參數的序列化結果，包含輸入路徑和識別資訊
    - task: funboost 任務狀態資訊，包含任務 ID 和執行狀態

    result_data 僅在 COMPLETE 狀態時包含，格式為：
    {'result': <JSON 序列化的執行結果>}
    執行結果包含每個命令的 stdout 和 stderr。

    事件發送失敗處理：
    - 如果事件發送失敗，不會影響推論任務的執行
    - 錯誤會被記錄到 funboost 的任務日誌中
    - 建議監控事件發送的成功率，確保狀態追蹤的完整性

    環境變數依賴：
    - UPLOAD_DATA_API_URL: 同步 API 基礎 URL，必須設置

    See Also
    --------
    DCOPEventRequest : 事件請求資料結構
    DCOPStatus : 狀態枚舉定義
    call_post_httpx : HTTP 請求發送函數

    Examples
    --------
    發送 RUNNING 狀態事件：

    >>> _send_status_event(
    ...     study_uid="1.2.3.4",
    ...     study_id="study-123",
    ...     status=DCOPStatus.STUDY_INFERENCE_RUNNING,
    ...     inference_cmd=inference_cmd,
    ...     params=params
    ... )

    發送 COMPLETE 狀態事件（包含結果）：

    >>> _send_status_event(
    ...     study_uid="1.2.3.4",
    ...     study_id="study-123",
    ...     status=DCOPStatus.STUDY_INFERENCE_COMPLETE,
    ...     inference_cmd=inference_cmd,
    ...     params=params,
    ...     result_data={'result': '{"status": "success"}'}
    ... )
    """
    from code_ai.task.task_dicom2nii import call_post_httpx
    
    # 構建 API URL
    upload_data_api_url = os.getenv("UPLOAD_DATA_API_URL")
    api_url = f"{upload_data_api_url}{SYNC_PROT_OPE_NO}"
    
    # 建立事件請求物件
    dcop_event = DCOPEventRequest(
        study_uid=study_uid,
        series_uid=None,  # Study 級別事件，series_uid 為 None
        study_id=study_id,
        ope_no=status.value,  # 操作編號對應狀態值
        tool_id="INFERENCE_TOOL",  # 工具識別碼
        params_data={
            "inference_item_cmd": inference_cmd.cmd_items,  # 推論命令列表
            "func_params": params.model_dump(),  # 任務參數序列化
            "task": fct.function_result_status.get_status_dict(),  # 任務狀態
        },
        result_data=result_data,  # 可選的結果資料
    )
    
    # 通過異步隊列發送事件
    call_post_httpx.push({
        "url": api_url,
        "data": dcop_event.model_dump_json(),
    })


def _execute_inference_commands(
    inference_cmd: InferenceCmd
) -> List[Tuple[str, str, str]]:
    """
    執行所有推論命令並收集執行結果。

    此函數依序執行推論命令列表中的每個命令，並收集標準輸出和標準錯誤。
    命令通過 subprocess 執行，支援 shell 命令和複雜的命令字串。

    Parameters
    ----------
    inference_cmd : InferenceCmd
        推論命令物件，包含要執行的命令列表。
        每個 InferenceCmdItem 包含：
        - cmd_str: 要執行的命令字串
        - 其他元資料（study_id, name, input_list, output_list 等）

    Returns
    -------
    List[Tuple[str, str, str]]
        執行結果列表，每個元組包含：
        - 第 0 個元素：命令字串
        - 第 1 個元素：標準輸出（stdout）的解碼字串
        - 第 2 個元素：標準錯誤（stderr）的解碼字串

    Notes
    -----
    執行機制：
    - 使用 subprocess.Popen 執行命令，支援 shell=True
    - 每個命令會等待完成（communicate() 會阻塞直到命令結束）
    - 標準輸出和錯誤都捕獲為位元組，然後解碼為 UTF-8 字串
    - 命令執行是同步的，前一個命令完成後才會執行下一個

    執行順序：
    - 命令按照 cmd_items 列表的順序依序執行
    - 前一個命令完成後才會執行下一個命令
    - 不支援並行執行（由 qps=1 配置保證）

    錯誤處理：
    - 目前不檢查命令的返回碼（returncode）
    - 錯誤輸出會包含在返回結果中，但不影響後續命令執行
    - 即使某個命令失敗，也會繼續執行後續命令
    - 階段三將添加錯誤處理邏輯，包括返回碼檢查和錯誤重試

    性能考量：
    - 命令依序執行，總執行時間為所有命令執行時間的總和
    - 如果某個命令執行時間很長，會阻塞整個推論流程
    - 建議在推論命令中設置超時機制，避免無限等待

    輸出處理：
    - stdout 和 stderr 都使用 UTF-8 解碼
    - 如果解碼失敗，可能會拋出 UnicodeDecodeError
    - 建議推論命令確保輸出使用 UTF-8 編碼

    See Also
    --------
    subprocess.Popen : Python 子進程執行函數
    InferenceCmdItem : 推論命令項目資料結構

    Examples
    --------
    >>> inference_cmd = InferenceCmd(cmd_items=[
    ...     InferenceCmdItem(cmd_str="python script.py", ...),
    ...     InferenceCmdItem(cmd_str="python another.py", ...)
    ... ])
    >>> results = _execute_inference_commands(inference_cmd)
    >>> print(len(results))
    2
    >>> print(results[0][0])  # 第一個命令字串
    python script.py
    >>> print(results[0][1])  # 第一個命令的標準輸出
    ...
    """
    result_list = []
    
    # 依序執行每個推論命令
    for inference_item in inference_cmd.cmd_items:
        # 執行命令並捕獲輸出
        process = subprocess.Popen(
            args=inference_item.cmd_str,
            shell=True,  # 允許 shell 命令
            stdout=subprocess.PIPE,  # 捕獲標準輸出
            stderr=subprocess.PIPE,  # 捕獲標準錯誤
        )
        stdout, stderr = process.communicate()  # 等待命令完成
        
        # 將結果加入列表（命令字串、stdout、stderr）
        result_list.append((
            inference_item.cmd_str,
            stdout.decode(),  # 解碼為 UTF-8 字串
            stderr.decode()   # 解碼為 UTF-8 字串
        ))
    
    return result_list

@Booster(
    BoosterParamsMyAI(
        queue_name="task_pipeline_inference_queue",
        user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
        qps=1,
    )
)
def task_pipeline_inference(func_params: Dict[str, Any]) -> str:
    """
    推論任務主函數 - 執行完整的推論管道流程。

    此函數是推論任務的入口點，負責協調整個推論流程：
    1. 驗證輸入參數
    2. 準備必要的目錄結構
    3. 分析 Study 並建立推論命令
    4. 儲存命令到檔案（用於審計）
    5. 發送 RUNNING 狀態事件
    6. 執行所有推論命令
    7. 發送 COMPLETE 狀態事件（包含結果）
    8. 返回執行結果

    Parameters
    ----------
    func_params : Dict[str, Any]
        任務參數字典，應包含以下鍵：
        - nifti_study_path (str, required): NIFTI Study 目錄路徑
        - dicom_study_path (str, required): DICOM Study 目錄路徑
        - study_uid (str, optional): Study UID，用於事件追蹤
        - study_id (str, optional): Study ID，用於事件追蹤

    Returns
    -------
    str
        JSON 序列化的執行結果字串。結果格式為：
        [
            (cmd_str_1, stdout_1, stderr_1),
            (cmd_str_2, stdout_2, stderr_2),
            ...
        ]

    Notes
    -----
    工作流程：
    1. **參數驗證**：使用 InferenceTaskParams 驗證輸入參數
    2. **目錄準備**：確保所有輸出目錄存在
    3. **命令建立**：根據 NIFTI 和 DICOM 路徑分析需要執行的推論任務
    4. **命令儲存**：將命令列表儲存到 JSON 檔案，用於審計和除錯
    5. **狀態通知**：如果提供了 study_uid 和 study_id，發送 RUNNING 事件
    6. **命令執行**：依序執行所有推論命令
    7. **結果通知**：如果提供了 study_uid 和 study_id，發送 COMPLETE 事件
    8. **結果返回**：返回序列化的執行結果

    狀態事件：
    - RUNNING 事件在命令執行前發送，通知系統推論任務已開始
    - COMPLETE 事件在命令執行後發送，包含執行結果

    錯誤處理：
    - 目前階段不包含錯誤處理（將在階段三添加）
    - 命令執行失敗不會中斷流程，錯誤資訊包含在返回結果中

    性能考量：
    - 命令依序執行，不並行（qps=1）
    - 每個命令會阻塞直到完成
    - 適合需要嚴格順序執行的推論任務
    - 總執行時間為所有命令執行時間的總和

    並發控制：
    - qps=1 確保每秒最多執行一個推論任務
    - 多個推論任務會排隊等待執行
    - 適合資源受限的環境，避免資源競爭

    See Also
    --------
    InferenceTaskParams : 輸入參數驗證模型
    _prepare_directories : 目錄準備函數
    _build_inference_commands : 命令建立函數
    _execute_inference_commands : 命令執行函數
    _send_status_event : 狀態事件發送函數

    Examples
    --------
    基本使用（包含事件追蹤）：

    >>> func_params = {
    ...     "nifti_study_path": "/path/to/nifti/study",
    ...     "dicom_study_path": "/path/to/dicom/study",
    ...     "study_uid": "1.2.3.4",
    ...     "study_id": "study-123"
    ... }
    >>> result = task_pipeline_inference(func_params)
    >>> print(result)
    '[["python script1.py", "output1", ""], ["python script2.py", "output2", ""]]'

    基本使用（不包含事件追蹤）：

    >>> func_params = {
    ...     "nifti_study_path": "/path/to/nifti/study",
    ...     "dicom_study_path": "/path/to/dicom/study"
    ... }
    >>> result = task_pipeline_inference(func_params)
    >>> # 不會發送狀態事件，但會執行推論命令
    """
    # 步驟 1: 驗證並解析輸入參數
    params = InferenceTaskParams.model_validate(func_params)
    
    # 步驟 2: 準備目錄結構
    directories = _prepare_directories()
    
    # 步驟 3: 建立推論命令
    inference_cmd = _build_inference_commands(
        params.nifti_study_path,
        params.dicom_study_path
    )
    
    # 步驟 4: 儲存命令到檔案（用於審計和除錯）
    _save_commands_to_file(
        inference_cmd,
        directories['cmd_tools'],
        params.dicom_study_path
    )
    
    # 步驟 5: 發送 RUNNING 狀態事件（如果應該發送）
    if _should_send_events(params.study_uid, params.study_id):
        _send_status_event(
            params.study_uid,
            params.study_id,
            DCOPStatus.STUDY_INFERENCE_RUNNING,
            inference_cmd,
            params
        )
    
    # 步驟 6: 執行推論命令
    result_list = _execute_inference_commands(inference_cmd)
    result = Serialization.to_json_str(result_list)
    
    # 步驟 7: 發送 COMPLETE 狀態事件（如果應該發送）
    if _should_send_events(params.study_uid, params.study_id):
        _send_status_event(
            params.study_uid,
            params.study_id,
            DCOPStatus.STUDY_INFERENCE_COMPLETE,
            inference_cmd,
            params,
            result_data={'result': result}
        )
    
    # 步驟 8: 返回執行結果
    return result


@Booster(
    BoosterParamsMyRABBITMQ(
        queue_name="task_subprocess_queue",
        concurrent_num=3,
        qps=1,
    )
)
def task_subprocess_inference(func_params: Dict[str, Any]) -> str:
    """
    子進程推論任務 - 執行單一命令並返回結果。

    此函數用於執行單一的推論命令，與 task_pipeline_inference 不同，
    它不處理完整的推論流程，只執行單一命令。通常用於：
    1. 測試單一推論命令
    2. 重新執行失敗的命令
    3. 並行執行多個獨立命令（concurrent_num=3）

    Parameters
    ----------
    func_params : Dict[str, Any]
        任務參數字典，應包含以下鍵：
        - cmd_str (str, required): 要執行的命令字串

    Returns
    -------
    str
        命令的標準輸出（stdout）解碼後的字串。
        如果命令執行失敗，標準錯誤會記錄到日誌，但不會包含在返回值中。

    Notes
    -----
    執行機制：
    - 使用 subprocess.Popen 執行命令，支援 shell=True
    - 命令會阻塞執行直到完成
    - 標準輸出和錯誤都捕獲，但只返回標準輸出

    日誌記錄：
    - 標準輸出記錄為 INFO 級別
    - 標準錯誤記錄為 WARN 級別
    - 日誌檔案：task_pipeline_inference_queue.log

    目錄準備：
    - 自動確保 cmd_tools 目錄存在
    - 目錄路徑：PATH_PROCESS/Deep_cmd_tools

    並發執行：
    - concurrent_num=3 允許最多 3 個任務並行執行
    - qps=1 限制每秒最多 1 個任務

    錯誤處理：
    - 目前不檢查命令返回碼（returncode）
    - 錯誤輸出記錄到日誌，但不影響返回值
    - 即使命令執行失敗，也會返回 stdout（可能為空字串）
    - 階段三將添加錯誤處理邏輯，包括返回碼檢查和異常處理

    返回值說明：
    - 只返回標準輸出（stdout），不包含標準錯誤（stderr）
    - 如果命令執行失敗，stdout 可能為空字串
    - 建議檢查返回值的長度，判斷命令是否成功執行

    See Also
    --------
    SubprocessTaskParams : 輸入參數驗證模型
    task_pipeline_inference : 完整的推論流程函數
    subprocess.Popen : Python 子進程執行函數

    Examples
    --------
    執行單一推論命令：

    >>> func_params = {
    ...     "cmd_str": "python /path/to/inference_script.py --input /path/to/input.nii.gz"
    ... }
    >>> output = task_subprocess_inference(func_params)
    >>> print(output)
    Inference completed successfully
    Processing time: 2.5s
    ...
    """
    # 步驟 1: 驗證並解析輸入參數
    params = SubprocessTaskParams.model_validate(func_params)
    
    # 步驟 2: 準備命令工具目錄
    path_process = os.getenv("PATH_PROCESS")
    path_cmd_tools = os.path.join(path_process, "Deep_cmd_tools")
    os.makedirs(path_cmd_tools, exist_ok=True)
    
    # 步驟 3: 執行命令
    cmd_str = params.cmd_str
    process = subprocess.Popen(
        args=cmd_str,
        shell=True,  # 允許 shell 命令
        stdout=subprocess.PIPE,  # 捕獲標準輸出
        stderr=subprocess.PIPE,  # 捕獲標準錯誤
    )
    stdout, stderr = process.communicate()  # 等待命令完成
    
    # 步驟 4: 記錄執行結果到日誌
    logger.info(stdout.decode())  # 標準輸出記錄為 INFO
    logger.warn(stderr.decode())  # 標準錯誤記錄為 WARN
    
    # 步驟 5: 返回標準輸出
    return stdout.decode()
