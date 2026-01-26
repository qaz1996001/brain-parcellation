import json
import logging
import re
import shutil
import os
import pathlib
import subprocess
from typing import List, Dict, Any

import httpx
import pandas as pd
import pydicom
from funboost import Booster
from pydicom import dcmread
from pyorthanc import Orthanc, Study

from code_ai.dicom2nii.convert import ModalityProcessingStrategy, MRAcquisitionTypeProcessingStrategy, \
    MRRenameSeriesProcessingStrategy
from code_ai.dicom2nii.convert import DwiProcessingStrategy, ADCProcessingStrategy, EADCProcessingStrategy, \
    SWANProcessingStrategy
from code_ai.dicom2nii.convert import ESWANProcessingStrategy, MRABrainProcessingStrategy, MRANeckProcessingStrategy
from code_ai.dicom2nii.convert import MRAVRBrainProcessingStrategy, MRAVRNeckProcessingStrategy, T1ProcessingStrategy
from code_ai.dicom2nii.convert import T2ProcessingStrategy, ASLProcessingStrategy, DSCProcessingStrategy
from code_ai.dicom2nii.convert import RestingProcessingStrategy, DTIProcessingStrategy, CVRProcessingStrategy
from code_ai.dicom2nii.convert import NullEnum
from code_ai.dicom2nii.convert import MRSeriesRenameEnum

from code_ai.dicom2nii.convert import dicom_rename_mr_postprocess
from code_ai.dicom2nii.convert import convert_nifti_postprocess
from code_ai.task.schema import intput_params
from code_ai.task.params import BoosterParamsMyRABBITMQ
from code_ai.utils.database import save_result_status_to_sqlalchemy
from backend.app.sync import urls as sync_urls
from backend.app.sync.schemas import DCOPEventRequest, DCOPStatus

logger = logging.getLogger(__name__)


def get_output_study(dicom_ds):
    """
    從 DICOM 資料集取得輸出研究資料夾名稱。

    此函數會從 DICOM 資料集中提取必要資訊，並產生標準化的研究資料夾名稱。
    如果 DICOM 資料集為 None 或無法產生有效的資料夾名稱，則返回 None。

    Parameters
    ----------
    dicom_ds : pydicom.dataset.FileDataset or None
        DICOM 資料集物件，包含 DICOM 標籤資訊。如果為 None，則直接返回 None。

    Returns
    -------
    str or None
        研究資料夾名稱，格式為 `{patient_id}_{study_date}_{modality}_{accession_number}`。
        如果無法產生有效的資料夾名稱，則返回 None。

    See Also
    --------
    get_study_folder_name : 實際產生研究資料夾名稱的函數。

    Notes
    -----
    此函數是 `get_study_folder_name` 的包裝函數，主要用於處理 None 值的情況。
    如果 DICOM 資料集中缺少必要的標籤（如 Study Date），則會返回 None。

    Examples
    --------
    >>> dicom_ds = dcmread('example.dcm', stop_before_pixels=True)
    >>> study_name = get_output_study(dicom_ds)
    >>> print(study_name)
    '12345678_20240101_MR_ACC123456'
    """
    if dicom_ds is None:
        return None
    study_folder_name = get_study_folder_name(dicom_ds)
    if not study_folder_name:
        return None
    return study_folder_name


def check_dicom_instance_number_at_last(output_study_instance: pathlib.Path) -> bool:
    """
    檢查 DICOM 實例是否為序列中的最後一個實例。

    此函數透過比較 DICOM 標籤中的 Instance Number (0020,0013) 和
    Images In Acquisition (0020,1002) 來判斷當前實例是否為序列的最後一個。

    Parameters
    ----------
    output_study_instance : pathlib.Path
        DICOM 實例檔案的路徑。必須指向有效的 DICOM 檔案。

    Returns
    -------
    bool
        如果實例編號等於序列中的影像總數，返回 True，表示這是序列的最後一個實例。
        如果檔案不存在、缺少必要的 DICOM 標籤，或實例編號不等於影像總數，則返回 False。

    Notes
    -----
    此函數用於判斷 DICOM 序列是否已完整接收。當處理大量 DICOM 檔案時，
    可以用此函數來確認某個序列的所有實例是否都已處理完成。

    DICOM 標籤說明：
    - (0020,0013) Instance Number: 當前實例在序列中的編號
    - (0020,1002) Images In Acquisition: 序列中的影像總數

    Examples
    --------
    >>> instance_path = pathlib.Path('study/series/instance_187.dcm')
    >>> is_last = check_dicom_instance_number_at_last(instance_path)
    >>> print(is_last)
    True  # 如果實例編號為 187，且序列總數為 192，則返回 False
    """
    if output_study_instance.exists():
        with open(output_study_instance, mode='rb') as dcm:
            dicom_ds = dcmread(dcm, stop_before_pixels=True)
            # (0020,0013)	Instance Number	187
            # (0020,1002)	Images In Acquisition	192
            instance_number_tag = dicom_ds.get((0x20, 0x13), False)
            images_acquisition_tag = dicom_ds.get((0x20, 0x1002), False)
            if instance_number_tag and images_acquisition_tag:
                instance_number = instance_number_tag.value
                images_acquisition = images_acquisition_tag.value
                if instance_number == images_acquisition:
                    return True
    return False


def get_series_folder_list(study_path: pathlib.Path, exclude_dicom_series) -> List[pathlib.Path]:
    """
    取得研究路徑下所有序列資料夾的列表，排除指定的序列。

    此函數會遍歷研究資料夾下的所有子資料夾，過濾掉 `.meta` 資料夾和
    在排除清單中的序列，返回有效的序列資料夾路徑列表。

    Parameters
    ----------
    study_path : pathlib.Path
        研究資料夾的路徑。此路徑必須存在且為目錄。
    exclude_dicom_series : set or list
        要排除的序列名稱集合或列表。如果序列資料夾名稱在此集合中，則不會被包含在結果中。

    Returns
    -------
    List[pathlib.Path]
        序列資料夾路徑的列表。每個元素都是一個 pathlib.Path 物件，指向一個序列資料夾。
        如果沒有符合條件的序列資料夾，則返回空列表。

    Notes
    -----
    此函數會自動排除 `.meta` 資料夾，因為這通常是系統產生的中繼資料資料夾，
    不應被視為 DICOM 序列資料夾。

    Examples
    --------
    >>> study_path = pathlib.Path('/data/study_001')
    >>> exclude_set = {'series_001', 'series_002'}
    >>> series_list = get_series_folder_list(study_path, exclude_set)
    >>> print([s.name for s in series_list])
    ['series_003', 'series_004']
    """
    series_folder_list = []
    for series_folder in study_path.iterdir():
        if series_folder.is_dir() and series_folder.name != '.meta':
            if series_folder.name not in exclude_dicom_series:
                series_folder_list.append(series_folder)
    return series_folder_list


def get_study_folder_name(dicom_ds):
    """
    從 DICOM 資料集產生標準化的研究資料夾名稱。

    此函數從 DICOM 資料集中提取關鍵資訊（患者 ID、研究日期、檢查類型、登錄號），
    並組合成標準化的資料夾名稱格式。

    Parameters
    ----------
    dicom_ds : pydicom.dataset.FileDataset
        DICOM 資料集物件，必須包含以下標籤：
        - (0008,0060) Modality: 檢查類型（如 MR、CT）
        - (0010,0020) Patient ID: 患者識別碼
        - (0008,0050) Accession Number: 登錄號
        - (0008,0020) Study Date: 研究日期（可選，但若缺失則返回 None）

    Returns
    -------
    str or None
        標準化的研究資料夾名稱，格式為 `{patient_id}_{study_date}_{modality}_{accession_number}`。
        如果 Study Date 標籤缺失，則返回 None。

    Notes
    -----
    此函數是產生研究資料夾名稱的核心邏輯。資料夾名稱的格式確保了：
    1. 唯一性：結合患者 ID、日期和登錄號
    2. 可讀性：使用下底線分隔各欄位
    3. 標準化：所有研究都使用相同的命名規則

    DICOM 標籤說明：
    - (0008,0060) Modality: 檢查類型，常見值包括 MR、CT、US 等
    - (0010,0020) Patient ID: 患者識別碼
    - (0008,0050) Accession Number: 登錄號，用於識別特定的檢查請求
    - (0008,0020) Study Date: 研究日期，格式通常為 YYYYMMDD

    Examples
    --------
    >>> dicom_ds = dcmread('example.dcm', stop_before_pixels=True)
    >>> folder_name = get_study_folder_name(dicom_ds)
    >>> print(folder_name)
    '12345678_20240101_MR_ACC123456'
    """
    # Implement actual logic based on DICOM attributes
    # 使用 str().strip() 清理 DICOM 標籤值，移除可能的控制字符（如 \r）
    modality = str(dicom_ds[0x08, 0x60].value).strip()
    patient_id = str(dicom_ds[0x10, 0x20].value).strip()
    accession_number = str(dicom_ds[0x08, 0x50].value).strip()
    study_date = dicom_ds.get((0x08, 0x20), None)
    if study_date is None:
        return None
    else:
        study_date = str(study_date.value).strip()
    return f'{patient_id}_{study_date}_{modality}_{accession_number}'


def rename_dicom_file(instance_path,
                      processing_strategy_list,
                      modality_processing_strategy,
                      mr_acquisition_type_processing_strategy):
    """
    根據 DICOM 檔案內容決定重新命名後的序列名稱和研究資料夾名稱。

    此函數使用策略模式來處理不同類型的 DICOM 序列。首先識別檢查類型（Modality）
    和 MR 取得類型（MR Acquisition Type），然後在處理策略列表中尋找匹配的策略，
    以決定序列的重新命名。

    Parameters
    ----------
    instance_path : pathlib.Path or str
        DICOM 實例檔案的路徑。必須指向有效的 DICOM 檔案。
    processing_strategy_list : List[MRRenameSeriesProcessingStrategy]
        處理策略列表，包含各種序列類型的處理策略（如 DWI、ADC、SWAN 等）。
        函數會依序檢查每個策略，直到找到匹配的策略。
    modality_processing_strategy : ModalityProcessingStrategy
        檢查類型處理策略，用於識別 DICOM 檔案的檢查類型（如 MR、CT）。
    mr_acquisition_type_processing_strategy : MRAcquisitionTypeProcessingStrategy
        MR 取得類型處理策略，用於識別 MR 影像的取得類型。

    Returns
    -------
    tuple or None
        如果找到匹配的處理策略，返回包含兩個元素的元組：
        - 第一個元素 (str): 重新命名後的序列名稱（series_enum.value）
        - 第二個元素 (str): 輸出研究資料夾名稱（output_study）
        
        如果 DICOM 資料集為 None，返回 `('', '')`。
        如果沒有找到匹配的策略，返回 None。

    Notes
    -----
    此函數使用策略模式來實現可擴展的序列重新命名邏輯。處理流程如下：
    1. 讀取 DICOM 檔案（不載入像素資料以提高效率）
    2. 識別檢查類型和 MR 取得類型
    3. 遍歷處理策略列表，尋找匹配的策略
    4. 如果找到匹配的策略且結果不為 NULL，則返回序列名稱和研究資料夾名稱

    此函數是 DICOM 檔案重新命名流程的核心，支援多種 MR 序列類型的自動識別和重新命名。

    See Also
    --------
    get_output_study : 取得輸出研究資料夾名稱的函數。

    Examples
    --------
    >>> instance_path = pathlib.Path('study/series/instance.dcm')
    >>> strategies = [DwiProcessingStrategy(), ADCProcessingStrategy()]
    >>> modality_strategy = ModalityProcessingStrategy()
    >>> mr_type_strategy = MRAcquisitionTypeProcessingStrategy()
    >>> result = rename_dicom_file(instance_path, strategies, modality_strategy, mr_type_strategy)
    >>> if result:
    ...     series_name, study_name = result
    ...     print(f"Series: {series_name}, Study: {study_name}")
    'Series: DWI, Study: 12345678_20240101_MR_ACC123456'
    """
    with open(instance_path, mode='rb') as dcm:
        dicom_ds = dcmread(dcm, stop_before_pixels=True, force=True)
    if dicom_ds is None:
        return tuple(['', ''])
    # Simulating renaming logic
    modality_enum = modality_processing_strategy.process(dicom_ds=dicom_ds)
    mr_acquisition_type_enum = mr_acquisition_type_processing_strategy.process(dicom_ds=dicom_ds)
    for processing_strategy in processing_strategy_list:
        if modality_enum == processing_strategy.modality:
            for mr_acquisition_type in processing_strategy.mr_acquisition_type:
                if mr_acquisition_type_enum == mr_acquisition_type:
                    series_enum = processing_strategy.process(dicom_ds=dicom_ds)
                    if series_enum is not NullEnum.NULL:
                        output_study = get_output_study(dicom_ds)
                        return series_enum.value, output_study
    return None


def copy_dicom_file(input_tuple, instance_path, output_path):
    """
    將 DICOM 檔案複製到重新命名後的目標路徑。

    此函數根據重新命名資訊（序列名稱和研究資料夾名稱）建立目標路徑，
    並將原始 DICOM 檔案複製到該路徑。如果目標檔案已存在，則直接返回路徑資訊。

    Parameters
    ----------
    input_tuple : tuple or None
        包含重新命名資訊的元組，格式為 `(rename_series, output_study)`：
        - rename_series (str): 重新命名後的序列名稱
        - output_study (str): 輸出研究資料夾名稱
        
        如果為 None 或序列名稱為空字串或研究名稱為 None，則返回 None。
    instance_path : pathlib.Path
        原始 DICOM 實例檔案的路徑。此檔案將被複製到目標位置。
    output_path : pathlib.Path
        輸出根目錄路徑。目標路徑將在此目錄下建立。

    Returns
    -------
    str or None
        如果複製成功或檔案已存在，返回 JSON 字串，包含兩個元素：
        - 第一個元素: 原始實例路徑的字串表示
        - 第二個元素: 目標實例路徑的字串表示
        
        如果輸入無效、無法建立目錄或複製失敗，則返回 None。

    Notes
    -----
    此函數會自動建立必要的目錄結構（如果不存在）。目標路徑結構為：
    `{output_path}/{output_study}/{rename_series}/{instance_path.name}`

    如果目標檔案已存在，函數不會重新複製，而是直接返回路徑資訊。
    這可以避免重複處理已存在的檔案。

    函數使用 `shutil.copyfileobj` 來複製檔案，這對於大型 DICOM 檔案更有效率。

    Examples
    --------
    >>> input_tuple = ('DWI', '12345678_20240101_MR_ACC123456')
    >>> instance_path = pathlib.Path('raw/study/series/instance.dcm')
    >>> output_path = pathlib.Path('/output')
    >>> result = copy_dicom_file(input_tuple, instance_path, output_path)
    >>> if result:
    ...     import json
    ...     paths = json.loads(result)
    ...     print(f"Copied from {paths[0]} to {paths[1]}")
    'Copied from raw/study/series/instance.dcm to /output/12345678_20240101_MR_ACC123456/DWI/instance.dcm'
    """
    if input_tuple is None or len(input_tuple[0]) == 0 or input_tuple[1] is None:
        return None
    rename_series = input_tuple[0]
    output_study = input_tuple[1]
    output_study_series = output_path.joinpath(output_study, rename_series)
    output_study_instance: pathlib.Path = output_study_series.joinpath(instance_path.name)
    if output_study_instance.exists():
        return json.dumps((str(instance_path), str(output_study_instance)))
    output_study_series.mkdir(exist_ok=True, parents=True)
    if output_study_series.is_dir():
        with open(instance_path, mode='rb') as instance:
            with open(output_study_instance, 'wb+') as output_instance:
                shutil.copyfileobj(instance, output_instance)
        return json.dumps((str(instance_path), str(output_study_instance)))
        # return output_study_instance
    return None


def file_processing(func_params: Dict[str, Any]):
    """
    對研究資料夾執行後處理操作。

    此函數使用指定的後處理管理器對研究資料夾進行後處理。
    後處理可能包括檔案重新命名、資料驗證、中繼資料更新等操作。

    Parameters
    ----------
    func_params : Dict[str, any]
        包含處理參數的字典，必須包含以下鍵：
        - 'study_folder_path' (pathlib.Path or str): 要處理的研究資料夾路徑
        - 'post_process_manager' (PostProcessManager): 後處理管理器實例
        
        如果 'study_folder_path' 為 None，函數會直接返回而不執行任何操作。

    Returns
    -------
    None
        此函數不返回任何值。

    Notes
    -----
    此函數是一個通用的後處理包裝函數，可以與不同的後處理管理器一起使用。
    常見的後處理操作包括：
    - DICOM 檔案重新命名和組織
    - NIfTI 檔案後處理和驗證
    - 中繼資料提取和更新

    See Also
    --------
    ConvertManager.dicom_post_process_manager : DICOM 後處理管理器
    ConvertManager.nifti_post_process_manager : NIfTI 後處理管理器

    Examples
    --------
    >>> study_path = pathlib.Path('/data/study_001')
    >>> manager = ConvertManager.dicom_post_process_manager
    >>> file_processing({'study_folder_path': study_path, 'post_process_manager': manager})
    """
    study_folder_path = func_params.get('study_folder_path')
    post_process_manager = func_params.get('post_process_manager')
    if study_folder_path is None or post_process_manager is None:
        return
    post_process_manager.post_process(study_folder_path)


@Booster(BoosterParamsMyRABBITMQ(queue_name='post_httpx_queue',
                                 qps=5, ))
def call_post_httpx(func_params: Dict[str, Any]):
    """
    透過 HTTP POST 請求發送 DCOP 事件到指定的 API 端點。

    此函數是一個分散式任務，使用 funboost 框架進行非同步處理。
    它會將 DCOP 事件（單個或列表）轉換為 JSON 格式並發送到指定的 URL。

    Parameters
    ----------
    func_params : Dict[str, any]
        包含請求參數的字典，必須包含以下鍵：
        - 'url' (str): 目標 API 端點的 URL
        - 'data' (str or List[str]): DCOP 事件的 JSON 字串或 JSON 字串列表
        
        如果 'data' 是列表，則會將所有事件合併為一個請求發送。
        如果 'data' 是單一字串，則會將其包裝在列表中發送。

    Returns
    -------
    None
        此函數不返回任何值。HTTP 回應會被記錄但不會返回。

    Notes
    -----
    此函數使用 httpx 客戶端發送 HTTP 請求，預設超時時間為 300 秒。
    函數會自動處理單個事件和多個事件的情況。

    此函數是一個分散式任務，透過 RabbitMQ 佇列進行非同步處理，
    每秒最多處理 5 個請求（qps=5）。

    See Also
    --------
    DCOPEventRequest : DCOP 事件請求的資料模型
    sync_urls.SYNC_PROT_OPE_NO : 同步協議操作號的 URL 端點

    Examples
    --------
    >>> event_json = '{"study_uid": "1.2.3", "series_uid": "1.2.4", ...}'
    >>> call_post_httpx.push({
    ...     'url': 'http://api.example.com/sync',
    ...     'data': event_json
    ... })
    """
    url = func_params['url']
    data = func_params['data']
    with httpx.Client(timeout=300) as clinet:
        if isinstance(data, list):
            dcop_event_list = [DCOPEventRequest.model_validate_json(temp).model_dump() for temp in data]
            clinet.post(url=url, json=dcop_event_list)
        else:
            dcop_event = DCOPEventRequest.model_validate_json(data).model_dump()
            clinet.post(url=url, json=[dcop_event])


@Booster(BoosterParamsMyRABBITMQ(queue_name='call_dcm2niix_queue',
                                 user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
                                 qps=10, ))
def call_dcm2niix(func_params: Dict[str, Any]):
    """
    呼叫 dcm2niix 工具將 DICOM 序列轉換為 NIfTI 格式。

    此函數是一個分散式任務，使用 funboost 框架進行非同步處理。
    它會執行 dcm2niix 命令列工具來轉換 DICOM 檔案，並處理輸出檔案的重新命名。

    Parameters
    ----------
    func_params : Dict[str, any]
        包含轉換參數的字典，會被驗證為 `CallDcm2niixParams` 物件。
        必須包含以下鍵：
        - 'output_series_file_path' (pathlib.Path): 輸出 NIfTI 檔案的完整路徑（包含檔名）
        - 'output_series_path' (pathlib.Path): 輸出序列的路徑（不含副檔名）
        - 'series_path' (pathlib.Path): 輸入 DICOM 序列的路徑

    Returns
    -------
    str
        如果轉換成功，返回輸出檔案的名稱（不含路徑）。
        如果轉換失敗或無法解析 dcm2niix 輸出，返回包含錯誤訊息的字串。

    Notes
    -----
    此函數使用 dcm2niix 工具進行 DICOM 到 NIfTI 的轉換。轉換流程如下：
    1. 建立輸出目錄（如果不存在）
    2. 執行 dcm2niix 命令，使用 gzip 壓縮（-z y）
    3. 從命令輸出中解析產生的檔案名稱
    4. 如果產生的檔案名稱與預期不符，則重新命名檔案
    5. 刪除不需要的 JSON 中繼資料檔案

    dcm2niix 命令參數說明：
    - `-z y`: 啟用 gzip 壓縮
    - `-f {name}`: 指定輸出檔案名稱
    - `-o {dir}`: 指定輸出目錄

    此函數是一個分散式任務，透過 RabbitMQ 佇列進行非同步處理，
    每秒最多處理 10 個請求（qps=10）。處理結果會自動記錄到資料庫。

    See Also
    --------
    intput_params.CallDcm2niixParams : 轉換參數的資料模型
    save_result_status_to_sqlalchemy : 將處理結果儲存到資料庫的函數

    Examples
    --------
    >>> params = {
    ...     'output_series_file_path': pathlib.Path('/output/DWI.nii.gz'),
    ...     'output_series_path': pathlib.Path('/output/DWI'),
    ...     'series_path': pathlib.Path('/input/series')
    ... }
    >>> result = call_dcm2niix.push(params)
    >>> print(result.result)
    'DWI.nii.gz'
    """
    task_params = intput_params.CallDcm2niixParams.model_validate(func_params,
                                                                  strict=False)
    output_series_file_path = task_params.output_series_file_path
    output_series_path = task_params.output_series_path
    series_path = task_params.series_path
    output_series_path.parent.mkdir(exist_ok=True, parents=True)
    cmd_str = f'dcm2niix -z y -f {output_series_path.name} -o {output_series_path.parent} {series_path}'
    process = subprocess.Popen(args=cmd_str, cwd='/', shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    pattern = re.compile(r"DICOM as (.*)\s[(]", flags=re.MULTILINE)
    match_result = pattern.search(stdout.decode())

    if match_result is None:
        return f'call_dcm2niix {stdout.decode()}'
    else:
        str_result = match_result.groups()[0]
        dcm2niix_output_path = pathlib.Path(f'{str_result}.nii.gz')
        if dcm2niix_output_path.name != output_series_path:
            try:
                # Rename the output file and corresponding JSON file
                dcm2niix_output_path.rename(output_series_file_path)
                dcm2niix_json_path = pathlib.Path(str(dcm2niix_output_path).replace('.nii.gz', '.json'))
                output_series_json_path = pathlib.Path(str(output_series_file_path).replace('.nii.gz', '.json'))
                if dcm2niix_json_path.exists():
                    dcm2niix_json_path.unlink()
                if output_series_json_path.exists():
                    output_series_json_path.unlink()
            except FileExistsError:
                print(rf'FileExistsError {series_path}')
        return output_series_path.name


@Booster(BoosterParamsMyRABBITMQ(queue_name='dicom_2_nii_file_queue',
                                 qps=10, ))
def dicom_2_nii_file(func_params: Dict[str, Any]):
    """
    將 DICOM 研究資料夾中的所有序列轉換為 NIfTI 格式。

    此函數是一個分散式任務，會遍歷研究資料夾中的所有序列，排除指定的序列類型，
    並將每個序列轉換為 NIfTI 格式。轉換完成後會執行 NIfTI 後處理操作。

    Parameters
    ----------
    func_params : Dict[str, any]
        包含轉換參數的字典，會被驗證為 `Dicom2NiiFileParams` 物件。
        必須包含以下鍵：
        - 'dicom_study_folder_path' (pathlib.Path): 輸入 DICOM 研究資料夾路徑
        - 'output_nifti_path' (pathlib.Path): 輸出 NIfTI 檔案的根目錄路徑

    Returns
    -------
    pathlib.Path or None
        如果轉換成功，返回 NIfTI 研究資料夾的路徑。
        如果輸入路徑為 None，則返回 None。

    Notes
    -----
    此函數的處理流程如下：
    1. 驗證輸入參數並取得研究資料夾路徑
    2. 遍歷研究資料夾中的所有序列（排除 `.meta` 資料夾）
    3. 檢查序列是否在排除清單中（`Dicm2NiixConverter.exclude_set`）
    4. 為每個序列建立轉換任務並提交到佇列
    5. 等待所有轉換任務完成
    6. 執行 NIfTI 後處理操作

    如果輸出檔案已存在但大小小於 500 位元組，會被視為無效檔案並刪除後重新轉換。

    此函數是一個分散式任務，透過 RabbitMQ 佇列進行非同步處理，
    每秒最多處理 10 個請求（qps=10）。

    See Also
    --------
    intput_params.Dicom2NiiFileParams : 轉換參數的資料模型
    call_dcm2niix : 執行實際轉換的函數
    ConvertManager.nifti_post_process_manager : NIfTI 後處理管理器
    Dicm2NiixConverter.exclude_set : 排除的序列類型集合

    Examples
    --------
    >>> params = {
    ...     'dicom_study_folder_path': pathlib.Path('/input/study_001'),
    ...     'output_nifti_path': pathlib.Path('/output/nifti')
    ... }
    >>> result = dicom_2_nii_file.push(params)
    >>> print(result.result)
    PosixPath('/output/nifti/study_001')
    """
    task_params = intput_params.Dicom2NiiFileParams.model_validate(func_params,
                                                                   strict=False)
    dicom_study_folder_path = task_params.dicom_study_folder_path
    output_nifti_path = task_params.output_nifti_path
    FILE_SIZE = 500
    if dicom_study_folder_path is None:
        return
    series_list = list(filter(lambda series_path: series_path.name != '.meta', dicom_study_folder_path.iterdir()))
    workflows = []
    for series_path in series_list:
        if series_path.name in Dicm2NiixConverter.exclude_set:
            continue
        output_series_path = pathlib.Path(
            f'{str(series_path).replace(str(dicom_study_folder_path.parent), str(output_nifti_path))}')
        output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
        call_dcm2niix_params = intput_params.CallDcm2niixParams(output_series_file_path=output_series_file_path,
                                                                output_series_path=output_series_path,
                                                                series_path=series_path)
        if output_series_file_path.exists():
            if output_series_file_path.stat().st_size < FILE_SIZE:
                output_series_file_path.unlink()
            #     result = call_dcm2niix.push(call_dcm2niix_params.get_str_dict())
            #     workflows.append(result)
            # else:
            #     continue
            result = call_dcm2niix.push(call_dcm2niix_params.get_str_dict())
            workflows.append(result)
        else:
            result = call_dcm2niix.push(call_dcm2niix_params.get_str_dict())
            workflows.append(result)
    [async_result.result for async_result in workflows]
    nifti_study_folder_path = output_nifti_path.joinpath(dicom_study_folder_path.name)
    file_processing(func_params=dict(study_folder_path=nifti_study_folder_path,
                                     post_process_manager=ConvertManager.nifti_post_process_manager))
    return nifti_study_folder_path


@Booster(BoosterParamsMyRABBITMQ(queue_name='dicom_2_nii_series_queue',
                                 qps=10, ))
def dicom_2_nii_series(func_params: Dict[str, Any]):
    """
    將單個 DICOM 序列轉換為 NIfTI 格式並發送轉換狀態事件。

    此函數是一個分散式任務，會將指定的 DICOM 序列轉換為 NIfTI 格式，
    並透過 HTTP POST 請求發送轉換狀態事件到 API 端點。

    Parameters
    ----------
    func_params : Dict[str, any]
        包含轉換參數的字典，會被驗證為 `Dicom2NiiSeriesParams` 物件。
        必須包含以下鍵：
        - 'output_dicom_path' (pathlib.Path): 輸入 DICOM 序列的路徑
        - 'output_nifti_path' (pathlib.Path): 輸出 NIfTI 檔案的根目錄路徑
        - 'study_uid' (str): 研究 UID
        - 'series_uid' (str): 序列 UID

    Returns
    -------
    pathlib.Path
        NIfTI 研究資料夾的路徑。

    Notes
    -----
    此函數的處理流程如下：
    1. 驗證輸入參數並取得序列路徑
    2. 檢查序列是否在排除清單中或路徑為 None
    3. 如果在排除清單中，建立跳過事件並發送
    4. 否則執行轉換：
       a. 建立轉換任務並提交到佇列
       b. 等待轉換完成
       c. 執行 NIfTI 後處理操作
       d. 建立完成事件並發送
    5. 透過 HTTP POST 發送事件到 API 端點

    如果輸出檔案已存在但大小小於 500 位元組，會被視為無效檔案並刪除後重新轉換。

    此函數是一個分散式任務，透過 RabbitMQ 佇列進行非同步處理，
    每秒最多處理 10 個請求（qps=10）。

    See Also
    --------
    intput_params.Dicom2NiiSeriesParams : 轉換參數的資料模型
    call_dcm2niix : 執行實際轉換的函數
    call_post_httpx : 發送 HTTP POST 請求的函數
    DCOPEventRequest : DCOP 事件請求的資料模型
    DCOPStatus : DCOP 狀態枚舉

    Examples
    --------
    >>> params = {
    ...     'output_dicom_path': pathlib.Path('/input/study/series'),
    ...     'output_nifti_path': pathlib.Path('/output/nifti'),
    ...     'study_uid': '1.2.3.4',
    ...     'series_uid': '1.2.3.5'
    ... }
    >>> result = dicom_2_nii_series.push(params)
    >>> print(result.result)
    PosixPath('/output/nifti/study')
    """
    task_params = intput_params.Dicom2NiiSeriesParams.model_validate(func_params)
    output_dicom_path = task_params.output_dicom_path
    output_nifti_path = task_params.output_nifti_path
    if output_dicom_path is None or output_nifti_path is None:
        raise ValueError("output_dicom_path and output_nifti_path are required")

    dicom_study_folder_path = output_dicom_path.parent
    series_path = output_dicom_path
    FILE_SIZE = 500

    UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
    nifti_study_folder_path = output_nifti_path.joinpath(dicom_study_folder_path.name)
    if (series_path.name in Dicm2NiixConverter.exclude_set) or (output_dicom_path is None):
        dcop_event = DCOPEventRequest(study_uid=str(task_params.study_uid or ""),
                                      series_uid=str(task_params.series_uid or ""),
                                      ope_no=DCOPStatus.SERIES_CONVERSION_SKIP.value,
                                      study_id=series_path.parent.name,
                                      tool_id='NIFTI_TOOL',
                                      params_data=task_params.get_str_dict(),
                                      result_data=None)
    else:
        output_series_path = pathlib.Path(
            f'{str(series_path).replace(str(dicom_study_folder_path.parent), str(output_nifti_path))}')
        output_series_file_path = pathlib.Path(f'{str(output_series_path)}.nii.gz')
        call_dcm2niix_params = intput_params.CallDcm2niixParams(output_series_file_path=output_series_file_path,
                                                                output_series_path=output_series_path,
                                                                series_path=series_path)
        if output_series_file_path.exists():
            if output_series_file_path.stat().st_size < FILE_SIZE:
                output_series_file_path.unlink()
            async_result = call_dcm2niix.push(call_dcm2niix_params.get_str_dict())
        else:
            async_result = call_dcm2niix.push(call_dcm2niix_params.get_str_dict())
        result = async_result.result

        file_processing(func_params=dict(study_folder_path=nifti_study_folder_path,
                                         post_process_manager=ConvertManager.nifti_post_process_manager))
        dcop_event = DCOPEventRequest(study_uid=str(task_params.study_uid or ""),
                                      series_uid=str(task_params.series_uid or ""),
                                      ope_no=DCOPStatus.SERIES_CONVERSION_COMPLETE.value,
                                      study_id=series_path.parent.name,
                                      tool_id='NIFTI_TOOL',
                                      params_data=task_params.get_str_dict(),
                                      result_data={'result': result})

    dcop_event_list_json = dcop_event.model_dump_json()
    call_post_httpx.push({'url': "{}{}".format(UPLOAD_DATA_API_URL, sync_urls.SYNC_PROT_OPE_NO),
                          'data': dcop_event_list_json
                          })
    return nifti_study_folder_path


@Booster(BoosterParamsMyRABBITMQ(queue_name='process_instances_queue',
                                 qps=100,
                                 log_level=logging.WARNING,
                                 # user_custom_record_process_info_func=save_result_status_to_sqlalchemy
                                 ))
def process_instances(func_params: Dict[str, Any]):
    """
    處理單個 DICOM 實例檔案：重新命名並複製到目標路徑。

    此函數是一個分散式任務，會對單個 DICOM 實例檔案執行以下操作：
    1. 根據 DICOM 內容決定重新命名後的序列名稱
    2. 將檔案複製到重新命名後的目標路徑

    Parameters
    ----------
    func_params : Dict[str, any]
        包含處理參數的字典，會被驗證為 `ProcessInstancesParams` 物件。
        必須包含以下鍵：
        - 'instance' (pathlib.Path): DICOM 實例檔案的路徑
        - 'output_dicom_path' (pathlib.Path): 輸出 DICOM 檔案的根目錄路徑

    Returns
    -------
    str or None
        如果處理成功，返回 JSON 字串，包含原始路徑和目標路徑。
        如果處理失敗（無法重新命名或複製），返回 None。

    Notes
    -----
    此函數使用 `ConvertManager` 中的處理策略來決定序列的重新命名。
    處理流程如下：
    1. 讀取 DICOM 檔案並識別序列類型
    2. 使用處理策略決定重新命名後的序列名稱
    3. 將檔案複製到目標路徑

    此函數是一個分散式任務，透過 RabbitMQ 佇列進行非同步處理，
    每秒最多處理 100 個請求（qps=100）。日誌級別設為 WARNING。

    See Also
    --------
    intput_params.ProcessInstancesParams : 處理參數的資料模型
    rename_dicom_file : 重新命名 DICOM 檔案的函數
    copy_dicom_file : 複製 DICOM 檔案的函數
    ConvertManager : 轉換管理器，包含處理策略

    Examples
    --------
    >>> params = {
    ...     'instance': pathlib.Path('/input/study/series/instance.dcm'),
    ...     'output_dicom_path': pathlib.Path('/output/dicom')
    ... }
    >>> result = process_instances.push(params)
    >>> if result.result:
    ...     import json
    ...     paths = json.loads(result.result)
    ...     print(f"Processed: {paths[0]} -> {paths[1]}")
    """
    task_params = intput_params.ProcessInstancesParams.model_validate(func_params,
                                                                      strict=False)
    instance = task_params.instance
    output_dicom_path = task_params.output_dicom_path
    rename_dicom_file_tuple = rename_dicom_file(instance,
                                                ConvertManager.processing_strategy_list,
                                                ConvertManager.modality_processing_strategy,
                                                ConvertManager.mr_acquisition_type_processing_strategy)
    copy_dicom_file_tuple = copy_dicom_file(rename_dicom_file_tuple,
                                            instance,
                                            output_dicom_path)
    if copy_dicom_file_tuple:
        return copy_dicom_file_tuple
    return None


def process_dir_next(sub_dir: pathlib.Path, output_dicom_path: pathlib.Path):
    """
    處理目錄中的 DICOM 檔案並執行後處理操作。

    此函數會從目錄中找到第一個 DICOM 實例檔案，讀取其 DICOM 標籤來決定研究資料夾名稱，
    然後執行 DICOM 後處理操作。

    Parameters
    ----------
    sub_dir : pathlib.Path
        要處理的子目錄路徑。函數會在此目錄中搜尋 DICOM 檔案。
    output_dicom_path : pathlib.Path
        輸出 DICOM 檔案的根目錄路徑。研究資料夾會在此目錄下建立。

    Returns
    -------
    pathlib.Path
        處理後的研究資料夾路徑。

    Notes
    -----
    此函數的處理流程如下：
    1. 在目錄中搜尋 `.dcm` 檔案
    2. 如果找不到 `.dcm` 檔案，則搜尋所有檔案並過濾出檔案（非目錄）
    3. 讀取第一個實例檔案的 DICOM 標籤（不載入像素資料）
    4. 根據 DICOM 標籤產生研究資料夾名稱
    5. 執行 DICOM 後處理操作

    此函數通常由 `process_dir` 函數呼叫，用於在處理完所有實例後執行後處理操作。

    See Also
    --------
    get_study_folder_name : 產生研究資料夾名稱的函數
    file_processing : 執行檔案後處理的函數
    ConvertManager.dicom_post_process_manager : DICOM 後處理管理器

    Examples
    --------
    >>> sub_dir = pathlib.Path('/input/study/series')
    >>> output_path = pathlib.Path('/output/dicom')
    >>> study_path = process_dir_next(sub_dir, output_path)
    >>> print(study_path)
    PosixPath('/output/dicom/12345678_20240101_MR_ACC123456')
    """
    instances_list = list(sub_dir.rglob('*.dcm'))
    if len(instances_list) == 0:
        instances_list = sorted(sub_dir.rglob('*'))
        instances_list = list(filter(lambda x: x.is_file(), instances_list))
        instance_path: pathlib.Path = instances_list[0]
    else:
        instance_path: pathlib.Path = instances_list[0]

    with open(instance_path, mode='rb') as dcm:
        dicom_ds = dcmread(dcm, stop_before_pixels=True)
        study_folder_name = get_study_folder_name(dicom_ds)
        study_folder_path = output_dicom_path.joinpath(study_folder_name)
        file_processing(func_params=dict(study_folder_path=study_folder_path,
                                         post_process_manager=ConvertManager.dicom_post_process_manager))
        dicom_study_folder_path = study_folder_path
    return dicom_study_folder_path


def get_orthanc_study_uid_series_uid(instance_path_str: str):
    """
    從 DICOM 實例檔案路徑取得對應的 Orthanc 研究 UID 和序列 UID。

    此函數會讀取 DICOM 檔案中的 Series Instance UID，然後連接到 Orthanc 伺服器，
    根據路徑結構推斷研究 UID，並在 Orthanc 中查找對應的序列 ID。

    Parameters
    ----------
    instance_path_str : str
        DICOM 實例檔案的完整路徑字串。路徑結構應為：
        `{base_path}/{study_uid}/{...}/{series}/{instance.dcm}`

    Returns
    -------
    tuple or None
        如果找到對應的序列，返回包含兩個元素的元組：
        - 第一個元素 (str): 研究 UID
        - 第二個元素 (str): Orthanc 序列 ID
        
        如果找不到對應的序列，返回 None。

    Notes
    -----
    此函數的處理流程如下：
    1. 讀取 DICOM 檔案並取得 Series Instance UID (0020,000E)
    2. 從路徑結構中推斷研究 UID（假設為路徑的第 4 層父目錄名稱）
    3. 連接到 Orthanc 伺服器（從環境變數 `UPLOAD_DATA_DICOM_SEG_URL` 取得 URL）
    4. 在研究的所有序列中查找匹配的 Series Instance UID
    5. 返回研究 UID 和序列 ID

    此函數依賴於路徑結構的一致性。如果路徑結構改變，可能需要調整推斷邏輯。

    See Also
    --------
    get_orthanc_series_uid : 批次取得多個序列的 Orthanc UID
    Orthanc : Orthanc 客戶端類別
    Study : Orthanc 研究類別

    Examples
    --------
    >>> instance_path = './raw_dicom/study_uid/patient/study/series/instance.dcm'
    >>> result = get_orthanc_study_uid_series_uid(instance_path)
    >>> if result:
    ...     study_uid, series_id = result
    ...     print(f"Study: {study_uid}, Series: {series_id}")
    'Study: study_uid, Series: series_id'
    """
    instance_path = pathlib.Path(instance_path_str)
    with open(instance_path_str, mode='rb') as f:
        dicom_ds = pydicom.dcmread(f)

    # (0020,000E)	Series Instance UID	1.2.840.113619.2.44.5554020.7707121.19025.1612063861.703
    series_sop_uid = dicom_ds[0x0020, 0x000E].value

    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    if not UPLOAD_DATA_DICOM_SEG_URL:
        raise ValueError("UPLOAD_DATA_DICOM_SEG_URL is required")
    # ./raw_dicom/ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30/10089413 GUO HSIOU HUA/21002010079 MRI Stroke Wall C C/MR 3D Ax SWAN/*.dcm
    client = Orthanc(str(UPLOAD_DATA_DICOM_SEG_URL), timeout=300)
    study_uid = instance_path.parent.parent.parent.parent.name
    study = Study(study_uid, client=client)
    series_filter = list(filter(lambda series: series.uid == series_sop_uid, study.series))
    if series_filter:
        return str(study_uid), str(series_filter[0].id_)
    else:
        return None


def get_orthanc_series_uid(study_uid: str,
                           series_dir_set: set):
    """
    批次取得多個序列目錄對應的 Orthanc 序列 UID。

    此函數會從多個序列目錄中讀取 DICOM 檔案的 Series Instance UID，
    然後連接到 Orthanc 伺服器，查找對應的序列 ID 和描述資訊。

    Parameters
    ----------
    study_uid : str
        Orthanc 研究 UID，用於識別要查詢的研究。
    series_dir_set : set
        序列目錄路徑的集合。每個目錄應包含至少一個 DICOM 檔案。

    Returns
    -------
    pd.DataFrame
        包含以下欄位的 DataFrame：
        - 'file_series_sop_uid' (str): 從檔案中讀取的 Series Instance UID
        - 'series_sop_uid' (str): Orthanc 中的 Series Instance UID（與 file_series_sop_uid 相同）
        - 'uid' (str): Orthanc 序列 ID
        - 'description' (str): 序列描述

        如果某個序列在 Orthanc 中找不到，則不會出現在結果 DataFrame 中。

    Notes
    -----
    此函數的處理流程如下：
    1. 連接到 Orthanc 伺服器（從環境變數 `UPLOAD_DATA_DICOM_SEG_URL` 取得 URL）
    2. 對每個序列目錄：
       a. 找到第一個 DICOM 檔案
       b. 讀取 Series Instance UID (0020,000E)
    3. 從 Orthanc 取得研究的所有序列資訊
    4. 使用 pandas DataFrame 合併檔案中的 Series Instance UID 和 Orthanc 中的序列資訊
    5. 返回合併後的 DataFrame

    此函數使用 pandas DataFrame 來處理資料合併，可以高效地處理多個序列的查詢。

    See Also
    --------
    get_orthanc_study_uid_series_uid : 取得單個實例的 Orthanc UID
    Orthanc : Orthanc 客戶端類別
    Study : Orthanc 研究類別

    Examples
    --------
    >>> study_uid = '1.2.3.4'
    >>> series_dirs = {
    ...     pathlib.Path('/data/series1'),
    ...     pathlib.Path('/data/series2')
    ... }
    >>> df = get_orthanc_series_uid(study_uid, series_dirs)
    >>> print(df[['file_series_sop_uid', 'uid']])
       file_series_sop_uid              uid
    0        1.2.3.4.5.6        series_id_1
    1        1.2.3.4.5.7        series_id_2
    """
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    if not UPLOAD_DATA_DICOM_SEG_URL:
        raise ValueError("UPLOAD_DATA_DICOM_SEG_URL is required")
    client = Orthanc(str(UPLOAD_DATA_DICOM_SEG_URL), timeout=300)
    series_sop_uid_list = []
    for series_dir in series_dir_set:
        series_path_list = list(series_dir.rglob('*.dcm'))
        instance_path_str = series_path_list[0]
        with open(instance_path_str, mode='rb') as f:
            dicom_ds = pydicom.dcmread(f)
        series_sop_uid = dicom_ds[0x0020, 0x000E].value
        series_sop_uid_list.append(series_sop_uid)
    study = Study(study_uid, client=client)
    series_dict_list = list(map(lambda x: {'series_sop_uid': x.uid,
                                           'uid': x.id_,
                                           'description': x.description, }, study.series))
    df = pd.DataFrame(series_sop_uid_list, columns=['file_series_sop_uid'])
    df1 = pd.DataFrame(series_dict_list)
    df2 = pd.merge(df, df1, left_on='file_series_sop_uid', right_on='series_sop_uid')
    return df2


@Booster(BoosterParamsMyRABBITMQ(queue_name='process_dir_queue',
                                 user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
                                 qps=10, ))
def process_dir(func_params: Dict[str, Any]):
    """
    處理目錄中的所有 DICOM 實例檔案並發送轉換完成事件。

    此函數是一個分散式任務，會處理目錄中的所有 DICOM 實例檔案，執行重新命名和複製操作，
    然後根據處理結果產生 DCOP 事件並發送到 API 端點。

    Parameters
    ----------
    func_params : Dict[str, any]
        包含處理參數的字典，會被驗證為 `Dicom2NiiParams` 物件。
        必須包含以下鍵：
        - 'sub_dir' (pathlib.Path): 要處理的子目錄路徑
        - 'output_dicom_path' (pathlib.Path): 輸出 DICOM 檔案的根目錄路徑

    Returns
    -------
    pathlib.Path
        處理後的 DICOM 研究資料夾路徑。

    Notes
    -----
    此函數的處理流程如下：
    1. 在目錄中搜尋所有 DICOM 實例檔案（`.dcm` 檔案或所有檔案）
    2. 為每個實例建立處理任務並提交到佇列（非同步處理）
    3. 等待所有處理任務完成（設定超時時間為 3600 秒）
    4. 收集處理結果並建立 DataFrame
    5. 從處理結果中提取序列資訊（Series SOP UID、Study UID 等）
    6. 根據 `rename_dicom_path` 去重，確保每個重新命名路徑只保留一筆記錄
    7. 對每個研究：
       a. 查詢 Orthanc 取得序列 UID 映射
       b. 為每個重新命名的序列產生 DCOP 事件
    8. 發送所有事件到 API 端點
    9. 執行 DICOM 後處理操作

    此函數使用 pandas DataFrame 來處理和組織處理結果，可以高效地處理大量實例檔案。

    此函數是一個分散式任務，透過 RabbitMQ 佇列進行非同步處理，
    每秒最多處理 10 個請求（qps=10）。處理結果會自動記錄到資料庫。

    See Also
    --------
    intput_params.Dicom2NiiParams : 處理參數的資料模型
    process_instances : 處理單個實例的函數
    process_dir_next : 執行後處理的函數
    get_orthanc_series_uid : 取得 Orthanc 序列 UID 的函數
    call_post_httpx : 發送 HTTP POST 請求的函數

    Examples
    --------
    >>> params = {
    ...     'sub_dir': pathlib.Path('/input/study/series'),
    ...     'output_dicom_path': pathlib.Path('/output/dicom')
    ... }
    >>> result = process_dir.push(params)
    >>> print(result.result)
    PosixPath('/output/dicom/12345678_20240101_MR_ACC123456')
    """
    UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
    task_params = intput_params.Dicom2NiiParams.model_validate(func_params,
                                                               strict=False)
    sub_dir = task_params.sub_dir
    output_dicom_path = task_params.output_dicom_path
    if sub_dir is None or output_dicom_path is None:
        raise ValueError("sub_dir and output_dicom_path are required")

    instances_list = sorted(sub_dir.rglob('*.dcm'))
    if len(instances_list) > 0:
        async_result_list = [process_instances.push(intput_params.ProcessInstancesParams(instance=instances,
                                                                                         output_dicom_path=output_dicom_path).get_str_dict())
                             for instances in instances_list]
    else:
        instances_list = sorted(sub_dir.rglob('*'))
        instances_list = list(filter(lambda x: x.is_file(), instances_list))
        async_result_list = [process_instances.push(intput_params.ProcessInstancesParams(instance=instances,
                                                                                         output_dicom_path=output_dicom_path).get_str_dict())
                             for instances in instances_list]
    for async_result in async_result_list:
        async_result.set_timeout(3600)

    result_list = [async_result.result for async_result in async_result_list]

    dicom_study_folder_path = process_dir_next(sub_dir, output_dicom_path)
    result_filter_list = list(filter(lambda x: x is not None, result_list))
    result_dict_list = list(map(lambda x: json.loads(x), result_filter_list))
    df = pd.DataFrame(result_dict_list, columns=['instance_path_str', 'rename_dicom_path'])
    df['instance_dir_path'] = df['instance_path_str'].map(lambda x: os.path.dirname(x))
    # 先不去重，保留所有記錄以便後續處理
    df['instance_dir_path'] = df['instance_dir_path'].map(lambda x: pathlib.Path(x))
    df['series_sop_uid'] = df['instance_path_str'].map(lambda x: pydicom.dcmread(x)[0x0020, 0x000E].value)
    # Platform
    # df['study_uid'] = df['instance_path_str'].map(lambda x: pathlib.Path(x).parent.parent.parent.parent.name)
    # Pc 4090
    df['study_uid'] = df['instance_path_str'].map(lambda x: pathlib.Path(x).parent.parent.name)
    df['study_id'] = df['rename_dicom_path'].map(lambda x: pathlib.Path(x).parent.parent.name)


    # 根據 rename_parent 去重，確保每個 rename 路徑只保留一筆記錄
    df['rename_parent'] = df['rename_dicom_path'].map(lambda x: str(pathlib.Path(x).parent))
    df.drop_duplicates(subset=['rename_parent'], inplace=True)

    study_uid_unique = df['study_uid'].unique()
    dcop_event_list = []
    for study_uid in study_uid_unique:
        df_study = df[df['study_uid'] == study_uid]
        # df 已經根據 rename_dicom_path 去重，所以 df_study 中每個 rename 路徑只有一筆記錄
        # 收集所有需要的 instance_dir_path（用於查詢 Orthanc）
        series_dir_set = set(df_study['instance_dir_path'].to_list())
        df2 = get_orthanc_series_uid(study_uid=study_uid, series_dir_set=series_dir_set)

        # 建立 series_sop_uid -> series_uid 的映射
        series_uid_map = {
            record['file_series_sop_uid']: record['uid']
            for record in df2.to_dict(orient='records')
        }
        if not series_uid_map:
            logger.warning("study %s 沒有對應的 Orthanc series", study_uid)
            continue

        # 為每個 rename_dicom_path 產生事件（df_study 已經去重，每個 rename 路徑只有一筆）
        for _, row in df_study.iterrows():
            series_sop_uid = row['series_sop_uid']
            series_uid = series_uid_map.get(series_sop_uid)
            if not series_uid:
                logger.warning("找不到 series_sop_uid=%s 在 study=%s 的 Orthanc mapping，已略過該 rename 路徑 %s",
                               series_sop_uid, study_uid, row['rename_dicom_path'])
                continue
            raw_parent = str(pathlib.Path(row['instance_dir_path']).parent)
            rename_parent = row['rename_parent']
            # 在 params_data 中包含 rename_dicom_path，以便後續查詢時可以區分不同的 rename 路徑
            dcop_event = DCOPEventRequest(
                study_uid=study_uid,
                series_uid=series_uid,
                ope_no=DCOPStatus.SERIES_TRANSFER_COMPLETE.value,
                study_id=row['study_id'],
                tool_id='DICOM_TOOL',
                params_data={
                    'rename_dicom_path': rename_parent,  # 在 params_data 中包含 rename_dicom_path
                },
                result_data={
                    'raw_dicom_path': raw_parent,
                    'rename_dicom_path': rename_parent,
                }
            )
            dcop_event_list.append(dcop_event.model_dump_json())
    call_post_httpx.push({'url': "{}{}".format(UPLOAD_DATA_API_URL, sync_urls.SYNC_PROT_OPE_NO),
                          'data': dcop_event_list
                          })
    return dicom_study_folder_path


@Booster(BoosterParamsMyRABBITMQ(queue_name='dicom_to_nii_queue',
                                 user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
                                 qps=10, ))
def dicom_to_nii(func_params: Dict[str, Any]):
    """
    將 DICOM 檔案轉換為 NIfTI 格式的主要入口函數。

    此函數是一個分散式任務，會根據輸入參數決定執行哪個轉換流程：
    1. 如果提供了 `sub_dir`，則執行完整的轉換流程（原始 DICOM -> 重新命名 DICOM -> NIfTI）
    2. 如果未提供 `sub_dir`，則只執行 DICOM 到 NIfTI 的轉換（重新命名 DICOM -> NIfTI）

    Parameters
    ----------
    func_params : Dict[str, any]
        包含轉換參數的字典，會被驗證為 `Dicom2NiiParams` 物件。
        必須包含以下鍵：
        - 'sub_dir' (pathlib.Path, optional): 輸入子目錄路徑。如果提供，則執行完整轉換流程。
        - 'output_dicom_path' (pathlib.Path): 輸出 DICOM 檔案的根目錄路徑
        - 'output_nifti_path' (pathlib.Path): 輸出 NIfTI 檔案的根目錄路徑

    Returns
    -------
    pathlib.Path
        轉換後的研究資料夾路徑。如果執行完整流程，返回 DICOM 研究資料夾路徑；
        如果只執行 DICOM 到 NIfTI 轉換，返回 NIfTI 研究資料夾路徑。

    Notes
    -----
    此函數的處理邏輯如下：
    - 如果 `sub_dir` 不為 None：
      1. 呼叫 `process_dir` 處理原始 DICOM 檔案（重新命名和複製）
      2. 等待處理完成（超時時間 3600 秒）
      3. 返回 DICOM 研究資料夾路徑
    
    - 如果 `sub_dir` 為 None：
      1. 建立 `Dicom2NiiFileParams` 參數物件
      2. 呼叫 `dicom_2_nii_file` 將重新命名後的 DICOM 轉換為 NIfTI
      3. 等待轉換完成（超時時間 3600 秒）
      4. 返回 NIfTI 研究資料夾路徑

    此函數是一個分散式任務，透過 RabbitMQ 佇列進行非同步處理，
    每秒最多處理 10 個請求（qps=10）。處理結果會自動記錄到資料庫。

    See Also
    --------
    intput_params.Dicom2NiiParams : 轉換參數的資料模型
    process_dir : 處理目錄中的所有 DICOM 實例檔案
    dicom_2_nii_file : 將 DICOM 研究資料夾轉換為 NIfTI

    Examples
    --------
    >>> # 完整轉換流程（原始 DICOM -> 重新命名 DICOM -> NIfTI）
    >>> params = {
    ...     'sub_dir': pathlib.Path('/input/raw_dicom'),
    ...     'output_dicom_path': pathlib.Path('/output/dicom'),
    ...     'output_nifti_path': pathlib.Path('/output/nifti')
    ... }
    >>> result = dicom_to_nii.push(params)
    >>> print(result.result)
    PosixPath('/output/dicom/12345678_20240101_MR_ACC123456')
    
    >>> # 只執行 DICOM 到 NIfTI 轉換
    >>> params = {
    ...     'sub_dir': None,
    ...     'output_dicom_path': pathlib.Path('/input/renamed_dicom'),
    ...     'output_nifti_path': pathlib.Path('/output/nifti')
    ... }
    >>> result = dicom_to_nii.push(params)
    >>> print(result.result)
    PosixPath('/output/nifti/12345678_20240101_MR_ACC123456')
    """
    task_params = intput_params.Dicom2NiiParams.model_validate(func_params,
                                                               strict=False)
    if task_params.output_dicom_path is None or task_params.output_nifti_path is None:
        raise ValueError("output_dicom_path and output_nifti_path are required")

    # 1. raw dicom -> rename dicom
    if task_params.sub_dir is not None:
        result = process_dir.push(func_params)
        result.set_timeout(3600)
        data = result.get()
    else:
        # rename dicom -> rename nifti
        dicom_2_nii_file_param = intput_params.Dicom2NiiFileParams(
            dicom_study_folder_path=task_params.output_dicom_path,
            output_nifti_path=task_params.output_nifti_path)
        result = dicom_2_nii_file.push(dicom_2_nii_file_param.get_str_dict())
        result.set_timeout(3600)
        data = result.get()

    return data


# @Booster('dicom_rename_queue',
#          broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10)
@Booster(BoosterParamsMyRABBITMQ(queue_name='dicom_rename_queue',
                                 qps=10, ))
def dicom_rename(func_params: Dict[str, Any]):
    """
    重新命名 DICOM 檔案的主要入口函數。

    此函數是一個分散式任務，會呼叫 `process_dir` 來處理目錄中的所有 DICOM 實例檔案，
    執行重新命名和複製操作。

    Parameters
    ----------
    func_params : Dict[str, any]
        包含處理參數的字典，會被驗證為 `Dicom2NiiParams` 物件。
        必須包含以下鍵：
        - 'sub_dir' (pathlib.Path): 要處理的子目錄路徑
        - 'output_dicom_path' (pathlib.Path): 輸出 DICOM 檔案的根目錄路徑

    Returns
    -------
    AsyncResult
        非同步處理結果物件，可以透過 `.result` 屬性取得最終結果（DICOM 研究資料夾路徑）。

    Notes
    -----
    此函數是 `process_dir` 的簡單包裝函數，專門用於 DICOM 檔案重新命名任務。
    它會將任務提交到 `process_dir` 佇列並返回非同步結果物件。

    此函數是一個分散式任務，透過 RabbitMQ 佇列進行非同步處理，
    每秒最多處理 10 個請求（qps=10）。

    See Also
    --------
    process_dir : 處理目錄中的所有 DICOM 實例檔案
    intput_params.Dicom2NiiParams : 處理參數的資料模型

    Examples
    --------
    >>> params = {
    ...     'sub_dir': pathlib.Path('/input/raw_dicom'),
    ...     'output_dicom_path': pathlib.Path('/output/dicom')
    ... }
    >>> result = dicom_rename.push(params)
    >>> print(result.result)
    PosixPath('/output/dicom/12345678_20240101_MR_ACC123456')
    """
    process_dir_result = process_dir.push(func_params)
    return process_dir_result


class ConvertManager:
    """
    轉換管理器類別，提供 DICOM 處理和轉換所需的策略和管理器。

    此類別是一個單例類別，包含所有 DICOM 處理和轉換所需的策略物件和後處理管理器。
    所有策略和管理器都是類別變數，在類別載入時初始化。

    Attributes
    ----------
    modality_processing_strategy : ModalityProcessingStrategy
        檢查類型處理策略，用於識別 DICOM 檔案的檢查類型（如 MR、CT）。
    mr_acquisition_type_processing_strategy : MRAcquisitionTypeProcessingStrategy
        MR 取得類型處理策略，用於識別 MR 影像的取得類型。
    processing_strategy_list : List[MRRenameSeriesProcessingStrategy]
        序列重新命名處理策略列表，包含各種 MR 序列類型的處理策略：
        - DwiProcessingStrategy: DWI 序列處理策略
        - ADCProcessingStrategy: ADC 序列處理策略
        - EADCProcessingStrategy: EADC 序列處理策略
        - SWANProcessingStrategy: SWAN 序列處理策略
        - ESWANProcessingStrategy: ESWAN 序列處理策略
        - MRABrainProcessingStrategy: MRA 腦部序列處理策略
        - MRANeckProcessingStrategy: MRA 頸部序列處理策略
        - MRAVRBrainProcessingStrategy: MRA VR 腦部序列處理策略
        - MRAVRNeckProcessingStrategy: MRA VR 頸部序列處理策略
        - T1ProcessingStrategy: T1 序列處理策略
        - T2ProcessingStrategy: T2 序列處理策略
        - ASLProcessingStrategy: ASL 序列處理策略
        - DSCProcessingStrategy: DSC 序列處理策略
        - RestingProcessingStrategy: Resting 序列處理策略
        - CVRProcessingStrategy: CVR 序列處理策略
        - DTIProcessingStrategy: DTI 序列處理策略
    dicom_post_process_manager : PostProcessManager
        DICOM 後處理管理器，用於執行 DICOM 檔案的後處理操作（如重新命名、組織等）。
    nifti_post_process_manager : PostProcessManager
        NIfTI 後處理管理器，用於執行 NIfTI 檔案的後處理操作（如驗證、中繼資料更新等）。

    Notes
    -----
    此類別使用策略模式來實現可擴展的序列處理邏輯。每個處理策略都實現了
    `MRRenameSeriesProcessingStrategy` 介面，可以根據 DICOM 標籤內容決定序列的重新命名。

    所有策略和管理器都是類別變數，在整個應用程式生命週期中共享同一個實例。
    這確保了處理邏輯的一致性和資源使用的效率。

    See Also
    --------
    ModalityProcessingStrategy : 檢查類型處理策略
    MRAcquisitionTypeProcessingStrategy : MR 取得類型處理策略
    MRRenameSeriesProcessingStrategy : MR 序列重新命名處理策略介面
    dicom_rename_mr_postprocess.PostProcessManager : DICOM 後處理管理器
    convert_nifti_postprocess.PostProcessManager : NIfTI 後處理管理器

    Examples
    --------
    >>> # 使用轉換管理器中的策略來處理 DICOM 檔案
    >>> result = rename_dicom_file(
    ...     instance_path,
    ...     ConvertManager.processing_strategy_list,
    ...     ConvertManager.modality_processing_strategy,
    ...     ConvertManager.mr_acquisition_type_processing_strategy
    ... )
    >>> # 使用後處理管理器執行後處理操作
    >>> file_processing({
    ...     'study_folder_path': study_path,
    ...     'post_process_manager': ConvertManager.dicom_post_process_manager
    ... })
    """
    modality_processing_strategy: ModalityProcessingStrategy = ModalityProcessingStrategy()
    mr_acquisition_type_processing_strategy: MRAcquisitionTypeProcessingStrategy = MRAcquisitionTypeProcessingStrategy()
    processing_strategy_list: List[MRRenameSeriesProcessingStrategy] = [DwiProcessingStrategy(),
                                                                        ADCProcessingStrategy(),
                                                                        EADCProcessingStrategy(),
                                                                        SWANProcessingStrategy(),
                                                                        ESWANProcessingStrategy(),
                                                                        MRABrainProcessingStrategy(),
                                                                        MRANeckProcessingStrategy(),
                                                                        MRAVRBrainProcessingStrategy(),
                                                                        MRAVRNeckProcessingStrategy(),
                                                                        T1ProcessingStrategy(),
                                                                        T2ProcessingStrategy(),
                                                                        ASLProcessingStrategy(),
                                                                        DSCProcessingStrategy(),
                                                                        RestingProcessingStrategy(),
                                                                        CVRProcessingStrategy(),
                                                                        DTIProcessingStrategy()]
    dicom_post_process_manager = dicom_rename_mr_postprocess.PostProcessManager()
    nifti_post_process_manager = convert_nifti_postprocess.PostProcessManager()


class Dicm2NiixConverter:
    """
    dcm2niix 轉換器配置類別，定義要排除的序列類型。

    此類別包含一個類別變數 `exclude_set`，用於指定在 DICOM 到 NIfTI 轉換過程中
    要排除的序列類型。這些序列通常不需要轉換為 NIfTI 格式，或者需要特殊處理。

    Attributes
    ----------
    exclude_set : set
        要排除的序列名稱集合。目前包含以下序列類型：
        - MRSeriesRenameEnum.MRAVR_BRAIN.value: MRA VR 腦部序列
        - MRSeriesRenameEnum.MRAVR_NECK.value: MRA VR 頸部序列
        
        其他被註解掉的序列類型（如 DSC、ASL 相關序列）目前不在排除清單中，
        但可以根據需要取消註解來啟用排除。

    Notes
    -----
    此類別用於配置 dcm2niix 轉換器的行為。當處理 DICOM 研究資料夾時，
    如果序列名稱在 `exclude_set` 中，則該序列不會被轉換為 NIfTI 格式。

    排除清單可以根據實際需求進行調整。某些序列類型可能因為以下原因被排除：
    - 不需要轉換為 NIfTI 格式
    - 需要特殊處理流程
    - 轉換後可能產生問題

    See Also
    --------
    MRSeriesRenameEnum : MR 序列重新命名枚舉
    dicom_2_nii_file : 將 DICOM 研究資料夾轉換為 NIfTI 的函數
    dicom_2_nii_series : 將單個 DICOM 序列轉換為 NIfTI 的函數

    Examples
    --------
    >>> # 檢查序列是否在排除清單中
    >>> if series_path.name in Dicm2NiixConverter.exclude_set:
    ...     print(f"序列 {series_path.name} 已被排除，不會進行轉換")
    ...     continue
    >>> # 添加新的排除序列類型
    >>> Dicm2NiixConverter.exclude_set.add('NEW_SERIES_TYPE')
    """
    exclude_set = {
        MRSeriesRenameEnum.MRAVR_BRAIN.value,
        MRSeriesRenameEnum.MRAVR_NECK.value,

        # DSCSeriesRenameEnum.DSC.value,
        # DSCSeriesRenameEnum.rCBV.value,
        # DSCSeriesRenameEnum.rCBF.value,
        # DSCSeriesRenameEnum.MTT.value,
        #
        # ASLSEQSeriesRenameEnum.ASLSEQ.value,
        # ASLSEQSeriesRenameEnum.ASLPROD.value,
        #
        # ASLSEQSeriesRenameEnum.ASLSEQATT.value,
        # ASLSEQSeriesRenameEnum.ASLSEQATT_COLOR.value,
        #
        # ASLSEQSeriesRenameEnum.ASLSEQCBF.value,
        # ASLSEQSeriesRenameEnum.ASLSEQCBF_COLOR.value,
        #
        # ASLSEQSeriesRenameEnum.ASLSEQPW.value,
    }
