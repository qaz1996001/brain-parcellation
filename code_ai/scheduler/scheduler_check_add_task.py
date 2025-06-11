import os
import json
import pathlib
import time
import subprocess
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import httpx
import nb_log

from funboost import BoosterParams, ConcurrentModeEnum, BrokerEnum, Booster, BrokerConnConfig
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import pydicom
from code_ai.task.schema.intput_params import Dicom2NiiParams
from code_ai.utils.database import get_sqla_helper
from code_ai.utils.model import RawDicomToNiiInference

# 設置日誌記錄器
logger = nb_log.LogManager('add_raw_dicom_to_nii_inference').get_logger_and_add_handlers(
    log_filename='add_raw_dicom_to_nii_inference.log'
)


def get_env_paths() -> Tuple[pathlib.Path,pathlib.Path,pathlib.Path]:
    """獲取環境變數中的路徑設置"""
    from code_ai import load_dotenv
    load_dotenv()

    path_raw_dicom = os.getenv("PATH_RAW_DICOM")
    path_rename_dicom = os.getenv("PATH_RENAME_DICOM")
    path_rename_nifti = os.getenv("PATH_RENAME_NIFTI")

    if not all([path_raw_dicom, path_rename_dicom, path_rename_nifti]):
        raise ValueError("環境變數缺失: 請確保設置了PATH_RAW_DICOM, PATH_RENAME_DICOM, PATH_RENAME_NIFTI")

    return (
        pathlib.Path(path_raw_dicom),
        pathlib.Path(path_rename_dicom),
        pathlib.Path(path_rename_nifti)
    )


def check_and_create_directories(paths: List[pathlib.Path]) -> None:
    """檢查並創建必要的目錄"""
    for path in paths:
        if not path.exists():
            logger.info(f"創建目錄: {path}")
            path.mkdir(parents=True, exist_ok=True)


def is_task_exists(session: Session, **filter_args) -> bool:
    """檢查任務是否已存在於資料庫中"""
    try:
        result = session.query(RawDicomToNiiInference).filter_by(**filter_args).one_or_none()
        return result is not None
    except SQLAlchemyError as e:
        logger.error(f"資料庫查詢錯誤: {str(e)}")
        raise


def add_task_record(session: Session, name: str, sub_dir: Optional[str],
                    output_dicom_path: str, output_nifti_path: str) -> None:
    """向資料庫添加任務記錄"""
    try:
        created_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_record = RawDicomToNiiInference(
            name=name,
            sub_dir=sub_dir,
            output_dicom_path=output_dicom_path,
            output_nifti_path=output_nifti_path,
            created_time=created_time_str
        )
        session.add(new_record)
        session.commit()
        logger.debug(f"添加任務記錄: {name}, {sub_dir}, {output_dicom_path}")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"添加任務記錄失敗: {str(e)}")
        raise


def post_study(study_uid_list : List[str]):
    from backend.app.sync import urls
    UPLOAD_DATA_API_URL = os.getenv("UPLOAD_DATA_API_URL")
    url = '{}{}'.format(UPLOAD_DATA_API_URL,urls.SYNC_PROT_STUDY)
    data = {"ids": study_uid_list}
    print(data)
    with httpx.Client(timeout=300) as clinet:
        rep = clinet.post(url=url, json = data)
        print(rep)


def process_dicom_to_nii_task(session: Session, input_dicom_path: pathlib.Path,
                              output_dicom_path: pathlib.Path,
                              output_nifti_path: pathlib.Path) -> Optional[Dicom2NiiParams]:
    """處理單個DICOM到NII的轉換任務"""


    task_params = Dicom2NiiParams(
        sub_dir=input_dicom_path,
        output_dicom_path=output_dicom_path,
        output_nifti_path=output_nifti_path,
    )

    # 檢查任務是否重複
    filter_args = {
        "sub_dir": str(task_params.sub_dir),
        "output_dicom_path": str(task_params.output_dicom_path),
        "output_nifti_path": str(task_params.output_nifti_path)
    }

    if is_task_exists(session, **filter_args):
        logger.info(f"跳過重複任務: {input_dicom_path}")
        return None

    # 發送任務
    try:
        logger.info(f"提交DICOM轉NII任務: {input_dicom_path}")
        # task = dicom_to_nii.push(task_params.get_str_dict())

        # 添加任務記錄
        add_task_record(
            session=session,
            name='dicom_to_nii_queue',
            sub_dir=str(task_params.sub_dir),
            output_dicom_path=str(task_params.output_dicom_path),
            output_nifti_path=str(task_params.output_nifti_path)
        )
        return task_params
    except Exception as e:
        logger.error(f"處理DICOM轉NII任務出錯: {str(e)}")
        return None


def process_nii_for_inference(session: Session, nifti_path: pathlib.Path,
                              dicom_path: pathlib.Path) -> bool:
    """處理NII文件的推理任務"""
    from code_ai.task.task_pipeline import task_pipeline_inference

    nifti_path_str = str(nifti_path)
    dicom_path_str = str(dicom_path)

    # 檢查任務是否重複
    filter_args = {
        "sub_dir": None,
        "output_dicom_path": dicom_path_str,
        "output_nifti_path": nifti_path_str
    }

    if is_task_exists(session, **filter_args):
        logger.info(f"跳過重複推理任務: {nifti_path}")
        return False

    # 發送推理任務
    try:
        logger.info(f"提交NII推理任務: {nifti_path}")
        task_data = {
            'nifti_study_path': nifti_path_str,
            'dicom_study_path': dicom_path_str,
        }
        task_pipeline_inference.push(task_data)

        # 添加任務記錄
        add_task_record(
            session=session,
            name='task_pipeline_inference_queue',
            sub_dir=None,
            output_dicom_path=dicom_path_str,
            output_nifti_path=nifti_path_str
        )
        return True
    except Exception as e:
        logger.error(f"提交NII推理任務出錯: {str(e)}")
        return False


def check_dicom_completeness(dicom_dir: str) -> bool:
    """
    檢查DICOM目錄的完整性

    進行以下檢查：
    1. 目錄是否存在
    2. 目錄中是否有DICOM文件
    3. DICOM文件是否包含必要的標籤
    4. DICOM文件數量是否與預期一致

    Args:
        dicom_dir: DICOM目錄路徑

    Returns:
        bool: 如果DICOM目錄完整則返回True，否則返回False
    """
    if not os.path.exists(dicom_dir):
        logger.error(f"DICOM目錄不存在: {dicom_dir}")
        return False

    try:
        # 檢查目錄中的DICOM文件
        dicom_files = [f for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]

        if len(dicom_files) == 0:
            logger.error(f"DICOM目錄為空: {dicom_dir}")
            return False

        # 讀取第一個文件以獲取系列信息
        first_dicom_path = os.path.join(dicom_dir, dicom_files[0])
        try:
            ds = pydicom.dcmread(first_dicom_path)

            # 檢查是否包含必要的標籤
            required_tags = ['StudyInstanceUID', 'SeriesInstanceUID']
            missing_tags = []

            for tag in required_tags:
                if not hasattr(ds, tag) and tag not in ds:
                    missing_tags.append(tag)

            if missing_tags:
                logger.warning(f"DICOM文件缺少必要標籤 {', '.join(missing_tags)}: {dicom_dir}")
                # 即使缺少某些標籤，我們可能仍然可以進行轉換，所以不立即返回False

            # 如果可能，檢查文件數量是否與預期一致
            expected_count = None

            # 嘗試從不同的標籤獲取圖像數量信息
            if hasattr(ds, 'ImagesInAcquisition') and ds.ImagesInAcquisition is not None:
                expected_count = int(ds.ImagesInAcquisition)
            elif hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames is not None:
                expected_count = int(ds.NumberOfFrames)
            elif hasattr(ds, 'AccessionNumber') and hasattr(ds, 'InstanceNumber'):
                # 如果有AccessionNumber和InstanceNumber，可以通過檢查InstanceNumber的最大值來估計
                try:
                    max_instance = 0
                    for f in dicom_files[:100]:  # 限制檢查文件數以提高效率
                        file_path = os.path.join(dicom_dir, f)
                        ds_temp = pydicom.dcmread(file_path, stop_before_pixels=True)
                        if hasattr(ds_temp, 'InstanceNumber'):
                            max_instance = max(max_instance, int(ds_temp.InstanceNumber))

                    if max_instance > 0:
                        expected_count = max_instance
                except Exception as e:
                    logger.warning(f"估計DICOM文件數量時出錯: {str(e)}")

            if expected_count is not None and len(dicom_files) < expected_count * 0.9:  # 允許10%的誤差
                logger.warning(f"DICOM文件數量不完整，預期約{expected_count}個，實際{len(dicom_files)}個: {dicom_dir}")
                return False

            # 基本檢查通過
            logger.info(f"DICOM目錄完整性檢查通過: {dicom_dir} (檔案數: {len(dicom_files)})")
            return True

        except Exception as e:
            logger.error(f"讀取DICOM文件出錯 {first_dicom_path}: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"檢查DICOM完整性出錯: {str(e)}")
        return False


def is_task_waiting_in_queue(session: Session, nifti_path: str, dicom_path: str) -> bool:
    """
    檢查任務是否正在等待隊列中處理

    檢查兩個位置：
    1. RawDicomToNiiInference 資料表中的記錄
    2. RabbitMQ 中的 task_pipeline_inference_queue 隊列

    Args:
        session: 數據庫會話
        nifti_path: NII文件路徑
        dicom_path: DICOM目錄路徑

    Returns:
        bool: 如果任務正在等待則返回True，否則返回False
    """
    try:
        # 1. 首先查詢資料庫中是否有對應的記錄
        result = session.query(RawDicomToNiiInference).filter(
            RawDicomToNiiInference.output_nifti_path == nifti_path,
            RawDicomToNiiInference.output_dicom_path == dicom_path,
            RawDicomToNiiInference.name == 'task_pipeline_inference_queue'
        ).one_or_none()

        # 檢查任務是否存在於資料庫中
        if result is not None:
            logger.info(f"任務在資料庫記錄中找到: {nifti_path}")
            return True

        # 2. 檢查RabbitMQ隊列中是否有對應的任務
        try:
            from code_ai.task.task_pipeline import task_pipeline_inference
            # 使用pika連接RabbitMQ並檢查隊列
            import pika
            import json

            # 連接到RabbitMQ
            connection_params = pika.ConnectionParameters(
                host=BrokerConnConfig.RABBITMQ_HOST,
                port=BrokerConnConfig.RABBITMQ_PORT,
                virtual_host = BrokerConnConfig.RABBITMQ_VIRTUAL_HOST,
                credentials=pika.PlainCredentials(
                    BrokerConnConfig.RABBITMQ_USER,
                    BrokerConnConfig.RABBITMQ_PASS
                )
            )
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()

            # 獲取隊列消息但不消費
            queue_name = 'task_pipeline_inference_queue'
            method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=False)

            # 如果隊列中有消息
            while method_frame:
                try:
                    # 解析消息內容
                    message = json.loads(body.decode('utf-8'))
                    task_body = json.loads(message.get('body', '{}'))

                    # 檢查是否為目標任務
                    if (task_body.get('nifti_study_path') == nifti_path and
                            task_body.get('dicom_study_path') == dicom_path):
                        logger.info(f"任務在RabbitMQ隊列中找到: {nifti_path}")
                        # 重新放回隊列中
                        channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
                        connection.close()
                        return True

                    # 不是目標任務，繼續檢查下一個
                    channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
                    method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=False)
                except (json.JSONDecodeError, KeyError) as e:
                    # 消息格式錯誤，繼續檢查下一個
                    logger.warning(f"解析RabbitMQ消息出錯: {str(e)}")
                    channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
                    method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=False)

            # 關閉連接
            connection.close()

        except Exception as e:
            logger.warning(f"檢查RabbitMQ隊列時出錯: {str(e)}")
            # 發生錯誤時，我們不應該中斷流程，而是繼續後續步驟

        # 沒有找到對應的記錄
        return False

    except SQLAlchemyError as e:
        logger.error(f"查詢任務隊列出錯: {str(e)}")
        raise


def run_command(cmd: str) -> Tuple[bool, str]:
    """
    執行命令行命令

    Args:
        cmd: 要執行的命令

    Returns:
        Tuple[bool, str]: 執行成功則返回(True, 輸出)，否則返回(False, 錯誤信息)
    """
    try:
        logger.info(f"執行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"命令執行失敗，返回代碼 {e.returncode}: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"執行命令時出錯: {str(e)}")
        return False, str(e)


def check_input_files_and_dicom(pipeline_task: Dict[str, Any]) -> bool:
    """
    檢查輸入文件列表，如果文件不存在則檢查DICOM目錄的完整性

    Args:
        pipeline_task: 包含input_list和input_dicom_dir的任務字典

    Returns:
        bool: 如果所有輸入文件存在或DICOM目錄完整則返回True，否則返回False
    """
    input_list = pipeline_task.get('input_list', [])
    input_dicom_dir = pipeline_task.get('input_dicom_dir', '')
    study_id = pipeline_task.get('study_id', 'unknown')
    name = pipeline_task.get('name', 'unknown')

    logger.info(f"檢查任務{study_id}({name})的輸入文件")

    # 檢查所有輸入文件是否存在
    missing_files = []
    for input_file in input_list:
        if not os.path.exists(input_file):
            missing_files.append(input_file)
            logger.warning(f"輸入文件不存在: {input_file}")

    # 如果所有文件都存在，返回True
    if not missing_files:
        logger.info(f"任務{study_id}({name})的所有輸入文件都存在")
        return True

    # 如果有缺失文件但沒有指定DICOM目錄，返回False
    if not input_dicom_dir:
        logger.error(f"任務{study_id}({name})缺少輸入文件且沒有指定DICOM目錄")
        return False

    # 檢查DICOM目錄的完整性
    logger.info(f"檢查DICOM目錄的完整性: {input_dicom_dir}")
    dicom_complete = check_dicom_completeness(input_dicom_dir)

    if dicom_complete:
        logger.info(f"DICOM目錄完整，可以進行轉換: {input_dicom_dir}")
    else:
        logger.error(f"DICOM目錄不完整或有問題: {input_dicom_dir}")

    return dicom_complete


def check_output_files_and_rerun(session: Session, pipeline_task: Dict[str, Any]) -> bool:
    """
    檢查輸出文件列表，如果文件不存在則檢查任務狀態並可能重新執行任務

    檢查流程：
    1. 檢查所有輸出文件是否存在
    2. 如果有文件不存在，檢查任務是否在資料庫記錄或RabbitMQ隊列中等待處理
    3. 如果不在等待中，檢查輸入文件和DICOM完整性
    4. 如果輸入文件齊全，重新執行命令並記錄

    Args:
        session: 數據庫會話
        pipeline_task: 包含output_list、study_id、name和cmd_str的任務字典

    Returns:
        bool: 如果所有輸出文件存在或任務成功重新啟動則返回True，否則返回False
    """
    output_list = pipeline_task.get('output_list', [])
    study_id = pipeline_task.get('study_id', 'unknown')
    name = pipeline_task.get('name', 'unknown')
    cmd_str = pipeline_task.get('cmd_str', '')

    logger.info(f"檢查任務{study_id}({name})的輸出文件")

    # 檢查所有輸出文件是否存在
    missing_files = []
    for output_file in output_list:
        if not os.path.exists(output_file):
            missing_files.append(output_file)
            logger.warning(f"輸出文件不存在: {output_file}")

    # 如果所有文件都存在，返回True
    if not missing_files:
        logger.info(f"任務{study_id}({name})的所有輸出文件都存在")
        return True

    # 如果有缺失的輸出文件，需要檢查任務狀態
    # 為簡化查詢，我們只使用第一個輸入和輸出路徑進行檢查
    if pipeline_task.get('input_list') and pipeline_task.get('input_dicom_dir'):
        nifti_path = pipeline_task['input_list'][0]
        dicom_path = pipeline_task['input_dicom_dir']

        # 檢查任務是否在資料庫或RabbitMQ隊列中等待處理
        is_waiting = is_task_waiting_in_queue(session, nifti_path, dicom_path)

        if is_waiting:
            logger.info(f"任務{study_id}({name})正在等待隊列中，不需要重新運行")
            return True

    # 如果任務不在等待隊列中且命令字符串存在，執行命令
    if cmd_str:
        logger.info(f"準備重新運行任務{study_id}({name})")

        # 首先檢查輸入文件是否存在
        inputs_exist = check_input_files_and_dicom(pipeline_task)
        if not inputs_exist:
            logger.error(f"任務{study_id}({name})輸入文件不存在且DICOM不完整，無法重新運行")
            return False

        # 執行命令
        success, output = run_command(cmd_str)

        if success:
            logger.info(f"成功重新運行任務{study_id}({name})")

            # 檢查命令執行後文件是否生成
            files_created = True
            for output_file in missing_files:
                if not os.path.exists(output_file):
                    logger.warning(f"命令執行後輸出文件仍不存在: {output_file}")
                    files_created = False

            # 在成功執行命令後，將任務記錄添加到資料庫中
            if files_created and pipeline_task.get('input_list') and pipeline_task.get('input_dicom_dir'):
                try:
                    nifti_path = pipeline_task['input_list'][0]
                    dicom_path = pipeline_task['input_dicom_dir']

                    # 添加任務記錄以防止重複運行
                    add_task_record(
                        session=session,
                        name='task_pipeline_inference_queue',
                        sub_dir=None,
                        output_dicom_path=dicom_path,
                        output_nifti_path=nifti_path
                    )
                    logger.info(f"為重新運行的任務{study_id}({name})添加資料庫記錄")
                except Exception as e:
                    logger.error(f"添加任務記錄時出錯: {str(e)}")

            return files_created
        else:
            logger.error(f"重新運行任務{study_id}({name})失敗: {output}")
            return False
    else:
        logger.error(f"任務{study_id}({name})缺少命令字符串，無法重新運行")
        return False


def load_pipeline_tasks(file_path: str) -> List[Dict[str, Any]]:
    """
    從文件中加載流水線任務配置

    Args:
        file_path: 任務配置文件的路徑

    Returns:
        List[Dict[str, Any]]: 任務配置列表
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"任務配置文件不存在: {file_path}")
            return []

        with open(file_path, 'r') as f:
            tasks = json.load(f)

        if not isinstance(tasks, list):
            logger.error(f"任務配置文件格式錯誤，應為列表: {file_path}")
            return []

        logger.info(f"成功加載 {len(tasks)} 個任務配置")
        return tasks

    except json.JSONDecodeError as e:
        logger.error(f"解析任務配置文件出錯: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"加載任務配置出錯: {str(e)}")
        return []


def process_pipeline_tasks(session: Session, tasks_file_path: str) -> Dict[str, Any]:
    """
    處理流水線任務，檢查輸入輸出文件並在必要時重新執行任務

    流程：
    1. 加載任務配置文件
    2. 檢查每個任務的輸入文件和DICOM完整性
    3. 檢查每個任務的輸出文件，如果缺失則查看任務狀態
    4. 如果任務不在等待中且輸入文件齊全，則重新執行任務

    Args:
        session: 數據庫會話
        tasks_file_path: 任務配置文件的路徑

    Returns:
        Dict[str, Any]: 處理結果統計
    """
    # 加載任務配置
    tasks = load_pipeline_tasks(tasks_file_path)
    if not tasks:
        return {"status": "error", "error": "無法加載任務配置或配置為空"}

    stats = {
        "total_tasks": len(tasks),
        "input_checks_passed": 0,
        "output_checks_passed": 0,
        "rerun_tasks": 0,
        "failed_tasks": 0,
        "waiting_tasks": 0
    }

    # 處理每個任務
    for task in tasks:
        study_id = task.get('study_id', 'unknown')
        name = task.get('name', 'unknown')

        try:
            logger.info(f"開始處理任務: {study_id}({name})")

            # 檢查輸入文件和DICOM
            inputs_ok = check_input_files_and_dicom(task)
            if inputs_ok:
                stats["input_checks_passed"] += 1
                logger.info(f"任務{study_id}({name})輸入檢查通過")
            else:
                logger.warning(f"任務{study_id}({name})輸入檢查失敗，跳過輸出檢查")
                stats["failed_tasks"] += 1
                continue

            # 檢查輸出文件，如果需要重新運行
            if task.get('input_list') and task.get('input_dicom_dir'):
                nifti_path = task['input_list'][0]
                dicom_path = task['input_dicom_dir']

                # 先檢查任務是否在等待中
                is_waiting = is_task_waiting_in_queue(session, nifti_path, dicom_path)
                if is_waiting:
                    logger.info(f"任務{study_id}({name})正在等待隊列中，標記為正在處理")
                    stats["waiting_tasks"] += 1
                    continue

            # 檢查輸出文件並可能重新運行任務
            outputs_ok = check_output_files_and_rerun(session, task)
            if outputs_ok:
                stats["output_checks_passed"] += 1
                logger.info(f"任務{study_id}({name})輸出檢查通過或成功重新運行")

                # 判斷是否是重新運行的任務
                rerun = False
                for output_file in task.get('output_list', []):
                    if os.path.exists(output_file) and os.path.getmtime(output_file) > time.time() - 300:  # 5分鐘內創建的文件
                        rerun = True
                        break

                if rerun:
                    stats["rerun_tasks"] += 1
                    logger.info(f"任務{study_id}({name})已成功重新運行")
            else:
                logger.warning(f"任務{study_id}({name})輸出檢查失敗")
                stats["failed_tasks"] += 1

        except Exception as e:
            logger.error(f"處理任務{study_id}({name})時出錯: {str(e)}", exc_info=True)
            stats["failed_tasks"] += 1

    logger.info(f"任務處理統計: {stats}")
    return stats


# @Booster(BoosterParams(
#     queue_name='add_raw_dicom_to_nii_inference_queue',
#     broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
#     concurrent_mode=ConcurrentModeEnum.SOLO,
#     qps=1,
#     log_level=20,  # INFO level
#     is_print_detail_exception=True
# ))
# def add_raw_dicom_to_nii_inference(tasks_file_path: Optional[str] = None) -> Dict[str, Any]:
#     """
#     主函數: 定時檢查並處理DICOM到NII的轉換和推理任務
#
#     Args:
#         tasks_file_path: 可選的任務配置文件路徑，用於檢查和重新運行特定任務
#
#     Returns:
#         Dict[str, Any]: 處理結果統計
#     """
#     start_time = time.time()
#     logger.info("開始執行DICOM到NII轉換和推理任務檢查")
#
#     try:
#         # 獲取環境路徑
#         input_dicom, output_dicom_path, output_nifti_path = get_env_paths()
#
#         # 檢查並創建必要的目錄
#         check_and_create_directories([output_dicom_path, output_nifti_path])
#
#         # 獲取資料庫連接
#         enginex, sqla_helper = get_sqla_helper()
#         session: Session = sqla_helper.session
#
#         # 如果提供了任務配置文件，處理特定任務
#         if tasks_file_path:
#             logger.info(f"使用任務配置文件: {tasks_file_path}")
#             pipeline_stats = process_pipeline_tasks(session, tasks_file_path)
#
#             elapsed_time = time.time() - start_time
#             result = {
#                 "status": "success",
#                 "pipeline_stats": pipeline_stats,
#                 "elapsed_time": elapsed_time
#             }
#
#             logger.info(f"完成特定任務處理，總耗時: {elapsed_time:.2f}秒")
#             return result
#
#         # 正常的DICOM到NII轉換處理
#         # 找出所有輸入目錄
#         input_dicom_list = []
#         if input_dicom.exists():
#             folders = sorted(input_dicom.iterdir())
#             input_dicom_list = [f for f in folders if f.is_dir()]
#
#         # 如果沒有子目錄，使用主目錄
#         if len(input_dicom_list) == 0:
#             input_dicom_list = [input_dicom]
#
#         logger.info(f"發現 {len(input_dicom_list)} 個DICOM目錄待處理")
#
#         # 處理所有DICOM到NII的轉換任務 - 批次處理版本
#         processed_tasks = []
#         # 先收集所有需要處理的任務
#         for input_dicom_path in input_dicom_list:
#             try:
#                 task = process_dicom_to_nii_task(
#                     session, input_dicom_path, output_dicom_path, output_nifti_path
#                 )
#                 if task:
#                     processed_tasks.append(task)
#             except Exception as e:
#                 logger.error(f"處理DICOM目錄出錯 {input_dicom_path}: {str(e)}")
#
#         # 為所有任務設置超時
#         for task in processed_tasks:
#             task.set_timeout(600)  # 10分鐘超時
#
#         # 批次等待所有任務完成
#         logger.info(f"已提交 {len(processed_tasks)} 個DICOM到NII轉換任務，等待所有任務完成...")
#         processed_results = []
#         if processed_tasks:
#             for task in processed_tasks:
#                 result = task.result
#                 processed_results.append(result)
#                 logger.debug(f"DICOM轉NII任務完成: {result}")
#             logger.info(f"完成 {len(processed_results)} 個DICOM到NII轉換任務")
#         else:
#             logger.info("沒有新的DICOM到NII轉換任務需要處理")
#
#         # 處理NII推理任務 - 批次處理版本
#         inference_tasks = []
#
#         # 首先處理新轉換的NII文件
#         if processed_results:
#             for result_path in processed_results:
#                 nifti_study_path = output_nifti_path.joinpath(os.path.basename(result_path))
#                 dicom_study_path = output_dicom_path.joinpath(nifti_study_path.name)
#
#                 # 檢查任務是否重複
#                 nifti_path_str = str(nifti_study_path)
#                 dicom_path_str = str(dicom_study_path)
#
#                 filter_args = {
#                     "sub_dir": None,
#                     "output_dicom_path": dicom_path_str,
#                     "output_nifti_path": nifti_path_str
#                 }
#
#                 if not is_task_exists(session, **filter_args):
#                     # 收集推理任務但暫不發送
#                     inference_tasks.append((nifti_study_path, dicom_study_path))
#
#         # 然後檢查現有的NII目錄中是否有未處理的文件
#         elif output_nifti_path.exists():
#             existing_nii_paths = sorted(output_nifti_path.iterdir())
#             logger.info(f"檢查 {len(existing_nii_paths)} 個現有NII文件")
#
#             for nifti_study_path in existing_nii_paths:
#                 if not nifti_study_path.is_dir():
#                     continue
#
#                 dicom_study_path = output_dicom_path.joinpath(nifti_study_path.name)
#
#                 # 檢查任務是否重複
#                 nifti_path_str = str(nifti_study_path)
#                 dicom_path_str = str(dicom_study_path)
#
#                 filter_args = {
#                     "sub_dir": None,
#                     "output_dicom_path": dicom_path_str,
#                     "output_nifti_path": nifti_path_str
#                 }
#
#                 if not is_task_exists(session, **filter_args):
#                     # 收集推理任務但暫不發送
#                     inference_tasks.append((nifti_study_path, dicom_study_path))
#
#         # 批次提交所有推理任務
#         logger.info(f"準備批次提交 {len(inference_tasks)} 個NII推理任務")
#         inference_count = 0
#         from code_ai.task.task_pipeline import task_pipeline_inference
#
#         for nifti_path, dicom_path in inference_tasks:
#             try:
#                 nifti_path_str = str(nifti_path)
#                 dicom_path_str = str(dicom_path)
#
#                 logger.info(f"提交NII推理任務: {nifti_path}")
#                 task_data = {
#                     'nifti_study_path': nifti_path_str,
#                     'dicom_study_path': dicom_path_str,
#                 }
#                 task_pipeline_inference.push(task_data)
#
#                 # 添加任務記錄
#                 add_task_record(
#                     session=session,
#                     name='task_pipeline_inference_queue',
#                     sub_dir=None,
#                     output_dicom_path=dicom_path_str,
#                     output_nifti_path=nifti_path_str
#                 )
#                 inference_count += 1
#             except Exception as e:
#                 logger.error(f"提交NII推理任務出錯: {str(e)}")
#
#         # 任務已全部提交，等待處理完成（這裡不需要等待結果，因為推理任務是異步的）
#         logger.info(f"已批次提交 {inference_count} 個NII推理任務")
#
#         elapsed_time = time.time() - start_time
#         logger.info(f"完成 {inference_count} 個NII推理任務提交，總耗時: {elapsed_time:.2f}秒")
#
#         return {
#             "status": "success",
#             "dicom_to_nii_count": len(processed_results),
#             "inference_count": inference_count,
#             "elapsed_time": elapsed_time
#         }
#
#     except Exception as e:
#         logger.error(f"執行檢查任務失敗: {str(e)}", exc_info=True)
#         return {
#             "status": "error",
#             "error": str(e),
#             "elapsed_time": time.time() - start_time
#         }


@Booster(BoosterParams(
    queue_name='add_raw_dicom_to_nii_inference_queue',
    broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM,
    concurrent_mode=ConcurrentModeEnum.SOLO,
    qps=1,
    log_level=20,  # INFO level
    is_print_detail_exception=True
))
def add_raw_dicom_to_nii_inference(tasks_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    主函數: 定時檢查並處理DICOM到NII的轉換和推理任務

    Args:
        tasks_file_path: 可選的任務配置文件路徑，用於檢查和重新運行特定任務

    Returns:
        Dict[str, Any]: 處理結果統計
    """
    start_time = time.time()
    logger.info("開始執行DICOM到NII轉換和推理任務檢查")

    try:
        # 獲取環境路徑
        input_dicom, output_dicom_path, output_nifti_path = get_env_paths()

        # 檢查並創建必要的目錄
        check_and_create_directories([output_dicom_path, output_nifti_path])

        # 獲取資料庫連接
        enginex, sqla_helper = get_sqla_helper()
        session: Session = sqla_helper.session

        # 如果提供了任務配置文件，處理特定任務
        if tasks_file_path:
            logger.info(f"使用任務配置文件: {tasks_file_path}")
            pipeline_stats = process_pipeline_tasks(session, tasks_file_path)

            elapsed_time = time.time() - start_time
            result = {
                "status": "success",
                "pipeline_stats": pipeline_stats,
                "elapsed_time": elapsed_time
            }

            logger.info(f"完成特定任務處理，總耗時: {elapsed_time:.2f}秒")
            return result

        # 正常的DICOM到NII轉換處理
        # 找出所有輸入目錄
        input_dicom_list = []
        if input_dicom.exists():
            folders = sorted(input_dicom.iterdir())
            input_dicom_list = [f for f in folders if f.is_dir()]

        # 如果沒有子目錄，使用主目錄
        if len(input_dicom_list) == 0:
            input_dicom_list = [input_dicom]

        logger.info(f"發現 {len(input_dicom_list)} 個DICOM目錄待處理")

        # 處理所有DICOM到NII的轉換任務 - 批次處理版本
        processed_tasks = []
        # 先收集所有需要處理的任務
        for input_dicom_path in input_dicom_list:
            try:
                task_params = process_dicom_to_nii_task(
                    session, input_dicom_path, output_dicom_path, output_nifti_path
                )
                if task_params:
                    processed_tasks.append(task_params)
            except Exception as e:
                logger.error(f"處理DICOM目錄出錯 {input_dicom_path}: {str(e)}")


        study_dir_name_list = list(map(lambda x:os.path.basename(x.sub_dir), processed_tasks))
        print(study_dir_name_list)
        post_study(study_dir_name_list)
        # 批次等待所有任務完成
        logger.info(f"已提交 {len(processed_tasks)} 個DICOM到NII轉換任務，等待所有任務完成...")
        # processed_results = []
        # if processed_tasks:
        #     for task in processed_tasks:
        #         result = task.result
        #         processed_results.append(result)
        #         logger.debug(f"DICOM轉NII任務完成: {result}")
        #     logger.info(f"完成 {len(processed_results)} 個DICOM到NII轉換任務")
        # else:
        #     logger.info("沒有新的DICOM到NII轉換任務需要處理")

        elapsed_time = time.time() - start_time

        return {
            "status": "success",
            "dicom_to_nii_count": len(input_dicom_list),
            "elapsed_time": elapsed_time
        }

    except Exception as e:
        logger.error(f"執行檢查任務失敗: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='處理DICOM到NII的轉換和推理任務')
    parser.add_argument('--tasks_file', type=str, help='任務配置文件的路徑，用於檢查特定任務')

    args = parser.parse_args()

    try:
        if args.tasks_file:
            result = add_raw_dicom_to_nii_inference.push(tasks_file_path=args.tasks_file)
        else:
            result = add_raw_dicom_to_nii_inference.push()

        print(f"任務提交結果: {result}")
    except Exception as e:
        print(f"提交任務失敗: {str(e)}")