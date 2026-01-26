import atexit
import copy
import functools
import threading
import time
import json

from db_libs.sqla_lib import SqlaReflectHelper
from funboost.core.serialization import Serialization
from sqlalchemy import create_engine
from funboost import FunctionResultStatus, funboost_config_deafult

from code_ai.utils.model import FunboostConsumeResult, Base


_thread_local_data = threading.local()
_thread_local_data.pending_objects = []
_thread_local_data.last_flush_time = time.time()

MAX_BATCH_SIZE = 100  # 批量提交的最大記錄數
FLUSH_INTERVAL = 5.0  # 強制刷新的時間間隔（秒）


def _flush_pending_objects():
    """將所有待處理的物件刷新到資料庫"""
    if not hasattr(_thread_local_data, 'pending_objects') or not _thread_local_data.pending_objects:
        return

    enginex, sqla_helper = get_sqla_helper()
    with sqla_helper.session as ss:
        try:
            ss.add_all(_thread_local_data.pending_objects)
            ss.commit()
            print(f"批量提交了 {len(_thread_local_data.pending_objects)} 條記錄")
        except Exception as e:
            ss.rollback()
            print(f"批量提交失敗: {str(e)}")
            # 如果批量失敗，嘗試逐個提交以保存盡可能多的記錄
            for obj in _thread_local_data.pending_objects:
                try:
                    with sqla_helper.session as individual_session:
                        individual_session.add(obj)
                        individual_session.commit()
                except Exception as individual_e:
                    print(f"單條記錄提交失敗: {str(individual_e)}")

    # 清空待處理物件列表
    _thread_local_data.pending_objects.clear()
    _thread_local_data.last_flush_time = time.time()


@functools.lru_cache()
def get_sqla_helper():
    enginex = create_engine(
        funboost_config_deafult.BrokerConnConfig.SQLACHEMY_ENGINE_URL,
        max_overflow=10,  # 超過連接池大小外最多建立的連接
        pool_size=50,  # 連接池大小
        pool_timeout=30,  # 池中沒有執行緒最多等待的時間，否則報錯
        pool_recycle=600,  # 多久之後對執行緒池中的執行緒進行一次連接的回收（重置）
        # echo=True
    )
    sqla_helper = SqlaReflectHelper(enginex)
    Base.metadata.create_all(enginex)
    # t_funboost_consume_results = sqla_helper.base_classes.funboost_consume_results
    return enginex, sqla_helper, #t_funboost_consume_results

def save_result_status_to_sqlalchemy_by_batch(function_result_status: FunctionResultStatus):
    if not hasattr(_thread_local_data, 'pending_objects'):
        _thread_local_data.pending_objects = []
        _thread_local_data.last_flush_time = time.time()

    # 處理狀態字典
    status_dict = function_result_status.get_status_dict()
    status_dict_new = copy.copy(status_dict)
    # 將字典類型的欄位轉換為JSON字串
    for k, v in status_dict.items():
        if isinstance(v, dict):
            status_dict_new[k] = json.dumps(v)
    # 建立模型物件並添加到待處理列表
    sa_model = FunboostConsumeResult(**status_dict_new)
    _thread_local_data.pending_objects.append(sa_model)
    # 檢查是否需要刷新（達到最大批量大小或超過時間間隔）
    current_time = time.time()
    if (len(_thread_local_data.pending_objects) >= MAX_BATCH_SIZE or
            current_time - _thread_local_data.last_flush_time > FLUSH_INTERVAL):
        _flush_pending_objects()


def save_result_status_to_sqlalchemy(function_result_status: FunctionResultStatus):
    """ function_result_status變數上有各種豐富的資訊，使用者可以使用其中的資訊
    使用者自定義記錄函數消費資訊的鉤子函數

    例如  @boost('test_user_custom', user_custom_record_process_info_func=save_result_status_to_sqlalchemy)
    """
    from code_ai.utils.model import FunboostConsumeResult
    from sqlalchemy import inspect
    
    enginex, sqla_helper = get_sqla_helper()
    with (sqla_helper.session as ss):
        status_dict = function_result_status.get_status_dict()
        
        # Get valid column names from the model
        mapper = inspect(FunboostConsumeResult)
        valid_columns = {c.key for c in mapper.columns}
        
        status_dict_new = {}
        for k, v in status_dict.items():
            if k not in valid_columns:
                continue  # Skip unknown fields (e.g., publish_time_format)
            if isinstance(v, (dict, list)):
                status_dict_new[k] = Serialization.to_json_str(v)
            else:
                status_dict_new[k] = v
        
        sa_model = FunboostConsumeResult(**status_dict_new)
        ss.add(sa_model)
        ss.commit()
        # sql = _gen_insert_sqlalchemy(status_dict) # 這種是sqlahemy sql方式插入.
        # print('sql',sql)
        # print('status_dict_new', status_dict_new)
        # ss.execute(sql, status_dict_new)
        # ss.merge(t_funboost_consume_results(**status_dict_new)) # 這種是orm方式插入.




def query_result_status_to_sqlalchemy():
    """
    select (_id,params ,queue_name ,result ,success )
    from funboost_consume_results
    where queue_name=:queue_name

    """
    pass

    # enginex, sqla_helper = get_sqla_helper()
    # with (sqla_helper.session as ss):





atexit.register(_flush_pending_objects)