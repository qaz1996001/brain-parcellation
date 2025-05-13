import copy
import functools
import threading
import time
from db_libs.sqla_lib import SqlaReflectHelper
from funboost.core.serialization import Serialization
from sqlalchemy import create_engine
from funboost import FunctionResultStatus, funboost_config_deafult

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
import json

Base = declarative_base()


class FunboostConsumeResult(Base):
    __tablename__ = 'funboost_consume_results'

    _id = Column(String, primary_key=True)
    function = Column(String)
    host_name = Column(String)
    host_process = Column(String)
    insert_minutes = Column(String)
    insert_time = Column(DateTime)
    insert_time_str = Column(String)
    msg_dict = Column(Text)  # 存儲為 JSON 字串
    params = Column(Text)  # 存儲為 JSON 字串
    params_str = Column(String)
    process_id = Column(Integer)
    publish_time = Column(Float)
    publish_time_str = Column(String)
    queue_name = Column(String)
    result = Column(Text)
    run_times = Column(Integer)
    script_name = Column(String)
    script_name_long = Column(String)
    success = Column(Boolean)  # SQLite INTEGER 轉為 Python Boolean
    task_id = Column(String)
    thread_id = Column(Integer)
    time_cost = Column(Float)
    time_end = Column(Float)
    time_start = Column(Float)
    total_thread = Column(Integer)
    utime = Column(String)
    exception = Column(Text)
    rpc_result_expire_seconds = Column(Integer)
    run_status = Column(String)
    # JSON 處理的 helper 方法
    @property
    def msg_dict_obj(self):
        """將 JSON 字串轉換為 Python 對象"""
        if self.msg_dict:
            return json.loads(self.msg_dict)
        return None

    @msg_dict_obj.setter
    def msg_dict_obj(self, value):
        """將 Python 對象轉換為 JSON 字串"""
        if value is not None:
            self.msg_dict = json.dumps(value)
        else:
            self.msg_dict = None

    @property
    def params_obj(self):
        """將 JSON 字串轉換為 Python 對象"""
        if self.params:
            return json.loads(self.params)
        return None

    @params_obj.setter
    def params_obj(self, value):
        """將 Python 對象轉換為 JSON 字串"""
        if value is not None:
            self.params = json.dumps(value)
        else:
            self.params = None

    def __repr__(self):
        return f"<FunboostConsumeResult(id='{self._id}', function='{self.function}', task_id='{self.task_id}')>"



_thread_local_data = threading.local()
_thread_local_data.pending_objects = []
_thread_local_data.last_flush_time = time.time()

MAX_BATCH_SIZE = 100  # 批量提交的最大记录数
FLUSH_INTERVAL = 5.0  # 强制刷新的时间间隔（秒）


def _flush_pending_objects():
    """将所有待处理的对象刷新到数据库"""
    if not hasattr(_thread_local_data, 'pending_objects') or not _thread_local_data.pending_objects:
        return

    enginex, sqla_helper = get_sqla_helper()
    with sqla_helper.session as ss:
        try:
            ss.add_all(_thread_local_data.pending_objects)
            ss.commit()
            print(f"批量提交了 {len(_thread_local_data.pending_objects)} 条记录")
        except Exception as e:
            ss.rollback()
            print(f"批量提交失败: {str(e)}")
            # 如果批量失败，尝试逐个提交以保存尽可能多的记录
            for obj in _thread_local_data.pending_objects:
                try:
                    with sqla_helper.session as individual_session:
                        individual_session.add(obj)
                        individual_session.commit()
                except Exception as individual_e:
                    print(f"单条记录提交失败: {str(individual_e)}")

    # 清空待处理对象列表
    _thread_local_data.pending_objects.clear()
    _thread_local_data.last_flush_time = time.time()


@functools.lru_cache()
def get_sqla_helper():
    enginex = create_engine(
        funboost_config_deafult.BrokerConnConfig.SQLACHEMY_ENGINE_URL,
        max_overflow=10,  # 超过连接池大小外最多创建的连接
        pool_size=50,  # 连接池大小
        pool_timeout=30,  # 池中没有线程最多等待的时间，否则报错
        pool_recycle=600,  # 多久之后对线程池中的线程进行一次连接的回收（重置）
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

    # 处理状态字典
    status_dict = function_result_status.get_status_dict()
    status_dict_new = copy.copy(status_dict)
    # 将字典类型的字段转换为JSON字符串
    for k, v in status_dict.items():
        if isinstance(v, dict):
            status_dict_new[k] = json.dumps(v)
    # 创建模型对象并添加到待处理列表
    sa_model = FunboostConsumeResult(**status_dict_new)
    _thread_local_data.pending_objects.append(sa_model)
    # 检查是否需要刷新（达到最大批量大小或超过时间间隔）
    current_time = time.time()
    if (len(_thread_local_data.pending_objects) >= MAX_BATCH_SIZE or
            current_time - _thread_local_data.last_flush_time > FLUSH_INTERVAL):
        _flush_pending_objects()


def save_result_status_to_sqlalchemy(function_result_status: FunctionResultStatus):
    """ function_result_status变量上有各种丰富的信息 ,用户可以使用其中的信息
    用户自定义记录函数消费信息的钩子函数

    例如  @boost('test_user_custom', user_custom_record_process_info_func=save_result_status_to_sqlalchemy)
    """
    enginex, sqla_helper = get_sqla_helper()
    with (sqla_helper.session as ss):
        status_dict = function_result_status.get_status_dict()
        status_dict_new = copy.copy(status_dict)
        for k, v in status_dict.items():
            if isinstance(v, (dict,list)):
                # status_dict_new[k] = json.dumps(v)
                status_dict_new[k] = Serialization.to_json_str(v)
        sa_model = FunboostConsumeResult(**status_dict_new)
        ss.add(sa_model)
        ss.commit()
        # sql = _gen_insert_sqlalchemy(status_dict) # 这种是sqlahemy sql方式插入.
        # print('sql',sql)
        # print('status_dict_new', status_dict_new)
        # ss.execute(sql, status_dict_new)
        # ss.merge(t_funboost_consume_results(**status_dict_new)) # 这种是orm方式插入.




def query_result_status_to_sqlalchemy():
    """
    select (_id,params ,queue_name ,result ,success )
    from funboost_consume_results
    where queue_name=:queue_name

    """
    pass

    # enginex, sqla_helper = get_sqla_helper()
    # with (sqla_helper.session as ss):





import atexit
atexit.register(_flush_pending_objects)
