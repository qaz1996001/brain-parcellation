import typing

from funboost import BrokerEnum, BoosterParams, ConcurrentModeEnum


class BoosterParamsMyRABBITMQ(BoosterParams): # 传这个类就可以少每次都亲自指定使用rabbitmq作为消息队列，和重试改为4次,和消费发布日志写入自定义.log文件。
    broker_kind : str     = BrokerEnum.RABBITMQ_AMQPSTORM
    concurrent_mode: str  = ConcurrentModeEnum.THREADING
    concurrent_num: int   = 10
    is_send_consumer_hearbeat_to_redis:bool = True

    max_retry_times : int = 3
    retry_interval: typing.Union[float, int] = 20
    is_push_to_dlx_queue_when_retry_max_times:bool = True

    # user_custom_record_process_info_func: typing.Callable = None  # 提供一个用户自定义的保存消息处理记录到某个地方例如mysql数据库的函数，函数仅仅接受一个入参，入参类型是 FunctionResultStatus，用户可以打印参数
    is_using_rpc_mode:bool = True
    rpc_result_expire_seconds: int = 1800

class BoosterParamsMyAI(BoosterParamsMyRABBITMQ): # 传这个类就可以少每次都亲自指定使用rabbitmq作为消息队列，和重试改为4次,和消费发布日志写入自定义.log文件。
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int  = 5
    qps           : int  = 1