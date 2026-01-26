import typing
from funboost import BrokerEnum, BoosterParams, ConcurrentModeEnum


class BoosterParamsMyRABBITMQ(
    BoosterParams
):  # 傳這個類別就可以少每次都親自指定使用 RabbitMQ 作為訊息佇列，和重試改為 4 次，和消費發佈日誌寫入自定義 .log 檔案。
    broker_kind: str = BrokerEnum.RABBITMQ_AMQPSTORM
    concurrent_mode: str = ConcurrentModeEnum.THREADING
    concurrent_num: int = 10
    is_send_consumer_hearbeat_to_redis: bool = True

    max_retry_times: int = 3
    retry_interval: typing.Union[float, int] = 20
    is_push_to_dlx_queue_when_retry_max_times: bool = True

    # user_custom_record_process_info_func: typing.Callable = None  # 提供一個使用者自定義的儲存訊息處理記錄到某個地方（例如 MySQL 資料庫）的函數，函數僅接受一個輸入參數，參數類型是 FunctionResultStatus，使用者可以列印參數
    is_using_rpc_mode: bool = True
    rpc_result_expire_seconds: int = 1800


class BoosterParamsMyAI(
    BoosterParamsMyRABBITMQ
):  # 傳這個類別就可以少每次都親自指定使用 RabbitMQ 作為訊息佇列，和重試改為 4 次，和消費發佈日誌寫入自定義 .log 檔案。
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int = 5
    qps: int = 1
