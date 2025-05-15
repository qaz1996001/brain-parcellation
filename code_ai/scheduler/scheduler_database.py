from funboost import BoosterParams, ConcurrentModeEnum, BrokerEnum, Booster
from code_ai import LOCAL_DB

@Booster(BoosterParams(queue_name='delete_old_date_queue',
                       broker_kind=BrokerEnum.LOCAL_PYTHON_QUEUE,
                       concurrent_mode=ConcurrentModeEnum.SOLO,
                       qps=1,))
def delete_old_date():
    pass