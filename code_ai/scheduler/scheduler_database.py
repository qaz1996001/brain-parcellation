from funboost import BoosterParams, ConcurrentModeEnum, BrokerEnum, Booster


@Booster(
    BoosterParams(
        queue_name="delete_old_date_queue",
        broker_kind=BrokerEnum.LOCAL_PYTHON_QUEUE,
        concurrent_mode=ConcurrentModeEnum.SOLO,
        qps=1,
    )
)
def delete_old_date():
    pass
