# CELERY_TASK_SERIALIZER = 'pickle'
# CELERY_ACCEPT_CONTENT = ['json', 'pickle']  # Allow both JSON and pickle
# CELERY_RESULT_SERIALIZER = 'pickle'
# CELERY_TASK_REJECT_ON_WORKER_LOST = 'true'
# CELERY_BROKER_HEARTBEAT = 0
# CELERY_BROKER_POOL_LIMIT = None
# CELERY_BROKER_TRANSPORT_OPTIONS = {'confirm_publish': True,
#                                    'max_retries': 10,  # 失败后最多重试 3 次
#                                    'interval_start': 30,  # 第一次重试间隔 0.2 秒
#                                    'interval_step': 30,  # 递增间隔
#                                    'interval_max': 30  # 最大重试间隔 5 秒
#                                    }
# CELERY_BROKER_CONNECTION_TIMEOUT = 30
# CELERY_BROKER_CONNECTION_RETRY = True
# CELERY_BROKER_CONNECTION_MAX_RETRIES = 100
#
# CELERY_WORKER_CANCEL_LONG_RUNNING_TASKS_ON_CONNECTION_LOSS = True
# CELERY_WORKER_MAX_TASKS_PER_CHILD = 500
# CELERY_WORKER_PREFETCH_MULTIPLIER = 2
# CELERY_TIME_LIMIT = 2 * 60 * 60
#
CELERY_TASK_SERIALIZER = 'pickle'
CELERY_ACCEPT_CONTENT = ['json', 'pickle']  # Allow both JSON and pickle
CELERY_RESULT_SERIALIZER = 'pickle'
CELERY_TASK_REJECT_ON_WORKER_LOST = 'true'
CELERY_BROKER_HEARTBEAT = 0
CELERY_BROKER_POOL_LIMIT = None
CELERY_BROKER_TRANSPORT_OPTIONS = {'confirm_publish': True,
                                   'max_retries': 10,  # 失败后最多重试 3 次
                                   'interval_start': 30,  # 第一次重试间隔 0.2 秒
                                   'interval_step': 30,  # 递增间隔
                                   'interval_max': 30  # 最大重试间隔 5 秒
                                   }
CELERY_BROKER_CONNECTION_TIMEOUT = 30
CELERY_BROKER_CONNECTION_RETRY = True
CELERY_BROKER_CONNECTION_MAX_RETRIES = 100

CELERY_WORKER_CANCEL_LONG_RUNNING_TASKS_ON_CONNECTION_LOSS = True
CELERY_WORKER_MAX_TASKS_PER_CHILD = 500
CELERY_WORKER_PREFETCH_MULTIPLIER = 2
CELERY_TIME_LIMIT = 2 * 60 * 60

