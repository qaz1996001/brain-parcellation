CELERY_TASK_SERIALIZER = 'pickle'
CELERY_ACCEPT_CONTENT = ['json', 'pickle']  # Allow both JSON and pickle
CELERY_RESULT_SERIALIZER = 'pickle'
CELERY_task_reject_on_worker_lost = 'true'
