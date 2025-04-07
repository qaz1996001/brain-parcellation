from celery import Celery

#  172.27.64.1 172.24.160.1 172.17.0.1
app = Celery('tasks',            
             broker='pyamqp://guest:guest@172.17.0.1:5672/celery',
            #  include=['task_infarct'
                    #   ],
             backend='redis://172.17.0.1:10079/1'
             )

# app = Celery('tasks',            
#              broker='pyamqp://guest:guest@localhost:5672/celery',
#              include=['task.task_infarct'
#                       ],
#              backend='redis://localhost:10079/1'
#              )

# app.config_from_object('celery_config')
# app.conf.task_default_retry_delay = 5  # Seconds delay between retries
# app.conf.task_max_retries = 5  # Maximum number of retries

# app.conf.task_routes = {
#     'task_infarct.infarct': {'queue': 'synthseg_queue'},
# }

# app.conf.task_queues = {
#     'synthseg_queue': {'routing_key': 'synthseg_queue'},  # 專屬synthseg_task
#     'default': {'routing_key': 'default'},               # 默認隊列
# }
