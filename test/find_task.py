import sys
from celery import Celery
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)




app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             backend='redis://localhost:10079/1'
             )
app.config_from_object('code_ai.celery_config')


if __name__ == '__main__':
    # 946a02a5-79ea-4b0f-9133-793fc2db9beb
    from celery.result import AsyncResult
    # # 获取任务的 ID
    task_id = '2e41f07d-28d0-41b4-bb1c-cd0142f5c98c'
    task_id = 'ef6db8ae-e16a-4c94-b375-18b97a0ecc7b'
    # 创建 AsyncResult 实例
    result = AsyncResult(task_id)
    # 2e41f07d-28d0-41b4-bb1c-cd0142f5c98c
    print('result',result,type(result))
    print('result status', result.status)
    print('result state', result.state)
