import sys
from celery import Celery
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)




app = Celery('tasks',
             broker='pyamqp://guest:guest@localhost:5672/celery',
             backend='redis://localhost:10079/1'
             )
app.config_from_object('code_ai.celery_config')

def get_result_by_id(task_id:str):
    result = AsyncResult(task_id)
    if result.children is not None and len(result.children) > 0 :
        get_result_by_id(result.children[0].task_id)
    print(result.task_id)
    result.retries
    print('result status', result.status)
    print('result state', result.state)
    print('result',result.result)
    print('result',result.name)
    print('*******************************')



if __name__ == '__main__':
    # 946a02a5-79ea-4b0f-9133-793fc2db9beb
    from celery.result import AsyncResult
    # # 获取任务的 ID
    task_id = ('ce29d361-cc0a-41f6-92f4-ee9366a3e070'
               )
    get_result_by_id(task_id)


    # 创建 AsyncResult 实例
    result = AsyncResult(task_id)
    # print('result id', result.task_id, )
    # print('result kwargs', result.kwargs, )
    # print('result args', result.args, )
    # print('result status', result.status)
    # print('result state', result.state)
    # print('result parent', result.parent)
    # print('result children', result.children)
    # for children in result.children:
    #     result = AsyncResult(children.task_id)
    #     print('result id', result.task_id,)
    #     print('result kwargs', result.kwargs, )
    #     print('result args', result.args, )
    #     print('result status', result.status)
    #     print('result state', result.state)
    #     print('result parent', result.parent)
    #     print('result children', result.children)



