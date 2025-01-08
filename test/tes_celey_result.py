import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
from code_ai.task.task_synthseg import build_celery_workflow


if __name__ == '__main__':
    from celery.result import AsyncResult

    # 获取任务的 ID
    task_id = ' b3acfe27-0a23-4f34-ad62-925e5277f80e'

    # 创建 AsyncResult 实例
    result = AsyncResult(task_id)
    print('result', result)
    print('parent', result.parent)
    print('app', result.app)
    print('children', result.children)
    result.successful()
    print(result.args)
    print(result.result)




