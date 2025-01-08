import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
from code_ai.task.task_synthseg import build_celery_workflow

workflow_group = group(
    simple_task.s(file) for file in ['01_input.txt', '02_input.txt', '03_input.txt']
)

result = workflow_group.apply_async()
print(result)

result = workflow_group.apply_async()
print(f"Task result status: {result.status}")


if __name__ == '__main__':
    from celery.result import AsyncResult

    # 获取任务的 ID
    task_id = '8bbe32d3-2846-4c81-9898-854bc8f192ea'

    # 创建 AsyncResult 实例
    result = AsyncResult(task_id)
    print(result.args)
    print(result.result)
    # 查询任务的状态
    # status = result.status  # 状态可以是 "PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY" 等

    # # 获取任务的结果
    # if result.ready():
    #     # 如果任务已完成
    #     result_value = result.result  # 获取任务结果
    # else:
    #     result_value = None  # 任务未完成
    #
    # print(f"Task Status: {status}, Result: {result_value}")




