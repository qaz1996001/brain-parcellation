from celery import Celery, group, chain, chord
from code_ai.celery_app import app


@app.task
def some_task():
    print("Task started")
    # 执行任务的代码
    print("Task completed")
    return "Success"  # 返回一个简单的字符串或字典

@app.task
def simple_task(file):
    # 处理文件
    return "Processed"

