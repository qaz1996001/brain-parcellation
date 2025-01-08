import argparse
import pathlib
import re
import sys
sys.path.append('/mnt/d/00_Chen/Task04_git')
print('path', sys.path)
import code_ai.task.task_01 as task_01
from celery import group


if __name__ == '__main__':
    workflow_group = group(
        task_01.simple_task.s(file) for file in ['a01_b.txt', 'a01_c.txt', 'a01_d1.txt', 'a01_e.txt']
    )
    result = workflow_group.apply_async()
    print('result',result)  # 期望返回 'Test Task Completed'



