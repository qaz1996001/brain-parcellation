#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import json
import os
import pathlib
import sqlite3

def main():
    from code_ai.task.task_pipeline import task_pipeline_inference
    # delete dicom seg
    database_path = pathlib.Path(__file__).parent.parent.joinpath('database.sqlite3').absolute()

    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT queue_name, params
                       FROM funboost_consume_results
                       WHERE queue_name = 'task_pipeline_inference_queue'
                         AND datetime(utime) >= datetime('now', '-1 day')
                       ''')

        result_list = cursor.fetchall()  # Use fetchall() since there might be multiple tables
    result_set = set(result_list)
    for result in result_set:
        print(result_set)
        params = json.loads(result[1])['func_params']
        # task_pipeline_result = task_pipeline_inference.push(params)
        # print(task_pipeline_result)

if __name__ == '__main__':
    main()