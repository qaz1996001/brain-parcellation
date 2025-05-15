#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import os
import sqlite3


def main():
    # delete dicom seg
    with sqlite3.connect(r'/mnt/d/00_Chen/Task04_git/database.sqlite3') as conn:
        cursor = conn.cursor()
        # Replace SHOW TABLES with the correct SQLite syntax
        cursor.execute('''SELECT name FROM sqlite_master WHERE type='table';''')

        result = cursor.fetchall()  # Use fetchall() since there might be multiple
        cursor.execute('''SELECT * FROM raw_dicom_to_inference;''')

        result = cursor.fetchall()  # Use fetchall() since there might be multiple tables
        print('result:', result)
        cursor.execute('''DELETE FROM raw_dicom_to_inference;''')

# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    main()