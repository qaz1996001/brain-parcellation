#!/bin/bash

# 設定 PYTHONPATH 為當前目錄
# export PYTHONPATH=$(pwd)
export PYTHONPATH=$(pwd)/code_ai

# 啟動 Funboost 佇列監聽
funboost start -m funboost consume_all_queues  --project_root_path=$(pwd) --import_modules_str code_ai.task --booster_dirs_str 

# python ./boosters_manager.py

