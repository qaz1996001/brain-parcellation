#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-12-17

ReRun 模組 API 路由路徑常數定義

此模組集中定義所有 ReRun 相關 API 端點的路由路徑，
便於路由註冊和客戶端呼叫的一致性管理。

路由路徑清單：
-----------
1. RERUN_PROT_STUDY_UID : 按 Study UID 重新執行
2. RERUN_PROT_STUDY_RENAME_ID : 按 Study Rename ID 重新執行  
3. RERUN_PROT_STUDY_FAIL : 失敗研究相關操作 (預留)

@author: sean Ho
"""

# API 路由前綴，所有 ReRun 相關端點都以 /rerun 開頭
prefix = "/rerun"

# 根據 Study UID 重新執行研究的端點
# 方法: POST
# 參數: 請求體包含 Study UID 列表
# 預期: {"ids": ["1.2.3.4.5", "1.2.3.4.6"]}
RERUN_PROT_STUDY_UID = f"{prefix}/study/by-uid"

# 根據 Study Rename ID 重新執行研究的端點
# 方法: POST
# 參數: JSON 陣列形式的 Study Rename ID 列表
# 預期: ["study_rename_001", "study_rename_002"]
RERUN_PROT_STUDY_RENAME_ID = f"{prefix}/study/by-rename_id"

# 失敗研究相關操作的端點 (預留用，未來可實現)
# 用途: 處理重新執行失敗的研究或失敗重試機制
RERUN_PROT_STUDY_FAIL = f"{prefix}/study/fail"
