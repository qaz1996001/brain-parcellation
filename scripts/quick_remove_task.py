#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速移除指定任務的腳本

直接執行此腳本並修改下方的 TASK_ID 即可
"""

import os

# 載入環境變數
import code_ai
code_ai.load_dotenv()

import redis

# ==================== 配置區 ====================
# 在這裡修改要刪除的任務 ID
TASK_ID = "77167315-6579-475f-8609-0f65b9f06a66"

# 是否為測試模式 (True: 只顯示不刪除, False: 實際刪除)
DRY_RUN = False
# ==============================================


def quick_remove_task(task_id: str, dry_run: bool = False) -> bool:
    """快速移除指定任務"""

    # Redis 連接設定
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_password = os.getenv("REDIS_PASSWORD", "")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB_FILTER_AND_RPC_RESULT", "3"))

    print("=" * 60)
    print("快速移除任務腳本")
    print("=" * 60)
    print(f"Redis: {redis_host}:{redis_port} (DB: {redis_db})")
    print(f"任務 ID: {task_id}")
    print(f"模式: {'測試模式 (不實際刪除)' if dry_run else '實際刪除'}")
    print("=" * 60)

    # 建構完整的 Redis key
    task_key = f"task_pipeline_inference_queue_result:{task_id}"

    try:
        # 連接 Redis
        redis_client = redis.Redis(
            host=redis_host,
            password=redis_password if redis_password else None,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )

        # 測試連接
        redis_client.ping()
        print("\n✓ Redis 連接成功")

        # 檢查任務是否存在
        exists = redis_client.exists(task_key)

        if not exists:
            print(f"\n✗ 任務不存在: {task_key}")
            print("\n正在搜尋所有相關任務...")

            # 列出所有任務
            all_keys = redis_client.keys("task_pipeline_inference_queue_result:*")

            if all_keys:
                print(f"\n找到 {len(all_keys)} 個任務:")
                for i, key in enumerate(all_keys, 1):
                    ttl = redis_client.ttl(key)
                    print(f"  {i}. {key}")
                    print(f"      TTL: {ttl}s")
                    if i >= 10:
                        print(f"  ... 還有 {len(all_keys) - 10} 個任務")
                        break
            else:
                print("  沒有找到任何任務")

            return False

        # 取得任務資訊
        ttl = redis_client.ttl(task_key)
        task_data = redis_client.get(task_key)

        print("\n找到任務:")
        print(f"  Key: {task_key}")
        print(f"  TTL: {ttl} 秒 ({ttl // 60} 分鐘)")

        if task_data:
            data_size = len(task_data)
            print(f"  資料大小: {data_size} 字元")

            # 顯示資料預覽
            if data_size > 0:
                preview_len = min(200, data_size)
                preview = task_data[:preview_len]
                if data_size > preview_len:
                    preview += "..."
                print(f"  資料預覽: {preview}")

        # 執行刪除
        if dry_run:
            print("\n" + "=" * 60)
            print("⚠️  測試模式: 不會實際刪除任務")
            print("=" * 60)
            print("如要實際刪除，請將 DRY_RUN 設為 False")
            return True
        else:
            print("\n正在刪除任務...")
            deleted = redis_client.delete(task_key)

            if deleted > 0:
                print("\n" + "=" * 60)
                print("✓ 任務刪除成功!")
                print("=" * 60)
                return True
            else:
                print("\n✗ 刪除失敗")
                return False

    except redis.ConnectionError as e:
        print(f"\n✗ Redis 連接失敗: {e}")
        print("\n請檢查:")
        print("  1. Redis 伺服器是否運行")
        print("  2. 環境變數 REDIS_HOST, REDIS_PORT, REDIS_PASSWORD 是否正確")
        return False
    except Exception as e:
        print(f"\n✗ 發生錯誤: {e}")
        return False
    finally:
        try:
            redis_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    success = quick_remove_task(TASK_ID, dry_run=DRY_RUN)

    if not success:
        print("\n任務移除失敗")
        exit(1)
    else:
        print("\n任務移除完成")
        exit(0)

