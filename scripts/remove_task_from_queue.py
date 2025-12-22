#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
移除指定任務的腳本

使用方式:
    python scripts/remove_task_from_queue.py --task-id "task_pipeline_inference_queue_result:77167315-6579-475f-8609-0f65b9f06a66"

或者:
    python scripts/remove_task_from_queue.py --task-id "77167315-6579-475f-8609-0f65b9f06a66"
"""

import argparse
import os
import sys

# 載入環境變數
import code_ai
code_ai.load_dotenv()

import redis


def remove_task_from_redis(task_id: str, dry_run: bool = False) -> bool:
    """從 Redis 移除指定的任務

    Args:
        task_id: 任務 ID，格式為完整 key 或只有 UUID 部分
        dry_run: 若為 True，只顯示操作但不實際執行

    Returns:
        是否成功移除任務
    """
    # 連接 Redis
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_password = os.getenv("REDIS_PASSWORD", "")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db_filter = int(os.getenv("REDIS_DB_FILTER_AND_RPC_RESULT", "3"))

    print(f"連接 Redis: {redis_host}:{redis_port}, DB: {redis_db_filter}")

    try:
        redis_client = redis.Redis(
            host=redis_host,
            password=redis_password,
            port=redis_port,
            db=redis_db_filter,
            decode_responses=True,
        )

        # 測試連接
        redis_client.ping()
        print("✓ Redis 連接成功")

    except Exception as e:
        print(f"✗ Redis 連接失敗: {e}")
        return False

    # 處理 task_id 格式
    if not task_id.startswith("task_pipeline_inference_queue_result:"):
        task_key = f"task_pipeline_inference_queue_result:{task_id}"
    else:
        task_key = task_id

    print(f"\n目標任務 Key: {task_key}")

    # 檢查任務是否存在
    try:
        exists = redis_client.exists(task_key)

        if not exists:
            print(f"✗ 任務不存在: {task_key}")
            print("\n嘗試搜尋相關的 key...")

            # 搜尋所有相關的 key
            pattern = "task_pipeline_inference_queue_result:*"
            all_keys = redis_client.keys(pattern)

            if all_keys:
                print(f"\n找到 {len(all_keys)} 個相關的任務:")
                for i, key in enumerate(all_keys[:10], 1):  # 只顯示前 10 個
                    ttl = redis_client.ttl(key)
                    print(f"  {i}. {key} (TTL: {ttl}s)")

                if len(all_keys) > 10:
                    print(f"  ... 還有 {len(all_keys) - 10} 個任務")
            else:
                print("  沒有找到任何相關任務")

            return False

        # 取得任務資訊
        task_data = redis_client.get(task_key)
        ttl = redis_client.ttl(task_key)

        print("\n任務資訊:")
        print("  存在: 是")
        print(f"  TTL: {ttl} 秒")
        if task_data:
            preview = task_data[:200] + "..." if len(task_data) > 200 else task_data
            print(f"  資料預覽: {preview}")

        # 執行刪除
        if dry_run:
            print(f"\n[DRY RUN] 將刪除任務: {task_key}")
            return True
        else:
            deleted_count = redis_client.delete(task_key)
            if deleted_count > 0:
                print(f"\n✓ 成功刪除任務: {task_key}")
                return True
            else:
                print(f"\n✗ 刪除失敗: {task_key}")
                return False

    except Exception as e:
        print(f"\n✗ 操作失敗: {e}")
        return False
    finally:
        redis_client.close()


def remove_inference_cache(study_uid: str, study_id: str, dry_run: bool = False) -> bool:
    """移除推論任務的快取

    Args:
        study_uid: Study UID
        study_id: Study ID
        dry_run: 若為 True，只顯示操作但不實際執行

    Returns:
        是否成功移除快取
    """
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_password = os.getenv("REDIS_PASSWORD", "")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db_filter = int(os.getenv("REDIS_DB_FILTER_AND_RPC_RESULT", "0"))

    print("\n移除推論快取...")
    print(f"連接 Redis: {redis_host}:{redis_port}, DB: {redis_db_filter}")

    try:
        redis_client = redis.Redis(
            host=redis_host,
            password=redis_password,
            port=redis_port,
            db=redis_db_filter,
            decode_responses=True,
        )

        redis_client.ping()
        print("✓ Redis 連接成功")

        # 推論快取 key 格式
        inference_key = f"inference_task:{study_uid},{study_id}"

        exists = redis_client.exists(inference_key)

        if not exists:
            print(f"✗ 推論快取不存在: {inference_key}")
            return False

        cache_value = redis_client.get(inference_key)
        ttl = redis_client.ttl(inference_key)

        print("\n快取資訊:")
        print(f"  Key: {inference_key}")
        print(f"  狀態: {cache_value}")
        print(f"  TTL: {ttl} 秒")

        if dry_run:
            print(f"\n[DRY RUN] 將刪除快取: {inference_key}")
            return True
        else:
            deleted_count = redis_client.delete(inference_key)
            if deleted_count > 0:
                print(f"\n✓ 成功刪除快取: {inference_key}")
                return True
            else:
                print(f"\n✗ 刪除失敗: {inference_key}")
                return False

    except Exception as e:
        print(f"\n✗ 操作失敗: {e}")
        return False
    finally:
        redis_client.close()


def list_all_tasks(pattern: str = "task_pipeline_inference_queue_result:*") -> None:
    """列出所有任務

    Args:
        pattern: Redis key 模式
    """
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_password = os.getenv("REDIS_PASSWORD", "")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db_filter = int(os.getenv("REDIS_DB_FILTER_AND_RPC_RESULT", "3"))

    print(f"連接 Redis: {redis_host}:{redis_port}, DB: {redis_db_filter}")

    try:
        redis_client = redis.Redis(
            host=redis_host,
            password=redis_password,
            port=redis_port,
            db=redis_db_filter,
            decode_responses=True,
        )

        redis_client.ping()
        print("✓ Redis 連接成功\n")

        # 搜尋所有相關的 key
        all_keys = redis_client.keys(pattern)

        if not all_keys:
            print(f"沒有找到符合模式的任務: {pattern}")
            return

        print(f"找到 {len(all_keys)} 個任務:\n")

        for i, key in enumerate(all_keys, 1):
            ttl = redis_client.ttl(key)
            task_data = redis_client.get(key)

            print(f"{i}. {key}")
            print(f"   TTL: {ttl} 秒")

            if task_data:
                preview = task_data[:100] + "..." if len(task_data) > 100 else task_data
                print(f"   資料: {preview}")

            print()

    except Exception as e:
        print(f"✗ 操作失敗: {e}")
    finally:
        redis_client.close()


def main():
    parser = argparse.ArgumentParser(
        description="移除 task_pipeline_inference 佇列中的指定任務"
    )

    parser.add_argument(
        "--task-id",
        type=str,
        help='任務 ID (完整格式或僅 UUID)，例如: "77167315-6579-475f-8609-0f65b9f06a66"',
    )

    parser.add_argument(
        "--study-uid",
        type=str,
        help="Study UID (用於移除推論快取)",
    )

    parser.add_argument(
        "--study-id",
        type=str,
        help="Study ID (用於移除推論快取)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有任務",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅顯示將執行的操作，不實際執行",
    )

    args = parser.parse_args()

    # 列出所有任務
    if args.list:
        list_all_tasks()
        return

    # 移除任務結果
    if args.task_id:
        success = remove_task_from_redis(args.task_id, dry_run=args.dry_run)
        if not success:
            sys.exit(1)

    # 移除推論快取
    if args.study_uid and args.study_id:
        success = remove_inference_cache(
            args.study_uid, args.study_id, dry_run=args.dry_run
        )
        if not success:
            sys.exit(1)

    # 如果沒有提供任何參數
    if not args.task_id and not (args.study_uid and args.study_id):
        parser.print_help()
        print("\n錯誤: 請提供 --task-id 或 (--study-uid 和 --study-id) 或 --list")
        sys.exit(1)


if __name__ == "__main__":
    main()

