#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的任務移除工具 - 同時處理 RabbitMQ 佇列和 Redis 結果

功能:
1. 從 RabbitMQ 佇列中移除等待中的任務
2. 從 Redis 中移除 RPC 結果
3. 從 Redis 中移除推論快取

使用方式:
    python scripts/remove_task_complete.py --task-id "77167315-6579-475f-8609-0f65b9f06a66"
"""

import argparse
import json
import os
import sys
from typing import Optional

# 載入環境變數
import code_ai
code_ai.load_dotenv()

import pika
import redis


class TaskRemover:
    """完整的任務移除工具"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.removed_tasks = []

        # RabbitMQ 配置
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "127.0.0.1")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
        self.rabbitmq_pass = os.getenv("RABBITMQ_PASS", "guest")
        self.rabbitmq_vhost = os.getenv("RABBITMQ_VIRTUAL_HOST", "/")

        # Redis 配置
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.redis_db_rpc = int(os.getenv("REDIS_DB_FILTER_AND_RPC_RESULT", "3"))
        self.redis_db_cache = int(os.getenv("REDIS_DB", "0"))

        print("=" * 70)
        print("完整任務移除工具")
        print("=" * 70)
        print(f"RabbitMQ: {self.rabbitmq_host}:{self.rabbitmq_port} (vhost: {self.rabbitmq_vhost})")
        print(f"Redis RPC: {self.redis_host}:{self.redis_port} (DB: {self.redis_db_rpc})")
        print(f"Redis Cache: {self.redis_host}:{self.redis_port} (DB: {self.redis_db_cache})")
        print(f"模式: {'🔍 測試模式 (不實際刪除)' if self.dry_run else '🗑️  實際刪除'}")
        print("=" * 70)

    def connect_rabbitmq(self) -> Optional[pika.BlockingConnection]:
        """連接 RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_pass)
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                virtual_host=self.rabbitmq_vhost,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            connection = pika.BlockingConnection(parameters)
            print("\n✓ RabbitMQ 連接成功")
            return connection
        except Exception as e:
            print(f"\n✗ RabbitMQ 連接失敗: {e}")
            return None

    def connect_redis(self, db: int) -> Optional[redis.Redis]:
        """連接 Redis"""
        try:
            redis_client = redis.Redis(
                host=self.redis_host,
                password=self.redis_password if self.redis_password else None,
                port=self.redis_port,
                db=db,
                decode_responses=True,
            )
            redis_client.ping()
            return redis_client
        except Exception as e:
            print(f"\n✗ Redis (DB {db}) 連接失敗: {e}")
            return None

    def find_and_remove_from_rabbitmq(
        self, task_id: str, queue_name: str = "task_pipeline_inference_queue"
    ) -> bool:
        """從 RabbitMQ 佇列中尋找並移除任務"""
        print(f"\n{'[1]'} 檢查 RabbitMQ 佇列: {queue_name}")
        print("-" * 70)

        connection = self.connect_rabbitmq()
        if not connection:
            return False

        try:
            channel = connection.channel()

            # 宣告佇列（確保存在）
            queue = channel.queue_declare(queue=queue_name, passive=True)
            message_count = queue.method.message_count

            print(f"佇列中有 {message_count} 個任務")

            if message_count == 0:
                print("✓ 佇列為空，無需處理")
                return True

            # 暫存需要重新放回的任務
            messages_to_keep = []
            found_target = False
            checked_count = 0

            print("\n正在檢查佇列中的任務...")

            # 逐一檢查佇列中的任務
            while True:
                method_frame, header_frame, body = channel.basic_get(
                    queue=queue_name, auto_ack=False
                )

                if method_frame is None:
                    break  # 佇列已空

                checked_count += 1

                try:
                    # 解析任務資料
                    message_data = json.loads(body.decode("utf-8"))

                    # funboost 的訊息格式可能包含 function_result_status_id
                    # 檢查是否包含目標 task_id
                    message_str = json.dumps(message_data)
                    contains_task_id = task_id in message_str

                    if contains_task_id:
                        found_target = True
                        print(f"\n✓ 找到目標任務 (第 {checked_count} 個)")
                        print(f"  訊息預覽: {message_str[:200]}...")

                        if self.dry_run:
                            print("  [DRY RUN] 將移除此任務")
                            # 測試模式：放回佇列
                            messages_to_keep.append((method_frame, header_frame, body))
                        else:
                            # 實際模式：確認刪除（不 ack，不重新放回）
                            channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                            print("  ✓ 已從佇列移除")
                            self.removed_tasks.append(
                                {"type": "rabbitmq", "queue": queue_name}
                            )
                    else:
                        # 保留其他任務
                        messages_to_keep.append((method_frame, header_frame, body))

                except json.JSONDecodeError:
                    # 無法解析的訊息也保留
                    messages_to_keep.append((method_frame, header_frame, body))
                except Exception as e:
                    print(f"  警告: 處理訊息時發生錯誤: {e}")
                    messages_to_keep.append((method_frame, header_frame, body))

            # 將保留的任務放回佇列
            if messages_to_keep:
                print(f"\n正在將 {len(messages_to_keep)} 個任務放回佇列...")
                for method_frame, header_frame, body in messages_to_keep:
                    channel.basic_publish(
                        exchange="",
                        routing_key=queue_name,
                        body=body,
                        properties=header_frame,
                    )
                    channel.basic_ack(delivery_tag=method_frame.delivery_tag)

            if not found_target:
                print(f"\n✗ 在 RabbitMQ 佇列中未找到任務 ID: {task_id}")
                print(f"   已檢查 {checked_count} 個任務")

            return found_target

        except pika.exceptions.ChannelClosedByBroker as e:
            print(f"\n✗ 佇列不存在或無權限: {e}")
            return False
        except Exception as e:
            print(f"\n✗ RabbitMQ 操作失敗: {e}")
            return False
        finally:
            connection.close()

    def remove_from_redis_rpc(self, task_id: str) -> bool:
        """從 Redis 移除 RPC 結果"""
        print(f"\n{'[2]'} 檢查 Redis RPC 結果 (DB {self.redis_db_rpc})")
        print("-" * 70)

        redis_client = self.connect_redis(self.redis_db_rpc)
        if not redis_client:
            return False

        try:
            task_key = f"task_pipeline_inference_queue_result:{task_id}"

            exists = redis_client.exists(task_key)

            if not exists:
                print(f"✗ RPC 結果不存在: {task_key}")
                return False

            ttl = redis_client.ttl(task_key)
            task_data = redis_client.get(task_key)

            print("✓ 找到 RPC 結果:")
            print(f"  Key: {task_key}")
            print(f"  TTL: {ttl} 秒 ({ttl // 60} 分鐘)")

            if task_data:
                data_size = len(task_data)
                print(f"  資料大小: {data_size} 字元")
                preview = task_data[:150] + "..." if len(task_data) > 150 else task_data
                print(f"  資料預覽: {preview}")

            if self.dry_run:
                print("\n  [DRY RUN] 將刪除此 RPC 結果")
                return True
            else:
                deleted = redis_client.delete(task_key)
                if deleted > 0:
                    print("\n  ✓ 已刪除 RPC 結果")
                    self.removed_tasks.append({"type": "redis_rpc", "key": task_key})
                    return True
                else:
                    print("\n  ✗ 刪除失敗")
                    return False

        except Exception as e:
            print(f"\n✗ Redis 操作失敗: {e}")
            return False
        finally:
            redis_client.close()

    def remove_inference_cache(
        self, study_uid: Optional[str] = None, study_id: Optional[str] = None
    ) -> bool:
        """從 Redis 移除推論快取"""
        if not study_uid or not study_id:
            print(f"\n{'[3]'} 跳過推論快取移除 (未提供 study_uid/study_id)")
            return True

        print(f"\n{'[3]'} 檢查 Redis 推論快取 (DB {self.redis_db_cache})")
        print("-" * 70)

        redis_client = self.connect_redis(self.redis_db_cache)
        if not redis_client:
            return False

        try:
            inference_key = f"inference_task:{study_uid},{study_id}"

            exists = redis_client.exists(inference_key)

            if not exists:
                print(f"✗ 推論快取不存在: {inference_key}")
                return False

            cache_value = redis_client.get(inference_key)
            ttl = redis_client.ttl(inference_key)

            print("✓ 找到推論快取:")
            print(f"  Key: {inference_key}")
            print(f"  狀態: {cache_value}")
            print(f"  TTL: {ttl} 秒 ({ttl // 60} 分鐘)")

            if self.dry_run:
                print("\n  [DRY RUN] 將刪除此快取")
                return True
            else:
                deleted = redis_client.delete(inference_key)
                if deleted > 0:
                    print("\n  ✓ 已刪除推論快取")
                    self.removed_tasks.append(
                        {"type": "redis_cache", "key": inference_key}
                    )
                    return True
                else:
                    print("\n  ✗ 刪除失敗")
                    return False

        except Exception as e:
            print(f"\n✗ Redis 操作失敗: {e}")
            return False
        finally:
            redis_client.close()

    def list_rabbitmq_queue(
        self, queue_name: str = "task_pipeline_inference_queue", limit: int = 10
    ) -> None:
        """列出 RabbitMQ 佇列中的任務"""
        print(f"\n列出 RabbitMQ 佇列: {queue_name}")
        print("=" * 70)

        connection = self.connect_rabbitmq()
        if not connection:
            return

        try:
            channel = connection.channel()
            queue = channel.queue_declare(queue=queue_name, passive=True)
            message_count = queue.method.message_count

            print(f"佇列中有 {message_count} 個任務\n")

            if message_count == 0:
                print("佇列為空")
                return

            displayed = 0
            messages_to_requeue = []

            while displayed < limit:
                method_frame, header_frame, body = channel.basic_get(
                    queue=queue_name, auto_ack=False
                )

                if method_frame is None:
                    break

                displayed += 1

                try:
                    message_data = json.loads(body.decode("utf-8"))
                    message_str = json.dumps(message_data, indent=2, ensure_ascii=False)
                    preview = message_str[:300] + "..." if len(message_str) > 300 else message_str

                    print(f"{displayed}. 任務:")
                    print(f"   {preview}")
                    print()

                except Exception as e:
                    print(f"{displayed}. 無法解析的任務: {e}\n")

                # 保留訊息以便重新放回
                messages_to_requeue.append((method_frame, header_frame, body))

            # 重新放回所有訊息
            for method_frame, header_frame, body in messages_to_requeue:
                channel.basic_publish(
                    exchange="", routing_key=queue_name, body=body, properties=header_frame
                )
                channel.basic_ack(delivery_tag=method_frame.delivery_tag)

            if message_count > limit:
                print(f"... 還有 {message_count - limit} 個任務未顯示")

        except Exception as e:
            print(f"✗ 操作失敗: {e}")
        finally:
            connection.close()

    def print_summary(self):
        """列印執行摘要"""
        print("\n" + "=" * 70)
        print("執行摘要")
        print("=" * 70)

        if not self.removed_tasks:
            print("未移除任何任務")
        else:
            print(f"成功移除 {len(self.removed_tasks)} 個項目:")
            for item in self.removed_tasks:
                if item["type"] == "rabbitmq":
                    print(f"  ✓ RabbitMQ 佇列: {item['queue']}")
                elif item["type"] == "redis_rpc":
                    print(f"  ✓ Redis RPC: {item['key']}")
                elif item["type"] == "redis_cache":
                    print(f"  ✓ Redis 快取: {item['key']}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="完整的任務移除工具 - RabbitMQ + Redis"
    )

    parser.add_argument(
        "--task-id",
        type=str,
        help='任務 ID (UUID 格式)，例如: "77167315-6579-475f-8609-0f65b9f06a66"',
    )

    parser.add_argument(
        "--study-uid", type=str, help="Study UID (用於移除推論快取)"
    )

    parser.add_argument(
        "--study-id", type=str, help="Study ID (用於移除推論快取)"
    )

    parser.add_argument(
        "--queue-name",
        type=str,
        default="task_pipeline_inference_queue",
        help="RabbitMQ 佇列名稱",
    )

    parser.add_argument(
        "--list", action="store_true", help="列出 RabbitMQ 佇列中的任務"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="測試模式，不實際刪除"
    )

    args = parser.parse_args()

    remover = TaskRemover(dry_run=args.dry_run)

    # 列出任務
    if args.list:
        remover.list_rabbitmq_queue(queue_name=args.queue_name)
        return

    # 移除任務
    if args.task_id:
        # 1. 從 RabbitMQ 移除
        remover.find_and_remove_from_rabbitmq(args.task_id, args.queue_name)

        # 2. 從 Redis RPC 移除
        remover.remove_from_redis_rpc(args.task_id)

        # 3. 從 Redis 快取移除（如果提供）
        if args.study_uid and args.study_id:
            remover.remove_inference_cache(args.study_uid, args.study_id)

        # 列印摘要
        remover.print_summary()
    else:
        parser.print_help()
        print("\n錯誤: 請提供 --task-id 或使用 --list")
        sys.exit(1)


if __name__ == "__main__":
    main()

