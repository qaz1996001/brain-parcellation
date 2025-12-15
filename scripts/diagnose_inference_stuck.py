#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
診斷工具：檢查 Study 300.50 卡住的原因

使用方式：
    python scripts/diagnose_inference_stuck.py [study_id]
"""

import os
import sys
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import redis
import pika

load_dotenv()


class InferenceStuckDiagnostic:
    """診斷推理任務卡住的工具"""

    def __init__(self):
        self.redis_host = os.getenv("REDIS_HOST")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_db = int(os.getenv("REDIS_DB_FILTER_AND_RPC_RESULT", 0))

        self.rabbitmq_user = os.getenv("RABBITMQ_USER")
        self.rabbitmq_pass = os.getenv("RABBITMQ_PASS")
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", 5672))
        self.rabbitmq_vhost = os.getenv("RABBITMQ_VIRTUAL_HOST")

        self.redis_client = None
        self.rabbitmq_connection = None

    def connect_redis(self):
        """連接 Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                password=self.redis_password,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
            )
            self.redis_client.ping()
            print(f"✅ Redis 連接成功: {self.redis_host}:{self.redis_port}/{self.redis_db}")
            return True
        except Exception as e:
            print(f"❌ Redis 連接失敗: {e}")
            return False

    def connect_rabbitmq(self):
        """連接 RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_pass)
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                virtual_host=self.rabbitmq_vhost,
                credentials=credentials,
            )
            self.rabbitmq_connection = pika.BlockingConnection(parameters)
            print(
                f"✅ RabbitMQ 連接成功: {self.rabbitmq_host}:{self.rabbitmq_port}/{self.rabbitmq_vhost}"
            )
            return True
        except Exception as e:
            print(f"❌ RabbitMQ 連接失敗: {e}")
            return False

    def check_redis_cache(self, study_uid: str, study_id: str):
        """檢查 Redis 快取狀態"""
        print("\n" + "=" * 60)
        print("🔍 檢查 Redis 快取狀態")
        print("=" * 60)

        if not self.redis_client:
            print("❌ Redis 未連接")
            return

        inference_task_key = f"inference_task:{study_uid},{study_id}"

        try:
            # 檢查 key 是否存在
            exists = self.redis_client.exists(inference_task_key)
            if exists:
                value = self.redis_client.get(inference_task_key)
                ttl = self.redis_client.ttl(inference_task_key)

                print(f"🔴 Redis 快取 KEY 存在:")
                print(f"   Key: {inference_task_key}")
                print(f"   Value: {value}")
                print(f"   TTL: {ttl} 秒 ({ttl/3600:.1f} 小時)")

                if ttl > 18000:  # 超過 5 小時
                    print(
                        f"\n⚠️  警告：TTL 過長 ({ttl/3600:.1f} 小時)，"
                        "這可能導致任務被永久跳過！"
                    )
                    print("   建議：手動刪除此 key 以重新執行推理")
                    print(f"   指令：redis-cli DEL {inference_task_key}")
                else:
                    print(f"\n✅ TTL 正常 ({ttl/3600:.1f} 小時)，將在到期後自動清除")
            else:
                print(f"✅ Redis 快取 KEY 不存在: {inference_task_key}")
                print("   任務可以正常執行")

        except Exception as e:
            print(f"❌ 檢查 Redis 快取失敗: {e}")

    def check_rabbitmq_queue(self):
        """檢查 RabbitMQ 佇列狀態"""
        print("\n" + "=" * 60)
        print("🔍 檢查 RabbitMQ 佇列狀態")
        print("=" * 60)

        if not self.rabbitmq_connection:
            print("❌ RabbitMQ 未連接")
            return

        try:
            channel = self.rabbitmq_connection.channel()
            queue_name = "task_pipeline_inference_queue"

            # 聲明佇列（如果不存在）
            method = channel.queue_declare(queue=queue_name, passive=True)
            message_count = method.method.message_count
            consumer_count = method.method.consumer_count

            print(f"📊 佇列: {queue_name}")
            print(f"   待處理訊息數: {message_count}")
            print(f"   消費者數量: {consumer_count}")

            if message_count > 0:
                print(f"\n⚠️  佇列中有 {message_count} 個待處理任務")
                if consumer_count == 0:
                    print("   🔴 警告：沒有消費者在線！")
                    print("   原因：funboost consumer 可能已停止")
                    print("   解決：sudo systemctl restart brain-parcellation.service")
            else:
                print("\n✅ 佇列為空")

            if consumer_count == 0:
                print("\n🔴 警告：沒有消費者連接到佇列！")
                print("   檢查：funboost consumer 是否正在運行")
                print("   指令：ps aux | grep funboost")

        except Exception as e:
            print(f"❌ 檢查 RabbitMQ 佇列失敗: {e}")

    def check_funboost_consumer(self):
        """檢查 funboost consumer 進程"""
        print("\n" + "=" * 60)
        print("🔍 檢查 Funboost Consumer 進程")
        print("=" * 60)

        import subprocess

        try:
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.split("\n")
            funboost_processes = [
                line for line in lines if "funboost" in line.lower() and "grep" not in line
            ]

            if funboost_processes:
                print(f"✅ 找到 {len(funboost_processes)} 個 funboost 進程:")
                for proc in funboost_processes:
                    print(f"   {proc[:120]}")
            else:
                print("🔴 未找到 funboost 進程！")
                print("   Consumer 可能已停止")
                print("   解決：sudo systemctl restart brain-parcellation.service")

        except Exception as e:
            print(f"❌ 檢查進程失敗: {e}")

    def check_database_status(self, study_uid: str, study_id: str):
        """檢查資料庫中的狀態記錄"""
        print("\n" + "=" * 60)
        print("🔍 檢查資料庫狀態記錄")
        print("=" * 60)

        try:
            import asyncio
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker

            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                print("❌ 未設定 DATABASE_URL")
                return

            async def query_status():
                engine = create_async_engine(database_url, echo=False)
                async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

                async with async_session() as session:
                    # 查詢最近的狀態
                    sql = text(
                        """
                        SELECT ope_no, tool_id, create_time, update_time, params_data
                        FROM dcop_event_bt
                        WHERE study_uid = :study_uid OR study_id = :study_id
                        ORDER BY create_time DESC
                        LIMIT 10
                    """
                    )
                    result = await session.execute(
                        sql, {"study_uid": study_uid, "study_id": study_id}
                    )
                    rows = result.fetchall()

                    if rows:
                        print(f"📋 最近 10 筆狀態記錄:")
                        for row in rows:
                            print(
                                f"   [{row[0]}] {row[1]} - {row[2]} (更新: {row[3]})"
                            )

                        # 檢查是否卡在 300.050
                        latest_ope_no = rows[0][0]
                        if latest_ope_no == "300.050":
                            print(
                                f"\n🔴 當前狀態: {latest_ope_no} (STUDY_INFERENCE_READY)"
                            )
                            print("   任務已進入推理佇列，但可能未被處理")
                        elif latest_ope_no == "300.100":
                            print(
                                f"\n🟡 當前狀態: {latest_ope_no} (STUDY_INFERENCE_QUEUED)"
                            )
                            print("   任務在佇列中等待")
                        elif latest_ope_no == "300.150":
                            print(
                                f"\n🟡 當前狀態: {latest_ope_no} (STUDY_INFERENCE_RUNNING)"
                            )
                            print("   任務正在執行中")
                        elif latest_ope_no == "300.300":
                            print(
                                f"\n✅ 當前狀態: {latest_ope_no} (STUDY_INFERENCE_COMPLETE)"
                            )
                            print("   任務已完成")
                    else:
                        print(f"⚠️  未找到 study_uid={study_uid} 或 study_id={study_id} 的記錄")

                await engine.dispose()

            asyncio.run(query_status())

        except Exception as e:
            print(f"❌ 檢查資料庫狀態失敗: {e}")

    def cleanup_redis_key(self, study_uid: str, study_id: str, force: bool = False):
        """清理 Redis 快取 key"""
        if not self.redis_client:
            print("❌ Redis 未連接")
            return

        inference_task_key = f"inference_task:{study_uid},{study_id}"

        try:
            exists = self.redis_client.exists(inference_task_key)
            if not exists:
                print(f"ℹ️  Key 不存在: {inference_task_key}")
                return

            if not force:
                confirm = input(f"\n⚠️  確定要刪除 Redis key: {inference_task_key}? (y/N): ")
                if confirm.lower() != "y":
                    print("❌ 已取消")
                    return

            self.redis_client.delete(inference_task_key)
            print(f"✅ 已刪除 Redis key: {inference_task_key}")
            print("   現在可以重新執行推理任務")

        except Exception as e:
            print(f"❌ 刪除 Redis key 失敗: {e}")

    def run_full_diagnostic(self, study_uid: str = None, study_id: str = None):
        """執行完整診斷"""
        print("\n" + "=" * 60)
        print("🏥 Study 300.50 卡住診斷工具")
        print("=" * 60)

        if study_uid:
            print(f"Study UID: {study_uid}")
        if study_id:
            print(f"Study ID: {study_id}")

        # 1. 連接中介軟體
        redis_ok = self.connect_redis()
        rabbitmq_ok = self.connect_rabbitmq()

        if not redis_ok and not rabbitmq_ok:
            print("\n❌ 無法連接任何中介軟體，請檢查設定")
            return

        # 2. 檢查 Redis 快取
        if redis_ok and study_uid and study_id:
            self.check_redis_cache(study_uid, study_id)

        # 3. 檢查 RabbitMQ 佇列
        if rabbitmq_ok:
            self.check_rabbitmq_queue()

        # 4. 檢查 funboost consumer
        self.check_funboost_consumer()

        # 5. 檢查資料庫狀態
        if study_uid or study_id:
            self.check_database_status(study_uid, study_id)

        # 6. 建議操作
        print("\n" + "=" * 60)
        print("💡 建議操作")
        print("=" * 60)

        if redis_ok and study_uid and study_id:
            inference_task_key = f"inference_task:{study_uid},{study_id}"
            if self.redis_client.exists(inference_task_key):
                print("1️⃣ 刪除 Redis 快取 key:")
                print(f"   python scripts/diagnose_inference_stuck.py --cleanup {study_uid} {study_id}")

        print("2️⃣ 重啟 funboost consumer:")
        print("   sudo systemctl restart brain-parcellation.service")

        print("3️⃣ 查看日誌:")
        print("   sudo journalctl -u brain-parcellation.service -f")

        print("\n" + "=" * 60)

    def close(self):
        """關閉連接"""
        if self.rabbitmq_connection:
            self.rabbitmq_connection.close()


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="診斷 Study 300.50 卡住的原因")
    parser.add_argument("--study-uid", help="Study UID")
    parser.add_argument("--study-id", help="Study ID")
    parser.add_argument(
        "--cleanup",
        nargs=2,
        metavar=("STUDY_UID", "STUDY_ID"),
        help="清理 Redis 快取 key",
    )

    args = parser.parse_args()

    diagnostic = InferenceStuckDiagnostic()

    try:
        if args.cleanup:
            study_uid, study_id = args.cleanup
            diagnostic.connect_redis()
            diagnostic.cleanup_redis_key(study_uid, study_id)
        else:
            diagnostic.run_full_diagnostic(args.study_uid, args.study_id)
    finally:
        diagnostic.close()


if __name__ == "__main__":
    main()

