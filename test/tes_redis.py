import redis
from celery.result import AsyncResult

def batch_read_redis_keys_and_values(redis_host, redis_port, redis_db, batch_size=100):
    # 连接到 Redis

    # 使用 scan_iter 扫描键，避免使用 keys() 导致阻塞
    cursor = r.scan_iter()
    batch = []

    for key in cursor:
        # 将键和值加入批次
        value = r.get(key)
        batch.append((key, value if value else None))
        # batch.append((key.decode('utf-8'), value if value else None))

        # 批次达到指定大小时返回
        if len(batch) >= batch_size:
            yield batch
            batch = []
    # 返回剩余的批次
    if batch:
        yield batch


# 使用示例
if __name__ == "__main__":
    redis_host = "localhost"
    redis_port = 10079
    redis_db = 1
    batch_size = 50  # 每次读取 50 条
    import pickle

    r = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)

    for batch in batch_read_redis_keys_and_values(redis_host, redis_port, redis_db, batch_size):
        for key, value in batch:
            key_type = r.type(key).decode()
            print(f"Key: {key}, Type: {key_type}")

            if key_type != "string":
                print(f"Skipping key {key} because it's not a string.")
                continue
            redis_result = pickle.loads(value)
            print(f"Decoded Redis Result: {redis_result}")
            break
        break

