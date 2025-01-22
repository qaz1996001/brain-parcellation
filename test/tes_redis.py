import redis


def batch_read_redis_keys_and_values(redis_host, redis_port, redis_db, batch_size=100):
    # 连接到 Redis
    r = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)

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

    for batch in batch_read_redis_keys_and_values(redis_host, redis_port, redis_db, batch_size):
        for key, value in batch:
            print(f"Key: {key}, {type(value)} Value: {pickle.loads(value)}")
            break
        break
