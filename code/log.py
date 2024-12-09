import os
import time
import logging
import sys
from functools import wraps
import traceback

LOGGING_FTAGE = True
log_dir1 = os.path.join(os.path.dirname(__file__), "logs")
today = time.strftime('%Y%m%d', time.localtime(time.time()))
full_path = os.path.join(log_dir1, today)
if not os.path.exists(full_path):
    os.makedirs(full_path)
log_path = os.path.join(full_path, "log_file.log")


def get_logger():
    # 获取logger实例，如果参数为空则返回root logger
    logger = logging.getLogger("facebook")
    if not logger.handlers:
        # 指定logger输出格式
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        # 文件日志
        file_handler = logging.FileHandler(log_path, encoding="utf8")
        file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
        # 为logger添加的日志处理器
        logger.addHandler(file_handler)
        if LOGGING_FTAGE:
            # 控制台日志
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter  # 也可以直接给formatter赋值
            logger.addHandler(console_handler)

            # 指定日志的最低输出级别，默认为WARN级别
        logger.setLevel(logging.INFO)
        # 添加下面一句，在记录日志之后移除句柄
    return logger


def logging_time(func):
    @wraps(func)
    def log(*args, **kwargs):
        try:
            start = time.time()
            # get_logger().info(f"{type(args[0]).__name__} {func.__name__} is start ")
            get_logger().info(f"{func.__name__} is start ")
            result = func(*args, **kwargs)
            end = time.time()
            # get_logger().info(f"{type(args[0]).__name__} {func.__name__} is end {end - start:.6f} sec")
            get_logger().info(f"{func.__name__} is end {end - start:.6f} sec")
            return result
        except Exception as e:
            get_logger().error(f"{args[0]} {func.__name__} is error,here are details:{traceback.format_exc()}")

    return log
