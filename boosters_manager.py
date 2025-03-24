from pathlib import Path

from funboost import BoostersManager, BoosterDiscovery




if __name__ == '__main__':
    """ 有的人不想这样写代码,一个个的函数亲自 .consume() 来启动消费,可以使用BoostersManager相关的方法来启动某些队列或者启动所有队列.
    mod1.fun1.consume()
    mod2.fun2a.consume()
    mod2.fun2b.consume()
    """

    BoosterDiscovery(project_root_path=Path(__file__).parent.parent,
                     booster_dirs=[Path(__file__).parent]).auto_discovery()  # 这个放在main里面运行,防止无限懵逼死循环

    # 选择启动哪些队列名消费
    # BoostersManager.consume(queue_names.q_test_queue_manager1,queue_names.q_test_queue_manager2a)

    # 选择启动哪些队列名消费,每个队列设置不同的消费进程数量
    # BoostersManager.m_consume(**{queue_names.q_test_queue_manager1: 2, queue_names.q_test_queue_manager2a: 3})

    # 启动所有队列名消费,在同一个进程内消费
    BoostersManager.consume_all()

    # 启动所有队列名消费,每个队列启动单独的n个进程消费
    # BoostersManager.m_consume_all(2)