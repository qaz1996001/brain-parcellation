

from celery import shared_task
from kombu import Connection, Exchange, Queue, Producer
from kombu.exceptions import NotBoundError

# RabbitMQ 鎖配置
RABBITMQ_URL = "amqp://guest:guest@localhost:5672//"

LOCK_NAME = "synthseg_task_lock"


def acquire_lock():
    """
    Attempt to acquire a RabbitMQ lock. Returns True if successful, otherwise False.
    """
    with Connection(RABBITMQ_URL) as conn:
        with conn.channel() as channel:
            # Define exchange and queue
            lock_exchange = Exchange(LOCK_NAME, type="direct", durable=True)
            lock_queue = Queue(LOCK_NAME, exchange=lock_exchange, routing_key=LOCK_NAME, durable=True)

            # Declare exchange and queue
            lock_exchange(channel).declare()
            lock_queue(channel).declare()

            bound_queue = lock_queue(channel)

            # Check if lock already exists
            if bound_queue.get() is None:  # If the queue is empty
                # Use a producer to put a message into the queue
                producer = Producer(channel)
                producer.publish(
                    "lock",
                    exchange=lock_exchange,
                    routing_key=LOCK_NAME,
                    declare=[bound_queue],
                    serializer="json"
                )
                return True
            else:
                return False  # Lock already exists



def release_lock():
    """
    Release RabbitMQ lock.
    """
    with Connection(RABBITMQ_URL) as conn:
        with conn.channel() as channel:
            # Define exchange and queue
            lock_exchange = Exchange(LOCK_NAME, type="direct", durable=True)
            lock_queue = Queue(LOCK_NAME, exchange=lock_exchange, routing_key=LOCK_NAME, durable=True)

            # Declare exchange and queue
            lock_exchange(channel).declare()
            lock_queue(channel).declare()

            bound_queue = lock_queue(channel)
            bound_queue.purge(channel)


if __name__ == '__main__':
    release_lock()
    print(10000000000000000)
    # if acquire_lock():
    #     print('acquire_lock')
    #     release_lock()
    #     print('release_lock')
    # else:
    #     print('eeeeeeeeeeee')


