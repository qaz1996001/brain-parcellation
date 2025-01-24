from celery import Task


class MyTask(Task):

    def run(self, *args, **kwargs):
        print("Hello World")
