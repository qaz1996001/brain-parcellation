
if __name__ == '__main__':
    from celery.result import AsyncResult
    from celery import Celery

    app = Celery('tasks',
                 broker='pyamqp://guest:guest@localhost:5672/celery',
                 backend='redis://localhost:10079/1'
                 )
    # app.conf.task_serializer = 'pickle'
    # app.conf.result_serializer = 'pickle'

    i = app.control.inspect()

    print(i)
    print('active',i.active())


    #
    # # 获取任务的 ID
    #
    # task_id = 'f33c0868-2bb3-4102-9d31-51c44004af4a'
    #
    # # 创建 AsyncResult 实例
    # result = AsyncResult(task_id)

    # print('result', result)
    # print('args',result.args)
    # print('parent', result.parent)
    # print('children', result.children)
    # result.successful()




