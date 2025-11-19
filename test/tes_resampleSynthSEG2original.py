
if __name__ == '__main__':
    app = Celery('tasks',
                 broker='pyamqp://guest:guest@localhost:5672/celery',
                 backend='redis://localhost:10079/1'
                 )
    # args = {}
    # app.config_from_object('code_ai.celery_config')
    # result = app.send_task('code_ai.task.task_synthseg.resample_to_original_task', args=(args, file_list),
    #                        queue='synthseg_queue',
    #                        routing_key='celery')
    # raw_file             =
    # resample_image_file
    # resample_seg_file
    # resampleSynthSEG2original(raw_file, resample_image_file, resample_seg_file)