from __future__ import annotations
import pathlib
from funboost import Booster, BrokerEnum, boost
from code_ai.task.schema import intput_params
from abc import ABC, abstractmethod
from typing import Any, Optional

# 假設 Funboost 的任務如下（其他任務請依照需求引入）

from code_ai.task.task_synthseg import resample_task, synthseg_task,process_synthseg_task,save_file_tasks,post_process_synthseg_task,resample_to_original_task
from typing import Optional, Any, Dict


# -----------------------------------------------------------------------------
# 定義基礎 Handler 介面與抽象類別
# -----------------------------------------------------------------------------


class Handler(ABC):
    """
    Handler 介面，宣告了 set_next 與 handle 方法，
    用以實作責任鏈模式。
    """

    @abstractmethod
    def set_next(self, handler: "Handler") -> "Handler":
        pass

    @abstractmethod
    def handle(self, request: Dict[str, Any]) -> Any:
        pass


class AbstractHandler(Handler):
    """
    提供預設的鏈接功能，若目前處理完後有下一個 handler，則轉交給下一個 handler 處理。
    """
    _next_handler: Optional[Handler] = None

    def set_next(self, handler: "Handler") -> "Handler":
        self._next_handler = handler
        return handler

    def handle(self, request: Dict[str, Any]) -> Any:
        if self._next_handler:
            return self._next_handler.handle(request)
        return None


# -----------------------------------------------------------------------------
# 實作具體任務的 Handler
# -----------------------------------------------------------------------------

class ResampleHandler(AbstractHandler):
    def handle(self, request: Dict[str, Any]) -> Any:
        print("Running resample_task...")

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        # 注意：此處 request 必須包含 'resample_task' 的參數
        result = resample_task.apply_async({'func_params':request})
        resample_file = result.get()  # 阻塞等待結果返回
        print("Resample task result:", resample_file)

        # # 更新下游任務所需參數
        # if 'synthseg_task' not in request:
        #     request['synthseg_task'] = {}
        # request['synthseg_task']['resample_file'] = resample_file
        # 若有下一個 handler，則交由下一個處理
        if self._next_handler:
            return self._next_handler.handle(request)
        return resample_file


class SynthSegHandler(AbstractHandler):
    def handle(self, request: Dict[str, Any]) -> Any:
        print("Running synthseg_task...")

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        result = synthseg_task.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回
        print("synthseg task result:", task_result)

        # # 更新下游任務所需參數
        # if 'synthseg_task' not in request:
        #     request['synthseg_task'] = {}
        # request['synthseg_task']['resample_file'] = resample_file
        # 若有下一個 handler，則交由下一個處理
        if self._next_handler:
            return self._next_handler.handle(request)
        return task_result



class ProcessSynthSegHandler(AbstractHandler):
    def handle(self, request: Dict[str, Any]) -> Any:
        print("Running process synthseg_task...")

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        result = process_synthseg_task.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回
        print("process synthseg task result:", task_result)

        # # 更新下游任務所需參數
        # if 'synthseg_task' not in request:
        #     request['synthseg_task'] = {}
        # request['synthseg_task']['resample_file'] = resample_file
        # 若有下一個 handler，則交由下一個處理
        if self._next_handler:
            return self._next_handler.handle(request)
        return task_result


class PostProcessSynthSegHandler(AbstractHandler):
    def handle(self, request: Dict[str, Any]) -> Any:
        print("Running post process synthseg_task...")

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        # 注意：此處 request 必須包含 'resample_task' 的參數
        result = post_process_synthseg_task.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回
        print("post process synthseg task result:", task_result)

        # # 更新下游任務所需參數
        # if 'synthseg_task' not in request:
        #     request['synthseg_task'] = {}
        # request['synthseg_task']['resample_file'] = resample_file
        # 若有下一個 handler，則交由下一個處理
        if self._next_handler:
            return self._next_handler.handle(request)
        return task_result


class ResampleToOriginalHandler(AbstractHandler):
    def handle(self, request: Dict[str, Any]) -> Any:
        print("Running resample_to_original_task...")

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        # 注意：此處 request 必須包含 'resample_task' 的參數
        result = resample_to_original_task.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回
        print("post resample_to_original_task result:", task_result)

        # # 更新下游任務所需參數
        # if 'synthseg_task' not in request:
        #     request['synthseg_task'] = {}
        # request['synthseg_task']['resample_file'] = resample_file
        # 若有下一個 handler，則交由下一個處理
        if self._next_handler:
            return self._next_handler.handle(request)
        return task_result



@boost('auto_inference_task',
       broker_kind=BrokerEnum.RABBITMQ_AMQPSTORM, qps=10,
       is_send_consumer_hearbeat_to_redis=True,
       is_push_to_dlx_queue_when_retry_max_times=True,
       is_using_rpc_mode=True)
def auto_inference_task(func_params: Dict[str, any]):
    task_params = intput_params.TaskInferenceParams.model_validate(func_params,
                                                                   strict=False)
