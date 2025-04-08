from __future__ import annotations
import pathlib
from code_ai.utils_inference import replace_suffix
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Sequence

from code_ai.task.schema.intput_params import SaveFileTaskParams
# 假設 Funboost 的任務如下（其他任務請依照需求引入）

from code_ai.task.task_synthseg import resample_task, synthseg_task,process_synthseg_task,save_file_tasks,post_process_synthseg_task,resample_to_original_task
from typing import Optional, Any, Dict

from code_ai.utils_inference import InferenceEnum


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

    @classmethod
    def generate_save_file_params(cls, subject_data: Dict,
                                      targets: Sequence[Union[str,InferenceEnum]] = (InferenceEnum.CMB,
                                                                                     InferenceEnum.DWI,
                                                                                     InferenceEnum.WMH_PVS,
                                                                                     InferenceEnum.Area,)
                                  ) -> List[SaveFileTaskParams]:
        results = []

        for key in targets:
            if key not in subject_data or subject_data[key] is None:
                continue
            input_path_list = subject_data[key].get("input_path_list", [])
            output_path = pathlib.Path(subject_data[key].get("output_path", []))
            if not input_path_list:
                continue
            for input_path_str in input_path_list:
                input_path = pathlib.Path(input_path_str)
                input_name = input_path.name
                resample = output_path / replace_suffix(input_name,'_resample.nii.gz')
                synthseg = output_path / replace_suffix(input_name,'_resample_synthseg.nii.gz')
                synthseg33 = output_path / replace_suffix(input_name,'_resample_synthseg33.nii.gz')
                david = output_path / replace_suffix(input_name,'_resample_david.nii.gz')
                wm = output_path / replace_suffix(input_name,'_resample_wm.nii.gz')
                if key == InferenceEnum.Area:
                    save_file = synthseg
                else:
                    save_file = output_path  / replace_suffix(input_name,f'_resample_{key}.nii.gz')
                results.append(
                    SaveFileTaskParams(
                        file=input_path,
                        resample_file=resample,
                        synthseg_file=synthseg,
                        synthseg33_file=synthseg33,
                        david_file=david,
                        wm_file=wm,
                        save_mode=key,
                        save_file_path=save_file
                    )
                )
        return results


# -----------------------------------------------------------------------------
# 實作具體任務的 Handler
# -----------------------------------------------------------------------------

class ResampleHandler(AbstractHandler):
    def handle(self, request: Dict[str, Any]) -> Any:

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        # 注意：此處 request 必須包含 'resample_task' 的參數
        result = resample_task.apply_async({'func_params':request})
        resample_file = result.get()  # 阻塞等待結果返回

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

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        result = synthseg_task.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回

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

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        result = process_synthseg_task.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回

        # # 更新下游任務所需參數
        # if 'synthseg_task' not in request:
        #     request['synthseg_task'] = {}
        # request['synthseg_task']['resample_file'] = resample_file
        # 若有下一個 handler，則交由下一個處理
        if self._next_handler:
            return self._next_handler.handle(request)
        return task_result


class SaveFileTaskHandler(AbstractHandler):
    def handle(self, request: Dict[str, Any]) -> Any:

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        # 注意：此處 request 必須包含 'resample_task' 的參數
        result = save_file_tasks.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回

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

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        # 注意：此處 request 必須包含 'resample_task' 的參數
        result = post_process_synthseg_task.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回

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

        # 使用 Funboost 的 apply_async 發送任務並等待結果
        # 注意：此處 request 必須包含 'resample_task' 的參數
        result = resample_to_original_task.apply_async({'func_params':request})
        task_result = result.get()  # 阻塞等待結果返回

        # # 更新下游任務所需參數
        # if 'synthseg_task' not in request:
        #     request['synthseg_task'] = {}
        # request['synthseg_task']['resample_file'] = resample_file
        # 若有下一個 handler，則交由下一個處理
        if self._next_handler:
            return self._next_handler.handle(request)
        return task_result