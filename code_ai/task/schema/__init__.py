"""Task schema 模組 - 定義所有任務參數模型"""

from .intput_params import (
    ResampleTaskParams,
    SynthsegTaskParams,
    ProcessSynthsegTaskParams,
    ResampleToOriginalTaskParams,
    SaveFileTaskParams,
    PostProcessSynthsegTaskParams,
    Dicom2NiiParams,
    Dicom2NiiFileParams,
    Dicom2NiiSeriesParams,
    ProcessInstancesParams,
    CallDcm2niixParams,
    TaskInferenceParams,
    StudyTaskInferenceParams,
    InferenceTaskParams,
    SubprocessTaskParams,
)

__all__ = [
    "ResampleTaskParams",
    "SynthsegTaskParams",
    "ProcessSynthsegTaskParams",
    "ResampleToOriginalTaskParams",
    "SaveFileTaskParams",
    "PostProcessSynthsegTaskParams",
    "Dicom2NiiParams",
    "Dicom2NiiFileParams",
    "Dicom2NiiSeriesParams",
    "ProcessInstancesParams",
    "CallDcm2niixParams",
    "TaskInferenceParams",
    "StudyTaskInferenceParams",
    "InferenceTaskParams",
    "SubprocessTaskParams",
]

