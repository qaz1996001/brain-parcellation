# rdxai_init_template.py - rdxai/__init__.py 範本

"""
RDX-AI: Medical Imaging AI Core Library
Provides base classes, utilities, and inference frameworks for medical imaging tasks.
"""

__version__ = "0.1.0"

# Base classes
from rdxai.base.inference_base import InferenceBase, InferencePipeline
from rdxai.base.pipeline_base import PipelineBase
from rdxai.base.model_config_base import ModelConfigBase

# Utilities
from rdxai.utils.dicom_utils import load_dicom, save_dicom_seg
from rdxai.utils.nifti_utils import load_nifti, save_nifti
from rdxai.utils.image_processing import normalize_image, resample_image

# Models
from rdxai.models.response_models import PredictionBaseResponse
from rdxai.models.config_models import ThreeStageModelConfig, NnUNetModelConfig

# Inference handlers
from rdxai.inference.nnunet_handler import NnUNetInferenceHandler
from rdxai.inference.model_loader import ModelLoader

# Exceptions
from rdxai.exceptions.custom_exceptions import (
    ModelLoadError,
    InferenceError,
    ValidationError,
)

__all__ = [
    "InferenceBase",
    "InferencePipeline",
    "PipelineBase",
    "ModelConfigBase",
    "load_dicom",
    "save_dicom_seg",
    "load_nifti",
    "save_nifti",
    "normalize_image",
    "resample_image",
    "PredictionBaseResponse",
    "ThreeStageModelConfig",
    "NnUNetModelConfig",
    "NnUNetInferenceHandler",
    "ModelLoader",
    "ModelLoadError",
    "InferenceError",
    "ValidationError",
]
