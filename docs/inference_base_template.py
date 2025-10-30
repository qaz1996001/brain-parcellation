# inference_base_template.py - rdxai 核心基類

"""
Base classes for inference pipelines.
All specialized inference classes should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ModelConfigBase(BaseModel):
    """Base configuration class for all models."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str
    model_path: Union[str, Path]
    device: str = "cuda"
    precision: str = "fp32"


class InferenceBase(ABC):
    """Abstract base class for all inference pipelines."""
    
    def __init__(self, config: ModelConfigBase, verbose: bool = False):
        """Initialize inference pipeline."""
        self.config = config
        self.verbose = verbose
        self.model = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the inference pipeline."""
        if not self._initialized:
            logger.info(f"Initializing {self.__class__.__name__}...")
            self.load_model()
            self._initialized = True
    
    @abstractmethod
    def load_model(self):
        """Load model weights and architecture."""
        pass
    
    @abstractmethod
    def preprocess(self, image: Any) -> Any:
        """Preprocess image for inference."""
        pass
    
    @abstractmethod
    def forward(self, preprocessed_image: Any) -> Any:
        """Run inference (forward pass)."""
        pass
    
    @abstractmethod
    def postprocess(self, model_output: Any) -> Any:
        """Process model output to final predictions."""
        pass
    
    def infer(self, image: Any, **kwargs) -> Any:
        """Standard inference workflow."""
        if not self._initialized:
            self.initialize()
        
        preprocessed = self.preprocess(image, **kwargs)
        model_output = self.forward(preprocessed)
        predictions = self.postprocess(model_output)
        return predictions
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            logger.debug("Cleaning up model resources...")


class InferencePipeline(ABC):
    """Multi-stage inference pipeline."""
    
    def __init__(self, verbose: bool = False):
        """Initialize pipeline."""
        self.verbose = verbose
        self.stages: Dict[str, InferenceBase] = {}
        self._initialized = False
    
    def add_stage(self, name: str, inference: InferenceBase):
        """Add inference stage to pipeline."""
        self.stages[name] = inference
        logger.debug(f"Added stage '{name}' to pipeline")
    
    def initialize(self):
        """Initialize all pipeline stages."""
        if not self._initialized:
            logger.info("Initializing pipeline stages...")
            for name, stage in self.stages.items():
                logger.info(f"Initializing stage: {name}")
                stage.initialize()
            self._initialized = True
    
    @abstractmethod
    def run(self, image: Any, **kwargs) -> Any:
        """Run the complete pipeline."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up all stages."""
        logger.debug("Cleaning up all pipeline stages...")
        for stage in self.stages.values():
            stage.cleanup()
