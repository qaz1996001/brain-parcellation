#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU device management and memory checking

Author: Architecture Redesign Team
Created: 2025-10-20
"""
import logging
import os
from typing import Optional

import pynvml


class GPUManager:
    """Manages GPU device allocation and memory availability checking

    Provides:
        - GPU memory usage monitoring via pynvml
        - TensorFlow GPU configuration
        - Automatic device selection and memory growth

    Example:
        >>> gpu_manager = GPUManager(gpu_id=0, memory_threshold=0.6)
        >>> if gpu_manager.check_gpu_available():
        ...     gpu_manager.configure_tensorflow()
        ...     # Proceed with inference
        ... else:
        ...     # Handle insufficient GPU memory
    """

    def __init__(self, gpu_id: int = 0, memory_threshold: float = 0.6):
        """Initialize GPUManager

        Args:
            gpu_id: GPU device ID to use (default: 0)
            memory_threshold: Maximum GPU memory usage allowed (0.0-1.0)
                            Pipeline will not run if usage exceeds this threshold

        Raises:
            ValueError: If gpu_id < 0 or memory_threshold not in (0.0, 1.0]
        """
        if gpu_id < 0:
            raise ValueError(f"gpu_id must be >= 0, got {gpu_id}")
        if not 0.0 < memory_threshold <= 1.0:
            raise ValueError(
                f"memory_threshold must be between 0.0 and 1.0, got {memory_threshold}"
            )

        self.gpu_id = gpu_id
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger(__name__)

        # Flag to track if pynvml is initialized
        self._nvml_initialized = False

    def _init_nvml(self) -> bool:
        """Initialize NVML (NVIDIA Management Library)

        Returns:
            True if initialization successful, False otherwise
        """
        if self._nvml_initialized:
            return True

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            return False

    def check_gpu_available(self) -> bool:
        """Check if GPU has sufficient memory available

        Returns:
            True if GPU memory usage < threshold, False otherwise

        Note:
            If NVML initialization fails or GPU is not available,
            returns False and logs error
        """
        if not self._init_nvml():
            return False

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Calculate GPU usage ratio
            gpu_usage = memory_info.used / memory_info.total

            available = gpu_usage < self.memory_threshold

            if available:
                self.logger.info(
                    f"GPU {self.gpu_id} available - "
                    f"Memory usage: {gpu_usage:.1%} / {self.memory_threshold:.1%} threshold"
                )
            else:
                self.logger.warning(
                    f"GPU {self.gpu_id} memory insufficient - "
                    f"Usage {gpu_usage:.1%} exceeds threshold {self.memory_threshold:.1%}"
                )

            return available

        except pynvml.NVMLError_GpuIsLost:
            self.logger.error(f"GPU {self.gpu_id} is lost or inaccessible")
            return False
        except pynvml.NVMLError_InvalidArgument:
            self.logger.error(f"Invalid GPU ID: {self.gpu_id}")
            return False
        except Exception as e:
            self.logger.error(f"GPU availability check failed: {e}")
            return False

    def get_gpu_info(self) -> Optional[dict]:
        """Get detailed GPU information

        Returns:
            Dictionary with GPU information or None if unavailable:
                - name: GPU device name
                - memory_total: Total memory in bytes
                - memory_used: Used memory in bytes
                - memory_free: Free memory in bytes
                - utilization: GPU utilization percentage
        """
        if not self._init_nvml():
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

            # Get GPU name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Get utilization (if supported)
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
            except:
                gpu_util = None

            return {
                'gpu_id': self.gpu_id,
                'name': name,
                'memory_total': memory_info.total,
                'memory_used': memory_info.used,
                'memory_free': memory_info.free,
                'memory_usage_ratio': memory_info.used / memory_info.total,
                'utilization_percent': gpu_util,
            }

        except Exception as e:
            self.logger.error(f"Failed to get GPU info: {e}")
            return None

    def configure_tensorflow(self) -> bool:
        """Configure TensorFlow for GPU usage

        Configures:
            - Visible devices (only specified GPU)
            - Memory growth (dynamic allocation)
            - Suppresses TensorFlow warnings

        Returns:
            True if configuration successful, False otherwise

        Note:
            Should be called after check_gpu_available() returns True
        """
        try:
            # Suppress TensorFlow logging
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            import tensorflow as tf

            # Suppress TensorFlow warnings
            tf_logger = tf.get_logger()
            tf_logger.setLevel(logging.ERROR)

            # Get available GPUs
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

            if not gpus:
                self.logger.warning("No GPU devices found by TensorFlow")
                return False

            if self.gpu_id >= len(gpus):
                self.logger.error(
                    f"GPU {self.gpu_id} not available. "
                    f"Found {len(gpus)} GPU(s): {gpus}"
                )
                return False

            # Select target GPU
            target_gpu = gpus[self.gpu_id]

            # Set visible devices (only target GPU)
            tf.config.experimental.set_visible_devices(
                devices=target_gpu,
                device_type='GPU'
            )

            # Enable memory growth (dynamic allocation)
            tf.config.experimental.set_memory_growth(target_gpu, True)

            self.logger.info(
                f"TensorFlow configured for GPU {self.gpu_id}: {target_gpu.name}"
            )

            return True

        except ImportError:
            self.logger.error("TensorFlow not installed")
            return False
        except Exception as e:
            self.logger.error(f"TensorFlow GPU configuration failed: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup NVML resources

        Should be called when done with GPU operations
        """
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                self.logger.debug("NVML shutdown successfully")
            except Exception as e:
                self.logger.warning(f"NVML shutdown failed: {e}")

    def __enter__(self):
        """Context manager entry"""
        self._init_nvml()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup NVML"""
        self.cleanup()
        return False

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"GPUManager(gpu_id={self.gpu_id}, "
            f"memory_threshold={self.memory_threshold})"
        )
