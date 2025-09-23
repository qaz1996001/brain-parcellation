"""
Unit tests for task base module.

Tests the functional task design with no abstract classes.
"""
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from code_ai.task.base import (
    TaskType,
    TaskConfig,
    TaskResult,
    execute_task,
    create_task,
    TASK_EXECUTORS,
)


class TestTaskType:
    """Test TaskType enumeration."""
    
    def test_task_types_defined(self):
        """Test all task types are defined."""
        expected_types = [
            "segmentation",
            "registration", 
            "parcellation",
            "wmh_detection",
            "cmb_detection",
            "dwi_analysis",
        ]
        
        for task_type in expected_types:
            assert hasattr(TaskType, task_type.upper())
            assert TaskType[task_type.upper()].value == task_type


class TestTaskConfig:
    """Test TaskConfig validation model."""
    
    def test_task_config_validation(self):
        """Test TaskConfig with valid data."""
        config = TaskConfig(
            name="test_task",
            task_type=TaskType.SEGMENTATION,
            input_files=["/path/to/input.nii.gz"],
            config={"batch_size": 1},
            depends_on=["previous_task"],
        )
        
        assert config.name == "test_task"
        assert config.task_type == TaskType.SEGMENTATION
        assert len(config.input_files) == 1
        assert config.config["batch_size"] == 1
        assert "previous_task" in config.depends_on
    
    def test_task_config_defaults(self):
        """Test TaskConfig with default values."""
        config = TaskConfig(
            name="test_task",
            task_type=TaskType.SEGMENTATION,
            input_files=["/path/to/input.nii.gz"],
        )
        
        assert config.config == {}
        assert config.depends_on == []


class TestTaskResult:
    """Test TaskResult model."""
    
    def test_task_result_success(self):
        """Test successful task result."""
        result = TaskResult(
            success=True,
            output_files={"output": "/path/to/output.nii.gz"},
            error=None,
            execution_time=1.5,
        )
        
        assert result.success is True
        assert result.output_files["output"] == "/path/to/output.nii.gz"
        assert result.error is None
        assert result.execution_time == 1.5
    
    def test_task_result_failure(self):
        """Test failed task result."""
        result = TaskResult(
            success=False,
            output_files={},
            error="Task execution failed",
            execution_time=0.5,
        )
        
        assert result.success is False
        assert result.output_files == {}
        assert result.error == "Task execution failed"
        assert result.execution_time == 0.5


class TestExecuteTask:
    """Test execute_task function."""
    
    @pytest.mark.asyncio
    async def test_execute_task_unknown_type(self):
        """Test execute_task with unknown task type."""
        # Create a fake task type not in registry
        config = TaskConfig(
            name="unknown_task",
            task_type="UNKNOWN_TYPE",  # type: ignore
            input_files=["/path/to/input.nii.gz"],
        )
        
        result = await execute_task(config)
        
        assert result.success is False
        assert "Unknown task type" in result.error
        assert result.execution_time == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_task_missing_input_files(self, tmp_path):
        """Test execute_task with missing input files."""
        config = TaskConfig(
            name="test_task",
            task_type=TaskType.SEGMENTATION,
            input_files=[str(tmp_path / "nonexistent.nii.gz")],
        )
        
        result = await execute_task(config)
        
        assert result.success is False
        assert "Input file not found" in result.error
        assert result.execution_time == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, tmp_path, monkeypatch):
        """Test successful task execution."""
        # Create a test file
        test_file = tmp_path / "input.nii.gz"
        test_file.touch()
        
        # Mock the task executor
        async def mock_executor(config):
            return TaskResult(
                success=True,
                output_files={"output": "/path/to/output.nii.gz"},
                error=None,
                execution_time=0.0,  # Will be overridden
            )
        
        monkeypatch.setitem(TASK_EXECUTORS, TaskType.SEGMENTATION, mock_executor)
        
        config = TaskConfig(
            name="test_task",
            task_type=TaskType.SEGMENTATION,
            input_files=[str(test_file)],
        )
        
        result = await execute_task(config)
        
        assert result.success is True
        assert "output" in result.output_files
        assert result.execution_time > 0  # Should be set by execute_task
    
    @pytest.mark.asyncio
    async def test_execute_task_exception_handling(self, tmp_path, monkeypatch):
        """Test execute_task exception handling."""
        # Create a test file
        test_file = tmp_path / "input.nii.gz"
        test_file.touch()
        
        # Mock executor that raises exception
        async def failing_executor(config):
            raise RuntimeError("Task processing failed")
        
        monkeypatch.setitem(TASK_EXECUTORS, TaskType.SEGMENTATION, failing_executor)
        
        config = TaskConfig(
            name="test_task",
            task_type=TaskType.SEGMENTATION,
            input_files=[str(test_file)],
        )
        
        result = await execute_task(config)
        
        assert result.success is False
        assert "Task processing failed" in result.error
        assert result.execution_time > 0


class TestCreateTask:
    """Test create_task factory function."""
    
    def test_create_task_factory(self):
        """Test create_task factory function."""
        params = {
            "input_files": ["/path/to/input.nii.gz"],
            "config": {"batch_size": 1},
        }
        
        task = create_task(TaskType.SEGMENTATION, params)
        
        assert isinstance(task, TaskConfig)
        assert task.task_type == TaskType.SEGMENTATION
        assert task.input_files == ["/path/to/input.nii.gz"]
        assert task.config["batch_size"] == 1
        assert task.name.startswith("segmentation_")  # Should include timestamp
    
    def test_create_task_unique_names(self):
        """Test that create_task generates unique names."""
        task1 = create_task(TaskType.SEGMENTATION, {"input_files": []})
        task2 = create_task(TaskType.SEGMENTATION, {"input_files": []})
        
        assert task1.name != task2.name


class TestTaskExecutors:
    """Test task executors registry (data-driven design)."""
    
    def test_task_executors_registry(self):
        """Test that task executors registry follows data-driven design."""
        # Verify registry is a dictionary
        assert isinstance(TASK_EXECUTORS, dict)
        
        # Verify all task types have executors
        for task_type in TaskType:
            assert task_type in TASK_EXECUTORS
            assert callable(TASK_EXECUTORS[task_type])
    
    def test_no_class_hierarchy(self):
        """Verify no abstract class hierarchy (functional design)."""
        # This test verifies the design principle is followed
        # Should not have Task abstract base class
        import inspect
        from code_ai.task import base
        
        # Check that we're not using abstract base classes
        module_classes = [
            obj for name, obj in inspect.getmembers(base)
            if inspect.isclass(obj) and obj.__module__ == base.__name__
        ]
        
        # Should only have Pydantic models and enums, no abstract base classes
        for cls in module_classes:
            assert not hasattr(cls, "__abstractmethods__"), f"{cls.__name__} is abstract - violates functional design"
