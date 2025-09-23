"""
Unit tests for pipeline base module.

Tests the functional pipeline design with data-driven approach.
"""
import pytest
from typing import Dict, Any

from code_ai.pipeline.base import (
    PipelineInput,
    PipelineResult,
    execute_pipeline,
    PIPELINE_REGISTRY,
)


class TestPipelineInput:
    """Test PipelineInput validation model."""
    
    def test_pipeline_input_validation(self):
        """Test PipelineInput with valid data."""
        input_data = PipelineInput(
            files={"T2_FLAIR": "/path/to/flair.nii.gz"},
            config={"threshold": 0.5},
            priority=2,
        )
        
        assert input_data.files["T2_FLAIR"] == "/path/to/flair.nii.gz"
        assert input_data.config["threshold"] == 0.5
        assert input_data.priority == 2
    
    def test_pipeline_input_defaults(self):
        """Test PipelineInput with default values."""
        input_data = PipelineInput(
            files={"T1": "/path/to/t1.nii.gz"}
        )
        
        assert input_data.config == {}
        assert input_data.priority == 1
    
    def test_pipeline_input_priority_validation(self):
        """Test PipelineInput priority validation."""
        with pytest.raises(ValueError):
            PipelineInput(
                files={"T1": "/path/to/t1.nii.gz"},
                priority=5  # Should be between 1-4
            )


class TestPipelineResult:
    """Test PipelineResult model."""
    
    def test_pipeline_result_success(self):
        """Test successful pipeline result."""
        result = PipelineResult(
            success=True,
            results={"task1": {"status": "completed"}},
            error=None,
        )
        
        assert result.success is True
        assert "task1" in result.results
        assert result.error is None
    
    def test_pipeline_result_failure(self):
        """Test failed pipeline result."""
        result = PipelineResult(
            success=False,
            results={},
            error="Pipeline execution failed",
        )
        
        assert result.success is False
        assert result.results == {}
        assert result.error == "Pipeline execution failed"


class TestExecutePipeline:
    """Test execute_pipeline function."""
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_unknown_type(self, pipeline_config):
        """Test execute_pipeline with unknown pipeline type."""
        input_data = PipelineInput(
            files={"T1": "/path/to/t1.nii.gz"}
        )
        
        result = await execute_pipeline(
            "UNKNOWN_PIPELINE",
            input_data,
            pipeline_config
        )
        
        assert result.success is False
        assert result.error == "Unknown pipeline type: UNKNOWN_PIPELINE"
        assert result.results == {}
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_exception_handling(self, pipeline_config, monkeypatch):
        """Test execute_pipeline exception handling."""
        # Mock a pipeline function that raises an exception
        async def failing_pipeline(input_data, config):
            raise RuntimeError("Pipeline processing failed")
        
        # Temporarily add to registry
        monkeypatch.setitem(PIPELINE_REGISTRY, "TEST_FAIL", failing_pipeline)
        
        input_data = PipelineInput(
            files={"T1": "/path/to/t1.nii.gz"}
        )
        
        result = await execute_pipeline(
            "TEST_FAIL",
            input_data,
            pipeline_config
        )
        
        assert result.success is False
        assert "Pipeline processing failed" in result.error
        assert result.results == {}


class TestPipelineRegistry:
    """Test pipeline registry pattern (data-driven design)."""
    
    def test_pipeline_registry_no_special_cases(self):
        """Test that pipeline registry uses data-driven approach."""
        # Verify registry is a dictionary (data-driven)
        assert isinstance(PIPELINE_REGISTRY, dict)
        
        # Verify all expected pipelines are registered
        expected_pipelines = ["WMH_PVS", "CMB", "DWI", "ANEURYSM"]
        for pipeline in expected_pipelines:
            assert pipeline in PIPELINE_REGISTRY
            
        # Verify all values are callable
        for pipeline_func in PIPELINE_REGISTRY.values():
            assert callable(pipeline_func)
    
    def test_no_if_else_chains(self):
        """Verify no if-else chains in pipeline selection (Linus style)."""
        # This test verifies the design principle is followed
        # The execute_pipeline function should use dictionary lookup
        # not if-else chains
        import inspect
        from code_ai.pipeline import base
        
        source = inspect.getsource(base.execute_pipeline)
        
        # Check for if-else chains (anti-pattern)
        assert source.count("elif") == 0, "Found elif chains - violates Linus style"
        
        # Verify dictionary lookup pattern
        assert "PIPELINE_REGISTRY[" in source, "Should use dictionary lookup"
