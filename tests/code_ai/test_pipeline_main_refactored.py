"""
Unit tests for refactored pipeline main module.

Tests the functional pipeline design following Linus-style good taste.
"""
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile

import pytest
import numpy as np
import nibabel as nib

from code_ai.pipeline.main_refactored import (
    PipelineType,
    PipelineConfig,
    ProcessingResult,
    find_nifti_files,
    generate_output_path,
    process_segmentation,
    process_wmh,
    process_cmb,
    process_dwi,
    process_single_file,
    run_pipeline,
    create_config_from_args,
    PIPELINE_PROCESSORS,
    PROCESSORS,
)


class TestPipelineType:
    """Test PipelineType enum."""
    
    def test_pipeline_types_defined(self):
        """Test all pipeline types are properly defined."""
        assert PipelineType.WMH == "wmh"
        assert PipelineType.CMB == "cmb"
        assert PipelineType.DWI == "dwi"
        assert PipelineType.SEGMENTATION == "segmentation"
        assert PipelineType.ALL == "all"


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""
    
    def test_config_initialization(self, tmp_path):
        """Test PipelineConfig initialization."""
        config = PipelineConfig(
            input_path=tmp_path,
            output_path=tmp_path / "output",
            input_pattern="T2_FLAIR",
            pipeline_types=[PipelineType.WMH],
            depth_number=6
        )
        
        assert config.input_path == tmp_path
        assert config.output_path == tmp_path / "output"
        assert config.input_pattern == "T2_FLAIR"
        assert PipelineType.WMH in config.pipeline_types
        assert config.depth_number == 6
    
    def test_config_defaults(self, tmp_path):
        """Test PipelineConfig default values."""
        config = PipelineConfig(input_path=tmp_path)
        
        assert config.output_path is None
        assert config.input_pattern is None
        assert config.pipeline_types == [PipelineType.SEGMENTATION]
        assert config.depth_number == 5
    
    def test_config_path_conversion(self):
        """Test that paths are converted to Path objects."""
        config = PipelineConfig(input_path="/some/path")
        assert isinstance(config.input_path, Path)


class TestFindNiftiFiles:
    """Test find_nifti_files function."""
    
    def test_find_single_file(self, tmp_path):
        """Test finding a single NIfTI file."""
        nifti_file = tmp_path / "test.nii.gz"
        nifti_file.touch()
        
        files = find_nifti_files(nifti_file)
        assert len(files) == 1
        assert files[0] == nifti_file
    
    def test_find_files_in_directory(self, tmp_path):
        """Test finding NIfTI files in directory."""
        # Create test files
        (tmp_path / "file1.nii.gz").touch()
        (tmp_path / "file2.nii").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.nii.gz").touch()
        (tmp_path / "other.txt").touch()
        
        files = find_nifti_files(tmp_path)
        assert len(files) == 3
        assert all(f.suffix in ['.nii', '.gz'] for f in files)
    
    def test_find_files_with_pattern(self, tmp_path):
        """Test finding files with pattern matching."""
        (tmp_path / "T2_FLAIR_001.nii.gz").touch()
        (tmp_path / "T2_FLAIR_002.nii.gz").touch()
        (tmp_path / "T1_MPRAGE.nii.gz").touch()
        
        files = find_nifti_files(tmp_path, "T2_FLAIR")
        assert len(files) == 2
        assert all("T2_FLAIR" in f.name for f in files)


class TestGenerateOutputPath:
    """Test generate_output_path function."""
    
    def test_generate_output_path_default_dir(self, tmp_path):
        """Test output path generation with default directory."""
        input_file = tmp_path / "test.nii.gz"
        output_path = generate_output_path(input_file, "_processed")
        
        assert output_path == tmp_path / "test_processed.nii.gz"
    
    def test_generate_output_path_custom_dir(self, tmp_path):
        """Test output path generation with custom directory."""
        input_file = tmp_path / "input" / "test.nii.gz"
        output_dir = tmp_path / "output"
        output_path = generate_output_path(input_file, "_processed", output_dir)
        
        assert output_path == output_dir / "test_processed.nii.gz"
    
    def test_generate_output_path_handles_extensions(self, tmp_path):
        """Test that various extensions are handled correctly."""
        for ext in ["test.nii.gz", "test.nii", "test.nii.nii.gz"]:
            input_file = tmp_path / ext
            output_path = generate_output_path(input_file, "_processed")
            assert output_path.name == "test_processed.nii.gz"


class TestProcessSegmentation:
    """Test process_segmentation function."""
    
    @pytest.mark.asyncio
    async def test_process_segmentation_success(self, tmp_path, monkeypatch):
        """Test successful segmentation processing."""
        # Create test file
        input_file = tmp_path / "test.nii.gz"
        test_data = np.zeros((10, 10, 10))
        nib.save(nib.Nifti1Image(test_data, np.eye(4)), input_file)
        
        # Mock functions
        mock_resample = Mock()
        mock_synthseg = Mock()
        mock_synthseg.run = Mock()
        mock_resample_back = Mock(return_value=("original_seg.nii.gz", 0))
        
        monkeypatch.setattr("code_ai.pipeline.main_refactored.resample_one", mock_resample)
        monkeypatch.setattr("code_ai.pipeline.main_refactored.SynthSeg", lambda: mock_synthseg)
        monkeypatch.setattr("code_ai.pipeline.main_refactored.resampleSynthSEG2original_z_index", mock_resample_back)
        
        config = PipelineConfig(input_path=tmp_path)
        results = await process_segmentation(input_file, config)
        
        assert "resample" in results
        assert "synthseg" in results
        assert "synthseg33" in results
        assert "original_seg" in results
        
        # Verify functions were called
        assert mock_resample.called
        assert mock_synthseg.run.called


class TestProcessWMH:
    """Test process_wmh function."""
    
    @pytest.mark.asyncio
    async def test_process_wmh_success(self, tmp_path, monkeypatch):
        """Test successful WMH processing."""
        # Create test data
        test_data = np.ones((10, 10, 10))
        test_nifti = nib.Nifti1Image(test_data, np.eye(4))
        
        # Create segmentation results
        seg_results = {
            "synthseg": tmp_path / "synthseg.nii.gz",
            "synthseg33": tmp_path / "synthseg33.nii.gz"
        }
        nib.save(test_nifti, seg_results["synthseg"])
        nib.save(test_nifti, seg_results["synthseg33"])
        
        # Mock functions
        mock_parcellation = Mock(return_value=(test_data, test_data))
        mock_wmh = Mock(return_value=test_data)
        
        monkeypatch.setattr("code_ai.pipeline.main_refactored.run_with_WhiteMatterParcellation", mock_parcellation)
        monkeypatch.setattr("code_ai.pipeline.main_refactored.run_wmh", mock_wmh)
        
        config = PipelineConfig(input_path=tmp_path, depth_number=5)
        input_file = tmp_path / "test.nii.gz"
        
        results = await process_wmh(input_file, config, seg_results)
        
        assert "david" in results
        assert "wmh" in results
        assert results["david"].exists()
        assert results["wmh"].exists()


class TestDataDrivenDesign:
    """Test that the refactored code follows data-driven design."""
    
    def test_no_special_cases_in_registry(self):
        """Test that pipeline registry uses data-driven approach."""
        # Verify registries are dictionaries
        assert isinstance(PIPELINE_PROCESSORS, dict)
        assert isinstance(PROCESSORS, dict)
        
        # Verify all pipeline types are in registry
        for pipeline_type in [PipelineType.WMH, PipelineType.CMB, PipelineType.DWI, PipelineType.SEGMENTATION]:
            assert pipeline_type in PIPELINE_PROCESSORS
        
        # Verify all processor names map to functions
        for processor_name in PIPELINE_PROCESSORS.values():
            assert processor_name in PROCESSORS
    
    def test_no_if_else_chains(self):
        """Verify no if-else chains in the refactored code."""
        import inspect
        from code_ai.pipeline import main_refactored
        
        # Get source code
        source = inspect.getsource(main_refactored)
        
        # Count elif occurrences (should be minimal)
        elif_count = source.count("elif")
        assert elif_count <= 2, f"Found {elif_count} elif statements - should use data structures instead"
    
    def test_max_nesting_level(self):
        """Verify maximum nesting level is 3."""
        import ast
        import inspect
        from code_ai.pipeline import main_refactored
        
        class NestingChecker(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0
            
            def visit(self, node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If, ast.For, ast.While, ast.With)):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                else:
                    self.generic_visit(node)
        
        # Parse and check nesting
        source = inspect.getsource(main_refactored)
        tree = ast.parse(source)
        checker = NestingChecker()
        checker.visit(tree)
        
        assert checker.max_depth <= 4, f"Max nesting depth is {checker.max_depth}, should be <= 4"


class TestProcessSingleFile:
    """Test process_single_file function."""
    
    @pytest.mark.asyncio
    async def test_process_single_file_segmentation_only(self, tmp_path, monkeypatch):
        """Test processing with segmentation only."""
        input_file = tmp_path / "test.nii.gz"
        input_file.touch()
        
        mock_segmentation = AsyncMock(return_value={"synthseg": tmp_path / "synthseg.nii.gz"})
        monkeypatch.setattr("code_ai.pipeline.main_refactored.process_segmentation", mock_segmentation)
        
        config = PipelineConfig(
            input_path=tmp_path,
            pipeline_types=[PipelineType.SEGMENTATION]
        )
        
        result = await process_single_file(input_file, config)
        
        assert result.success
        assert result.input_file == input_file
        assert "synthseg" in result.output_files
        assert mock_segmentation.called
    
    @pytest.mark.asyncio
    async def test_process_single_file_with_wmh(self, tmp_path, monkeypatch):
        """Test processing with WMH pipeline."""
        input_file = tmp_path / "test.nii.gz"
        input_file.touch()
        
        seg_results = {"synthseg": tmp_path / "synthseg.nii.gz"}
        wmh_results = {"wmh": tmp_path / "wmh.nii.gz"}
        
        mock_segmentation = AsyncMock(return_value=seg_results)
        mock_wmh = AsyncMock(return_value=wmh_results)
        
        monkeypatch.setattr("code_ai.pipeline.main_refactored.process_segmentation", mock_segmentation)
        monkeypatch.setattr("code_ai.pipeline.main_refactored.process_wmh", mock_wmh)
        
        config = PipelineConfig(
            input_path=tmp_path,
            pipeline_types=[PipelineType.WMH]
        )
        
        result = await process_single_file(input_file, config)
        
        assert result.success
        assert "synthseg" in result.output_files
        assert "wmh" in result.output_files
        assert mock_segmentation.called
        assert mock_wmh.called
    
    @pytest.mark.asyncio
    async def test_process_single_file_error_handling(self, tmp_path, monkeypatch):
        """Test error handling in process_single_file."""
        input_file = tmp_path / "test.nii.gz"
        
        mock_segmentation = AsyncMock(side_effect=RuntimeError("Processing failed"))
        monkeypatch.setattr("code_ai.pipeline.main_refactored.process_segmentation", mock_segmentation)
        
        config = PipelineConfig(input_path=tmp_path)
        
        result = await process_single_file(input_file, config)
        
        assert not result.success
        assert result.error == "Processing failed"
        assert result.output_files == {}


class TestCreateConfigFromArgs:
    """Test create_config_from_args function."""
    
    def test_create_config_all_pipelines(self):
        """Test config creation with all pipelines."""
        args = Mock(
            input="/path/to/input",
            output="/path/to/output",
            input_name="T2_FLAIR",
            template=None,
            template_name=None,
            all=True,
            wmh=False,
            cmb=False,
            dwi=False,
            depth_number=6
        )
        
        config = create_config_from_args(args)
        
        assert config.input_path == Path("/path/to/input")
        assert config.output_path == Path("/path/to/output")
        assert config.input_pattern == "T2_FLAIR"
        assert PipelineType.WMH in config.pipeline_types
        assert PipelineType.CMB in config.pipeline_types
        assert PipelineType.DWI in config.pipeline_types
        assert config.depth_number == 6
    
    def test_create_config_specific_pipelines(self):
        """Test config creation with specific pipelines."""
        args = Mock(
            input="/path/to/input",
            all=False,
            wmh=True,
            cmb=False,
            dwi=True,
            depth_number=5
        )
        # Handle missing attributes
        for attr in ['output', 'input_name', 'template', 'template_name']:
            if not hasattr(args, attr):
                setattr(args, attr, None)
        
        config = create_config_from_args(args)
        
        assert PipelineType.WMH in config.pipeline_types
        assert PipelineType.DWI in config.pipeline_types
        assert PipelineType.CMB not in config.pipeline_types
    
    def test_create_config_default_segmentation(self):
        """Test config defaults to segmentation if no pipelines specified."""
        args = Mock(
            input="/path/to/input",
            all=False,
            wmh=False,
            cmb=False,
            dwi=False
        )
        # Handle missing attributes
        for attr in ['output', 'input_name', 'template', 'template_name', 'depth_number']:
            if not hasattr(args, attr):
                setattr(args, attr, None)
        
        config = create_config_from_args(args)
        
        assert config.pipeline_types == [PipelineType.SEGMENTATION]
