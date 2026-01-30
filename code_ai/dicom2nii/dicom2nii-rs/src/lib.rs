//! # dicom2nii
//!
//! A high-performance DICOM to NIfTI converter with Python bindings.
//!
//! This library provides:
//! - DICOM file parsing and metadata extraction
//! - Automatic series type recognition (T1, T2, DWI, ADC, SWAN, etc.)
//! - DICOM to NIfTI conversion
//! - Post-processing utilities
//!
//! ## Features
//!
//! - **Fast**: Written in Rust with parallel processing support
//! - **Accurate**: Supports 15+ MR series recognition strategies
//! - **Flexible**: CLI tool, Python bindings, and C FFI
//!
//! ## Example
//!
//! ```rust,ignore
//! use dicom2nii::ConvertManager;
//!
//! let manager = ConvertManager::new();
//! let result = manager.process_directory("input", "output")?;
//! println!("Converted {} series", result.successful);
//! ```
//!
//! ## Modules
//!
//! - [`config`]: Configuration and enum definitions
//! - [`utils`]: Utility functions (filesystem, parallel processing, regex)
//! - [`cli`]: Command-line interface
//! - [`dicom`]: DICOM file parsing and metadata extraction
//! - [`nifti`]: NIfTI file handling via dcm2niix
//! - [`strategies`]: Processing strategies for different series types
//! - [`manager`]: High-level conversion management

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod utils;
pub mod cli;
pub mod dicom;
pub mod nifti;
pub mod strategies;
pub mod manager;

// Conditional compilation for Python bindings
#[cfg(feature = "python")]
pub mod python;

// Conditional compilation for C FFI
#[cfg(feature = "ffi")]
pub mod ffi;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get the library version
pub fn version() -> &'static str {
    VERSION
}

// Re-export commonly used types from config
pub use config::{
    ASLSeriesRename, BValue, BodyPart, CTSeriesRename, Contrast, DSCSeriesRename, DTISeries,
    EchoTime, EnumExt, ImageOrientation, MRAcquisitionType, MRSeriesRename, Modality, NullEnum,
    RepetitionTime, SeriesRename, SeriesType, T1SeriesRename, T2SeriesRename,
};

// Re-export configuration settings
pub use config::{
    get_config, init_config, init_default_config, Settings,
    GeneralSettings, FilesystemSettings, Dcm2niixSettings,
    ThresholdSettings, FileSizeThresholds, TrTeThreshold,
    DwiSettings, OrientationSettings, DicomTagSettings,
    ExclusionSettings, OutputSettings, PostprocessSettings,
};

// Re-export CLI types
pub use cli::{Cli, Commands};

// Re-export utility functions
pub use utils::fs::{collect_dicom_files, ensure_dir, is_dicom_file, DicomFileInfo};
pub use utils::parallel::{ParallelConfig, ParallelExecutor, ProcessingStats};
pub use utils::regex_patterns::PATTERNS;

// Re-export DICOM types
pub use dicom::{DicomMetadata, DicomReader};

// Re-export NIfTI types
pub use nifti::NiftiConverter;

// Re-export strategy types
pub use strategies::{ProcessingStrategy, ProcessingResult, StrategyRegistry};
pub use strategies::mr::{T1ProcessingStrategy, DwiProcessingStrategy, AdcProcessingStrategy};

// Re-export manager types
pub use manager::{ConvertManager, SeriesOrganizer};
