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
//! use dicom2nii::cli::{Cli, run};
//! use clap::Parser;
//!
//! let cli = Cli::parse();
//! run(cli)?;
//! ```
//!
//! ## Modules
//!
//! - [`config`]: Configuration and enum definitions
//! - [`utils`]: Utility functions (filesystem, parallel processing, regex)
//! - [`cli`]: Command-line interface
//! - [`dicom`]: DICOM file parsing (TODO)
//! - [`nifti`]: NIfTI file handling (TODO)
//! - [`strategies`]: Processing strategies for different series types (TODO)
//! - [`postprocess`]: Post-processing utilities (TODO)
//! - [`manager`]: High-level conversion management (TODO)

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod utils;
pub mod cli;

// TODO: Implement these modules
// pub mod dicom;
// pub mod nifti;
// pub mod strategies;
// pub mod postprocess;
// pub mod manager;

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

// Re-export CLI types
pub use cli::{Cli, Commands};

// Re-export utility functions
pub use utils::fs::{collect_dicom_files, ensure_dir, is_dicom_file, DicomFileInfo};
pub use utils::parallel::{ParallelConfig, ParallelExecutor, ProcessingStats};
pub use utils::regex_patterns::PATTERNS;
