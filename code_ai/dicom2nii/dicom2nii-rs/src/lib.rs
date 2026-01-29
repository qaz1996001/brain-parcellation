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
//! use dicom2nii::manager::ConvertManager;
//!
//! let manager = ConvertManager::new("./input", "./output");
//! let results = manager.run(4)?;
//! println!("Converted {} studies", results.len());
//! ```
//!
//! ## Modules
//!
//! - [`config`]: Configuration and enum definitions
//! - [`dicom`]: DICOM file parsing (TODO)
//! - [`nifti`]: NIfTI file handling (TODO)
//! - [`strategies`]: Processing strategies for different series types (TODO)
//! - [`postprocess`]: Post-processing utilities (TODO)
//! - [`manager`]: High-level conversion management (TODO)

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;

// TODO: Implement these modules
// pub mod dicom;
// pub mod nifti;
// pub mod strategies;
// pub mod postprocess;
// pub mod manager;
// pub mod utils;
// pub mod cli;

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

// Re-export commonly used types
pub use config::{
    Contrast, ImageOrientation, MRAcquisitionType, MRSeriesRename, Modality, SeriesRename,
    T1SeriesRename, T2SeriesRename,
};
