//! Utility modules for DICOM to NIfTI conversion
//!
//! This module provides common utilities:
//! - [`fs`]: Filesystem operations (directory traversal, file management)
//! - [`parallel`]: Parallel processing utilities
//! - [`regex_patterns`]: Pre-compiled regex patterns for DICOM parsing

pub mod fs;
pub mod parallel;
pub mod regex_patterns;

// Re-export commonly used items
pub use fs::{collect_dicom_files, ensure_dir, is_dicom_file, DicomFileInfo};
pub use parallel::{ParallelConfig, ParallelExecutor};
pub use regex_patterns::PATTERNS;
