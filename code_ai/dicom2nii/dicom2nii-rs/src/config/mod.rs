//! Configuration module for DICOM to NIfTI conversion
//!
//! This module provides:
//! - Enum definitions for series naming, orientations, modalities, etc.
//! - Constants for DICOM tags and processing parameters
//!
//! # Example
//!
//! ```rust
//! use dicom2nii::config::{Modality, ImageOrientation, T1SeriesRename};
//!
//! let modality = Modality::MR;
//! let orientation = ImageOrientation::AXI;
//! let series = T1SeriesRename::T1BRAVO_AXI;
//!
//! println!("Series: {} ({} {})", series, modality, orientation);
//! ```

pub mod enums;

// Re-export all enums for convenience
pub use enums::*;
