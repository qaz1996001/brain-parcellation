//! Configuration module for DICOM to NIfTI conversion
//!
//! This module provides:
//! - Enum definitions for series naming, orientations, modalities, etc.
//! - Settings management and TOML configuration loading
//! - Constants for DICOM tags and processing parameters
//!
//! # Example
//!
//! ```rust,ignore
//! use dicom2nii::config::{Modality, ImageOrientation, T1SeriesRename, Settings};
//!
//! // Load configuration
//! let settings = Settings::load_or_default();
//!
//! // Use enums
//! let modality = Modality::MR;
//! let orientation = ImageOrientation::AXI;
//! let series = T1SeriesRename::T1BRAVO_AXI;
//!
//! println!("Series: {} ({} {})", series, modality, orientation);
//! println!("Workers: {}", settings.general.default_workers);
//! ```

pub mod enums;
pub mod settings;

// Re-export all enums for convenience
pub use enums::*;

// Re-export settings types
pub use settings::{
    get_config, init_config, init_default_config,
    Settings, GeneralSettings, FilesystemSettings, Dcm2niixSettings,
    ThresholdSettings, FileSizeThresholds, MrParameterThresholds, TrTeThreshold,
    DwiSettings, OrientationSettings, DicomTagSettings,
    PatternSettings, ExclusionSettings, OutputSettings,
    PostprocessSettings, Nii2dcmSettings, ModalitySettings, SeriesTypeSettings,
};
