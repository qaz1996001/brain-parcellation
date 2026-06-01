//! Conversion management module
//!
//! This module provides the main conversion manager that orchestrates
//! the DICOM to NIfTI conversion pipeline.

pub mod convert_manager;
pub mod series_organizer;

pub use convert_manager::ConvertManager;
pub use series_organizer::SeriesOrganizer;
