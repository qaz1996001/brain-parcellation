//! DICOM processing module
//!
//! This module provides DICOM file reading and metadata extraction.

pub mod metadata;
pub mod reader;

pub use metadata::DicomMetadata;
pub use reader::DicomReader;
