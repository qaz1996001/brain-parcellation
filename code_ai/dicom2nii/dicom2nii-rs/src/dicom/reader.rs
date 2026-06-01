//! DICOM file reader
//!
//! This module provides functions for reading DICOM files.

use crate::dicom::metadata::DicomMetadata;
use anyhow::{Context, Result};
use dicom::object::{open_file, FileDicomObject, InMemDicomObject};
use std::path::Path;

/// DICOM file reader
pub struct DicomReader;

impl DicomReader {
    /// Read a DICOM file and return the object
    pub fn read_file(path: impl AsRef<Path>) -> Result<FileDicomObject<InMemDicomObject>> {
        let path = path.as_ref();
        open_file(path)
            .with_context(|| format!("Failed to read DICOM file: {:?}", path))
    }

    /// Read a DICOM file and extract metadata
    pub fn read_metadata(path: impl AsRef<Path>) -> Result<DicomMetadata> {
        let obj = Self::read_file(path)?;
        DicomMetadata::from_dicom(&obj)
    }

    /// Check if a file is a valid DICOM file
    pub fn is_valid_dicom(path: impl AsRef<Path>) -> bool {
        Self::read_file(path).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_file() {
        assert!(!DicomReader::is_valid_dicom("/nonexistent/file.dcm"));
    }
}
