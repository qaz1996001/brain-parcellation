//! Filesystem utilities for DICOM processing
//!
//! This module provides functions for:
//! - Traversing directories to find DICOM files
//! - Managing output directories
//! - File validation and information extraction

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use walkdir::{DirEntry, WalkDir};

/// DICOM file magic bytes (DICM at offset 128)
const DICOM_MAGIC: &[u8; 4] = b"DICM";
const DICOM_MAGIC_OFFSET: usize = 128;

/// Information about a DICOM file
#[derive(Debug, Clone)]
pub struct DicomFileInfo {
    /// Full path to the file
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// Last modification time
    pub modified: Option<SystemTime>,
    /// Parent directory name (often series folder)
    pub parent_name: Option<String>,
}

impl DicomFileInfo {
    /// Create a new DicomFileInfo from a path
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).context("Failed to get file metadata")?;

        let parent_name = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .map(String::from);

        Ok(Self {
            path: path.to_path_buf(),
            size: metadata.len(),
            modified: metadata.modified().ok(),
            parent_name,
        })
    }
}

/// Check if a file is likely a DICOM file
///
/// This function checks:
/// 1. File extension (.dcm, .DCM, or no extension)
/// 2. DICOM magic bytes at offset 128 (DICM)
///
/// # Arguments
/// * `path` - Path to the file to check
///
/// # Returns
/// * `true` if the file appears to be a DICOM file
/// * `false` otherwise
pub fn is_dicom_file(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();

    // Must be a file
    if !path.is_file() {
        return false;
    }

    // Check extension first (fast path)
    if let Some(ext) = path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        if ext_lower == "dcm" {
            return true;
        }
        // Skip obviously non-DICOM files
        if matches!(
            ext_lower.as_str(),
            "txt" | "json" | "xml" | "csv" | "nii" | "gz" | "zip" | "tar" | "png" | "jpg" | "jpeg"
        ) {
            return false;
        }
    }

    // Check DICOM magic bytes
    check_dicom_magic(path).unwrap_or(false)
}

/// Check for DICOM magic bytes (DICM at offset 128)
fn check_dicom_magic(path: &Path) -> Result<bool> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read first 132 bytes (128 preamble + 4 magic)
    let mut buffer = [0u8; DICOM_MAGIC_OFFSET + 4];
    let bytes_read = reader.read(&mut buffer)?;

    if bytes_read < DICOM_MAGIC_OFFSET + 4 {
        return Ok(false);
    }

    // Check magic bytes
    Ok(&buffer[DICOM_MAGIC_OFFSET..] == DICOM_MAGIC)
}

/// Collect all DICOM files from a directory
///
/// # Arguments
/// * `root` - Root directory to search
/// * `recursive` - Whether to search subdirectories
///
/// # Returns
/// * `Result<Vec<DicomFileInfo>>` - List of DICOM file information
pub fn collect_dicom_files(root: impl AsRef<Path>, recursive: bool) -> Result<Vec<DicomFileInfo>> {
    let root = root.as_ref();

    if !root.exists() {
        anyhow::bail!("Directory does not exist: {:?}", root);
    }

    if !root.is_dir() {
        anyhow::bail!("Path is not a directory: {:?}", root);
    }

    let walker = if recursive {
        WalkDir::new(root)
    } else {
        WalkDir::new(root).max_depth(1)
    };

    let files: Vec<DicomFileInfo> = walker
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| is_dicom_file(e.path()))
        .filter_map(|e| DicomFileInfo::from_path(e.path()).ok())
        .collect();

    Ok(files)
}

/// Collect all subdirectories (study/series folders)
///
/// # Arguments
/// * `root` - Root directory to search
/// * `depth` - Maximum depth to search (1 = immediate children only)
///
/// # Returns
/// * `Result<Vec<PathBuf>>` - List of subdirectory paths
pub fn collect_subdirectories(root: impl AsRef<Path>, depth: usize) -> Result<Vec<PathBuf>> {
    let root = root.as_ref();

    if !root.exists() {
        anyhow::bail!("Directory does not exist: {:?}", root);
    }

    let dirs: Vec<PathBuf> = WalkDir::new(root)
        .min_depth(1)
        .max_depth(depth)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_dir())
        .map(|e| e.path().to_path_buf())
        .collect();

    Ok(dirs)
}

/// Ensure a directory exists, creating it if necessary
///
/// # Arguments
/// * `path` - Path to the directory
///
/// # Returns
/// * `Result<PathBuf>` - The path to the directory
pub fn ensure_dir(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();

    if !path.exists() {
        fs::create_dir_all(path).context(format!("Failed to create directory: {:?}", path))?;
    }

    Ok(path.to_path_buf())
}

/// Copy a file to a destination, creating parent directories if needed
///
/// # Arguments
/// * `src` - Source file path
/// * `dst` - Destination file path
///
/// # Returns
/// * `Result<u64>` - Number of bytes copied
pub fn copy_file(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<u64> {
    let src = src.as_ref();
    let dst = dst.as_ref();

    // Ensure parent directory exists
    if let Some(parent) = dst.parent() {
        ensure_dir(parent)?;
    }

    fs::copy(src, dst).context(format!("Failed to copy {:?} to {:?}", src, dst))
}

/// Move a file to a destination, creating parent directories if needed
///
/// # Arguments
/// * `src` - Source file path
/// * `dst` - Destination file path
pub fn move_file(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
    let src = src.as_ref();
    let dst = dst.as_ref();

    // Ensure parent directory exists
    if let Some(parent) = dst.parent() {
        ensure_dir(parent)?;
    }

    fs::rename(src, dst).context(format!("Failed to move {:?} to {:?}", src, dst))
}

/// Remove a file if it exists
///
/// # Arguments
/// * `path` - Path to the file
///
/// # Returns
/// * `Result<bool>` - true if file was removed, false if it didn't exist
pub fn remove_file_if_exists(path: impl AsRef<Path>) -> Result<bool> {
    let path = path.as_ref();

    if path.exists() {
        fs::remove_file(path).context(format!("Failed to remove file: {:?}", path))?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Remove a directory and all its contents if it exists
///
/// # Arguments
/// * `path` - Path to the directory
///
/// # Returns
/// * `Result<bool>` - true if directory was removed, false if it didn't exist
pub fn remove_dir_if_exists(path: impl AsRef<Path>) -> Result<bool> {
    let path = path.as_ref();

    if path.exists() {
        fs::remove_dir_all(path).context(format!("Failed to remove directory: {:?}", path))?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Get the size of a file in bytes
///
/// # Arguments
/// * `path` - Path to the file
///
/// # Returns
/// * `Result<u64>` - File size in bytes
pub fn file_size(path: impl AsRef<Path>) -> Result<u64> {
    let metadata = fs::metadata(path.as_ref())?;
    Ok(metadata.len())
}

/// Get the size of a file in kilobytes
///
/// # Arguments
/// * `path` - Path to the file
///
/// # Returns
/// * `Result<u64>` - File size in KB
pub fn file_size_kb(path: impl AsRef<Path>) -> Result<u64> {
    Ok(file_size(path)? / 1024)
}

/// Check if a file is smaller than a threshold (useful for filtering invalid files)
///
/// # Arguments
/// * `path` - Path to the file
/// * `min_size_kb` - Minimum size in KB
///
/// # Returns
/// * `bool` - true if file is smaller than threshold
pub fn is_file_too_small(path: impl AsRef<Path>, min_size_kb: u64) -> bool {
    file_size_kb(path).map(|size| size < min_size_kb).unwrap_or(true)
}

/// List all files with a specific extension in a directory
///
/// # Arguments
/// * `dir` - Directory to search
/// * `extension` - File extension (without dot, e.g., "nii" or "gz")
/// * `recursive` - Whether to search subdirectories
///
/// # Returns
/// * `Result<Vec<PathBuf>>` - List of matching file paths
pub fn list_files_with_extension(
    dir: impl AsRef<Path>,
    extension: &str,
    recursive: bool,
) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    let ext_lower = extension.to_lowercase();

    let walker = if recursive {
        WalkDir::new(dir)
    } else {
        WalkDir::new(dir).max_depth(1)
    };

    let files: Vec<PathBuf> = walker
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext.to_string_lossy().to_lowercase() == ext_lower)
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    Ok(files)
}

/// List all NIfTI files (.nii, .nii.gz) in a directory
///
/// # Arguments
/// * `dir` - Directory to search
/// * `recursive` - Whether to search subdirectories
///
/// # Returns
/// * `Result<Vec<PathBuf>>` - List of NIfTI file paths
pub fn list_nifti_files(dir: impl AsRef<Path>, recursive: bool) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();

    let walker = if recursive {
        WalkDir::new(dir)
    } else {
        WalkDir::new(dir).max_depth(1)
    };

    let files: Vec<PathBuf> = walker
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| is_nifti_file(e.path()))
        .map(|e| e.path().to_path_buf())
        .collect();

    Ok(files)
}

/// Check if a file is a NIfTI file
fn is_nifti_file(path: &Path) -> bool {
    let path_str = path.to_string_lossy().to_lowercase();
    path_str.ends_with(".nii") || path_str.ends_with(".nii.gz")
}

/// Get unique parent directories from a list of file paths
///
/// # Arguments
/// * `files` - List of file paths
///
/// # Returns
/// * `HashSet<PathBuf>` - Set of unique parent directories
pub fn get_unique_parents(files: &[PathBuf]) -> HashSet<PathBuf> {
    files
        .iter()
        .filter_map(|f| f.parent())
        .map(|p| p.to_path_buf())
        .collect()
}

/// Generate output path based on DICOM metadata
///
/// Format: `{patient_id}_{study_date}_{modality}_{accession_number}`
///
/// # Arguments
/// * `base_path` - Base output directory
/// * `patient_id` - Patient ID
/// * `study_date` - Study date (YYYYMMDD)
/// * `modality` - Modality (CT, MR)
/// * `accession_number` - Accession number
///
/// # Returns
/// * `PathBuf` - Generated output path
pub fn generate_study_output_path(
    base_path: impl AsRef<Path>,
    patient_id: &str,
    study_date: &str,
    modality: &str,
    accession_number: &str,
) -> PathBuf {
    let folder_name = format!(
        "{}_{}_{}_{}",
        sanitize_filename(patient_id),
        sanitize_filename(study_date),
        sanitize_filename(modality),
        sanitize_filename(accession_number)
    );
    base_path.as_ref().join(folder_name)
}

/// Sanitize a string for use as a filename
///
/// Replaces invalid characters with underscores
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_ensure_dir() {
        let temp = tempdir().unwrap();
        let new_dir = temp.path().join("test_dir");

        assert!(!new_dir.exists());
        ensure_dir(&new_dir).unwrap();
        assert!(new_dir.exists());
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("test123"), "test123");
        assert_eq!(sanitize_filename("test/file"), "test_file");
        assert_eq!(sanitize_filename("test:file"), "test_file");
        assert_eq!(sanitize_filename("test file"), "test_file");
    }

    #[test]
    fn test_generate_study_output_path() {
        let path = generate_study_output_path(
            "/output",
            "PAT001",
            "20240101",
            "MR",
            "ACC123"
        );
        assert_eq!(
            path.to_string_lossy(),
            "/output/PAT001_20240101_MR_ACC123"
        );
    }

    #[test]
    fn test_is_nifti_file() {
        assert!(is_nifti_file(Path::new("test.nii")));
        assert!(is_nifti_file(Path::new("test.nii.gz")));
        assert!(is_nifti_file(Path::new("TEST.NII.GZ")));
        assert!(!is_nifti_file(Path::new("test.dcm")));
        assert!(!is_nifti_file(Path::new("test.txt")));
    }
}
