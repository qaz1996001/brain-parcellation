//! NIfTI converter using dcm2niix
//!
//! This module provides integration with dcm2niix for converting
//! DICOM files to NIfTI format.

use crate::config::{get_config, Dcm2niixSettings};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tracing::{debug, info, warn};

/// NIfTI converter using dcm2niix
pub struct NiftiConverter {
    /// Path to dcm2niix executable
    dcm2niix_path: PathBuf,
    /// Dcm2niix settings
    settings: Dcm2niixSettings,
}

impl NiftiConverter {
    /// Create a new NIfTI converter
    pub fn new() -> Self {
        let config = get_config();
        Self {
            dcm2niix_path: PathBuf::from(&config.dcm2niix.executable),
            settings: config.dcm2niix.clone(),
        }
    }

    /// Create a new NIfTI converter with custom dcm2niix path
    pub fn with_path(dcm2niix_path: impl AsRef<Path>) -> Self {
        let config = get_config();
        Self {
            dcm2niix_path: dcm2niix_path.as_ref().to_path_buf(),
            settings: config.dcm2niix.clone(),
        }
    }

    /// Check if dcm2niix is available
    pub fn is_available(&self) -> bool {
        Command::new(&self.dcm2niix_path)
            .arg("-v")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    /// Get dcm2niix version
    pub fn version(&self) -> Result<String> {
        let output = Command::new(&self.dcm2niix_path)
            .arg("-v")
            .output()
            .context("Failed to run dcm2niix")?;

        let version = String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .unwrap_or("unknown")
            .to_string();

        Ok(version)
    }

    /// Check if compression is enabled based on settings
    fn is_compressed(&self) -> bool {
        self.settings.compression.to_lowercase() == "y"
    }

    /// Convert DICOM files to NIfTI
    ///
    /// # Arguments
    /// * `input_files` - List of DICOM file paths
    /// * `output_dir` - Output directory for NIfTI files
    /// * `output_name` - Base name for output files
    ///
    /// # Returns
    /// Path to the generated NIfTI file
    pub fn convert(
        &self,
        input_files: &[&Path],
        output_dir: impl AsRef<Path>,
        output_name: &str,
    ) -> Result<PathBuf> {
        let output_dir = output_dir.as_ref();

        if input_files.is_empty() {
            bail!("No input files provided");
        }

        // Create output directory if needed
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;

        // Get input directory (dcm2niix needs a directory, not individual files)
        let input_dir = input_files[0].parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid input file path"))?;

        // Build dcm2niix command
        let mut cmd = Command::new(&self.dcm2niix_path);

        // Add standard options
        cmd.arg("-z").arg(&self.settings.compression);
        cmd.arg("-f").arg(output_name);
        cmd.arg("-o").arg(output_dir);

        // Add extra arguments from settings
        for arg in &self.settings.extra_args {
            cmd.arg(arg);
        }

        // Add input directory
        cmd.arg(input_dir);

        debug!("Running dcm2niix: {:?}", cmd);

        // Run conversion
        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute dcm2niix")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            warn!("dcm2niix stderr: {}", stderr);
            warn!("dcm2niix stdout: {}", stdout);
            bail!("dcm2niix failed with exit code: {:?}", output.status.code());
        }

        // Find the generated NIfTI file
        let extension = if self.is_compressed() { ".nii.gz" } else { ".nii" };
        let expected_path = output_dir.join(format!("{}{}", output_name, extension));

        if expected_path.exists() {
            info!("Created NIfTI file: {:?}", expected_path);
            return Ok(expected_path);
        }

        // Try to find any generated NIfTI file
        let nifti_files: Vec<_> = std::fs::read_dir(output_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map_or(false, |ext| ext == "nii" || ext == "gz")
            })
            .collect();

        if let Some(first_nifti) = nifti_files.first() {
            info!("Found NIfTI file: {:?}", first_nifti);
            return Ok(first_nifti.clone());
        }

        bail!("No NIfTI file generated")
    }

    /// Convert DICOM directory to NIfTI
    pub fn convert_directory(
        &self,
        input_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        output_name: &str,
    ) -> Result<PathBuf> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        // Create output directory if needed
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;

        // Build dcm2niix command
        let mut cmd = Command::new(&self.dcm2niix_path);

        cmd.arg("-z").arg(&self.settings.compression);
        cmd.arg("-f").arg(output_name);
        cmd.arg("-o").arg(output_dir);

        // Add extra arguments
        for arg in &self.settings.extra_args {
            cmd.arg(arg);
        }

        cmd.arg(input_dir);

        debug!("Running dcm2niix: {:?}", cmd);

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute dcm2niix")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("dcm2niix failed: {}", stderr);
        }

        // Find the generated NIfTI file
        let extension = if self.is_compressed() { ".nii.gz" } else { ".nii" };
        let expected_path = output_dir.join(format!("{}{}", output_name, extension));

        if expected_path.exists() {
            return Ok(expected_path);
        }

        // Try to find any generated file
        let nifti_files: Vec<_> = std::fs::read_dir(output_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().contains(".nii"))
            .collect();

        nifti_files.first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No NIfTI file generated"))
    }

    /// Batch convert multiple series
    pub fn batch_convert(
        &self,
        conversions: &[(PathBuf, PathBuf, String)], // (input_dir, output_dir, name)
    ) -> Vec<Result<PathBuf>> {
        conversions
            .iter()
            .map(|(input, output, name)| {
                self.convert_directory(input, output, name)
            })
            .collect()
    }
}

impl Default for NiftiConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for NIfTI conversion
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Compress output (.nii.gz)
    pub compress: bool,
    /// Generate BIDS sidecar JSON
    pub bids_sidecar: bool,
    /// Output filename format
    pub filename_format: String,
    /// Additional dcm2niix flags
    pub additional_flags: Vec<String>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            compress: true,
            bids_sidecar: true,
            filename_format: "%p_%s".to_string(),
            additional_flags: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = NiftiConverter::new();
        // Just test that it can be created
        assert!(converter.dcm2niix_path.to_string_lossy().contains("dcm2niix"));
    }
}
