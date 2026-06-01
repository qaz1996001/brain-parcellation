//! DICOM to NIfTI conversion manager
//!
//! This module provides the main conversion manager that orchestrates
//! the complete pipeline: DICOM reading -> Series recognition -> NIfTI conversion.

use crate::config::{get_config, Settings};
use crate::dicom::{DicomMetadata, DicomReader};
use crate::nifti::NiftiConverter;
use crate::strategies::{ProcessingResult, StrategyRegistry};
use crate::utils::fs::{collect_dicom_files, ensure_dir, DicomFileInfo};
use crate::utils::parallel::{ParallelConfig, ParallelExecutor};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn, error};

/// Result of processing a single DICOM series
#[derive(Debug, Clone)]
pub struct SeriesConversionResult {
    /// Series UID
    pub series_uid: String,
    /// Identified series name
    pub series_name: String,
    /// Number of files in the series
    pub file_count: usize,
    /// Output NIfTI path
    pub output_path: Option<PathBuf>,
    /// Whether conversion was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Result of processing a complete study
#[derive(Debug)]
pub struct StudyConversionResult {
    /// Study folder name
    pub study_folder: String,
    /// Patient ID
    pub patient_id: Option<String>,
    /// Study date
    pub study_date: Option<String>,
    /// Series conversion results
    pub series_results: Vec<SeriesConversionResult>,
    /// Total files processed
    pub total_files: usize,
    /// Total series identified
    pub total_series: usize,
    /// Successful conversions
    pub successful: usize,
    /// Failed conversions
    pub failed: usize,
}

/// DICOM to NIfTI conversion manager
pub struct ConvertManager {
    /// Strategy registry for series recognition
    strategy_registry: Arc<StrategyRegistry>,
    /// NIfTI converter
    converter: NiftiConverter,
    /// Parallel executor for processing
    parallel_executor: ParallelExecutor,
    /// Configuration settings
    config: &'static Settings,
}

impl ConvertManager {
    /// Create a new conversion manager
    pub fn new() -> Self {
        Self::with_config(get_config())
    }

    /// Create a new conversion manager with custom configuration
    pub fn with_config(config: &'static Settings) -> Self {
        let parallel_config = ParallelConfig::default();

        Self {
            strategy_registry: Arc::new(StrategyRegistry::default()),
            converter: NiftiConverter::new(),
            parallel_executor: ParallelExecutor::new(parallel_config),
            config,
        }
    }

    /// Process a directory of DICOM files
    pub fn process_directory(
        &self,
        input_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
    ) -> Result<StudyConversionResult> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        info!("Processing DICOM directory: {:?}", input_dir);

        // Collect all DICOM files
        let dicom_files = collect_dicom_files(input_dir, true)?;
        let total_files = dicom_files.len();

        if total_files == 0 {
            warn!("No DICOM files found in {:?}", input_dir);
            return Ok(StudyConversionResult {
                study_folder: input_dir.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                patient_id: None,
                study_date: None,
                series_results: vec![],
                total_files: 0,
                total_series: 0,
                successful: 0,
                failed: 0,
            });
        }

        info!("Found {} DICOM files", total_files);

        // Group files by series
        let series_groups = self.group_by_series(&dicom_files)?;
        let total_series = series_groups.len();

        info!("Identified {} series", total_series);

        // Process each series
        let mut series_results = Vec::with_capacity(total_series);
        let mut patient_id = None;
        let mut study_date = None;
        let mut study_folder = String::new();

        for (series_uid, files) in series_groups {
            // Read metadata from first file to identify series
            let first_file = &files[0];
            let metadata = match DicomReader::read_metadata(&first_file.path) {
                Ok(m) => m,
                Err(e) => {
                    error!("Failed to read metadata from {:?}: {}", first_file.path, e);
                    series_results.push(SeriesConversionResult {
                        series_uid: series_uid.clone(),
                        series_name: "unknown".to_string(),
                        file_count: files.len(),
                        output_path: None,
                        success: false,
                        error: Some(format!("Failed to read metadata: {}", e)),
                    });
                    continue;
                }
            };

            // Capture study info from first series
            if patient_id.is_none() {
                patient_id = metadata.patient_id.clone();
                study_date = metadata.study_date.clone();
                study_folder = metadata.generate_study_folder();
            }

            // Identify series using strategy registry
            let processing_result = self.strategy_registry.process(&metadata)?;

            let series_name = match &processing_result {
                ProcessingResult::Matched { series_name, .. } => series_name.clone(),
                ProcessingResult::Skip { reason } => {
                    debug!("Skipping series {}: {}", series_uid, reason);
                    continue;
                }
                ProcessingResult::NotMatched => {
                    debug!("Series {} did not match any strategy", series_uid);
                    "unknown".to_string()
                }
            };

            info!("Series {} identified as: {}", series_uid, series_name);

            // Create output directory structure
            let series_output_dir = output_dir
                .join(&study_folder)
                .join(&series_name);

            if let Err(e) = ensure_dir(&series_output_dir) {
                error!("Failed to create output directory: {}", e);
                series_results.push(SeriesConversionResult {
                    series_uid: series_uid.clone(),
                    series_name,
                    file_count: files.len(),
                    output_path: None,
                    success: false,
                    error: Some(format!("Failed to create output directory: {}", e)),
                });
                continue;
            }

            // Convert to NIfTI
            let file_paths: Vec<&Path> = files.iter().map(|f| f.path.as_path()).collect();
            let nifti_result = self.converter.convert(
                &file_paths,
                &series_output_dir,
                &series_name,
            );

            match nifti_result {
                Ok(output_path) => {
                    info!("Successfully converted to: {:?}", output_path);
                    series_results.push(SeriesConversionResult {
                        series_uid: series_uid.clone(),
                        series_name,
                        file_count: files.len(),
                        output_path: Some(output_path),
                        success: true,
                        error: None,
                    });
                }
                Err(e) => {
                    error!("Failed to convert series {}: {}", series_uid, e);
                    series_results.push(SeriesConversionResult {
                        series_uid: series_uid.clone(),
                        series_name,
                        file_count: files.len(),
                        output_path: None,
                        success: false,
                        error: Some(format!("Conversion failed: {}", e)),
                    });
                }
            }
        }

        let successful = series_results.iter().filter(|r| r.success).count();
        let failed = series_results.iter().filter(|r| !r.success).count();

        Ok(StudyConversionResult {
            study_folder,
            patient_id,
            study_date,
            series_results,
            total_files,
            total_series,
            successful,
            failed,
        })
    }

    /// Group DICOM files by series UID
    fn group_by_series(&self, files: &[DicomFileInfo]) -> Result<HashMap<String, Vec<DicomFileInfo>>> {
        let mut groups: HashMap<String, Vec<DicomFileInfo>> = HashMap::new();

        for file in files {
            // Read metadata to get series UID
            let metadata = DicomReader::read_metadata(&file.path)?;

            // Use series instance UID as key, or generate one from series description + number
            let series_key = metadata.series_description
                .as_ref()
                .map(|d| format!("{}_{}", d, metadata.series_number.unwrap_or(0)))
                .unwrap_or_else(|| format!("series_{}", metadata.series_number.unwrap_or(0)));

            groups.entry(series_key).or_default().push(file.clone());
        }

        // Sort files within each series by instance creation time
        for files in groups.values_mut() {
            files.sort_by(|a, b| a.path.cmp(&b.path));
        }

        Ok(groups)
    }

    /// Process a single DICOM file and return its identified series name
    pub fn identify_series(&self, path: impl AsRef<Path>) -> Result<ProcessingResult> {
        let metadata = DicomReader::read_metadata(path)?;
        self.strategy_registry.process(&metadata)
    }

    /// Get the strategy registry
    pub fn strategy_registry(&self) -> &StrategyRegistry {
        &self.strategy_registry
    }
}

impl Default for ConvertManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_manager_creation() {
        let manager = ConvertManager::new();
        assert!(!manager.strategy_registry().strategies().is_empty());
    }
}
