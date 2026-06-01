//! Series organizer for DICOM files
//!
//! This module provides utilities for organizing DICOM files
//! based on identified series names.

use crate::dicom::{DicomMetadata, DicomReader};
use crate::strategies::{ProcessingResult, StrategyRegistry};
use crate::utils::fs::ensure_dir;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Series organizer for renaming and organizing DICOM files
pub struct SeriesOrganizer {
    /// Strategy registry for series identification
    strategy_registry: Arc<StrategyRegistry>,
    /// Whether to copy files instead of moving
    copy_mode: bool,
    /// Whether to use series description as subfolder
    use_series_subfolder: bool,
}

impl SeriesOrganizer {
    /// Create a new series organizer
    pub fn new(strategy_registry: Arc<StrategyRegistry>) -> Self {
        Self {
            strategy_registry,
            copy_mode: false,
            use_series_subfolder: true,
        }
    }

    /// Set copy mode (true = copy, false = move)
    pub fn with_copy_mode(mut self, copy: bool) -> Self {
        self.copy_mode = copy;
        self
    }

    /// Set whether to use series subfolders
    pub fn with_series_subfolder(mut self, use_subfolder: bool) -> Self {
        self.use_series_subfolder = use_subfolder;
        self
    }

    /// Organize a single DICOM file
    pub fn organize_file(
        &self,
        source_path: impl AsRef<Path>,
        output_base: impl AsRef<Path>,
    ) -> Result<Option<PathBuf>> {
        let source_path = source_path.as_ref();
        let output_base = output_base.as_ref();

        // Read metadata
        let metadata = DicomReader::read_metadata(source_path)
            .with_context(|| format!("Failed to read DICOM: {:?}", source_path))?;

        // Identify series
        let result = self.strategy_registry.process(&metadata)?;

        let series_name = match result {
            ProcessingResult::Matched { series_name, .. } => series_name,
            ProcessingResult::Skip { reason } => {
                debug!("Skipping {:?}: {}", source_path, reason);
                return Ok(None);
            }
            ProcessingResult::NotMatched => {
                debug!("No match for {:?}", source_path);
                return Ok(None);
            }
        };

        // Build output path
        let study_folder = metadata.generate_study_folder();
        let mut output_dir = output_base.join(&study_folder);

        if self.use_series_subfolder {
            output_dir = output_dir.join(&series_name);
        }

        ensure_dir(&output_dir)?;

        // Generate output filename
        let filename = source_path.file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid source path"))?;
        let output_path = output_dir.join(filename);

        // Copy or move file
        if self.copy_mode {
            fs::copy(source_path, &output_path)
                .with_context(|| format!("Failed to copy {:?}", source_path))?;
        } else {
            fs::rename(source_path, &output_path)
                .with_context(|| format!("Failed to move {:?}", source_path))?;
        }

        info!("Organized {:?} -> {:?}", source_path, output_path);

        Ok(Some(output_path))
    }

    /// Organize multiple DICOM files
    pub fn organize_files(
        &self,
        source_paths: &[PathBuf],
        output_base: impl AsRef<Path>,
    ) -> Result<OrganizeResult> {
        let output_base = output_base.as_ref();
        let mut result = OrganizeResult::default();

        for path in source_paths {
            match self.organize_file(path, output_base) {
                Ok(Some(output)) => {
                    result.organized.push(output);
                    result.success_count += 1;
                }
                Ok(None) => {
                    result.skipped.push(path.clone());
                    result.skip_count += 1;
                }
                Err(e) => {
                    warn!("Failed to organize {:?}: {}", path, e);
                    result.failed.push((path.clone(), e.to_string()));
                    result.fail_count += 1;
                }
            }
        }

        Ok(result)
    }

    /// Group files by identified series
    pub fn group_by_series(
        &self,
        files: &[PathBuf],
    ) -> Result<HashMap<String, Vec<PathBuf>>> {
        let mut groups: HashMap<String, Vec<PathBuf>> = HashMap::new();

        for path in files {
            let metadata = match DicomReader::read_metadata(path) {
                Ok(m) => m,
                Err(e) => {
                    warn!("Failed to read {:?}: {}", path, e);
                    continue;
                }
            };

            let result = self.strategy_registry.process(&metadata)?;

            let series_name = match result {
                ProcessingResult::Matched { series_name, .. } => series_name,
                _ => "unknown".to_string(),
            };

            groups.entry(series_name).or_default().push(path.clone());
        }

        Ok(groups)
    }
}

/// Result of organizing multiple files
#[derive(Debug, Default)]
pub struct OrganizeResult {
    /// Successfully organized files
    pub organized: Vec<PathBuf>,
    /// Skipped files (no match)
    pub skipped: Vec<PathBuf>,
    /// Failed files with error messages
    pub failed: Vec<(PathBuf, String)>,
    /// Success count
    pub success_count: usize,
    /// Skip count
    pub skip_count: usize,
    /// Fail count
    pub fail_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_organizer_creation() {
        let registry = Arc::new(StrategyRegistry::default());
        let organizer = SeriesOrganizer::new(registry);
        assert!(!organizer.copy_mode);
        assert!(organizer.use_series_subfolder);
    }
}
