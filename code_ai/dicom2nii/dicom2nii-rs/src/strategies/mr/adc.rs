//! ADC (Apparent Diffusion Coefficient) series processing strategy
//!
//! This module implements the strategy for identifying ADC MRI series
//! including standard ADC and exponential ADC (eADC).

use crate::config::MRSeriesRename;
use crate::dicom::DicomMetadata;
use crate::strategies::traits::{ProcessingResult, ProcessingStrategy};
use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// Pattern for ADC series description
    static ref ADC_PATTERN: Regex = Regex::new(r"(?i).*(ADC|APPARENT\s*DIFFUSION).*").unwrap();

    /// Pattern for eADC (exponential ADC)
    static ref EADC_PATTERN: Regex = Regex::new(r"(?i).*(EADC|EXP.*ADC).*").unwrap();
}

/// ADC series processing strategy
pub struct AdcProcessingStrategy {
    pattern: Regex,
}

impl AdcProcessingStrategy {
    /// Create a new ADC processing strategy
    pub fn new() -> Self {
        Self {
            pattern: ADC_PATTERN.clone(),
        }
    }

    /// Determine the ADC series type
    fn get_adc_series_name(&self, metadata: &DicomMetadata) -> MRSeriesRename {
        let series_desc = metadata.series_description.as_deref().unwrap_or("");

        // Check for eADC first (more specific)
        if EADC_PATTERN.is_match(series_desc) {
            return MRSeriesRename::EADC;
        }

        // Default to standard ADC
        MRSeriesRename::ADC
    }
}

impl Default for AdcProcessingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessingStrategy for AdcProcessingStrategy {
    fn name(&self) -> &'static str {
        "ADC"
    }

    fn priority(&self) -> i32 {
        25 // Higher priority than DWI (lower number = higher priority)
    }

    fn supports_modality(&self, modality: &str) -> bool {
        modality.eq_ignore_ascii_case("MR")
    }

    fn process(&self, metadata: &DicomMetadata) -> Result<ProcessingResult> {
        // Check series description
        let series_desc = match &metadata.series_description {
            Some(desc) => desc,
            None => return Ok(ProcessingResult::NotMatched),
        };

        // Check if this matches ADC pattern
        if !self.pattern.is_match(series_desc) {
            return Ok(ProcessingResult::NotMatched);
        }

        // ADC is typically a derived image
        // We still process it even if it's not marked as derived
        // since some scanners don't properly set the image type

        // Get the series name
        let series = self.get_adc_series_name(metadata);

        Ok(ProcessingResult::matched(series.to_string()))
    }

    fn get_pattern(&self) -> Option<&Regex> {
        Some(&self.pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adc_pattern() {
        assert!(ADC_PATTERN.is_match("ADC"));
        assert!(ADC_PATTERN.is_match("AX ADC"));
        assert!(ADC_PATTERN.is_match("APPARENT DIFFUSION"));
        assert!(!ADC_PATTERN.is_match("DWI"));
    }

    #[test]
    fn test_eadc_pattern() {
        assert!(EADC_PATTERN.is_match("eADC"));
        assert!(EADC_PATTERN.is_match("EADC"));
        assert!(EADC_PATTERN.is_match("EXP ADC"));
        assert!(!EADC_PATTERN.is_match("ADC"));
    }
}
