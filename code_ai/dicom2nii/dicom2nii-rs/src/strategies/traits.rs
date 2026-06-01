//! Strategy traits for DICOM series processing
//!
//! This module defines the core traits for implementing
//! processing strategies for different series types.

use crate::dicom::DicomMetadata;
use anyhow::Result;
use std::sync::Arc;

/// Result of processing a DICOM file
#[derive(Debug, Clone)]
pub enum ProcessingResult {
    /// Successfully matched and identified series
    Matched {
        /// The identified series name
        series_name: String,
        /// Additional metadata for the series
        metadata: Option<String>,
    },
    /// Did not match this strategy
    NotMatched,
    /// Matched but should be skipped
    Skip {
        /// Reason for skipping
        reason: String,
    },
}

impl ProcessingResult {
    /// Create a matched result
    pub fn matched(series_name: impl Into<String>) -> Self {
        ProcessingResult::Matched {
            series_name: series_name.into(),
            metadata: None,
        }
    }

    /// Create a matched result with metadata
    pub fn matched_with_metadata(series_name: impl Into<String>, metadata: impl Into<String>) -> Self {
        ProcessingResult::Matched {
            series_name: series_name.into(),
            metadata: Some(metadata.into()),
        }
    }

    /// Create a skip result
    pub fn skip(reason: impl Into<String>) -> Self {
        ProcessingResult::Skip {
            reason: reason.into(),
        }
    }

    /// Check if this is a match
    pub fn is_matched(&self) -> bool {
        matches!(self, ProcessingResult::Matched { .. })
    }

    /// Get the series name if matched
    pub fn series_name(&self) -> Option<&str> {
        match self {
            ProcessingResult::Matched { series_name, .. } => Some(series_name),
            _ => None,
        }
    }
}

/// Trait for processing strategies
pub trait ProcessingStrategy: Send + Sync {
    /// Get the strategy name
    fn name(&self) -> &'static str;

    /// Get the priority (lower = higher priority)
    fn priority(&self) -> i32 {
        100
    }

    /// Check if this strategy supports the given modality
    fn supports_modality(&self, modality: &str) -> bool;

    /// Process a DICOM file and return the result
    fn process(&self, metadata: &DicomMetadata) -> Result<ProcessingResult>;

    /// Get the regex pattern for matching series description
    fn get_pattern(&self) -> Option<&regex::Regex> {
        None
    }
}

/// Registry for managing processing strategies
pub struct StrategyRegistry {
    strategies: Vec<Arc<dyn ProcessingStrategy>>,
}

impl StrategyRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Create a registry with default MR strategies
    pub fn with_default_mr_strategies() -> Self {
        let mut registry = Self::new();

        // Add strategies in priority order
        registry.register(Arc::new(super::mr::dwi::DwiProcessingStrategy::new()));
        registry.register(Arc::new(super::mr::adc::AdcProcessingStrategy::new()));
        registry.register(Arc::new(super::mr::t1::T1ProcessingStrategy::new()));

        // Sort by priority
        registry.sort_by_priority();

        registry
    }

    /// Register a new strategy
    pub fn register(&mut self, strategy: Arc<dyn ProcessingStrategy>) {
        self.strategies.push(strategy);
    }

    /// Sort strategies by priority
    pub fn sort_by_priority(&mut self) {
        self.strategies.sort_by_key(|s| s.priority());
    }

    /// Process metadata through all strategies
    pub fn process(&self, metadata: &DicomMetadata) -> Result<ProcessingResult> {
        let modality = metadata.modality.as_deref().unwrap_or("");

        for strategy in &self.strategies {
            if !strategy.supports_modality(modality) {
                continue;
            }

            match strategy.process(metadata)? {
                ProcessingResult::Matched { series_name, metadata } => {
                    return Ok(ProcessingResult::Matched { series_name, metadata });
                }
                ProcessingResult::Skip { reason } => {
                    return Ok(ProcessingResult::Skip { reason });
                }
                ProcessingResult::NotMatched => continue,
            }
        }

        Ok(ProcessingResult::NotMatched)
    }

    /// Get all registered strategies
    pub fn strategies(&self) -> &[Arc<dyn ProcessingStrategy>] {
        &self.strategies
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::with_default_mr_strategies()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_result_matched() {
        let result = ProcessingResult::matched("T1_AXI");
        assert!(result.is_matched());
        assert_eq!(result.series_name(), Some("T1_AXI"));
    }

    #[test]
    fn test_processing_result_not_matched() {
        let result = ProcessingResult::NotMatched;
        assert!(!result.is_matched());
        assert_eq!(result.series_name(), None);
    }
}
