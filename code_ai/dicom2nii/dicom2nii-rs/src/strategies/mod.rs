//! Processing strategies for DICOM series recognition
//!
//! This module provides the strategy pattern implementation for
//! identifying and processing different MRI and CT series types.

pub mod traits;
pub mod mr;

pub use traits::{ProcessingStrategy, ProcessingResult, StrategyRegistry};
pub use mr::t1::T1ProcessingStrategy;
pub use mr::dwi::DwiProcessingStrategy;
pub use mr::adc::AdcProcessingStrategy;
