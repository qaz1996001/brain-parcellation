//! Command-line argument definitions
//!
//! This module defines all CLI arguments and subcommands using clap.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// DICOM to NIfTI converter - High-performance medical image conversion tool
///
/// This tool provides functionality to:
/// - Rename and organize DICOM files by series type
/// - Convert DICOM to NIfTI format
/// - Post-process NIfTI files
/// - Generate statistics reports
#[derive(Parser, Debug)]
#[command(name = "dicom2nii")]
#[command(author = "Brain Parcellation Team")]
#[command(version)]
#[command(about = "High-performance DICOM to NIfTI converter", long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output (debug logging)
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Number of worker threads (default: number of CPUs, max 8)
    #[arg(short = 'j', long, default_value = "4", global = true)]
    pub jobs: usize,

    /// Disable progress bar
    #[arg(long, global = true)]
    pub no_progress: bool,

    /// Output format for logs
    #[arg(long, default_value = "text", global = true)]
    pub log_format: LogFormat,
}

/// Available subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Rename and organize DICOM files by series type
    ///
    /// This command analyzes DICOM files and organizes them into
    /// standardized folder structures based on series type (T1, T2, DWI, etc.)
    Rename(RenameArgs),

    /// Convert DICOM files to NIfTI format
    ///
    /// Uses dcm2niix for conversion and applies post-processing
    Convert(ConvertArgs),

    /// Run the full pipeline: rename + convert + postprocess
    ///
    /// This is the recommended command for most use cases
    Pipeline(PipelineArgs),

    /// Post-process existing NIfTI files
    ///
    /// Apply post-processing to already converted NIfTI files
    Postprocess(PostprocessArgs),

    /// Generate statistics report for DICOM or NIfTI files
    Stats(StatsArgs),

    /// Convert NIfTI back to DICOM format
    Nii2dcm(Nii2dcmArgs),

    /// List supported series types
    ListSeries,

    /// Validate DICOM files without processing
    Validate(ValidateArgs),

    /// Show version and build information
    Info,
}

/// Arguments for the rename command
#[derive(Parser, Debug)]
pub struct RenameArgs {
    /// Input directory containing raw DICOM files
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for organized DICOM files
    #[arg(short, long)]
    pub output: PathBuf,

    /// Overwrite existing output files
    #[arg(long)]
    pub overwrite: bool,

    /// Copy files instead of moving them
    #[arg(long)]
    pub copy: bool,

    /// Skip DICOM validation
    #[arg(long)]
    pub skip_validation: bool,

    /// Only process specific modalities (CT, MR)
    #[arg(long, value_delimiter = ',')]
    pub modality: Option<Vec<String>>,

    /// Exclude specific series patterns (regex)
    #[arg(long)]
    pub exclude: Option<String>,

    /// Run in dry-run mode (no actual file operations)
    #[arg(long)]
    pub dry_run: bool,
}

/// Arguments for the convert command
#[derive(Parser, Debug)]
pub struct ConvertArgs {
    /// Input directory containing organized DICOM files
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for NIfTI files
    #[arg(short, long)]
    pub output: PathBuf,

    /// Skip DICOM rename step (input is already organized)
    #[arg(long)]
    pub skip_rename: bool,

    /// Skip post-processing step
    #[arg(long)]
    pub skip_postprocess: bool,

    /// Overwrite existing output files
    #[arg(long)]
    pub overwrite: bool,

    /// Keep intermediate JSON files from dcm2niix
    #[arg(long)]
    pub keep_json: bool,

    /// Compression level (0-9, higher = smaller but slower)
    #[arg(long, default_value = "6")]
    pub compression: u8,

    /// Output filename format (see dcm2niix documentation)
    #[arg(long, default_value = "%f")]
    pub filename_format: String,
}

/// Arguments for the pipeline command
#[derive(Parser, Debug)]
pub struct PipelineArgs {
    /// Input directory containing raw DICOM files
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for final NIfTI files
    #[arg(short, long)]
    pub output: PathBuf,

    /// Intermediate directory for organized DICOM files
    /// (default: creates temp directory)
    #[arg(long)]
    pub intermediate: Option<PathBuf>,

    /// Keep intermediate DICOM files after conversion
    #[arg(long)]
    pub keep_intermediate: bool,

    /// Overwrite existing output files
    #[arg(long)]
    pub overwrite: bool,

    /// Generate statistics report after processing
    #[arg(long)]
    pub stats: bool,

    /// Statistics output format
    #[arg(long, default_value = "csv")]
    pub stats_format: OutputFormat,
}

/// Arguments for the postprocess command
#[derive(Parser, Debug)]
pub struct PostprocessArgs {
    /// Input directory containing NIfTI files
    #[arg(short, long)]
    pub input: PathBuf,

    /// Processing mode
    #[arg(long, default_value = "all")]
    pub mode: PostprocessMode,

    /// Remove small/invalid files
    #[arg(long)]
    pub cleanup: bool,

    /// Minimum file size in KB (files smaller than this are removed)
    #[arg(long, default_value = "100")]
    pub min_size_kb: u64,
}

/// Arguments for the stats command
#[derive(Parser, Debug)]
pub struct StatsArgs {
    /// Directory to analyze
    #[arg(short, long)]
    pub path: PathBuf,

    /// Output format
    #[arg(short, long, default_value = "csv")]
    pub format: OutputFormat,

    /// Output file path (default: auto-generated)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Recursive directory scan
    #[arg(short, long)]
    pub recursive: bool,

    /// Include file hashes in output
    #[arg(long)]
    pub include_hash: bool,
}

/// Arguments for the nii2dcm command
#[derive(Parser, Debug)]
pub struct Nii2dcmArgs {
    /// Input NIfTI file
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for DICOM files
    #[arg(short, long)]
    pub output: PathBuf,

    /// Reference DICOM file for metadata
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// Patient ID to use (overrides reference)
    #[arg(long)]
    pub patient_id: Option<String>,

    /// Study description
    #[arg(long)]
    pub study_description: Option<String>,

    /// Series description
    #[arg(long)]
    pub series_description: Option<String>,
}

/// Arguments for the validate command
#[derive(Parser, Debug)]
pub struct ValidateArgs {
    /// Directory to validate
    #[arg(short, long)]
    pub path: PathBuf,

    /// Recursive directory scan
    #[arg(short, long)]
    pub recursive: bool,

    /// Show detailed validation results
    #[arg(long)]
    pub detailed: bool,

    /// Output validation report to file
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Log output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    /// Human-readable text format
    Text,
    /// JSON format for machine parsing
    Json,
    /// Compact format with minimal output
    Compact,
}

impl Default for LogFormat {
    fn default() -> Self {
        LogFormat::Text
    }
}

/// Output format for reports
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// CSV format
    Csv,
    /// JSON format
    Json,
    /// Excel format (.xlsx)
    Excel,
    /// Plain text table
    Table,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Csv
    }
}

/// Post-processing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PostprocessMode {
    /// Process all supported series types
    All,
    /// Only process DWI/ADC series
    Dwi,
    /// Only process SWAN series
    Swan,
    /// Only process T1 series
    T1,
    /// Only process T2 series
    T2,
}

impl Default for PostprocessMode {
    fn default() -> Self {
        PostprocessMode::All
    }
}

impl Cli {
    /// Get the effective number of workers (capped at reasonable maximum)
    pub fn effective_workers(&self) -> usize {
        self.jobs.min(16).max(1)
    }

    /// Check if progress bar should be shown
    pub fn show_progress(&self) -> bool {
        !self.quiet && !self.no_progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::try_parse_from(["dicom2nii", "info"]).unwrap();
        assert!(matches!(cli.command, Commands::Info));
    }

    #[test]
    fn test_rename_args() {
        let cli = Cli::try_parse_from([
            "dicom2nii",
            "rename",
            "-i",
            "/input",
            "-o",
            "/output",
            "--dry-run",
        ])
        .unwrap();

        if let Commands::Rename(args) = cli.command {
            assert_eq!(args.input, PathBuf::from("/input"));
            assert_eq!(args.output, PathBuf::from("/output"));
            assert!(args.dry_run);
        } else {
            panic!("Expected Rename command");
        }
    }

    #[test]
    fn test_pipeline_args() {
        let cli = Cli::try_parse_from([
            "dicom2nii",
            "-j",
            "8",
            "pipeline",
            "-i",
            "/input",
            "-o",
            "/output",
            "--keep-intermediate",
        ])
        .unwrap();

        assert_eq!(cli.jobs, 8);
        assert_eq!(cli.effective_workers(), 8);

        if let Commands::Pipeline(args) = cli.command {
            assert!(args.keep_intermediate);
        } else {
            panic!("Expected Pipeline command");
        }
    }

    #[test]
    fn test_effective_workers_capping() {
        let cli = Cli::try_parse_from(["dicom2nii", "-j", "100", "info"]).unwrap();
        assert_eq!(cli.effective_workers(), 16); // Capped at 16
    }
}
