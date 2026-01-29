//! DICOM to NIfTI CLI tool
//!
//! This binary provides command-line access to the dicom2nii library.
//!
//! # Usage
//!
//! ```bash
//! # Run full pipeline
//! dicom2nii-cli pipeline -i ./raw_dicom -o ./nifti
//!
//! # Rename and organize DICOM files
//! dicom2nii-cli rename -i ./raw_dicom -o ./organized_dicom
//!
//! # Convert DICOM to NIfTI
//! dicom2nii-cli convert -i ./organized_dicom -o ./nifti
//!
//! # Generate statistics
//! dicom2nii-cli stats -p ./nifti -f csv
//!
//! # List supported series types
//! dicom2nii-cli list-series
//!
//! # Show version and info
//! dicom2nii-cli info
//! ```

use clap::Parser;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use dicom2nii::cli::{self, args::LogFormat, Cli};

fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    let log_level = if cli.verbose {
        Level::DEBUG
    } else if cli.quiet {
        Level::ERROR
    } else {
        Level::INFO
    };

    // Set up subscriber based on log format
    match cli.log_format {
        LogFormat::Json => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(log_level)
                .json()
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
        LogFormat::Compact => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(log_level)
                .compact()
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
        LogFormat::Text => {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(log_level)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
        }
    }

    // Set up global thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.effective_workers())
        .build_global()?;

    // Run the command
    cli::run(cli)
}
