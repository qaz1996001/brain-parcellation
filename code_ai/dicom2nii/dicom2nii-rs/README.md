# dicom2nii-rs

A high-performance DICOM to NIfTI converter written in Rust with Python bindings.

## Features

- **Fast**: Written in Rust with parallel processing support via Rayon
- **Accurate**: Supports 15+ MR series recognition strategies (T1, T2, DWI, ADC, SWAN, MRA, ASL, DSC, etc.)
- **Flexible**: CLI tool, Python bindings (PyO3), and C FFI
- **Configurable**: All parameters configurable via TOML file

## Installation

### From Source

```bash
cargo build --release
```

### With Python Bindings

```bash
maturin develop --release
```

## Usage

### CLI

```bash
# Full pipeline: raw DICOM -> organized DICOM -> NIfTI
dicom2nii pipeline -i ./raw_dicom -o ./nifti

# Rename and organize DICOM files
dicom2nii rename -i ./raw_dicom -o ./organized_dicom

# Convert DICOM to NIfTI
dicom2nii convert -i ./organized_dicom -o ./nifti

# List supported series types
dicom2nii list-series

# Show version and info
dicom2nii info
```

### Python

```python
from dicom2nii import ConvertManager

manager = ConvertManager()
result = manager.process_directory("input", "output")
print(f"Converted {result.successful} series")
```

## Configuration

All parameters are configurable via `config.toml`. See the included configuration file for all available options.

Key configuration sections:
- `[general]` - General settings (workers, log level)
- `[dcm2niix]` - dcm2niix integration settings
- `[thresholds]` - File size and MR parameter thresholds
- `[dwi]` - DWI b-value settings
- `[orientation]` - Image orientation detection settings
- `[patterns]` - Regex patterns for series matching

## Supported Series Types

### MR Series
- T1 weighted (with FLAIR, CUBE, BRAVO variants)
- T2 weighted (with FLAIR, CUBE variants)
- DWI (b0, b1000, DTI)
- ADC / eADC
- SWAN / SWI
- MRA / MRV
- ASL
- DSC
- And more...

### CT Series
- NCCT
- CTA
- CTP

## Architecture

```
src/
├── config/      # Configuration and enum definitions
├── cli/         # Command-line interface
├── dicom/       # DICOM parsing and metadata extraction
├── strategies/  # Processing strategies for series recognition
├── manager/     # Conversion orchestration
├── nifti/       # NIfTI conversion via dcm2niix
└── utils/       # Utilities (filesystem, parallel processing)
```

## Dependencies

- [dicom-rs](https://github.com/Enet4/dicom-rs) - DICOM parsing
- [dcm2niix](https://github.com/rordenlab/dcm2niix) - DICOM to NIfTI conversion
- [rayon](https://github.com/rayon-rs/rayon) - Parallel processing
- [clap](https://github.com/clap-rs/clap) - CLI parsing
- [pyo3](https://github.com/PyO3/pyo3) - Python bindings

## License

MIT
