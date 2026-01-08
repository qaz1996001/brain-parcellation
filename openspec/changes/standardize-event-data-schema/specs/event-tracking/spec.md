# Specification: Event Tracking

## ADDED Requirements

### Requirement: Schema-Based Event Data Validation

The system SHALL validate `params_data` and `result_data` JSON fields against operation-specific schemas before persisting events to `dcop_event_bt`.

#### Scenario: Valid params_data for STUDY_TRANSFERRING

- **GIVEN** an event creation request for STUDY_TRANSFERRING operation
- **WHEN** params_data contains {sub_dir, output_dicom_path, output_nifti_path}
- **AND** all paths are absolute strings
- **THEN** the params_data SHALL be validated successfully
- **AND** the event SHALL be persisted with validated data

#### Scenario: Invalid params_data structure rejected

- **GIVEN** an event creation request for STUDY_TRANSFERRING operation
- **WHEN** params_data is missing required field "output_dicom_path"
- **THEN** validation SHALL fail with clear error message
- **AND** event creation SHALL raise ValidationError
- **AND** no database record SHALL be created

#### Scenario: Type validation for numeric fields

- **GIVEN** an event creation request for SERIES_CONVERSION_COMPLETE operation
- **WHEN** result_data contains series_count as string "5" instead of integer
- **THEN** Pydantic SHALL coerce string to integer if possible
- **AND** event SHALL be persisted with correct type

### Requirement: Operation-Specific Schemas

The system SHALL provide distinct schema classes for each operation type's params_data and result_data.

#### Scenario: Study transfer parameters schema

- **GIVEN** STUDY_TRANSFERRING operation
- **WHEN** creating params_data
- **THEN** StudyTransferParams schema SHALL require:
  - sub_dir: str (source directory path)
  - output_dicom_path: str (DICOM output path)
  - output_nifti_path: str (NIFTI output path)
  - tool_id: str (inherited from base)
  - ope_no: str (inherited from base)

#### Scenario: Study transfer result schema

- **GIVEN** STUDY_TRANSFER_COMPLETE operation
- **WHEN** creating result_data
- **THEN** StudyTransferResult schema SHALL require:
  - status: Literal["success", "error", "warning"]
  - study_id: str
  - series_count: int
  - total_files: int
  - total_size_mb: float
  - output_paths: Dict[str, str]
- **AND** MAY include optional fields:
  - skipped_series: List[str]
  - errors: List[str]
  - message: str

#### Scenario: Series conversion parameters schema

- **GIVEN** SERIES_CONVERTING operation
- **WHEN** creating params_data
- **THEN** SeriesConversionParams schema SHALL require:
  - study_uid: str
  - series_uid: str
  - output_dicom_path: str
  - output_nifti_path: str
- **AND** MAY include optional:
  - sub_dir: str

#### Scenario: Series conversion result schema

- **GIVEN** SERIES_CONVERSION_COMPLETE operation
- **WHEN** creating result_data
- **THEN** SeriesConversionResult schema SHALL include:
  - study_uid: str
  - series_uid: str
  - nifti_file_path: str (generated output)
  - series_description: str
  - modality: str
  - file_size_mb: float
  - conversion_duration_sec: float
- **AND** MAY include optional:
  - json_file_path: str (metadata)

### Requirement: Schema Registry

The system SHALL maintain a registry mapping operation codes (ope_no) to their corresponding schema classes.

#### Scenario: Lookup params schema by operation

- **GIVEN** ope_no value "100.100" (STUDY_TRANSFERRING)
- **WHEN** calling get_params_schema("100.100")
- **THEN** SHALL return StudyTransferParams class
- **AND** returned class SHALL be usable for validation

#### Scenario: Lookup result schema by operation

- **GIVEN** ope_no value "100.200" (STUDY_TRANSFER_COMPLETE)
- **WHEN** calling get_result_schema("100.200")
- **THEN** SHALL return StudyTransferResult class
- **AND** returned class SHALL be usable for validation

#### Scenario: Unknown operation code fallback

- **GIVEN** ope_no value "999.999" (unregistered)
- **WHEN** calling get_params_schema("999.999")
- **THEN** SHALL return BaseParamsData (fallback schema)
- **AND** SHALL log warning about unregistered operation
- **AND** validation SHALL use minimal base schema

### Requirement: Path Validation

The system SHALL validate that file path fields in params_data and result_data conform to absolute path requirements.

#### Scenario: Absolute path requirement

- **GIVEN** params_data with sub_dir field
- **WHEN** sub_dir value is relative path "data/study"
- **THEN** validation SHALL fail
- **AND** error message SHALL indicate "Path must be absolute"

#### Scenario: Path existence validation

- **GIVEN** params_data with output_dicom_path
- **WHEN** output_dicom_path is absolute but parent directory doesn't exist
- **THEN** validation SHALL warn (not fail)
- **AND** SHALL log warning for operational monitoring

#### Scenario: Path serialization

- **GIVEN** validated params_data with Path objects
- **WHEN** serializing to JSON for database storage
- **THEN** all Path objects SHALL convert to strings
- **AND** string representation SHALL be absolute path

### Requirement: Base Schema Inheritance

All operation-specific schemas SHALL inherit from base classes that provide common fields and behavior.

#### Scenario: Common params fields

- **GIVEN** any params schema (StudyTransferParams, SeriesConversionParams, etc.)
- **WHEN** instantiated
- **THEN** SHALL include from BaseParamsData:
  - timestamp: datetime (auto-generated)
  - tool_id: str
  - ope_no: str
- **AND** SHALL support json_encoders for Path serialization

#### Scenario: Common result fields

- **GIVEN** any result schema (StudyTransferResult, SeriesConversionResult, etc.)
- **WHEN** instantiated
- **THEN** SHALL include from BaseResultData:
  - timestamp: datetime (auto-generated)
  - status: Literal["success", "error", "warning"]
  - message: Optional[str]
- **AND** SHALL support json_encoders for Path serialization

### Requirement: Backward Compatibility

The system SHALL support reading existing events with unvalidated JSON data during transition period.

#### Scenario: Read legacy event without validation

- **GIVEN** existing dcop_event_bt record with unstructured params_data
- **WHEN** querying event via DCOPEventModel
- **THEN** SHALL return raw JSON dict without validation errors
- **AND** SHALL not attempt to coerce to schema
- **AND** application code SHALL handle both schema and raw dict

#### Scenario: New events always validated

- **GIVEN** new event creation request via DCOPEventModel.create_event_ope_no
- **WHEN** params_data or result_data provided
- **THEN** SHALL validate against registered schema
- **AND** SHALL fail fast on validation errors
- **AND** SHALL not persist invalid data

### Requirement: Consistent Query Fields

The system SHALL enforce consistent field naming across all schemas to enable reliable SQL JSON queries.

#### Scenario: Standard path field names

- **GIVEN** any schema with file path fields
- **THEN** SHALL use consistent naming:
  - Input paths: sub_dir, output_dicom_path, output_nifti_path
  - Output paths: nifti_file_path, dicom_file_path, json_file_path
- **AND** SHALL not use abbreviations (rename_dicom vs output_dicom)

#### Scenario: Standard identifier fields

- **GIVEN** any schema with entity identifiers
- **THEN** SHALL use consistent naming:
  - study_uid (not study_id when referring to UID)
  - series_uid (not series_id when referring to UID)
  - study_id (when referring to internal ID)
- **AND** types SHALL match database column types

#### Scenario: Standard metric fields

- **GIVEN** result schemas with measurements
- **THEN** SHALL use consistent units and naming:
  - Sizes: *_size_mb (megabytes as float)
  - Counts: *_count (integer)
  - Durations: *_duration_sec (seconds as float)
- **AND** SHALL include unit in field name for clarity

### Requirement: Error Handling

The system SHALL provide clear, actionable error messages when schema validation fails.

#### Scenario: Missing required field error

- **GIVEN** params_data missing "output_dicom_path"
- **WHEN** validation fails
- **THEN** error message SHALL state "Field required: output_dicom_path"
- **AND** SHALL include schema class name
- **AND** SHALL include operation code (ope_no) for context

#### Scenario: Type mismatch error

- **GIVEN** result_data with series_count as string "invalid"
- **WHEN** validation fails (cannot coerce to int)
- **THEN** error message SHALL state "Invalid integer: series_count='invalid'"
- **AND** SHALL suggest correct type

#### Scenario: Multiple validation errors

- **GIVEN** params_data with multiple invalid fields
- **WHEN** validation fails
- **THEN** SHALL return all validation errors (not just first)
- **AND** SHALL group errors by field name
- **AND** SHALL preserve order for deterministic testing

### Requirement: Schema Documentation

Each schema class SHALL include docstrings documenting its purpose, required fields, and usage examples.

#### Scenario: Schema class documentation

- **GIVEN** any schema class (e.g., StudyTransferParams)
- **WHEN** accessed via help() or IDE inspection
- **THEN** SHALL include:
  - Class-level docstring explaining purpose
  - Field-level descriptions via Field(description=...)
  - Example usage in docstring
- **AND** documentation SHALL match actual implementation

#### Scenario: Field constraints documentation

- **GIVEN** schema field with validation constraints
- **WHEN** constraint is violated
- **THEN** error message SHALL reference documented constraint
- **AND** suggestion SHALL align with field description
