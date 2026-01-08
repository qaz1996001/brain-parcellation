# Design: Event Data Schema Standardization

## Context

The medical imaging pipeline processes DICOM studies through multiple stages: transfer, conversion, inference. Each stage creates events in `dcop_event_bt` with `params_data` and `result_data` JSON fields. Currently, these fields have no enforced structure, making the data difficult to query, validate, and maintain.

### Current Pain Points

1. **Inconsistent Structures**:
   - `params_data` sometimes dict, sometimes dict with nested objects
   - `result_data` can be dict, list of dicts, or nested arbitrarily
   - No standard for path representations (strings vs Path objects)

2. **Query Complexity**:
   - SQL JSON queries fail when structure changes
   - Can't reliably extract specific fields (e.g., all output_dicom_path values)
   - Joins on JSON fields are unreliable

3. **Validation Gaps**:
   - Missing required fields discovered at runtime
   - Type errors (expecting dict, got list) cause crashes
   - No schema evolution strategy

## Goals / Non-Goals

### Goals
- Define Pydantic schemas for all event types' params_data and result_data
- Enable SQL queries on JSON fields with predictable structure
- Validate data at event creation time (fail fast)
- Support backward compatibility during migration
- Document expected structure for each operation type

### Non-Goals
- Change database schema from JSON to relational tables (future consideration)
- Retrofit all historical data (validate forward only)
- Support arbitrary JSON (must conform to schemas)
- Create generic schema (each operation has specific needs)

## Decisions

### Schema Organization

Create separate schema modules:

```
backend/app/sync/schemas/
├── __init__.py
├── params/                  # Input parameter schemas
│   ├── base.py             # Base class with common fields
│   ├── transfer.py         # STUDY_TRANSFERRING, SERIES_TRANSFERRING
│   ├── conversion.py       # STUDY_CONVERTING, SERIES_CONVERTING
│   └── inference.py        # STUDY_INFERENCE_READY
└── results/                 # Output result schemas
    ├── base.py             # Base class with common fields
    ├── transfer.py         # TRANSFER_COMPLETE results
    ├── conversion.py       # CONVERSION_COMPLETE results
    └── inference.py        # INFERENCE results
```

### Base Schema Design

**Rationale**: Martin Fowler's "intentional architecture" - shared structure should be explicit.

```python
class BaseParamsData(BaseModel):
    """Base class for all params_data schemas."""
    # All operations have these
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_id: str
    ope_no: str

    class Config:
        # Allow reading from dict or object
        from_attributes = True
        # Ser ialize paths as strings
        json_encoders = {Path: str}

class BaseResultData(BaseModel):
    """Base class for all result_data schemas."""
    # All results have these
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["success", "error", "warning"]
    message: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {Path: str}
```

### Operation-Specific Schemas

**Example: Study Transfer**

```python
class StudyTransferParams(BaseParamsData):
    """Parameters for STUDY_TRANSFERRING operation."""
    sub_dir: str  # Source directory
    output_dicom_path: str
    output_nifti_path: str

    @field_validator('sub_dir', 'output_dicom_path', 'output_nifti_path')
    @classmethod
    def validate_path_exists(cls, v: str) -> str:
        """Validate that paths are absolute and exist."""
        path = Path(v)
        if not path.is_absolute():
            raise ValueError(f"Path must be absolute: {v}")
        return str(path)

class StudyTransferResult(BaseResultData):
    """Results from STUDY_TRANSFER_COMPLETE operation."""
    study_id: str
    series_count: int
    total_files: int
    total_size_mb: float
    output_paths: Dict[str, str]  # {series_uid: path}
    skipped_series: List[str] = []  # UIDs of skipped series
    errors: List[str] = []  # Error messages if any
```

**Example: Series Conversion**

```python
class SeriesConversionParams(BaseParamsData):
    """Parameters for SERIES_CONVERTING operation."""
    study_uid: str
    series_uid: str
    output_dicom_path: str
    output_nifti_path: str
    sub_dir: Optional[str] = None

class SeriesConversionResult(BaseResultData):
    """Results from SERIES_CONVERSION_COMPLETE operation."""
    study_uid: str
    series_uid: str
    nifti_file_path: str  # Generated .nii.gz path
    json_file_path: Optional[str] = None  # Metadata JSON
    series_description: str
    modality: str
    file_size_mb: float
    conversion_duration_sec: float
```

### Schema Registration System

**Rationale**: Linus's "good taste" - mapping between ope_no and schema should be obvious.

```python
# backend/app/sync/schemas/registry.py

PARAMS_SCHEMA_REGISTRY: Dict[str, Type[BaseParamsData]] = {
    DCOPStatus.STUDY_TRANSFERRING.value: StudyTransferParams,
    DCOPStatus.SERIES_TRANSFERRING.value: SeriesTransferParams,
    DCOPStatus.STUDY_CONVERTING.value: StudyConversionParams,
    DCOPStatus.SERIES_CONVERTING.value: SeriesConversionParams,
    DCOPStatus.STUDY_INFERENCE_READY.value: StudyInferenceParams,
}

RESULT_SCHEMA_REGISTRY: Dict[str, Type[BaseResultData]] = {
    DCOPStatus.STUDY_TRANSFER_COMPLETE.value: StudyTransferResult,
    DCOPStatus.SERIES_TRANSFER_COMPLETE.value: SeriesTransferResult,
    DCOPStatus.STUDY_CONVERSION_COMPLETE.value: StudyConversionResult,
    DCOPStatus.SERIES_CONVERSION_COMPLETE.value: SeriesConversionResult,
}

def get_params_schema(ope_no: str) -> Type[BaseParamsData]:
    """Get schema class for params_data based on ope_no."""
    return PARAMS_SCHEMA_REGISTRY.get(ope_no, BaseParamsData)

def get_result_schema(ope_no: str) -> Type[BaseResultData]:
    """Get schema class for result_data based on ope_no."""
    return RESULT_SCHEMA_REGISTRY.get(ope_no, BaseResultData)
```

### Model Integration

Update `DCOPEventModel.create_event_ope_no` to validate schemas:

```python
@classmethod
async def create_event_ope_no(
    cls,
    tool_id: str,
    study_uid: str,
    series_uid: str,
    study_id: str,
    ope_no: str,
    result_data: Optional[Dict[str, Any]],
    params_data: Optional[Dict[str, Any]],
    session: Session | AsyncSession = None
):
    # Validate params_data against schema
    if params_data:
        ParamsSchema = get_params_schema(ope_no)
        validated_params = ParamsSchema(**params_data)
        params_data = validated_params.model_dump()

    # Validate result_data against schema
    if result_data:
        ResultSchema = get_result_schema(ope_no)
        validated_result = ResultSchema(**result_data)
        result_data = validated_result.model_dump()

    # Rest of existing logic...
```

## Alternatives Considered

### Alternative 1: Relational Tables

**Rejected**: Too disruptive. Would require major schema migration and breaking changes across entire system. JSON columns work fine with proper structure.

### Alternative 2: Generic Schema with Discriminator

**Rejected**: Violates "explicit is better than implicit". Each operation has specific needs; forcing them into generic structure creates awkward compromises.

### Alternative 3: No Validation

**Rejected**: Current state. Leads to runtime errors and debugging difficulty.

## Risks / Trade-offs

### Risk: Breaking Existing Queries

**Mitigation**:
- Maintain backward compatibility helper methods
- Gradual rollout with feature flag
- Comprehensive test coverage for all query patterns

### Risk: Schema Evolution

**Mitigation**:
- Pydantic supports optional fields for graceful degradation
- Version schemas if major changes needed
- Use migration pattern for structural changes

### Trade-off: Validation Overhead

**Accepted**: Small performance cost at event creation time is worth the benefit of catching errors early and enabling reliable queries.

### Trade-off: More Code

**Accepted**: Explicit schemas are self-documenting and reduce long-term maintenance burden, following Fowler's "clarity over cleverness" principle.

## Migration Plan

### Phase 1: Add Schemas (Non-Breaking)

1. Create schema modules
2. Add validation to new event creations only
3. Update tests to use schemas
4. Deploy without affecting existing code

### Phase 2: Migrate Service Layer

1. Update `sync/service.py` to use typed schemas
2. Update `study/service.py` to use typed schemas
3. Update task modules in `code_ai/task/`
4. Add integration tests

### Phase 3: Update Queries

1. Audit all JSON field queries
2. Update to use consistent field names
3. Add query helper methods
4. Remove legacy query patterns

### Rollback Strategy

If issues arise:
1. Feature flag to disable validation
2. Fall back to raw dict handling
3. Log validation errors without failing
4. Fix schemas based on real data patterns

## Open Questions

1. Should we validate on read as well as write?
   - **Decision**: Write-only for now, read validation optional

2. How to handle legacy data that doesn't conform?
   - **Decision**: Validation applies to new data only, legacy data accessed permissively

3. Should schemas enforce business rules or just structure?
   - **Decision**: Structure only; business rules stay in service layer
