# Change: Standardize Event Data Schema for dcop_event_bt

## Why

The `dcop_event_bt` table's `params_data` and `result_data` JSON columns currently lack structure and standards, making it extremely difficult to:

- **Query specific information**: No consistent field names across different event types
- **Debug issues**: Inconsistent data formats (dict vs list, nested structures vary)
- **Maintain code**: Developers must remember ad-hoc structures for each operation type
- **Analyze workflows**: Difficult to trace parameter flow through the medical imaging pipeline
- **Validate data**: No schema validation means corrupted or incomplete data goes undetected

This technical debt violates Martin Fowler's principle of "intentional architecture" and Linus Torvalds' emphasis on "good taste in data structures." The current state creates cognitive load and reduces code quality.

## What Changes

Introduce a structured, validated schema system for `params_data` and `result_data` that:

1. **Defines clear schemas** for each operation type (STUDY_NEW, SERIES_CONVERTING, etc.)
2. **Separates concerns** between input parameters and output results
3. **Enables querying** with consistent field names and types
4. **Validates data** before persistence to catch errors early
5. **Documents structure** so developers know what to expect

### Schema Organization

- **params_data**: Input parameters for the operation (what triggers it)
  - File paths (input sources)
  - Configuration settings
  - Identifiers (study_id, series_uid)

- **result_data**: Output results from the operation (what it produced)
  - Generated file paths (outputs)
  - Status indicators
  - Metrics (counts, sizes, durations)
  - Error information if applicable

### Breaking Changes

**BREAKING**: Existing code that directly accesses JSON fields must be updated to use schema classes. Migration path provided for backward compatibility during transition.

## Impact

- **Affected specs**: event-tracking (new capability)
- **Affected code**:
  - `backend/app/sync/model.py` - Add schema validation
  - `backend/app/sync/service.py` - Use typed schemas instead of raw dicts
  - `backend/app/study/service.py` - Update event creation calls
  - `backend/app/listen/service.py` - Update event creation calls
  - `code_ai/task/task_dicom2nii.py` - Use typed parameter schemas
  - `code_ai/task/task_pipeline.py` - Use typed parameter schemas
  - Database queries using JSON fields - Add schema-aware access patterns

## Benefits

Following clean code principles (Fowler, Martin):
- **Single Responsibility**: Each schema class handles one event type
- **Self-Documenting**: Type hints and Pydantic models make intent clear
- **Fail Fast**: Validation catches bad data at creation time, not query time
- **Testable**: Schema classes can be unit tested independently
- **Queryable**: Consistent naming enables SQL JSON queries

Following pragmatic design (Linus):
- **Simple**: One schema per operation, no over-engineering
- **Practical**: Works with existing JSON columns, no migration needed
- **Maintainable**: Changes to one event type don't affect others
- **Debuggable**: Clear structure makes issues obvious
