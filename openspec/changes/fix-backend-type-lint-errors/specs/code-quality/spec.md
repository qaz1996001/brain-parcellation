# Code Quality: Type Safety and Lint Compliance

## ADDED Requirements

### Requirement: Backend Module Type Safety

All backend modules (`listen`, `rerun`, `series`, `services`, `study`, `sync`) SHALL pass type checking with `ty check` without errors.

#### Scenario: Type check passes for all modules
- **WHEN** running `uvx ty check backend/app/{listen,rerun,series,services,study,sync}/`
- **THEN** the command exits with 0 errors (warnings acceptable)

### Requirement: Backend Module Lint Compliance

All backend modules SHALL pass linting with `ruff check` without errors.

#### Scenario: Lint check passes for all modules
- **WHEN** running `uvx ruff check backend/app/{listen,rerun,series,services,study,sync}/`
- **THEN** the command exits with 0 errors

### Requirement: Optional Type Handling

Functions receiving Optional parameters SHALL handle None cases explicitly before use.

#### Scenario: None guard before list operations
- **WHEN** a function receives an `Optional[list]` parameter
- **AND** the function performs `len()`, subscript `[]`, or iteration on the list
- **THEN** the function SHALL check for None before the operation

### Requirement: Deprecated API Avoidance

Code SHALL NOT use deprecated standard library APIs.

#### Scenario: datetime.utcnow replacement
- **WHEN** code needs current UTC time
- **THEN** code SHALL use `datetime.now(timezone.utc)` instead of deprecated `datetime.utcnow()`
