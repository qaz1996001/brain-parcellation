# Change: Fix Type and Lint Errors in Backend Modules

## Why

The backend modules (`listen`, `rerun`, `series`, `services`, `study`, `sync`) contain type checking errors (ty) and linting issues (ruff) that prevent clean builds. These issues include:
- Missing imports (`os` not imported)
- Unused imports
- Incorrect type annotations (Optional types not handled)
- Deprecated API usage (`datetime.utcnow`)
- Invalid type forms in dynamic code
- Incorrect argument types for method calls

Fixing these ensures code quality, maintainability, and prevents runtime errors.

## What Changes

### listen module (8 errors)
- Fix `OrthancID` type alias usage in schemas.py (cannot call `Annotated` type directly)
- Fix missing URL imports in service.py (import from `backend.app.sync.urls` instead of `.urls`)
- Fix missing `DCOPEventModel` import in service.py
- Fix httpx `post()` data argument type

### rerun module (2 errors)
- Fix `Column[str]` to `str` type conversion for `study_uid`
- Fix httpx `post()` data argument type

### series module (7 errors)
- Add null checks before `len()` calls on Optional lists
- Fix iteration over Optional types
- Fix `file_name` Optional type handling

### services module (4 errors)
- Add null checks for Optional `study_id`, `result_data`, `params_data`
- Fix Path.joinpath argument type

### study module (9+ errors)
- Fix dynamic type form issues in deps.py (suppress with type: ignore)
- Add missing `os` import in service.py

### sync module (9 warnings + 3 errors)
- Replace deprecated `datetime.utcnow` with `datetime.now(timezone.utc)`
- Remove unused imports (`Generator`, `Union`)

## Impact

- Affected code: `backend/app/{listen,rerun,series,services,study,sync}/`
- No behavioral changes - all fixes are type safety and lint compliance
- No API changes
- No database schema changes
