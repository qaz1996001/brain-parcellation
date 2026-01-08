## ADDED Requirements

### Requirement: Repository Layer Database Operations

The `DCOPEventDicomService.Repo` class SHALL encapsulate all database query logic, providing semantic methods for data access.

#### Scenario: Retrieve studies ready for transfer
- **WHEN** the service needs to check studies ready for transfer
- **THEN** it SHALL call `Repo.get_studies_ready_for_transfer()` instead of executing raw SQL
- **AND** the repository method SHALL return typed results

#### Scenario: Query series by status
- **WHEN** the service needs to query series with a specific status
- **THEN** it SHALL call `Repo.get_series_by_status(status, study_uid=None)`
- **AND** the method SHALL support optional filtering by `study_uid`
- **AND** the method SHALL return `List[Row]` with typed access to result columns

#### Scenario: Query pending completion studies
- **WHEN** the service needs to find studies pending completion
- **THEN** it SHALL call `Repo.get_studies_pending_completion(status)`
- **AND** the repository SHALL encapsulate the complex subquery logic


### Requirement: Batch Database Operations

The service layer SHALL use batch operations for multiple record creation instead of commit-per-iteration patterns.

#### Scenario: Create multiple DCOP events
- **WHEN** `post_ope_no_task()` receives a list of `DCOPEventRequest` objects
- **THEN** it SHALL collect all records and call `create_many()` once
- **AND** it SHALL NOT call `session.commit()` inside a loop

#### Scenario: Add multiple study records
- **WHEN** `add_study_new()` creates STUDY_NEW and STUDY_TRANSFERRING records
- **THEN** it SHALL batch all records and commit once
- **AND** database round-trips SHALL be O(1) not O(n)


### Requirement: Status URL Mapping Data Structure

The service SHALL use a declarative data structure for status-to-URL mappings following Linus's "good data structure" principle.

#### Scenario: Status URL lookup
- **WHEN** `get_check_url_by_ope_no()` is called with a status value
- **THEN** it SHALL look up the endpoint from `STATUS_URL_MAP` dictionary
- **AND** it SHALL return `None` for unmapped statuses
- **AND** it SHALL NOT use procedural `match`/`case` statements for this mapping


## MODIFIED Requirements

### Requirement: Service Layer Session Management

The service layer SHALL NOT directly manipulate SQLAlchemy Session objects; all database operations SHALL go through Repository methods.

#### Scenario: Record creation
- **WHEN** the service needs to create a database record
- **THEN** it SHALL call `await self.create(data=..., auto_commit=True)`
- **AND** it SHALL NOT call `session.add()` or `session.commit()` directly

#### Scenario: Batch record creation
- **WHEN** the service needs to create multiple records
- **THEN** it SHALL call `await self.create_many(data=..., auto_commit=True)`
- **AND** it SHALL NOT use `session.add_all()` with manual commit

#### Scenario: Record query with pagination
- **WHEN** the service needs paginated query results
- **THEN** it SHALL call `await self.list_and_count(...)` for single-query pagination
- **AND** it SHALL NOT execute separate count and data queries


### Requirement: Unified Session Context Management

The service SHALL use a consistent pattern for session context management across all methods.

#### Scenario: Session acquisition
- **WHEN** a service method needs database access
- **THEN** it SHALL use `async with self.session_manager.get_session() as session:`
- **AND** it SHALL NOT accept `session: AsyncSession` as a method parameter
- **AND** it SHALL NOT access `self.repository.session` directly


## REMOVED Requirements

### Requirement: Raw SQL in Service Layer

**Reason**: Raw SQL strings in service methods violate Repository pattern separation of concerns.

**Migration**: All `text("SELECT...")` statements moved to Repository custom methods with proper encapsulation and type safety.

#### Scenario: Legacy raw SQL queries
- **WHEN** `text()` SQL was previously used in service methods
- **THEN** it SHALL be replaced with Repository method calls
- **AND** the Repository SHALL own all SQL query definitions
