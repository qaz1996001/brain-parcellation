# Specification: RFC 2119 Requirement Keywords

## ADDED Requirements

### Requirement: RFC 2119 Keyword Usage
The meta-framework SHALL adopt RFC 2119 standardized keywords for requirement specification to eliminate ambiguity and establish clear conformance levels.

#### Scenario: Writing an Absolute Requirement
- **GIVEN** a critical functional requirement
- **WHEN** writing the requirement specification
- **THEN** the SHALL or MUST keyword SHALL be used to indicate absolute necessity
- **AND** implementation SHALL be mandatory for conformance
- **EXAMPLE**: "The system SHALL encrypt all patient data at rest using AES-256."

#### Scenario: Writing a Recommended Requirement
- **GIVEN** a strongly recommended but not critical requirement
- **WHEN** writing the requirement specification
- **THEN** the SHOULD keyword SHALL be used
- **AND** deviation SHALL require documented justification
- **EXAMPLE**: "The system SHOULD log all security-relevant events to a centralized logging service."

#### Scenario: Writing an Optional Requirement
- **GIVEN** an optional feature or capability
- **WHEN** writing the requirement specification
- **THEN** the MAY or COULD keyword SHALL be used
- **AND** implementation SHALL be at the discretion of the implementer
- **EXAMPLE**: "The system MAY provide a dark mode user interface theme."

### Requirement: RFC 2119 Prohibition Keywords
The meta-framework SHALL support RFC 2119 prohibition keywords (MUST NOT, SHALL NOT) for explicitly forbidden behaviors.

#### Scenario: Specifying Prohibited Behavior
- **GIVEN** a critical safety or security constraint
- **WHEN** writing the constraint specification
- **THEN** the MUST NOT or SHALL NOT keyword SHALL be used
- **AND** violation SHALL result in non-conformance
- **EXAMPLE**: "The system SHALL NOT store patient passwords in plaintext."

### Requirement: Keyword Consistency Guidelines
The meta-framework SHALL provide guidelines for consistent RFC 2119 keyword usage across all requirements documentation.

#### Scenario: Requirement Review for Keyword Consistency
- **GIVEN** a requirements document under review
- **WHEN** checking keyword usage
- **THEN** the following rules SHALL apply:
  - SHALL, MUST: Use for critical, mandatory requirements
  - SHOULD, RECOMMENDED: Use for strong recommendations
  - MAY, OPTIONAL, COULD: Use for truly optional features
  - MUST NOT, SHALL NOT: Use for absolute prohibitions
  - SHOULD NOT, NOT RECOMMENDED: Use for discouraged but not forbidden behaviors
- **AND** weak language (e.g., "try to", "as much as possible") SHALL be avoided
- **AND** ambiguous language (e.g., "normally", "usually") SHALL be replaced with RFC 2119 keywords

### Requirement: RFC 2119 Integration in Templates
All meta-framework requirement templates SHALL include RFC 2119 keyword usage examples and guidelines.

#### Scenario: Using System Requirements Template
- **GIVEN** a new project using `TEMPLATE_SYSTEM_PRD_SR_SD.md`
- **WHEN** writing system requirements
- **THEN** the template SHALL:
  - Provide RFC 2119 keyword reference section
  - Include properly formatted examples using SHALL, SHOULD, MAY
  - Explain the conformance implications of each keyword
  - Provide guidance on selecting appropriate keywords for different requirement types (functional, performance, security, usability)

### Requirement: RFC 2119 Validation Support
The meta-framework SHALL provide guidance for validating requirements against RFC 2119 keyword usage best practices.

#### Scenario: Automated Requirement Quality Check
- **GIVEN** a requirements document
- **WHEN** performing quality validation
- **THEN** validation guidelines SHALL check for:
  - Presence of RFC 2119 keywords in all normative requirements
  - Absence of ambiguous language ("should be", "needs to", "must be able to")
  - Consistency between requirement criticality and keyword choice
  - Proper capitalization of keywords for normative use
- **AND** warnings SHALL be provided for weak or ambiguous phrasing

## MODIFIED Requirements

### Requirement: Requirement Writing Guidelines (Enhanced)
~~The meta-framework SHALL provide requirement writing guidelines based on SMART principles (Specific, Measurable, Achievable, Relevant, Testable).~~

**UPDATED**: The requirement writing guidelines SHALL now additionally incorporate:
- RFC 2119 keyword selection criteria
- Mapping from requirement type to appropriate keyword
- Examples of converting ambiguous requirements to RFC 2119 format
- Common mistakes and how to avoid them

#### Scenario: Converting Ambiguous to Precise Requirements
- **GIVEN** an ambiguous requirement like "The system should probably validate user input"
- **WHEN** applying RFC 2119 guidelines
- **THEN** the requirement SHALL be rewritten as:
  - "The system SHALL validate all user input against defined schemas" (if mandatory)
  - OR "The system SHOULD validate user input to prevent injection attacks" (if recommended)
  - OR "The system MAY provide additional input validation hints to users" (if optional)

### Requirement: AI Collaboration Patterns (Updated)
~~The meta-framework SHALL guide AI agents to write clear, testable requirements following SMART principles.~~

**UPDATED**: AI agents SHALL additionally:
- Use RFC 2119 keywords in all generated requirements
- Select appropriate keywords based on requirement criticality
- Avoid ambiguous language in requirement specifications
- Validate requirement phrasing for RFC 2119 compliance

#### Scenario: AI-Generated Requirements Review
- **GIVEN** requirements generated by an AI agent
- **WHEN** reviewing for quality
- **THEN** all requirements SHALL use RFC 2119 keywords
- **AND** keyword choice SHALL match the stated requirement criticality
- **AND** no ambiguous language SHALL be present

## Dependencies
- Integrates with all requirement templates in the meta-framework
- Supports IEC-62304 requirement rigor (Class A/B/C keyword selection)
- Enhances ISO 29148 requirement quality attributes

## Validation
- All requirement examples SHALL use RFC 2119 keywords correctly
- Templates SHALL provide clear guidance on keyword selection
- AI collaboration patterns SHALL enforce RFC 2119 usage
- Migration guide SHALL show before/after examples of requirement refinement
