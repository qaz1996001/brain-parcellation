# Specification: IEC-62304 Compliance Capability

## ADDED Requirements

### Requirement: Medical Device Software Lifecycle Support
The meta-framework SHALL support IEC-62304 medical device software lifecycle processes, including software development planning, requirements analysis, architectural design, detailed design, unit implementation and verification, integration and testing, system testing, and release.

#### Scenario: Medical Device Software Project Initialization
- **GIVEN** a new medical device software project
- **WHEN** the team initializes the meta-framework
- **THEN** IEC-62304 compliant templates SHALL be available for:
  - Software Requirements Specification (SRS)
  - Software Design Description (SDD)
  - Software Test Plan (STP)
  - SOUP (Software of Unknown Provenance) management
  - Risk management bridge to ISO 14971

### Requirement: Software Safety Classification
The meta-framework SHALL provide guidance for IEC-62304 software safety classification (Class A, B, or C) based on potential harm to patients.

#### Scenario: Safety Class Assessment
- **GIVEN** a medical device software component
- **WHEN** the team performs safety classification
- **THEN** the framework SHALL provide:
  - Classification decision tree based on harm severity
  - Documentation template for classification rationale
  - Class-specific requirement rigor guidelines
- **AND** Class A (no injury/harm) SHALL have minimum documentation requirements
- **AND** Class B (non-serious injury) SHALL have moderate documentation requirements
- **AND** Class C (death or serious injury) SHALL have maximum documentation requirements

### Requirement: SOUP Management
The meta-framework SHALL provide templates for managing Software of Unknown Provenance (third-party libraries, open-source components, COTS software).

#### Scenario: SOUP Component Evaluation
- **GIVEN** a third-party software component to be integrated
- **WHEN** the team evaluates the SOUP component
- **THEN** the SOUP template SHALL capture:
  - Component name, version, and supplier
  - Intended use within the medical device software
  - Known anomalies and their potential impact
  - Safety classification contribution
  - Verification and validation evidence
- **AND** traceability to requirements and risk controls SHALL be established

### Requirement: IEC-62304 Compliance Mapping
The meta-framework SHALL provide a compliance mapping template that demonstrates conformance to IEC-62304 requirements.

#### Scenario: Regulatory Submission Preparation
- **GIVEN** a completed medical device software project
- **WHEN** the team prepares regulatory submission documentation
- **THEN** the compliance mapping template SHALL:
  - List all IEC-62304 process requirements
  - Map each requirement to project artifacts (documents, records, evidence)
  - Identify any deviations with documented rationale
  - Support FDA 510(k), EU MDR, and PMDA submissions

### Requirement: Risk Control Traceability
The meta-framework SHALL extend traceability matrices to include risk controls per IEC-62304 requirements.

#### Scenario: Risk-to-Requirement-to-Test Traceability
- **GIVEN** a software hazard identified in risk analysis
- **WHEN** risk controls are implemented
- **THEN** the traceability matrix SHALL link:
  - Hazardous situation to risk control measure
  - Risk control measure to software requirement(s)
  - Software requirement to design element
  - Design element to implementation
  - Implementation to verification test case(s)
- **AND** the trace SHALL be bidirectional for impact analysis

## REMOVED Requirements

### Requirement: ISO 20816 Vibration Monitoring Compliance
~~The meta-framework SHALL support ISO 20816 mechanical vibration monitoring software compliance.~~

**Rationale**: Replacing vibration monitoring standard with medical device software standard to better serve medical software development teams.

### Requirement: Vibration Measurement System Templates
~~The meta-framework SHALL provide templates for vibration measurement system design, algorithm validation, and calibration procedures.~~

**Rationale**: Domain-specific to mechanical vibration monitoring, not applicable to medical device software.

### Requirement: ISO 20816 Equipment Classification
~~The meta-framework SHALL support ISO 20816 equipment classification (Group 1-4) based on machine power and speed.~~

**Rationale**: Replaced by IEC-62304 software safety classification (Class A/B/C) which is more appropriate for software systems.

## MODIFIED Requirements

### Requirement: Standards Compliance Framework (Updated)
The meta-framework SHALL support multiple international standards for software development ~~, including ISO/IEC/IEEE 29148 (requirements engineering) and ISO 20816 (mechanical vibration monitoring)~~.

**NEW**: The framework SHALL now support:
- ISO/IEC/IEEE 29148:2018 (requirements engineering) - retained
- IEC-62304:2006+A1:2015 (medical device software lifecycle) - added
- RFC 2119 (requirement keywords) - added
- ISO/IEC 27001:2022 (information security) - added
- ISO 9001:2015 (quality management) - added as optional

#### Scenario: Standard Selection for Medical Device Project
- **GIVEN** a new medical device software project
- **WHEN** the team selects applicable standards
- **THEN** the framework SHALL recommend:
  - IEC-62304 (mandatory for medical device software)
  - ISO 29148 (optional but recommended for requirements quality)
  - RFC 2119 (recommended for requirement precision)
  - ISO 27001 (mandatory if handling patient data)
  - ISO 14971 (mandatory for risk management, referenced but not templated in detail)
  - ISO 9001 (optional for QMS integration)

### Requirement: Traceability Matrix Structure (Extended)
~~The meta-framework SHALL provide traceability matrix templates linking requirements to design, implementation, and test cases.~~

**UPDATED**: The traceability matrix SHALL now additionally link:
- Risk controls to software requirements (IEC-62304 requirement)
- SOUP components to requirements and risks
- Security controls to requirements (ISO 27001)
- Quality process to requirements (ISO 9001, if applicable)

#### Scenario: Medical Device Traceability Audit
- **GIVEN** a regulatory audit of medical device software
- **WHEN** auditors request traceability evidence
- **THEN** the extended traceability matrix SHALL demonstrate:
  - All hazards have identified risk controls
  - All risk controls are implemented in software requirements
  - All software requirements are designed, coded, and tested
  - All SOUP components are evaluated and integrated correctly
  - All security requirements are implemented and verified

## Dependencies
- Requires RFC 2119 keyword integration for requirement precision
- Requires ISO 27001 security templates for patient data protection
- References ISO 14971 for risk management (bridge template provided)

## Validation
- All IEC-62304 templates SHALL be validated against the standard's requirements
- Compliance mapping template SHALL cover all applicable IEC-62304 clauses
- SOUP management template SHALL support evaluation of common medical software dependencies (databases, web frameworks, cryptography libraries)
- Traceability examples SHALL demonstrate complete risk-to-test chains
